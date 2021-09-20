import os
#os.environ['TRANSFORMERS_CACHE'] = '/data/qin/cache/'
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import gc
gc.enable()
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

from seqeval.metrics import classification_report,f1_score
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
import pickle
from model import *
from dataset import *
from utils import *
from datasets import load_metric

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def get_dataloader(num_workers,dataset, batch_size, max_len, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_len,
        pad_token_id=pad_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        #shuffle=True, #####?????
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def train(args, model, train_dataset,valid_dataset,grads_means, grads_fishers, ewc_lambda, onerun):
    # total step
    step_tot = (len(
        train_dataset) // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch
    warmup_steps_total = step_tot * args.warmup_steps
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length,
                                      train_dataset.tokenizer.pad_token_id,train_sampler)
    valid_dataloader = get_dataloader(args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length,
                                      valid_dataset.tokenizer.pad_token_id,valid_sampler)

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    base_optimizer_arguments = {"lr":args.lr, "eps":args.adam_epsilon, "correct_bias":False}
    optimizer = AdamW
    optimizer = OSS(
        params=optimizer_grouped_parameters,
        optim=optimizer,
        **base_optimizer_arguments)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps_total,
                                                num_training_steps=step_tot)
    # distributed training
    model = ShardedDDP(model, optimizer)
    model.train()
    scaler = ShardedGradScaler()
    # scaler = None

    startepoch = 0
    Best_F1 = 0.0

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_rouge1': [],
        'best_val_rouge1': Best_F1
    }
    global_step = 0
    #lm_lambda = 0.5
    for i in range(startepoch, startepoch + args.max_epoch):
        if i < 32:
            thisevalstep = args.eval_step * 10
        elif i >= 32 and i < 42:
            thisevalstep = args.eval_step * 4
        else:
            thisevalstep = args.eval_step
        if i > 4:
        #if i > 32:
            break
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        allewcloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    loss = model(inputs)
            else:
                loss = model(inputs)
            finalloss = loss
            ewcloss = torch.tensor(0.0).to(args.device)
            for gg in range(len(grads_means)):
                one_grad_mean = grads_means[gg]
                one_grad_fisher = grads_fishers[gg]
                ewcloss += param_loss(model, one_grad_mean, one_grad_fisher, ewc_lambda, args)
            finalloss += ewcloss
            if scaler is not None:
                scaler.scale(finalloss).backward()
            else:
                finalloss.backward()
            allloss.append(loss.item())
            allewcloss.append(ewcloss.item())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    optimizer.clip_grad_norm(args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    #logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (global_step, global_step/step_tot, np.average(allloss)))
                    # logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (
                    #     global_step, global_step / step_tot, np.average(allloss)))
                    logger.info("step: %d, shcedule: %.3f, loss: %.6f, ewcloss: %.6f" % (
                        global_step, global_step / step_tot, np.average(allloss), np.average(allewcloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    #####eval
                    #dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i)
                    #print("only eval every epoch")
                    print("not eval!!!")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            if i >= 2:
            #if i >= 32:
                dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun)
                model.train()

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")

    torch.cuda.empty_cache()
    del optimizer, scheduler, scaler, train_dataloader, valid_dataloader,
    gc.collect()

def dooneeval(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    logger.info("Do one eval!")
    allytrue = []
    allypred = []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    logger.info("valid_rouge1: %f", rouge_score["rouge1"].mid.fmeasure)
    logger.info("valid_rouge2: %f", rouge_score["rouge2"].mid.fmeasure)
    logger.info("valid_rougeL: %f", rouge_score["rougeL"].mid.fmeasure)

    result_dict['val_rouge1'].append(rouge_score["rouge1"].mid.fmeasure)
    if result_dict['val_rouge1'][-1] > result_dict['best_val_rouge1']:
        logger.info("{} epoch, best epoch was updated! val_rouge1: {: >4.5f}".format(i, result_dict['val_rouge1'][-1]))
        result_dict["best_val_rouge1"] = result_dict['val_rouge1'][-1]
        if not os.path.exists(args.tosavepath):
            os.mkdir(args.tosavepath)
        if not os.path.exists(args.tosavepath + "/" + args.taskfold):
            os.mkdir(args.tosavepath + "/" + args.taskfold)
        if not os.path.exists(args.tosavepath + "/" + args.taskfold + "/" + str(onerun)):
            os.mkdir(args.tosavepath + "/" + args.taskfold + "/" + str(onerun))
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            't5-large-sum': model_to_save.model.state_dict()
        }
        torch.save(ckpt, os.path.join(args.tosavepath + "/" + args.taskfold + "/" + str(onerun), "bestckpt"))

def test(args, test_dataset, onerun):

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length,
                                      test_dataset.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
    model = T5forSummarization(args, t5model, tokenizer)
    allckpt = torch.load(args.tosavepath +  "/" + args.taskfold + "/" + str(onerun) + "/bestckpt")
    model.model.load_state_dict(allckpt['t5-large-sum'])
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    scaler = ShardedGradScaler()
    #scaler = None
    allytrue = []
    allypred = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen,target,preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Test Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    logger.info("test_rouge1: %f", rouge_score["rouge1"].mid.fmeasure)
    logger.info("test_rouge2: %f", rouge_score["rouge2"].mid.fmeasure)
    logger.info("test_rougeL: %f", rouge_score["rougeL"].mid.fmeasure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--startindex", dest="startindex", type=int,
                        default=0, help="start index")
    parser.add_argument("--taskindex", dest="taskindex", type=int,
                        default=0, help="task index")
    parser.add_argument("--ewc_lambda", dest="ewc_lambda", type=float,
                        default=1.0, help='ewc loss lambda')

    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                        default=16, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                        default=24, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                        default=24, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers", dest="num_workers", type=int,
                        default=4, help="dataloader num_workers")

    parser.add_argument("--save_step", dest="save_step", type=int,
                        default=100000, help="step to save")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=100, help="how many steps to eval")

    parser.add_argument("--tosavepath", dest="tosavepath", type=str,
                        default="t5_sum_ckpt", help="ckpt dir to save")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")


    parser.add_argument("--model", dest="model", type=str,
                        default="T5NER", help="{T5NER}")
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="t5-base", help="{t5-base,google/t5-v1_1-base}")
    parser.add_argument("--train_file_name", dest="train_file_name", type=str,
                        default="data_conll/newtrain.txt", help="train data file path")
    parser.add_argument("--valid_file_name", dest="valid_file_name", type=str,
                        default="data_conll/newvalid.txt", help="valid data file path")
    parser.add_argument("--test_file_name", dest="test_file_name", type=str,
                        default="data_conll/newtest.txt", help="test data file path")
    parser.add_argument("--train_sample", action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=128, help="max sentence length")

    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=float,
                        default=0.1, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1.0, help="max grad norm")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")

    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="../t5_ckpt_1_0622_bak/t5_ckpt/ckpt_of_step_100000",
                        help="The path of lm_adapted model")
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=100, help="The number of prompt")
    parser.add_argument("--ifckpt_onlymodel", dest="ifckpt_onlymodel", type=int,
                        default=1, help="If ckpt only contains model. Default: True, only contains model")
    args = parser.parse_args()

    # print args
    print(args)
    # set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    initialseed = args.seed
    seed_everything(args)

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        with open("./log/trainner_log", 'a+') as f:
            f.write(str(time.ctime()) + "\n")
            f.write(str(args) + "\n")
            f.write("----------------------------------------------------------------------------\n")

    #runtimes = 3
    runtimes = 3
    alltaskfold = ["cnndm", "wikihow", "xsum"]
    alltaskname = ["cnn daily mail ", "wiki how", "extreme summarization"]
    allgentasktoken = ["summerizationcnndm", "summerizationwikihow", "summerizationxsum"]
    tasknum = len(alltaskfold)
    dataprefix = "../data/"
    fewshotnum = 16 #####use which num??
    if args.local_rank != -1:
        torch.distributed.barrier()
    ewc_lambda = args.ewc_lambda
    startindex = args.startindex
    for onerun in range(startindex, startindex + 1):
    # startindex = 0
    # for onerun in range(startindex, runtimes):
        logger.info(onerun)
        args.seed = initialseed + onerun * 100
        seed_everything(args)
        logger.info("new seed %s", args.seed)
        allindex = [i for i in range(tasknum)]
        #print(allindex)
        random.shuffle(allindex)
        #logger.info(allindex)
        newtaskname = [alltaskname[i] for i in allindex]
        print(newtaskname)
        newtaskfold = [alltaskfold[i] for i in allindex]
        print(newtaskfold)
        newtgentasktokens = [allgentasktoken[i] for i in allindex]
        print(newtgentasktokens)
        ######first,handle full data to get few shot data
        getnewfew = False
        #getnewfew = True
        if getnewfew:
            for j in range(len(newtaskfold)):
                onefold = newtaskfold[j]
                if not os.path.exists(dataprefix+onefold+"/"+str(onerun)+"_"+str(args.seed)):
                    os.mkdir(dataprefix+onefold+"/"+str(onerun)+"_"+str(args.seed))
                thispath = dataprefix+onefold+"/"+str(onerun)+"_"+str(args.seed)
                logger.info(thispath)
                #######handle full to get fewshot
                getfewshot(dataprefix+onefold,thispath,fewshotnum)
        #continue
        ####after get data, we run the model sequentially to get the performance of every single model
        globaltokenizer = None
        newfilefolder = "newdata"
        if not os.path.exists(newfilefolder):
            os.mkdir(newfilefolder)
        grads_means = []
        grads_fishers = []
        tostart = 0
        toend = args.taskindex + 1
        #for j in range(tostart,len(newtaskname)):
        for j in range(tostart, toend):
            thistaskname = newtaskname[j]
            thistaskfold = newtaskfold[j]
            args.taskfold = thistaskfold
            thistrainfilename = dataprefix + thistaskfold +"/" +str(onerun)+"_"+str(args.seed) + "/train.txt"
            thisvalidfilename = dataprefix + thistaskfold +"/" +str(onerun)+"_"+str(args.seed) + "/valid.txt"
            #thistestfilename = dataprefix + thistaskfold +"/" +str(onerun)+"_"+str(args.seed) + "/test.txt"
            if not os.path.exists(newfilefolder + "/" + thistaskfold):
                os.mkdir(newfilefolder + "/" + thistaskfold)

            allpretest = []
            for kk in range(0, j + 1):
                onepretaskfold = newtaskfold[kk]
                onepretestfile = dataprefix + onepretaskfold + "/" + str(onerun) + "_" + str(args.seed) + "/test.txt"
                allpretest.append(onepretestfile)

            newtestfile = newfilefolder + "/" + thistaskfold + "/" + "test.txt"
            f = open(newtestfile, 'w')
            for aa in range(len(allpretest)):
                oneprefile = allpretest[aa]
                for line in open(oneprefile, 'r'):
                    f.write(line)
            f.close()

            args.train_file_name = thistrainfilename
            args.valid_file_name = thisvalidfilename
            args.test_file_name = newtestfile
            t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
            # print(t5model.get_input_embeddings().weight[2040])
            tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
            #print(len(tokenizer))
            gentasktoken = newtgentasktokens[j]
            answertoken = "__ans__"
            tokenizer.add_tokens(gentasktoken)
            #print(len(tokenizer))
            logger.info(
                'gen token = {} , gen token id = {}'.format(gentasktoken,
                                                            tokenizer.convert_tokens_to_ids(gentasktoken)))
            special_tokens = {"ans_token": answertoken}
            tokenizer.add_tokens(list(special_tokens.values()))
            special_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()}
            globaltokenizer = tokenizer
            #print(len(tokenizer))
            #print(special_token_ids)
            model = T5forSummarization(args, t5model, tokenizer)
            if j == 0:
                #######just use t5
                logger.info("We just use the t5 ckpt of google")
            else:
                ######load t5 ckpt
                logger.info("use t5 ckpt of the previous task")
                ###the ckpt path of previous model
                #allckpt = torch.load(args.tosavepath +  "/" + args.taskfold + "/bestckpt")
                logger.info("Previous ckpt fold name: %s", newtaskfold[j - 1])
                t5modeltckpt = torch.load(args.tosavepath +  "/" + newtaskfold[j - 1] + "/" + str(onerun) + "/bestckpt")
                print("load previous ckpt!!!")
                model.model.load_state_dict(t5modeltckpt['t5-large-sum'])
            ###put to gpu
            model.to(args.device)

            train_dataset = T5SummarizationDataset(args.train_file_name, args.max_length, tokenizer, gentasktoken, answertoken)
            valid_dataset = T5SummarizationDataset(args.valid_file_name, args.max_length, tokenizer, gentasktoken, answertoken)
            test_dataset = T5SummarizationDataset(args.test_file_name, args.max_length, tokenizer, gentasktoken, answertoken)

            logger.info("Finish prepare model and dataset")
            logger.info("Start training")
            # if args.local_rank != -1:
            #     torch.distributed.init_process_group(backend="nccl")
            #     torch.distributed.barrier()
            if j >= toend - 1:
                train(args, model, train_dataset, valid_dataset, grads_means, grads_fishers, ewc_lambda, onerun)
            logger.info("Finish training")

            logger.info("Calculate Fisher of EWC")
            t5modeltckptforewc = torch.load(args.tosavepath + "/" + newtaskfold[j] + "/" + str(onerun) + "/bestckpt")
            logger.info("load ckpt for ewc!!!")
            model.model.load_state_dict(t5modeltckptforewc['t5-large-sum'])
            ewc_sampler = SequentialSampler(train_dataset)
            ewc_dataloader = get_dataloader(args.num_workers, train_dataset, 1, args.max_length,
                                            train_dataset.tokenizer.pad_token_id, ewc_sampler)
            grad_params = get_grad_params(model)
            grad_mean = copy_param_data(grad_params)
            grad_fisher = gen_fisher(model, args, ewc_dataloader)
            grads_means.append(grad_mean)
            grads_fishers.append(grad_fisher)
            logger.info("Finish calculate Fisher")
            torch.cuda.empty_cache()
            del model
            gc.collect()
            if j >= toend - 1:
                if args.local_rank in [0, -1]:
                    logger.info("Start testing")
                    logger.info("Testing...")
                    test(args, test_dataset, onerun)
                    logger.info("Finish testing!")
            torch.cuda.empty_cache()
            del tokenizer
            gc.collect()
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()








