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

def train(args, model, train_dataset, valid_ner, valid_class, valid_sum, onerun):
    # total step
    step_tot = (len(
        train_dataset) // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch
    warmup_steps_total = step_tot * args.warmup_steps
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)
    valid_ner_sampler = SequentialSampler(valid_ner)
    valid_class_sampler = SequentialSampler(valid_class)
    valid_sum_sampler = SequentialSampler(valid_sum)

    train_dataloader = get_dataloader(args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length,
                                      train_dataset.tokenizer.pad_token_id,train_sampler)
    valid_ner_dataloader = get_dataloader(args.num_workers, valid_ner, args.valid_size_per_gpu, args.max_length,
                                      valid_ner.tokenizer.pad_token_id,valid_ner_sampler)
    valid_class_dataloader = get_dataloader(args.num_workers, valid_class, args.valid_size_per_gpu, args.max_length,
                                          valid_class.tokenizer.pad_token_id, valid_class_sampler)
    valid_sum_dataloader = get_dataloader(args.num_workers, valid_sum, args.valid_size_per_gpu, args.max_length,
                                            valid_sum.tokenizer.pad_token_id, valid_sum_sampler)

    base_optimizer_arguments = {"lr": args.lr, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                "weight_decay": args.weight_decay,
                                "scale_parameter": False, "relative_step": False}
    optimizer = Adafactor
    optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                    **base_optimizer_arguments)
    # distributed training
    model = ShardedDDP(model, optimizer)
    model.train()
    scaler = ShardedGradScaler()
    scheduler = None

    startepoch = 0
    Best_Sum = 0.0
    Best_acc = 0.0
    Best_F1 = 0.0

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_rouge1': [],
        'best_val_rouge1': Best_Sum,
        'acc': [],
        'best_acc': Best_acc,
        'val_F1': [],
        'best_val_F1': Best_F1
    }
    global_step = 0
    #lm_lambda = 0.5
    lm_lambda = args.lm_lambda
    for i in range(startepoch, startepoch + args.max_epoch):
        #if i > 1:
        #    break
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        alllmloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            inputs_lm = {"input_ids": batch[4].to(args.device), "attention_mask": batch[5].to(args.device),
                         "target_ids": batch[6].to(args.device), "target_mask": batch[7].to(args.device)}
            if scaler is not None:
                with autocast():
                    loss = model(inputs)
                    lmloss = model(inputs_lm) * lm_lambda
                    # if args.gradient_accumulation_steps > 1:
                    #     loss = loss / args.gradient_accumulation_steps
                    #     lmloss = lmloss / args.gradient_accumulation_steps
            else:
                loss = model(inputs)
                lmloss = model(inputs_lm) * lm_lambda
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps
                #     lmloss = lmloss / args.gradient_accumulation_steps
            finalloss = loss + lmloss
            if scaler is not None:
                scaler.scale(finalloss).backward()
            else:
                finalloss.backward()
            allloss.append(loss.item())
            alllmloss.append(lmloss.item())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    #scaler.unscale_(optimizer)
                    #optimizer.clip_grad_norm(args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    #logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (global_step, global_step/step_tot, np.average(allloss)))
                    logger.info("step: %d, shcedule: %.3f, loss: %.6f, lmloss: %.6f" % (
                        global_step, global_step / step_tot, np.average(allloss), np.average(alllmloss)))

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            if i >= 0:
                if i >= 128:
                    dooneevalforner(model,valid_ner_dataloader,args,result_dict,optimizer,scaler,i,onerun)
                if i < 128:
                    dooneevalforclass(model,valid_class_dataloader,args,result_dict,optimizer,scaler,i,onerun)
                if i < 80:
                    dooneevalforsum(model, valid_sum_dataloader, args, result_dict, optimizer, scaler, i, onerun)
                model.train()

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")

def dooneevalforner(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    logger.info("Do one eval for ner!")
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
                    tarres, predres = getonebatchresult(sen, target, preds)
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = getonebatchresult(sen, target, preds)
                allytrue.extend(tarres)
                allypred.extend(predres)
    f1score = f1_score(allytrue, allypred)
    logger.info('----NER Conll Validation Results Summary----')
    logger.info(len(allypred))
    logger.info("F1: %f",f1score)

    result_dict['val_F1'].append(f1score)
    if result_dict['val_F1'][-1] > result_dict['best_val_F1']:
        logger.info("{} epoch, best ner epoch was updated! valid_F1: {: >4.5f}".format(i,result_dict['val_F1'][-1]))
        result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        if not os.path.exists(args.tosavepath):
            os.mkdir(args.tosavepath)
        if not os.path.exists(args.tosavepath + "/" + "NER"):
            os.mkdir(args.tosavepath + "/" + "NER")
        if not os.path.exists(args.tosavepath + "/" + "NER" + "/" + str(onerun)):
            os.mkdir(args.tosavepath + "/" + "NER" + "/" + str(onerun))
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(ckpt, os.path.join(args.tosavepath + "/" + 'NER' + "/" + str(onerun), "bestckpt"))

def dooneevalforclass(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    logger.info("Do one eval for class!")
    allnum = 0
    correctnum = 0
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    thisbatchnum = len(sen)
                    for k in range(thisbatchnum):
                        allnum += 1
                        if target[k] == preds[k]:
                            correctnum += 1
            else:
                sen, target, preds = model._generative_step(inputs)
                thisbatchnum = len(sen)
                for k in range(thisbatchnum):
                    allnum += 1
                    if target[k].lower() == preds[k].lower():
                        correctnum += 1
    accuracy = float(correctnum) / float(allnum)
    logger.info('----Classification ag news Validation Results Summary----')
    logger.info("allnum: %d", allnum)
    logger.info("correctnum: %d",correctnum)
    logger.info("Accuray: %f",accuracy)

    result_dict['acc'].append(accuracy)
    if result_dict['acc'][-1] > result_dict['best_acc']:
        logger.info("{} epoch, best class epoch was updated! valid_acc: {: >4.5f}".format(i, result_dict['acc'][-1]))
        result_dict["best_acc"] = result_dict['acc'][-1]
        if not os.path.exists(args.tosavepath):
            os.mkdir(args.tosavepath)
        if not os.path.exists(args.tosavepath + "/" + "Class"):
            os.mkdir(args.tosavepath + "/" + "Class")
        if not os.path.exists(args.tosavepath + "/" + "Class" + "/" + str(onerun)):
            os.mkdir(args.tosavepath + "/" + "Class" + "/" + str(onerun))
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(ckpt, os.path.join(args.tosavepath + "/" + "Class" + "/" + str(onerun), "bestckpt"))

def dooneevalforsum(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    logger.info("Do one eval for sum!")
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
    logger.info('----Summarization Validation Results Summary----')
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
        if not os.path.exists(args.tosavepath + "/" + "Sum"):
            os.mkdir(args.tosavepath + "/" + "Sum")
        if not os.path.exists(args.tosavepath + "/" + "Sum" + "/" + str(onerun)):
            os.mkdir(args.tosavepath + "/" + "Sum" + "/" + str(onerun))
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(ckpt, os.path.join(args.tosavepath + "/" + "Sum" + "/" + str(onerun), "bestckpt"))

def testner(args, test_ner, onerun):

    test_sampler = SequentialSampler(test_ner)
    test_dataloader = get_dataloader(args.num_workers, test_ner, args.test_size_per_gpu, args.max_length,
                                      test_ner.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
    model = T5forAll(args, t5model, test_ner.tokenizer)

    allckpt = torch.load(args.tosavepath + "/" + "NER" + "/" + str(onerun) + "/bestckpt")
    model.promptnumber = allckpt["promptnumber"]
    model.promptembedding = allckpt["promptembedding"]
    logger.info("load finished for ner!")

    model.to(args.device)
    model.eval()
    allytrue = []
    allypred = []
    scaler = ShardedGradScaler()

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen,target,preds = model._generative_step(inputs)
                    tarres, predres = getonebatchresult(sen,target,preds)
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = getonebatchresult(sen, target, preds)
                allytrue.extend(tarres)
                allypred.extend(predres)
    report = classification_report(allytrue, allypred, digits=4)
    logger.info('----NER Test Results Summary----')
    logger.info("\n%s", report)

def testclass(args, test_class, onerun):

    test_sampler = SequentialSampler(test_class)
    test_dataloader = get_dataloader(args.num_workers, test_class, args.test_size_per_gpu, args.max_length,
                                      test_class.tokenizer.pad_token_id, test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
    model = T5forAll(args, t5model, test_class.tokenizer)

    allckpt = torch.load(args.tosavepath + "/" + "Class" + "/" + str(onerun) + "/bestckpt")
    model.promptnumber = allckpt["promptnumber"]
    model.promptembedding = allckpt["promptembedding"]
    logger.info("load finished for class!")

    model.to(args.device)
    model.eval()
    scaler = ShardedGradScaler()
    allnum = 0
    correctnum = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen,target,preds = model._generative_step(inputs)
                    thisbatchnum = len(sen)
                    for k in range(thisbatchnum):
                        allnum += 1
                        if target[k] == preds[k]:
                            correctnum += 1
            else:
                sen, target, preds = model._generative_step(inputs)
                thisbatchnum = len(sen)
                for k in range(thisbatchnum):
                    allnum += 1
                    if target[k] == preds[k]:
                        correctnum += 1
    accuracy = float(correctnum) / float(allnum)
    logger.info('----Classification Test Results Summary----')
    logger.info("test_allnum: %d", allnum)
    logger.info("test_correctnum: %d", correctnum)
    logger.info("test_accuracy: %f", accuracy)

def testsum(args, test_sum, onerun):

    test_sampler = SequentialSampler(test_sum)
    test_dataloader = get_dataloader(args.num_workers, test_sum, args.test_size_per_gpu, args.max_length,
                                      test_sum.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
    model = T5forAll(args, t5model, test_sum.tokenizer)

    allckpt = torch.load(args.tosavepath + "/" + "Sum" + "/" + str(onerun) + "/bestckpt")
    model.promptnumber = allckpt["promptnumber"]
    model.promptembedding = allckpt["promptembedding"]

    logger.info("load finished for summarization!")

    model.to(args.device)
    model.eval()
    scaler = ShardedGradScaler()
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
    logger.info('----Summarization Test Results Summary----')
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
    parser.add_argument("--lm_lambda", dest="lm_lambda", type=float,
                        default=0.25, help='language model loss lambda')
    parser.add_argument("--startindex", dest="startindex", type=int,
                        default=0, help="start index")
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

    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="t5_ckpt", help="ckpt dir to save")
    parser.add_argument("--tosavepath", dest="tosavepath", type=str,
                        default="t5_pt_ckpt", help="ckpt dir to save")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")


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
    allgentasktoken = ["summarizationcnndm", "classifyagnews", "__nerco__"]

    if args.local_rank != -1:
        torch.distributed.barrier()

    startindex = args.startindex

    for onerun in range(startindex, startindex+1):

        logger.info(onerun)
        args.seed = initialseed + onerun * 100
        seed_everything(args)
        logger.info("new seed %s", args.seed)

        args.taskfold = "Multitask"

        tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
        for j in range(0, 3):
            gentasktoken = allgentasktoken[j]
            tokenizer.add_tokens(gentasktoken)
            logger.info(
                'gen token = {} , gen token id = {}'.format(gentasktoken,
                                                            tokenizer.convert_tokens_to_ids(gentasktoken)))
        answertoken = "__ans__"
        special_tokens = {"ans_token": answertoken}
        tokenizer.add_tokens(list(special_tokens.values()))

        thistrainfile = "alldata/" + str(onerun) + "/train.txt"

        validforner = "alldata/" + str(onerun) + "/NERvalid/valid.txt"
        testforner = "alldata/" + str(onerun) + "/NERtest/test.txt"
        validforclass = "alldata/" + str(onerun) + "/Classvalid/valid.txt"
        testforclass = "alldata/" + str(onerun) + "/Classtest/test.txt"
        validforsum = "alldata/" + str(onerun) + "/Sumvalid/valid.txt"
        testforsum = "alldata/" + str(onerun) + "/Sumtest/test.txt"

        t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/data/qin/cache/")
        model = T5forAll(args, t5model, tokenizer)
        promptnumber = args.prompt_number
        promptembedding = getpromptembedding(model, tokenizer, promptnumber)
        # print(promptembedding)
        model.set_prompt_embedding(promptnumber, promptembedding)

        ###put to gpu
        model.to(args.device)

        train_dataset = T5SummarizationDataset(thistrainfile, args.max_length, tokenizer, allgentasktoken, answertoken)

        valid_ner = T5SummarizationDataset(validforner, args.max_length, tokenizer, allgentasktoken, answertoken)
        valid_class = T5SummarizationDataset(validforclass, args.max_length, tokenizer, allgentasktoken, answertoken)
        valid_sum = T5SummarizationDataset(validforsum, args.max_length, tokenizer, allgentasktoken, answertoken)

        test_ner = T5SummarizationDataset(testforner, args.max_length, tokenizer, allgentasktoken, answertoken)
        test_class = T5SummarizationDataset(testforclass, args.max_length, tokenizer, allgentasktoken, answertoken)
        test_sum = T5SummarizationDataset(testforsum, args.max_length, tokenizer, allgentasktoken, answertoken)

        logger.info("Finish prepare model and dataset")
        logger.info("Start training")

        train(args, model, train_dataset, valid_ner, valid_class, valid_sum, onerun)
        logger.info("Finish training")

        torch.cuda.empty_cache()
        del model
        gc.collect()

        if args.local_rank in [0, -1]:
            logger.info("Start testing")
            logger.info("Testing...")
            testner(args, test_ner, onerun)
            testclass(args, test_class, onerun)
            testsum(args, test_sum, onerun)
            logger.info("Finish testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()








