import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import logging
import matplotlib.pyplot as plt
from apex import amp
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
from model import *
from dataset import *
from seqeval.metrics import classification_report,f1_score
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def getonebatchresult(sen,target,preds):
    #typedic = {"org": "ORG", "location": "LOC", "person": "PER", "mix": "MISC"}
    typedic = {"org": "ORG", "money": "MONEY", "country": "GPE", "time": "TIME", "law": "LAW", "fact": "FAC",
               "thing": "EVENT", "measure": "QUANTITY",
               "order": "ORDINAL", "art": "WORK_OF_ART", "location": "LOC", "language": "LANGUAGE", "person": "PERSON",
               "product": "PRODUCT", "num": "CARDINAL", "national": "NORP", "date": "DATE", "per": "PERCENT"}
    # typedic = {"money": "MONEY", "country": "GPE", "time": "TIME", "law": "LAW", "fact": "FAC",
    #            "thing": "EVENT", "measure": "QUANTITY",
    #            "order": "ORDINAL", "art": "WORK_OF_ART", "language": "LANGUAGE",
    #            "product": "PRODUCT", "num": "CARDINAL", "national": "NORP", "date": "DATE", "per": "PERCENT"}
    sennum = len(sen)
    restar = []
    respred = []
    for i in range(sennum):
        thissen, thistar, thispred = sen[i], target[i], preds[i]

        thissenlow = thissen.lower()

        sensplit = thissen.split(' ')
        sensplitlow = thissenlow.split(' ')

        tarres = ['O' for j in range(len(sensplit))]
        predres = ['O' for j in range(len(sensplit))]

        if thistar == 'end' and thispred == 'end':
            restar.append(tarres)
            respred.append(predres)
            continue

        if len(thistar) > 0 and thistar[-1] == ';':
            thistar = thistar[:-1]

        tarsplit1 = thistar.split(';')

        if thistar != 'end':
            for j in range(len(tarsplit1)):
                tarsplit2 = tarsplit1[j].split('!')
                if len(tarsplit2) != 2:
                    # logger.error('len error!')
                    # logger.error(sensplit)
                    # logger.error(thissen)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                entity = tarsplit2[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit2[1].strip(' ')
                if type not in typedic:
                    # logger.error('dic error!')
                    # logger.error(type)
                    # logger.error(sensplit)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                #if thissen.find(entity) == -1:
                if thissenlow.find(entitylow) == -1:
                    # logger.error('find error!')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                trueindex = -100
                #entitysplit = entity.split(' ')
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    #if sensplit[k] == entitysplit[0]:
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    # logger.error('error! Not find this entity')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                for k in range(trueindex, trueindex + len(entitysplit)):
                    if k == trueindex:
                        tarres[k] = 'B-' + typedic[type]
                    else:
                        tarres[k] = 'I-' + typedic[type]

        if len(thispred) > 0 and thispred[-1] == ';':
            thispred = thispred[:-1]

        tarsplit3 = thispred.split(';')

        if thispred != "end":
            for j in range(len(tarsplit3)):
                tarsplit4 = tarsplit3[j].split('!')
                if len(tarsplit4) != 2:
                    # logger.error('len error!')
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                entity = tarsplit4[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit4[1].strip(' ')
                if type not in typedic:
                    # logger.error('dic error!')
                    # logger.error(type)
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                #if thissen.find(entity) == -1:
                if thissenlow.find(entitylow) == -1:
                    # logger.error('find error!')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                trueindex = -100
                #entitysplit = entity.split(' ')
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    #if sensplit[k] == entitysplit[0]:
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            #if sensplit[k + l] != entitysplit[l]:
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    # logger.error('error! Not find this entity')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                else:
                    for k in range(trueindex, trueindex + len(entitysplit)):
                        if k == trueindex:
                            predres[k] = 'B-' + typedic[type]
                        else:
                            predres[k] = 'I-' + typedic[type]
        restar.append(tarres)
        respred.append(predres)
    return restar, respred

#tosavepath = "./t5ner_ckpt"
tosavepath = "./t5ner_onto_ckpt"

def dooneeval(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
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
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(f1score)

    result_dict['val_F1'].append(f1score)
    if result_dict['val_F1'][-1] > result_dict['best_val_F1']:
        logger.info("{} epoch, best epoch was updated! valid_F1: {: >4.5f}".format(i,result_dict['val_F1'][-1]))
        result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        if not os.path.exists(tosavepath):
            os.mkdir(tosavepath)
        if not os.path.exists(tosavepath + "/" + args.save_dir):
            os.mkdir(tosavepath + "/" + args.save_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            # 'bert-base': model.module.model.bert.state_dict(),
            't5-base-ner': model_to_save.model.state_dict(),
        }
        torch.save(ckpt, os.path.join(tosavepath + "/" + args.save_dir, "ckptofT5ner_best"))

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

def test(args, test_dataset):

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length,
                                      test_dataset.tokenizer.pad_token_id,test_sampler)
    model = T5forNER(args, tokenizer)
    allckpt = torch.load(tosavepath +  "/" + args.save_dir + "/ckptofT5ner_best")
    model.model.load_state_dict(allckpt['t5-base-ner'])
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    allytrue = []
    allypred = []
    scaler = ShardedGradScaler()
    # scaler = None
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
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
    print(len(allypred))
    report = classification_report(allytrue, allypred, digits=4)
    logger.info("\n%s", report)

def train(args, model, train_dataset,valid_dataset,test_dataset):
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
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # # base_optimizer_arguments = {"lr": args.lr, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
    # #                             #"weight_decay": args.weight_decay,
    # #                             "scale_parameter": False, "relative_step": False}
    # base_optimizer_arguments = { "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,"scale_parameter": False, "relative_step": True, "warmup_init": True}
    # optimizer = Adafactor
    # optimizer = OSS(params=optimizer_grouped_parameters, optim=optimizer,
    #                 **base_optimizer_arguments)
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
        'val_F1': [],
        'best_val_F1': Best_F1
    }
    global_step = 0
    for i in range(startepoch, startepoch + args.max_epoch):
        if i < 6:
            thisevalstep = args.eval_step * 3
        else:
            thisevalstep = args.eval_step
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    loss = model(inputs)
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
            else:
                loss = model(inputs)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            allloss.append(loss.item())

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
                    logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (global_step, global_step/step_tot, np.average(allloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    #####eval
                    print("one eval step!")
                    #dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i)
                    #model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            #if i > 64:
            if i > 24:
                dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i)
                ###save after evert epoch for load
                # if not os.path.exists(tosavepath):
                #     os.mkdir(tosavepath)
                # if not os.path.exists(tosavepath + "/" + args.save_dir):
                #     os.mkdir(tosavepath + "/" + args.save_dir)
                # model_to_save = model.module if hasattr(model, 'module') else model
                # ckpt = {
                #     # 'bert-base': model.module.model.bert.state_dict(),
                #     't5-base-ner': model_to_save.model.state_dict(),
                # }
                # torch.save(ckpt, os.path.join(tosavepath + "/" + args.save_dir, "ckptofT5ner_" + str(i)))
                model.train()

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
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
                        default=10000, help="step to save")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=10, help="how many steps to eval")

    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="t5_ckpt", help="ckpt dir to save")
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
                        default=0.01, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=float,
                        default=0.1, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1.0, help="max grad norm")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--load_ckpt", dest="load_ckpt", type=int,
                        default=0, help="whether load ckpt before training")
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="../t5_ckpt_1_0622_bak/t5_ckpt/ckpt_of_step_100000",
                        help="The path of lm_adapted model")
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
    #set_seed(args)
    seed_everything(args)

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        with open("./log/trainner_log", 'a+') as f:
            f.write(str(time.ctime()) + "\n")
            f.write(str(args) + "\n")
            f.write("----------------------------------------------------------------------------\n")

    # Model and dataset
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    #inputs = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
    #rint(inputs)
    #origin = tokenizer.convert_ids_to_tokens(inputs.squeeze())
    #print(origin)
    #'''
    if args.model == "T5NER":
        #t5model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        model = T5forNER(args,tokenizer)
        #####load model from conll ckpt
        t5conllckpt = torch.load("t5ner_ckpt/t5ner_ckpt/ckptofT5ner_best")
        print("load conll ckpt!!!")
        model.model.load_state_dict(t5conllckpt['t5-base-ner'])
        model.to(args.device)
        train_dataset = T5NERDatasetConll(args.train_file_name, args.max_length, tokenizer)
        valid_dataset = T5NERDatasetConll(args.valid_file_name, args.max_length, tokenizer)
        test_dataset = T5NERDatasetConll(args.test_file_name, args.max_length, tokenizer)
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {T5}")

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    train(args, model, train_dataset,valid_dataset,test_dataset)
    if args.local_rank in [0, -1]:
        test(args,test_dataset)
    logger.info("Finish training and testing!")
    #'''


