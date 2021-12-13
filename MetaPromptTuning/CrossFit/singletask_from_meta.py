# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch

import pandas as pd

from run_singletask_ddp_prompt import run

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--task_dir", default="data", required=True)
    parser.add_argument("--task_name", default="boolq", required=True)
    parser.add_argument("--identifier", default="T5-large", required=True)
    parser.add_argument("--train_file", default="data", required=False)
    parser.add_argument("--dev_file", default="data", required=False)
    parser.add_argument("--test_file", default="data", required=False)
    parser.add_argument("--dataset", default="nlp_forest_single", required=False)

    
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--predict_checkpoint", type=str, default="best-model.pt")

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=128)  ###change to 128?
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-1, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--warmup_proportion", default=0.01, type=float,
    #                     help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--total_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10000000000)

    # Other parameters
    parser.add_argument("--quiet", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=100,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # to tune
    parser.add_argument("--learning_rate_list", nargs="*", type=float, default=[])
    parser.add_argument("--bsz_list", nargs="*", type=int, default=[])

    parser.add_argument('--cache_dir', type=str, default="/data/qin/cache/")
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=10, help="how many steps to log")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="/data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    #parser.add_argument("--model", default="facebook/bart-base", required=False)
    parser.add_argument("--model", default="google/t5-v1_1-large", required=False)
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=100, help="The number of prompt")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="4", help="gpu id")

    args = parser.parse_args()
    if args.local_rank in [0, -1]:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            print("Output directory () already exists and is not empty.")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)
    #exit - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.n_gpu = torch.cuda.device_count()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    logger.info("args.device: %s", args.device)

    seed_everything(args)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_dir` must be specified.")
        if not args.dev_file:
            raise ValueError("If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.test_file:
            raise ValueError("If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))

    files = sorted(os.listdir(args.task_dir))
    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue
        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)

    logger.info("Fine-tuning the following samples: {}".format(prefixes))

    if args.local_rank in [0, -1]:
        df = pd.DataFrame(columns=["prefix", "lr", "bsz", "dev_performance", "test_performance"])

    if args.local_rank != -1:
        torch.distributed.barrier()

    for prefix in prefixes:
        args.train_file = os.path.join(args.task_dir, prefix + "_train.tsv")
        args.dev_file = os.path.join(args.task_dir, prefix + "_dev.tsv")
        args.test_file = os.path.join(args.task_dir, prefix + "_test.tsv")

        best_dev_performance = -1.0
        best_config = None
        for lr in args.learning_rate_list:
            for bsz in args.bsz_list:
                if args.local_rank in [0, -1]:
                    logger.info("Running ... prefix={}, lr={}, bsz={} ...".format(prefix, lr, bsz))
                args.learning_rate = lr
                args.train_batch_size = bsz
                args.prefix = prefix + "_" + str(lr) + "_" + str(bsz)
                dev_performance, test_performance = run(args, logger)

                if args.local_rank in [0, -1]:
                    df.loc[len(df.index)] = [prefix, lr, bsz, dev_performance, test_performance]
                    logger.info("prefix={}, lr={}, bsz={}, dev_performance={}, test_performance={}".format(prefix, lr, bsz,dev_performance, test_performance))
                    df.to_csv(os.path.join(args.output_dir, "result.csv"))

                    if dev_performance > best_dev_performance:
                        best_dev_performance = dev_performance
                        best_config = [prefix, lr, bsz, dev_performance, test_performance]
        if args.local_rank in [0, -1]:
            best_config[0] = best_config[0] + "_best"
            df.loc[len(df.index)] = best_config
            df.to_csv(os.path.join(args.output_dir, "result.csv"))

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

if __name__=='__main__':
    main()
