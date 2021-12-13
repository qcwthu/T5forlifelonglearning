import os
import numpy as np
import torch

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from dataloader.fewshot_gym_multitask import NLPFewshotGymMultiTaskData

from bart import MyBart
from T5Prompt import T5PromptModel
from utils import freeze_embeds, trim_batch, get_tasks_list, getpromptembedding

from tqdm import tqdm

from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.cuda.amp import autocast as autocast

def run(args, logger):
    #tokenizer = BartTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    # print(tokenizer.pad_token_id)
    # print("aaa")
    # exit -1
    train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    logger.info("Training on the following tasks: {}".format(train_tasks))

    dev_tasks = get_tasks_list(args.custom_tasks_splits, "dev")
    logger.info("Dev on the following tasks: {}".format(dev_tasks))

    train_data = NLPFewshotGymMultiTaskData(logger, args, args.train_dir, tasks=train_tasks, data_split="all", data_type="train", is_training=True)
    dev_data = NLPFewshotGymMultiTaskData(logger, args, args.train_dir, tasks=dev_tasks, data_split="all", data_type="dev", is_training=False)

    train_data.load_dataset(tokenizer)
    if args.local_rank != -1:
        train_data.load_dataloader(ifrandom=False)
    else:
        train_data.load_dataloader(ifrandom=True)

    dev_data.load_dataset(tokenizer)
    # if args.local_rank != -1:
    #     dev_data.load_dataloader(ifrandom=False)
    # else:
    #     dev_data.load_dataloader(ifrandom=True)
    dev_data.load_dataloader() #######dev ---> is_training = False

    if args.do_train:
        inermodel = T5ForConditionalGeneration.from_pretrained(args.model, cache_dir=args.cache_dir)
        model = T5PromptModel(args, inermodel)
        if args.checkpoint is not None:
            promptckpt = torch.load(args.checkpoint)
            promptnumber = args.prompt_number
            promptnumber_ckpt = promptckpt['promptnumber']
            assert promptnumber == promptnumber_ckpt
            promptembedding = promptckpt['promptembedding']
        else:
            logger.info("try to initialize prompt embeddings")
            promptnumber = args.prompt_number
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, train_tasks)
        ######initialize prompt

        model.set_prompt_embedding(promptnumber, promptembedding)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        # logger.info(args.n_gpu)
        # if args.n_gpu > 1:
        #     logger.info("use dataparallel")
        #     model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device(args.device))

        if args.local_rank != -1:
            torch.distributed.barrier()

        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        # #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # base_optimizer_arguments = {"lr":args.learning_rate, "eps":args.adam_epsilon, "correct_bias":False}
        # optimizer = AdamW
        # optimizer = OSS(
        #     params=optimizer_grouped_parameters,
        #     optim=optimizer,
        #     **base_optimizer_arguments)

        base_optimizer_arguments = {"lr": args.learning_rate, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                    "weight_decay": args.weight_decay,
                                    "scale_parameter": False, "relative_step": False}
        optimizer = Adafactor
        optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                        **base_optimizer_arguments)

        step_tot = (len(
            train_data.dataset) // args.gradient_accumulation_steps // args.train_batch_size // args.n_gpu) * args.num_train_epochs

        logger.info(step_tot)
        logger.info(len(train_data.dataset))
        logger.info(args.gradient_accumulation_steps)
        logger.info(args.train_batch_size)
        logger.info(args.n_gpu)
        logger.info(args.num_train_epochs)
        #
        # warmup_steps = int(step_tot * 0.06)
        # scheduler =  get_linear_schedule_with_warmup(optimizer,
        #                                 #num_warmup_steps=args.warmup_steps,
        #                                 #num_training_steps=args.total_steps)
        #                                 num_warmup_steps=warmup_steps,
        #                                 num_training_steps=step_tot)
        scheduler = None
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):

    model = ShardedDDP(model, optimizer)
    model.train()
    scaler = ShardedGradScaler()
    global_step = 0
    global_batch = 0
    train_losses = []
    train_losses_usedforeval = []
    best_accuracy = -1.0
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        #for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch)):
        logger.info(len(train_data.dataloader))
        for step, batch in enumerate(train_data.dataloader):
            global_batch += 1
            # if global_step >= 20:
            #     stop_training = True
            #     break
            if torch.cuda.is_available():
                batch = [b.to(torch.device(args.device)) for b in batch]
            
            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
            if scaler is not None:
                with autocast():
                    loss = model(input_ids=batch[0], attention_mask=batch[1],
                                 decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                 is_training=True)
            else:
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                             is_training=True)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break

            train_losses.append(loss.detach().cpu())
            train_losses_usedforeval.append(loss.detach().cpu())

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            #if step % args.gradient_accumulation_steps == 0 or step == len(train_data.dataloader) - 1:
            if global_batch % args.gradient_accumulation_steps == 0:
                global_step += 1
                if scaler is not None:
                    #scaler.unscale_(optimizer)
                    #optimizer.clip_grad_norm(args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enough gradients
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    logger.info("Step %d Global step %d Train loss %.2f on epoch=%d" % (global_batch, global_step, np.mean(train_losses), epoch))
                    train_losses = []

                if args.local_rank in [0, -1] and global_step % args.eval_period == 0:
                    model.eval()
                    curr_em = inference(args, model if args.n_gpu==1 else model.module, dev_data, scaler)
                    logger.info("Global step %d Train loss %.2f %s %s on epoch=%d" % (global_step, np.mean(train_losses_usedforeval), dev_data.metric, curr_em, epoch))
                    train_losses_usedforeval = []
                    if best_accuracy < curr_em:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        best_model_state_dict = {
                            "promptnumber": model_to_save.promptnumber,
                            "promptembedding": model_to_save.promptembedding
                        }
                        torch.save(best_model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                                (dev_data.metric, best_accuracy, curr_em, epoch, global_step))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                    model.train()
        if stop_training:
            break
    if args.local_rank in [0, -1]:
        logger.info("save last model!")
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        #model_state_dict = {k:v.cpu() for (k, v) in model_to_save.model.state_dict().items()}
        torch.save(ckpt, os.path.join(args.output_dir, "last-model.pt"))

def inference(args, model, dev_data, scaler, save_predictions=False, verbose=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    with torch.no_grad():
        for i, batch in enumerate(dev_data.dataloader):
            if torch.cuda.is_available():
                batch = [b.to(torch.device(args.device)) for b in batch]
            pad_token_id = dev_data.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            if scaler is not None:
                with autocast():
                    outputs = model._generative_step(input_ids=batch[0],
                                             attention_mask=batch[1],
                                             num_beams=dev_data.args.num_beams,
                                             max_length=dev_data.args.max_output_length)
            else:
                outputs = model._generative_step(input_ids=batch[0],
                                                 attention_mask=batch[1],
                                                 num_beams=dev_data.args.num_beams,
                                                 max_length=dev_data.args.max_output_length)
            for input_, output in zip(batch[0], outputs):
                pred = dev_data.decode(output)
                predictions.append(pred)
                #print(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return dev_data.evaluate(predictions, verbose=verbose)
