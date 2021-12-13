import gc
import os
import numpy as np
import torch

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from dataloader.fewshot_gym_singletask import NLPFewshotGymSingleTaskData

from bart import MyBart
from T5Prompt import T5PromptModel
from utils import freeze_embeds, trim_batch, getpromptembedding

from tqdm import tqdm

from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.cuda.amp import autocast as autocast


def run(args, logger):
    #tokenizer = BartTokenizer.from_pretrained(args.model)
    tokenizer = T5Tokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
    dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)

    train_data.load_dataset(tokenizer)
    #train_data.load_dataloader()
    if args.local_rank != -1:
        train_data.load_dataloader(ifrandom=False)
    else:
        train_data.load_dataloader(ifrandom=True)

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    best_dev_performance = -1.0
    best_model_state_dict = None
    test_performance = -1.0

    if args.do_train:
        inermodel = T5ForConditionalGeneration.from_pretrained(args.model, cache_dir=args.cache_dir)
        model = T5PromptModel(args, inermodel)
        if args.checkpoint is not None and args.checkpoint != "None":
            logger.info("load prompt embedding from ckpt")
            promptckpt = torch.load(args.checkpoint)
            promptnumber = args.prompt_number
            promptnumber_ckpt = promptckpt['promptnumber']
            assert promptnumber == promptnumber_ckpt
            promptembedding = promptckpt['promptembedding']
        else:
            logger.info("try to initialize prompt embeddings")
            promptnumber = args.prompt_number
            thistask = [args.task_name]
            logger.info("task name: %s" % (args.task_name))
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistask)

        model.set_prompt_embedding(promptnumber, promptembedding)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        # if args.n_gpu>1:
        #     model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # scheduler =  get_linear_schedule_with_warmup(optimizer,
        #                                 num_warmup_steps=args.warmup_steps,
        #                                 num_training_steps=args.total_steps)
        base_optimizer_arguments = {"lr": args.learning_rate, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                    "weight_decay": args.weight_decay,
                                    "scale_parameter": False, "relative_step": False}
        optimizer = Adafactor
        optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                        **base_optimizer_arguments)
        scheduler = None
        #model = ShardedDDP(model, optimizer)
        best_dev_performance, best_model_state_dict = train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.local_rank in [0, -1]:
        if args.do_predict:
            if args.do_train and best_model_state_dict is not None:

                logger.info("Loading checkpoint from best ckpt on the fly")

                promptnumber = args.prompt_number
                promptnumber_ckpt = best_model_state_dict['promptnumber']
                assert promptnumber == promptnumber_ckpt

                promptembedding = best_model_state_dict['promptembedding']

            else:

                checkpoint = os.path.join(args.output_dir, args.predict_checkpoint)

                logger.info("Loading checkpoint from {}".format(checkpoint))

                promptckptfortest = torch.load(checkpoint)

                promptnumber = args.prompt_number
                promptnumber_ckpt = promptckptfortest['promptnumber']
                assert promptnumber == promptnumber_ckpt

                promptembedding = promptckptfortest['promptembedding']

            model.set_prompt_embedding(promptnumber, promptembedding)

            if torch.cuda.is_available():
                #model.to(torch.device("cuda"))
                model.to(torch.device(args.device))
            model.eval()

            data_type = "test" if "test" in args.test_file else "dev"

            #dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)
            test_data = NLPFewshotGymSingleTaskData(logger, args, args.test_file, data_type=data_type, is_training=False)

            test_data.load_dataset(tokenizer)
            test_data.load_dataloader()
            scaler = ShardedGradScaler()

            test_performance = inference(args, model, test_data, scaler, save_predictions=True, verbose=True)
            logger.info("%s on %s data: %.4f" % (test_data.metric, test_data.data_type, test_performance))
            del scaler, test_data

    torch.cuda.empty_cache()
    del model, optimizer, scheduler, train_data, dev_data
    gc.collect()
    return best_dev_performance, test_performance

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):

    model = ShardedDDP(model, optimizer)
    model.train()
    scaler = ShardedGradScaler()
    global_step = 0
    global_batch = 0
    train_losses = []
    train_losses_usedforeval = []

    best_performance = -1.0
    stop_training=False
    best_model_state_dict = None

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        #for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch), disable=args.quiet):
        #logger.info(len(train_data.dataloader))
        for step, batch in enumerate(train_data.dataloader):
            global_batch += 1
            #logger.info(batch[0].shape)
            if torch.cuda.is_available():
                #batch = [b.to(torch.device("cuda")) for b in batch]
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
            #loss.backward()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_batch % args.gradient_accumulation_steps == 0:
                global_step += 1
                if scaler is not None:
                    # scaler.unscale_(optimizer)
                    # optimizer.clip_grad_norm(args.max_grad_norm)
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
                #if global_step % args.eval_period == 0:
                    model.eval()
                    curr_performance = inference(args, model if args.n_gpu==1 else model.module, dev_data, scaler)
                    # if args.local_rank in [0, -1]:
                    #     logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (global_step,np.mean(train_losses_usedforeval),dev_data.metric,curr_performance, epoch))
                    #     train_losses_usedforeval = []
                    logger.info("Global step %d Train loss %.2f %s %s on epoch=%d" % (global_step, np.mean(train_losses_usedforeval), dev_data.metric, curr_performance, epoch))
                    train_losses_usedforeval = []
                    if best_performance < curr_performance:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        best_model_state_dict = {
                            "promptnumber": model_to_save.promptnumber,
                            "promptembedding": model_to_save.promptembedding
                        }
                        # if args.local_rank in [0, -1]:
                        #     torch.save(best_model_state_dict, os.path.join(args.output_dir, "best-model.pt"))  #####存了
                        #     logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" %
                        #                 (dev_data.metric, best_performance, curr_performance, epoch, global_step))
                        torch.save(best_model_state_dict, os.path.join(args.output_dir, "best-model.pt"))  #####存了
                        logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" %
                                    (dev_data.metric, best_performance, curr_performance, epoch, global_step))
                        best_performance = curr_performance
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break

                    model.train()

            if global_step >= args.total_steps:
                stop_training = True
                break
                
        if stop_training:
            break
    if args.local_rank in [0, -1]:
        logger.info("save last model!")
        model_to_save = model.module if hasattr(model, 'module') else model
        lastckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(lastckpt, os.path.join(args.output_dir, "last-model.pt"))
    del scaler
    return best_performance, best_model_state_dict

def inference(args, model, dev_data, scaler, save_predictions=False, verbose=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    with torch.no_grad():
        #print(len(dev_data.dataloader))
        #print(args.device)
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
        #print(predictions)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return dev_data.evaluate(predictions, verbose=verbose)
