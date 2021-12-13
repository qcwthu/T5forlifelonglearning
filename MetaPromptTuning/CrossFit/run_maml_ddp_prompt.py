import os
import numpy as np
import torch
import higher

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from dataloader.fewshot_gym_metalearn import NLPFewshotGymMetaLearningData

from bart import MyBart
from T5Prompt import T5PromptModel
from utils import freeze_embeds, trim_batch, get_tasks_list, getpromptembedding

from tqdm import tqdm

from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.cuda.amp import autocast as autocast

def run(args, logger):
    #tokenizer = BartTokenizer.from_pretrained(args.model)
    tokenizer = T5Tokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    logger.info("Training on the following tasks: {}".format(train_tasks))

    train_data = NLPFewshotGymMetaLearningData(logger, args, args.train_dir, tasks=train_tasks, data_type="train", is_training=True)
    # dev_data = NLPFewshotGymMetaLearningData(logger, args, args.train_dir, tasks=DEFAULT_SPLIT["dev"], data_type="dev", is_training=False)
    dev_data = None

    train_data.load_dataset(tokenizer)
    #train_data.load_dataloader()
    if args.local_rank != -1:
        train_data.load_dataloader(ifrandom=False)
    else:
        train_data.load_dataloader(ifrandom=True)

    # dev_data.load_dataset(tokenizer)
    # dev_data.load_dataloader()

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

        model.set_prompt_embedding(promptnumber, promptembedding)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        # if args.n_gpu>1:
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
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):

    model = ShardedDDP(model, optimizer)
    model.train()
    scaler = ShardedGradScaler()
    global_batch = 0
    global_step = 0
    train_losses = []
    dev_losses = []
    best_accuracy = -1.0
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch)):

            global_batch += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device(args.device)) for b in batch[0]]
            
            pad_token_id = train_data.tokenizer.pad_token_id

            # train batch
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            # dev batch
            batch[4], batch[5] = trim_batch(batch[4], pad_token_id, batch[5])
            batch[6], batch[7] = trim_batch(batch[6], pad_token_id, batch[7])

            inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
            with higher.innerloop_ctx(
                model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                #logger.info("train batch")
                #logger.info(batch[0].shape)
                #####这个地方要不要改成一个一个算，多叠加几个?
                # thistrainloss = []
                # for aa in range(args.inner_bsz):
                #     train_loss = fnet(input_ids=batch[0][aa:aa + 1, :], attention_mask=batch[1][aa:aa + 1, :],
                #                       decoder_input_ids=batch[2][aa:aa + 1, :],
                #                       decoder_attention_mask=batch[3][aa:aa + 1, :],
                #                       is_training=True)
                #
                #     if torch.isnan(train_loss).data:
                #         logger.info("Stop training because loss=%s" % (train_loss.data))
                #         stop_training = True
                #         break  # does this ever happen?
                #     thistrainloss.append(train_loss.detach().cpu())
                #     diffopt.step(train_loss)
                # if stop_training:
                #     break
                # train_losses.append(np.average(thistrainloss))


                train_loss = fnet(input_ids=batch[0], attention_mask=batch[1],
                            decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                            is_training=True)
                if torch.isnan(train_loss).data:
                    logger.info("Stop training because loss=%s" % (train_loss.data))
                    stop_training=True
                    break # does this ever happen?
                train_losses.append(train_loss.detach().cpu())
                diffopt.step(train_loss)


                # print("dev batch")
                dev_loss = fnet(input_ids=batch[4], attention_mask=batch[5],
                            decoder_input_ids=batch[6], decoder_attention_mask=batch[7],
                            is_training=True)
                dev_losses.append(dev_loss.detach().cpu())

                dev_loss.backward()

            if global_batch % args.gradient_accumulation_steps == 0:
                global_step += 1
                # if args.local_rank in [0, -1]:
                #     logger.info(global_step)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enough gradients
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()

                if args.local_rank in [0, -1] and global_step % args.eval_period == 0:
                #     model.eval()
                #     curr_em = inference(model if args.n_gpu==1 else model.module, dev_data)
                #     logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                #             global_step,
                #             np.mean(train_losses),
                #             dev_data.metric,
                #             curr_em,
                #             epoch))
                    logger.info("global step: {}; train loss: {}; dev loss: {}".format(global_step, np.mean(train_losses), np.mean(dev_losses)))
                    #logger.info("train loss: {}; dev loss: {}".format(np.mean(train_losses), np.mean(dev_losses)))
                    train_losses = []
                    dev_losses = []
                #     if best_accuracy < curr_em:
                #         model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                #         torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                #         logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                #                 (dev_data.metric, best_accuracy, curr_em, epoch, global_step))
                #         best_accuracy = curr_em
                #         wait_step = 0
                #         stop_training = False
                #     else:
                #         wait_step += 1
                #         if wait_step >= args.wait_step:
                #             stop_training = True
                #             break
                #     model.train()

            if global_step >= args.total_steps:
                stop_training = True
                break

        if stop_training:
            break
    if args.local_rank in [0, -1]:
        logger.info("save model!")
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        #model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
        torch.save(ckpt, os.path.join(args.output_dir, "last-model.pt"))

def inference(model, dev_data, save_predictions=False, verbose=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        pad_token_id = dev_data.tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 decoder_start_token_id=model.config.bos_token_id,
                                 early_stopping=dev_data.gen_early_stop,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return dev_data.evaluate(predictions, verbose=verbose)
