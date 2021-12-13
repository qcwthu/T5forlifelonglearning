
####跑完"ag_news" "wiki_split" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc"
#TASKS=("ag_news" "wiki_split" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")

#TASKS=("ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")

#TASKS=("glue-cola")
TASKS=("ethos-disability" "ethos-sexual_orientation" "ai2_arc"
"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa"
"glue-qnli" "hatexplain" "circa")

#CHECKPOINT="None"
CHECKPOINT="models/upstream-multitask/last-model.pt"

#size="base"
size="large"

#IDENTIFIER="T5-"$size
IDENTIFIER="T5-"$size-"multi"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 29544 tune_hps_singletask_ddp_prompt.py \
        --task_dir data/${TASK}/ \
        --task_name ${TASK} \
        --identifier $IDENTIFIER \
        --checkpoint $CHECKPOINT \
        --do_train \
        --do_predict \
        --learning_rate_list 5e-1 4e-1 3e-1 2e-1 \
        --bsz_list 8 \
        --predict_batch_size 32 \
        --total_steps 3000 \
        --eval_period 100 \
        --warmup_steps 100 \
        --num_train_epochs 3000.0 \
        --gradient_accumulation_steps 2 \
        --output_dir models/${IDENTIFIER}/singletask-${TASK} \
        --cuda 5,6 \
        --lm_adapted_path /data/qin/lm_adapted_t5model/torch_ckpt/$size/pytorch_model.bin \
        --model google/t5-v1_1-$size \
        --prompt_number 100
      echo "++++++++++++++++++++++++++++++"
      ps aux | grep tune_hps_singletask_ddp_prompt.py | awk '{print $2}' | xargs kill -9
done