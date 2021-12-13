#TASKS=("ag_news" "quoref" "wiki_split" "ethos-disability" "yelp_polarity" "superglue-rte" "glue-cola" "ethos-sexual_orientation"
#"blimp-sentential_negation_npi_scope" "ai2_arc" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present"
#"tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa" "glue-qnli" "hatexplain" "circa")

#TASKS=("ag_news" "wiki_split" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")
###ag从21,wiki从100,ethos从13,air2从21
#TASKS=("ag_news" "ai2_arc")
TASKS=("ag_news" "ai2_arc")
#TASKS=("wiki_split" "ethos-disability"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")

CHECKPOINT="None"
size="large"
IDENTIFIER="T5-"$size"-ft"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 1 --master_port 29566 tune_hps_singletask_ddp.py \
        --task_dir data/${TASK}/ \
        --task_name ${TASK} \
        --identifier $IDENTIFIER \
        --checkpoint $CHECKPOINT \
        --do_train \
        --do_predict \
        --learning_rate_list 5e-4 3e-4 2e-4 1e-4 \
        --bsz_list 8 \
        --predict_batch_size 32 \
        --total_steps 1000 \
        --eval_period 100 \
        --warmup_steps 100 \
        --num_train_epochs 1000.0 \
        --gradient_accumulation_steps 2 \
        --output_dir models/${IDENTIFIER}/singletask-${TASK} \
        --cuda 7 \
        --model google/t5-v1_1-$size
      echo "++++++++++++++++++++++++++++++"
      ps aux | grep tune_hps_singletask_ddp.py | awk '{print $2}' | xargs kill -9
done
