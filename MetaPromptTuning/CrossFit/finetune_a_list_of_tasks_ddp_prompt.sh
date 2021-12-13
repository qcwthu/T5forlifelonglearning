TASKS=("boolq" "boolq")
CHECKPOINT="None"
IDENTIFIER="T5-Large"
echo "111"
for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"

  python -m torch.distributed.launch --nproc_per_node 4 --master_port 29544 tune_hps_singletask_ddp_prompt.py \
        --task_dir data/${TASK}/ \
        --task_name ${TASK} \
        --checkpoint $CHECKPOINT \
        --do_train \
        --do_predict \
        --learning_rate_list 1e-5 2e-5 5e-5 \
        --bsz_list 2 4 8 \
        --total_steps 1000 \
        --eval_period 100 \
        --warmup_steps 100 \
        --output_dir models/${IDENTIFIER}/singletask-${TASK} \
        --cuda 2,3,6,7 \
        --lm_adapted_path /data/qin/lm_adapted_t5model/torch_ckpt/base/pytorch_model.bin \
        --model google/t5-v1_1-base \
        --prompt_number 100
done