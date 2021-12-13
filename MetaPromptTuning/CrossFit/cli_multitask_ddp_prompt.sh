size="large"
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29530 cli_multitask_ddp_prompt.py \
        --output_dir models/upstream-multitask-temp \
        --identifier $size \
        --do_train \
        --prompt_number 100 \
        --train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --cuda 0 \
        --model google/t5-v1_1-$size \
        --lm_adapted_path /data/qin/lm_adapted_t5model/torch_ckpt/$size/pytorch_model.bin