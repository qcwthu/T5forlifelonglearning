python -m torch.distributed.launch --nproc_per_node 2 --master_port 29530 cli_multitask_ddp.py \
        --output_dir models/upstream-multitask \
        --do_train