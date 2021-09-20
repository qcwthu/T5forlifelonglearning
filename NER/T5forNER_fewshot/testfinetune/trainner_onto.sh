#learnrate=(5e-5 3e-5 2e-5)
learnrate=(5e-5)
for onerate in ${learnrate[@]}
do
      python -m torch.distributed.launch --nproc_per_node 1 --master_port 29511 main_tryparallel_fairscale_ontonotes.py \
        --cuda 5 \
        --lr $onerate \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --batch_size_per_gpu 8 \
        --valid_size_per_gpu 16 \
        --test_size_per_gpu 16 \
        --gradient_accumulation_steps 1 \
        --max_epoch 256 \
        --num_workers 4 \
        --save_step 100000 \
        --eval_step 100000 \
        --save_dir t5ner_onto_ckpt_$onerate \
        --seed 42 \
        --model T5NER \
        --model_name google/t5-v1_1-large \
        --train_file_name ../../ontonotes_fewshot/train.txt \
        --valid_file_name ../../ontonotes_fewshot/valid.txt \
        --test_file_name ../../ontonotes_fewshot/test.txt \
        --train_sample \
        --max_length 128 \
        --adam_epsilon 1e-8 \
        --warmup_steps 0.01 \
        --load_ckpt 0 \
        --use_lm_adapted 0 \
        --lm_adapted_path ../t5_ckpt_1_0622_bak/t5_ckpt/ckpt_of_step_100000
    echo "++++++++++++++++++++++++++++++"
    ps aux | grep main_tryparallel_fairscale_ontonotes.py | awk '{print $2}' | xargs kill -9
done


