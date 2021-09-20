#learnrate=(5e-5 3e-5 2e-5)
learnrate=(5e-5)
#allewclambda=(0.6 1.5 3.0 6.0)
allewclambda=(0.6)
#allewclambda=(1.5 3.0 6.0)
#allewclambda=(3.0 6.0)
for onerate in ${learnrate[@]}
do
  for onelambda in ${allewclambda[@]}
  do
      python -m torch.distributed.launch --nproc_per_node 1 --master_port 29202 testmas.py \
        --cuda 5 \
        --lr $onerate \
        --ewc_lambda $onelambda \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --batch_size_per_gpu 4 \
        --valid_size_per_gpu 8 \
        --test_size_per_gpu 12 \
        --gradient_accumulation_steps 2 \
        --max_epoch 256 \
        --num_workers 4 \
        --save_step 100000 \
        --eval_step 100000 \
        --save_dir t5ner_onto_ckpt_$onerate \
        --seed 42 \
        --model T5NER \
        --model_name google/t5-v1_1-large \
        --train_file_name ontonotes_fewshot/train.txt \
        --valid_file_name ontonotes_fewshot/valid.txt \
        --test_file_name all_fewshot/test.txt \
        --train_sample \
        --max_length 128 \
        --adam_epsilon 1e-8 \
        --warmup_steps 0.01 \
        --load_ckpt 0 \
        --use_lm_adapted 0 \
        --lm_adapted_path ../t5_ckpt_1_0622_bak/t5_ckpt/ckpt_of_step_100000
    echo "++++++++++++++++++++++++++++++"
    ps aux | grep testmas.py | awk '{print $2}' | xargs kill -9
  done
done


