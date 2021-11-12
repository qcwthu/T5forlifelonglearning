learnrate=(3e-5)
alllambda=(0.1)
for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
      echo "------------------------------"
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 29600 T5Finetune.py \
          --cuda 6 \
          --lr $onerate \
          --weight_decay 0.01 \
          --max_grad_norm 1.0 \
          --batch_size_per_gpu 8 \
          --valid_size_per_gpu 16 \
          --test_size_per_gpu 16 \
          --gradient_accumulation_steps 1 \
          --max_epoch 1280 \
          --num_workers 0 \
          --save_step 100000 \
          --eval_step 100000 \
          --tosavepath t5_classify_ckpt \
          --seed 42 \
          --model T5SentenceClassify \
          --model_name google/t5-v1_1-large \
          --train_sample \
          --max_length 128 \
          --adam_epsilon 1e-8 \
          --warmup_steps 0.01 \

    echo "++++++++++++++++++++++++++++++"
    ps aux | grep T5Finetune.py | awk '{print $2}' | xargs kill -9
  done
done



