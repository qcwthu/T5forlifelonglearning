learnrate=(5e-1)
#alllambda=(0.25)
alllambda=(0.10)
for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
      echo "------------------------------"
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 29581 PromptFinetune.py \
          --cuda 1 \
          --lr $onerate \
          --weight_decay 1e-5 \
          --max_grad_norm 1.0 \
          --batch_size_per_gpu 4 \
          --valid_size_per_gpu 16 \
          --test_size_per_gpu 16 \
          --gradient_accumulation_steps 2 \
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
          --use_lm_adapted 1 \
          --lm_adapted_path  /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin \
          --prompt_number 300 \
          --ifckpt_onlymodel 1

    echo "++++++++++++++++++++++++++++++"
    ps aux | grep PromptFinetune.py | awk '{print $2}' | xargs kill -9
  done
done



