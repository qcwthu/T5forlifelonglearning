#learnrate=(5e-1 4e-1 3e-1)
#alllambda=(0.10 0.20)
learnrate=(5e-1)
alllambda=(0.20)
#learnrate=(5e-1)
#alllambda=(0.10)
for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
      echo "------------------------------"
      echo 't5ner_real_ckpt_'$onelambda'_'$onerate
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 29560 testEWC.py \
          --cuda 6 \
          --lr $onerate \
          --lm_lambda $onelambda \
          --weight_decay 1e-5 \
          --max_grad_norm 1.0 \
          --batch_size_per_gpu 2 \
          --valid_size_per_gpu 12 \
          --test_size_per_gpu 12 \
          --gradient_accumulation_steps 4 \
          --max_epoch 1280 \
          --num_workers 4 \
          --save_step 100000 \
          --eval_step 100000 \
          --save_dir t5ner_real_ckpt_$onelambda'_'$onerate \
          --seed 42 \
          --model T5NER \
          --model_name google/t5-v1_1-large \
          --train_file_name ./ontonotes_fewshot/train_onlyonto.txt \
          --valid_file_name ./ontonotes_fewshot/valid_onlyonto.txt \
          --test_file_name ./ontonotes_fewshot/test_all.txt \
          --train_sample \
          --max_length 128 \
          --adam_epsilon 1e-8 \
          --warmup_steps 0.01 \
          --load_ckpt 0 \
          --use_lm_adapted 1 \
          --lm_adapted_path  /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin \
          --prompt_number 300 \
          --ifckpt_onlymodel 1 \
          --use_pre_prompt 1 \
          --pre_prompt_path  ./conll_ckpt/t5ner_ckpt_0.25_0729bak1/ckptofT5ner_best

    echo "++++++++++++++++++++++++++++++"
    #ps aux | grep testEWC.py | awk '{print $2}' | xargs kill -9
  done
done



