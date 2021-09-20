#learnrate=(5e-1)
learnrate=(5e-1)
#learnrate=(0.55 0.45 0.60)   #####0.55和0.45跑完了，0.60没有
#alllambda=(0.25)
#alllambda=(0.10 0.25)
alllambda=(0.10)
allstartindex=(0 1 2)
alltask=(0 1 2)
for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
    for onestartindex in ${allstartindex[@]}
    do
      for onetask in ${alltask[@]}
      do
              echo "------------------------------"
              python -m torch.distributed.launch --nproc_per_node 1 --master_port 29900 Summarization.py \
                --cuda 2 \
                --lr $onerate \
                --lm_lambda $onelambda \
                --startindex $onestartindex \
                --taskindex $onetask \
                --weight_decay 1e-5 \
                --max_grad_norm 1.0 \
                --batch_size_per_gpu 1 \
                --valid_size_per_gpu 8 \
                --test_size_per_gpu 8 \
                --gradient_accumulation_steps 4 \
                --max_epoch 1280 \
                --num_workers 0 \
                --save_step 100000 \
                --eval_step 100000 \
                --save_dir t5ner_onto_ckpt \
                --seed 42 \
                --model T5Summarization \
                --model_name google/t5-v1_1-large \
                --train_file_name ../T5forNER/ontonotes_fewshot/train.txt \
                --valid_file_name ../T5forNER/ontonotes_fewshot/valid.txt \
                --test_file_name ../T5forNER/ontonotes_fewshot/test.txt \
                --train_sample \
                --max_length 512 \
                --adam_epsilon 1e-8 \
                --warmup_steps 0.01 \
                --use_lm_adapted 1 \
                --lm_adapted_path  /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin \
                --prompt_number 300 \
                --ifckpt_onlymodel 1
          echo "++++++++++++++++++++++++++++++"
          ps aux | grep Summarization.py | awk '{print $2}' | xargs kill -9
      done
    done
  done
done



