#learnrate=(5e-1 3e-1 4e-1)
#promptnumber=(300 320 400)
learnrate=(5e-1)
promptnumber=(320)

for onenum in ${promptnumber[@]}
do
  for onerate in ${learnrate[@]}
  do
    echo "------------------------------"
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 29515 main_prompt_adafactor.py \
            --cuda 3,6 \
            --lr $onerate \
            --weight_decay 1e-5 \
            --max_grad_norm 1.0 \
            --batch_size_per_gpu 8 \
            --valid_size_per_gpu 12 \
            --test_size_per_gpu 16 \
            --gradient_accumulation_steps 1 \
            --max_epoch 1280 \
            --num_workers 4 \
            --save_step 100000 \
            --eval_step 100000 \
            --save_dir t5ner_ckpt \
            --seed 42 \
            --model T5NER \
            --model_name google/t5-v1_1-large \
            --train_file_name ../T5forNER/conll_fewshot/train.txt \
            --valid_file_name ../T5forNER/conll_fewshot/valid.txt \
            --test_file_name ../T5forNER/conll_fewshot/test.txt \
            --train_sample \
            --max_length 128 \
            --adam_epsilon 1e-8 \
            --warmup_steps 0.01 \
            --load_ckpt 0 \
            --use_lm_adapted 1 \
            --lm_adapted_path  /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin \
            --prompt_number $onenum \
            --ifckpt_onlymodel 1
    echo "++++++++++++++++++++++++++++++"
    ps aux | grep main_prompt_adafactor.py | awk '{print $2}' | xargs kill -9
  done
done


