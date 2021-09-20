learnrate=(3e-1 5e-1 4e-1 2e-1)
promptnumber=(300 320 400 200)

for onenum in ${promptnumber[@]}
do
  for onerate in ${learnrate[@]}
  do
    echo "------------------------------"
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 29505 main_prompt_adafactor_ontonotes.py \
            --cuda 4,2 \
            --lr $onerate \
            --weight_decay 1e-5 \
            --max_grad_norm 1.0 \
            --batch_size_per_gpu 8 \
            --valid_size_per_gpu 16 \
            --test_size_per_gpu 16 \
            --gradient_accumulation_steps 1 \
            --max_epoch 1280 \
            --num_workers 4 \
            --save_step 100000 \
            --eval_step 100000 \
            --save_dir t5ner_onto_ckpt \
            --seed 42 \
            --model T5NER \
            --model_name google/t5-v1_1-large \
            --train_file_name ../T5forNER/ontonotes_fewshot/train.txt \
            --valid_file_name ../T5forNER/ontonotes_fewshot/valid.txt \
            --test_file_name ../T5forNER/ontonotes_fewshot/test.txt \
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
    ps aux | grep main_prompt_adafactor_ontonotes.py | awk '{print $2}' | xargs kill -9
  done
done


