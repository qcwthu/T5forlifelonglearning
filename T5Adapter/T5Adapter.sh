learnrate=(1e-4)
#learnrate=(5e-5)
allstartindex=(0 1 2)
for onerate in ${learnrate[@]}
do
  for onestartindex in ${allstartindex[@]}
  do
              echo "------------------------------"
              python -m torch.distributed.launch --nproc_per_node 1 --master_port 29943 T5Adapter.py \
                --cuda 1 \
                --lr $onerate \
                --startindex $onestartindex \
                --weight_decay 0.01 \
                --max_grad_norm 1.0 \
                --batch_size_per_gpu 1 \
                --valid_size_per_gpu 8 \
                --test_size_per_gpu 8 \
                --gradient_accumulation_steps 4 \
                --max_epoch 80 \
                --num_workers 0 \
                --save_step 100000 \
                --eval_step 100000 \
                --seed 42 \
                --model_name google/t5-v1_1-large \
                --train_file_name ../T5forNER/ontonotes_fewshot/train.txt \
                --valid_file_name ../T5forNER/ontonotes_fewshot/valid.txt \
                --test_file_name ../T5forNER/ontonotes_fewshot/test.txt \
                --train_sample \
                --max_length 512 \
                --adam_epsilon 1e-8 \
                --warmup_steps 0.01
              echo "++++++++++++++++++++++++++++++"
              ps aux | grep T5Adapter.py | awk '{print $2}' | xargs kill -9
  done
done



