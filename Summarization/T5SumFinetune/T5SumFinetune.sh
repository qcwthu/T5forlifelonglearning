learnrate=(5e-5)
alllambda=(0.10)
allstartindex=(0 1 2)
#alltask=(0 1 2)
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
                python -m torch.distributed.launch --nproc_per_node 1 --master_port 29903 T5SumFinetune.py \
                  --cuda 3 \
                  --lr $onerate \
                  --startindex $onestartindex \
                  --taskindex $onetask \
                  --weight_decay 0.01 \
                  --max_grad_norm 1.0 \
                  --batch_size_per_gpu 1 \
                  --valid_size_per_gpu 8 \
                  --test_size_per_gpu 8 \
                  --gradient_accumulation_steps 4 \
                  --max_epoch 32 \
                  --num_workers 0 \
                  --save_step 100000 \
                  --eval_step 100000 \
                  --seed 42 \
                  --model T5Summarization \
                  --model_name google/t5-v1_1-large \
                  --train_sample \
                  --max_length 512 \
                  --adam_epsilon 1e-8 \
                  --warmup_steps 0.01
            echo "++++++++++++++++++++++++++++++"
            ps aux | grep T5SumFinetune.py | awk '{print $2}' | xargs kill -9
      done
    done
  done
done



