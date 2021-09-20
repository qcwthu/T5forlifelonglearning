echo "Run T5 conll"
cd /data/qin/T5/T5forNER/fewshot/testfinetune
bash trainner.sh
echo "Finish T5 conll"
echo "----------------------"

echo "Run T5 ontonotes"
cd /data/qin/T5/T5forNER/fewshot/testfinetune
bash trainner_onto.sh
echo "Finish T5 ontonotes"
echo "----------------------"

echo "Run T5 testall"
cd /data/qin/T5/T5forNER/fewshot/testfinetune
bash test_finetune_all.sh
echo "Finish T5 testall"
echo "----------------------"

echo "Run T5 ewc"
cd /data/qin/T5/T5forNER/fewshot/testewc
bash testewc.sh
echo "Finish T5 ewc"
echo "----------------------"

echo "Run T5 mas"
cd /data/qin/T5/T5forNER/fewshot/testmas
bash testmas.sh
echo "Finish T5 mas"
echo "----------------------"
