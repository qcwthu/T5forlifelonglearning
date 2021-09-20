#echo "Run single ontonotes"
#bash trainner_prompt_ontonotes.sh
#echo "Finish single ontonotes"
#echo "----------------------"

#echo "Run prompt fine tune"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testfinetune_prompt
#bash testfinetune.sh
#echo "Finish prompt fine tune"
#echo "----------------------"

#echo "Run prompt multi task"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testmultitask_prompt
#bash testmultitask.sh
#echo "Finish prompt multi task"
#echo "----------------------"

#echo "Run EWC"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testEWC
#bash testEWC.sh
#echo "Finish EWC"
#echo "----------------------"
#
#echo "Run MAS"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testMAS
#bash testMAS.sh
#echo "Finish MAS"
#echo "----------------------"
#
#echo "Test real data"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testrealdata
#bash testrealdata.sh
#echo "Finish real data"
#echo "----------------------"
#
#echo "Add pseudo samples"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_1
#bash testpseudodata_onlypesudo.sh
#echo "Finish add pseudo samples"
#echo "----------------------"
#
#echo "Add pseudo samples and kl divergence loss"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples
#echo "Finish pseudo samples and kl divergence loss"
#echo "----------------------"
#
#echo "Run single ontonotes"
#cd /data/qin/T5/Prompt_fewshot_withlmloss
#bash trainner_prompt_ontonotes.sh
#echo "Finish single ontonotes"
#echo "----------------------"

#echo "Run prompt fine tune"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testfinetune_prompt
#bash testfinetune.sh
#echo "Finish prompt fine tune"
#echo "----------------------"
#
#echo "Add pseudo samples"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_1
#bash testpseudodata_onlypesudo.sh
#echo "Finish add pseudo samples"
#echo "----------------------"

####----------------------------------------------0826
#echo "Test real data"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testrealdata
#bash testrealdata.sh
#echo "Finish real data"
#echo "----------------------"

#echo "Add pseudo samples"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_1
#bash testpseudodata_onlypesudo.sh
#echo "Finish add pseudo samples"
#echo "----------------------"

#echo "Add pseudo samples and kl divergence loss"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples
#bash testpseudodata.sh
#echo "Finish pseudo samples and kl divergence loss"
#echo "----------------------"

###0918
#echo "Run prompt fine tune"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testfinetune_prompt
#bash testfinetune.sh
#echo "Finish prompt fine tune"
#echo "----------------------"

#echo "Run prompt multi task"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testmultitask_prompt
#bash testmultitask.sh
#echo "Finish prompt multi task"
#echo "----------------------"
#
#echo "Run EWC"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testEWC
#bash testEWC.sh
#echo "Finish EWC"
#echo "----------------------"

echo "Run MAS"
cd /data/qin/T5/Prompt_fewshot_withlmloss/testMAS
bash testMAS.sh
echo "Finish MAS"
echo "----------------------"

###############################################
echo "Test real data"
cd /data/qin/T5/Prompt_fewshot_withlmloss/testrealdata
bash testrealdata.sh
echo "Finish real data"
echo "----------------------"

echo "Add pseudo samples"
cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_1
bash testpseudodata_onlypesudo.sh
echo "Finish add pseudo samples"
echo "----------------------"

echo "Add pseudo samples and kl divergence loss"
cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples
echo "Finish pseudo samples and kl divergence loss"
echo "----------------------"