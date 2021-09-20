#echo "Run pseudo+kl_gradient"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_combinekl_agem
#bash testcombineklandagem.sh
#echo "Finish pseudo+kl_gradient"
#echo "----------------------"

#echo "Add pseudo samples and kl divergence loss"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples
#bash testpseudodata.sh
#echo "Finish pseudo samples and kl divergence loss"
#echo "----------------------"

######0825

#echo "Run prompt multi task"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testmultitask_prompt
#bash testmultitask.sh
#echo "Finish prompt multi task"
#echo "----------------------"
#
#echo "Run single ontonotes"
#cd /data/qin/T5/Prompt_fewshot_withlmloss
#bash trainner_prompt.sh
#echo "Finish single ontonotes"
#echo "----------------------"
#
#echo "Add pseudo samples"
#cd /data/qin/T5/Prompt_fewshot_withlmloss
#bash test_generation.sh
#echo "Finish add pseudo samples"
#echo "----------------------"
#
#echo "Run momentum changenew"
#cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_trymomentum_changenew
#bash testmomentum_changenew.sh
#echo "Finish momentum changenew"
#echo "----------------------"


###0826 0827
echo "Run momentum changenew"
cd /data/qin/T5/Prompt_fewshot_withlmloss/testpseudosamples_trymomentum_changenew
bash testmomentum_changenew.sh
echo "Finish momentum changenew"
echo "----------------------"