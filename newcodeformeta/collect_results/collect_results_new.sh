allinnerlr=(2e-5 3e-5 5e-5)
allgradient=(1 2 4)
allstep=(2500 5000 10000)
alloutlr=(2e-1 3e-1 5e-1)

size="large"
onepath=""
for onelr in ${allinnerlr[@]}
do
  for oneg in ${allgradient[@]}
  do
    for onestep in ${allstep[@]}
    do
      for outerlr in ${alloutlr[@]}
      do
        onepath=$onepath" "../models/T5-$size-maml-$onelr-$oneg-$onestep-$outerlr
        #echo $onepath
      done
    done
  done
done

onepath=../models/T5-$size-maml" "../models/T5-$size-fomaml" "../models/T5-$size-multi
python collect_results_new.py --logs_dir $onepath