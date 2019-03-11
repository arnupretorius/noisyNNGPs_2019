#!/bin/bash
for weight_var in `seq 1.0 0.01 2.0`
do
    for mu_2 in `seq 1.0 0.01 2.0`
    do
        python run_experiments.py \
      --num_train=1000 \
      --num_eval=1000 \
      --hparams="nonlinearity=relu,depth=10,weight_var=$weight_var,bias_var=0.0, mu_2=$mu_2" \
      --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 
        
    done
done
