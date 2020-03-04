#!/usr/bin/env bash

set -eu

for it in 1 2 3 4 5 6 7 8 9 10; do
    for tr in trial_1 trial_2 trial_3; do
        for meth in finetune lottery lr_finetune lr_lottery reinit; do
            echo "/home/gnmt_tpu/gnmt_tpu/model/main.sh ./iterative.py --method ${meth} --trial ${tr} --version iterative_1 --retrain-epochs ${it} --base-dir gs://renda/gnmt_results/v2/base"
        done
        done
    done
done
