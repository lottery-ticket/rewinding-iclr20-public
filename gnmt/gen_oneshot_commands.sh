#!/usr/bin/env bash

set -eu

for it in 1 2 3 4 5 6 7 8 9 10; do
    for tr in trial_1 trial_2 trial_3; do
        for den in 80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74 8.59 6.87 5.5 4.4 3.52 2.81 2.25 1.8 1.44 1.15 0.92 0.74 0.59 0.47 0.38; do
            for meth in finetune lottery lr_finetune lr_lottery reinit; do
                echo "/home/gnmt_tpu/gnmt_tpu/model/main.sh ./run.py ${meth} --trial ${tr} --version oneshot_1 --retrain-epochs ${it} --density ${den} --base-dir ${DATA_DIR:-gs://renda}/gnmt_results/v2/base"
            done
        done
    done
done
