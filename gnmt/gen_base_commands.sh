#!/usr/bin/env bash

set -eu

for tr in `seq 1 10`; do
    tr="trial_${tr}"
    echo "/home/gnmt_tpu/gnmt_tpu/model/main.sh ./run.py train --trial ${tr} --version base_1"
done
