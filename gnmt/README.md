# GNMT

The code in this folder was used to generate the GNMT data for the paper.

## Setup

To set up the environment, install [nvidia-docker 1.0](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-1.0)), along with a NVIDIA driver compatible with CUDA 10.0 (i.e., >= 410.48).
Run `./build_docker.sh`, which will build a Docker image.
Launch that Docker image with `./connect_docker.sh`, which will launch a persistent Docker container with this `gnmt` directory mounted under `/home/gnmt_tpu/gnmt_tpu`, and launch a tmux session in that Docker container.
To train a CIFAR-10 network, consult the README in the [gpu-src directory](./gpu-src/).
To train an ImageNet network, consult the README in the [tpu-src/models directory](./tpu-src/models/).

Start a TPU v2-8 or v3-8, using Tensorflow 1.12.
Set the variable `MLP_TPU_NAME` to refer to the name of this TPU.
Set up a gcloud bucket that you can write to and read from within the Docker instance (take a look at [docker/posthoc_setup.sh](docker/posthoc_setup.sh) with a [private key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys) to see how we do this).

Download WMT16, as described [here](https://github.com/mlperf/training/tree/69dbe8b/rnn_translator#steps-to-download-and-verify-data)

## Initial training

To train a network, run:
```
export MLP_PATH_GCS_NMT=WHEREVER_WMT_IS_LOCATED
export DATA_DIR=gs://${YOUR_BUCKET_HERE}

model/run.py train --trial trial_1 --version v1
# print the eval results
gsutil cat gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/trial_1/eval_14/bleu
```

## Rewinding weights and learning rate
```
model/run.py lottery --trial trial_1 --version v1 --retrain-epochs 5 --density 80.0 --base-dir "gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/base"
# print the eval results
gsutil cat gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/lottery/retrain_5/density_80.0/trial_1/eval_14/bleu
```

## Rewinding just learning rate
```
model/run.py lr_finetune --trial trial_1 --version v1 --retrain-epochs 5 --density 80.0 --base-dir "gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/base"
# print the eval results
gsutil cat gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/lr_finetune/retrain_5/density_80.0/trial_1/eval_14/bleu
```

## Standard fine-tuning
```
model/run.py finetune --trial trial_1 --version v1 --retrain-epochs 5 --density 80.0 --base-dir "gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/base"
# print the eval results
gsutil cat gs://${YOUR_BUCKET_HERE}/gnmt_results/v1/finetune/retrain_5/density_80.0/trial_1/eval_14/bleu
```
