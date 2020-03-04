# Comparing Rewinding and Fine-tuning in Neural Network Pruning

This is a public release of the code for *Comparing Rewinding and Fine-tuning in Neural Network Pruning*
The data and plotting code is in [plot/](./plot).
The code in this repo was used to generate all of the data and plots in the paper.

The ResNet-20 implementation (using GPUs) can be found in `gpu-src/official/resnet`, and is based off of [the implementation provided alongside Tensorflow](https://github.com/tensorflow/models/tree/v1.13.0/official/resnet).
The VGG-16 implementation (using GPUs) can be found in `gpu-src/official/vgg`, and is a fork of the ResNet-20 implementation, modified to reflect a VGG-16 with batch norm and a with single fully connected layer at the end.
The ResNet-50 implementation (using TPUs) can be found in `tpu-src/models/official/resnet` and is based off of [the implementation provided alongside Tensorflow](https://github.com/tensorflow/tpu/tree/98497e0b/models/official/resnet).
The GNMT implementation (using TPUs) can be found in `gnmt/`, and us based off of [the implementation used for Google's MLPerf 0.5 submission](https://github.com/mlperf/training_results_v0.5/tree/7238ee7/v0.5.0/google/cloud_v3.8/gnmt-tpuv3-8).

The rewinding and pruning code can be found in `lottery/lottery/lottery.py` and `lottery/lottery/prune_functions.py`.

## Running the code

To set up the environment, install [nvidia-docker 1.0](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-1.0)), along with a NVIDIA driver compatible with CUDA 10.0 (i.e., >= 410.48).
Run `./build_docker.sh`, which will build a Docker image.
Launch that Docker image with `./connect_docker.sh`, which will launch a persistent Docker container with this `vision` directory mounted under `/home/lth/lth`, and launch a tmux session in that Docker container.
To train a CIFAR-10 network, consult the README in the [gpu-src directory](./gpu-src/).
To train an ImageNet network, consult the README in the [tpu-src/models directory](./tpu-src/models/).
