#!/usr/bin/env bash

cd "$(dirname $0)"
DIR="$(pwd)"
NAME="crf-gnmt"

export ROOT_INSTALL_FILE=docker/root_setup.sh
export PROJECT_NAME=gnmt_tpu
export PROJECT_NAME_UPPERCASE=GNMT_TPU
export USER_INSTALL_FILE=docker/user_setup.sh

docker build . -f docker/_docker/Dockerfile -t "${NAME}:latest" \
       --build-arg ROOT_INSTALL_FILE=docker/root_setup.sh \
       --build-arg PROJECT_NAME=gnmt_tpu \
       --build-arg PROJECT_NAME_UPPERCASE=GNMT_TPU \
       --build-arg USER_INSTALL_FILE=docker/user_setup.sh \
       --build-arg UID=`id -u`
