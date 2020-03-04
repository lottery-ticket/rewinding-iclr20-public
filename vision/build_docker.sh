#!/usr/bin/env bash

cd "$(dirname $0)"
DIR="$(pwd)"
NAME="crf-vision"

export ROOT_INSTALL_FILE=docker/root_setup.sh
export PROJECT_NAME=lth
export PROJECT_NAME_UPPERCASE=LTH
export USER_INSTALL_FILE=docker/user_setup.sh

docker build . -f docker/_docker/Dockerfile -t "${NAME}:latest" \
       --build-arg ROOT_INSTALL_FILE=docker/root_setup.sh \
       --build-arg PROJECT_NAME=lth \
       --build-arg PROJECT_NAME_UPPERCASE=LTH \
       --build-arg USER_INSTALL_FILE=docker/user_setup.sh \
       --build-arg UID=`id -u`
