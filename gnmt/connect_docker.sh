#!/usr/bin/env bash

cd "$(dirname $0)"
DIR="$(pwd)"
NAME="crf-gnmt"

function container_id() {
    sudo docker ps -q --filter 'name=^'"${NAME}"'$'
}


function get_sudo() {
    if ! sudo -S true < /dev/null 2> /dev/null; then
        echo "sudo access required for docker:"
        sudo true
    fi
}


get_sudo
CONTAINER="$(container_id)"

if [[ -z "${CONTAINER}" ]]; then
    read -p "Container is not currently running. Would you like to start it? (y/n) " -r

    if [[ !($REPLY =~ ^[Yy]) ]]; then
	echo "Not starting."
	exit 1
    fi

    nvidia-docker run --rm --name "${NAME}" -d -v `pwd`:/home/gnmt_tpu/gnmt_tpu "${NAME}:latest" tail -f /dev/null
    CONTAINER="$(container_id)"
    sudo docker exec "${CONTAINER}" bash -l /home/gnmt_tpu/gnmt_tpu/docker/posthoc_setup.sh
fi


docker exec -it "${CONTAINER}" bash -l /home/gnmt_tpu/gnmt_tpu/misc/tmux_attach.sh
echo "Disconnected. Container is still running in the background! Stop it with ${DIR}/stop_docker.sh"
