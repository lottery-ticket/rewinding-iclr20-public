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

if [[ ! -z "${CONTAINER}" ]]; then
    sudo docker stop "${CONTAINER}"
fi
