#!/usr/bin/env bash

pip install --user /home/lth/lth/lottery

if [ ! -z "${LOCAL_DOCKER+x}" ]; then
   exit
fi

gcloud config set pass_credentials_to_gsutil false
export GOOGLE_APPLICATION_CREDENTIALS=/home/lth/lth/private-key.json
cat <(echo "$GOOGLE_APPLICATION_CREDENTIALS") <(yes) | gsutil config -e
gcloud auth activate-service-account --key-file "${GOOGLE_APPLICATION_CREDENTIALS}"
