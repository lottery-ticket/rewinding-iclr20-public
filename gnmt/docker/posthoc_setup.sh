gcloud config set pass_credentials_to_gsutil false
export GOOGLE_APPLICATION_CREDENTIALS=/home/gnmt_tpu/gnmt_tpu/private-key.json
cat <(echo "$GOOGLE_APPLICATION_CREDENTIALS") <(yes) | gsutil config -e
gcloud auth activate-service-account --key-file "${GOOGLE_APPLICATION_CREDENTIALS}"
