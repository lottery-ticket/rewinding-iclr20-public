export RUN_VENV="/home/gnmt_tpu/tpu_run_env"

function run() {
    echo export MLP_TPU_TF_VERSION=1.12
    echo export MLP_TF_PIP_LINE=tensorflow==1.12.0
    echo export MLP_CIDR_SIZE=29
    echo export MLP_TPU_VERSION=v3-8
    echo export MLP_TPU_SIDECAR=N
    echo export MLP_TPU_PREEMPT="--preemptible"
    echo export BENCHMARK=gnmt
    echo export MLP_OVERRIDE_TPU=N

    echo "export RUN_VENV='${RUN_VENV}'"
    echo "source ${RUN_VENV}/bin/activate"
    echo 'export GOOGLE_APPLICATION_CREDENTIALS=/home/gnmt_tpu/gnmt_tpu/private-key.json'
}

run >> /home/gnmt_tpu/.bash_profile
run >> /home/gnmt_tpu/.bashrc

tmp="$(mktemp)"
run > "$tmp"
. "$tmp"



virtualenv -p python3 ${RUN_VENV:?}
source ${RUN_VENV:?}/bin/activate

pip install --upgrade --progress-bar=off pyyaml==3.13 oauth2client==4.1.3 google-api-python-client==1.7.4 google-cloud==0.34.0
pip install --progress-bar=off mlperf_compliance==0.0.10
pip install --progress-bar=off cloud-tpu-profiler==1.12

# Note: this could be over-ridden later
TF_TO_INSTALL=${MLP_TF_PIP_LINE:?}
pip install --progress-bar=off $TF_TO_INSTALL
pip install sacrebleu==1.2.11
