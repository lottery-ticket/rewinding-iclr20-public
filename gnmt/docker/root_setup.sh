export DEBIAN_FRONTEND=noninteractive

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get install -y apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install -y google-cloud-sdk

set -e

# Not sure why this happens... but it sometimes causes errors if not...
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -

apt-get install -y --quiet expect
apt-get install -y --quiet python3-pip virtualenv python-virtualenv
