set -ex

wget --quiet -O /tmp/install_conda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
    sh /tmp/install_conda.sh -b -p /opt/conda && \
    rm /tmp/install_conda.sh

/opt/conda/bin/conda create --name lth python=3.6
/opt/conda/bin/conda install -q -n lth tensorflow-gpu=1.13.1
/opt/conda/bin/conda install -q -n lth \
                     ipython=7.3.0 \
                     keras=2.2.4 \
                     matplotlib=3.0.3 \
                     numpy=1.16.2 \
                     notebook=5.7.6 \
                     pandas=0.24.2 \
                     scikit-learn=0.20.3 \
                     scipy=1.2.1 \
                     seaborn=0.9.0 \
                     tqdm=4.28.1

echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcsfuse htop
