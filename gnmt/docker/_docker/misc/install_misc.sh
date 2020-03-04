pip install --user boto3 docker-py google-api-python-client
mkdir -p "${HOME}/.config"
curl -L https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-242.0.0-linux-x86_64.tar.gz | tar xz

# non-login shell
echo 'export PATH=$PATH:~/google-cloud-sdk/bin' >> ~/.bash_profile

# login shell
echo 'export PATH=$PATH:~/google-cloud-sdk/bin' >> ~/.bashrc
