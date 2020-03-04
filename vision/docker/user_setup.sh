# non-login shell
echo 'export PYTHONPATH=/home/lth/lth:$PYTHONPATH' >> /home/lth/.bash_profile
echo 'source /opt/conda/bin/activate lth' >> /home/lth/.bash_profile
echo 'export GOOGLE_APPLICATION_CREDENTIALS=/home/lth/lth/private-key.json' >> /home/lth/.bash_profile

# login shell
echo 'export PYTHONPATH=/home/lth/lth:$PYTHONPATH' >> /home/lth/.bashrc
echo 'source /opt/conda/bin/activate lth' >> /home/lth/.bashrc
echo 'export GOOGLE_APPLICATION_CREDENTIALS=/home/lth/lth/private-key.json' >> /home/lth/.bashrc

source /opt/conda/bin/activate lth
pip install --upgrade --user pyhamcrest pip google-api-python-client oauth2client
