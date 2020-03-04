#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color

if [[ ! -x "$(command -v jupyter)" ]]; then
    echo -e "${RED}jupyter not installed${NC}"
    exit
fi
# generate notebook config
jupyter notebook --generate-config

# patch notebook to broadcast, rather than just running locally
patch ~/.jupyter/jupyter_notebook_config.py <<EOF
204c204
< #c.NotebookApp.ip = 'localhost'
---
> c.NotebookApp.ip = '0.0.0.0'
EOF
