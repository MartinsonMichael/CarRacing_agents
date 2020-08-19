#!/bin/bash

#if [[ $# != 2 ]]; then
#	echo "please pass 2 arguments, DEVICE and NAME"
#	exit 0
#fi

wandb_key=$(cat WANDB_API_KEY.txt) 

#  -e WANDB_API_KEY=$wandb_key \
#  -e DEVICE=$1 \
#  -e NAME=$2 \

docker run -it \
  --gpus all \
  -v "$PWD":/src \
  car-racing-tf1x:latest bash -c \
    "echo 'Use python3.6 version:' && \
    python3.6 --version && \
    cd open-ai-rnd && python3.6 run_atari.py"