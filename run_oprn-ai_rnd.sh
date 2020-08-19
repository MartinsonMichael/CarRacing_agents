#!/bin/bash

if [[ $# != 1 ]]; then
	echo "please pass 1 arguments, CUDA device (it 0, 1, etc.)"
	exit 0
fi

wandb_key=$(cat WANDB_API_KEY.txt)

docker run -it \
  --gpus all \
  -v "$PWD":/src \
  -e WANDB_API_KEY=$wandb_key \
  -e CUDA_VISIBLE_DEVICES=$1 \
  car-racing-tf1x:latest bash -c \
    "echo 'Use python3.6 version:' && \
    python3.6 --version && \
    cd open-ai-rnd && python3.6 run_atari.py"