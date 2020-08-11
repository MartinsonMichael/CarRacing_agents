#!/bin/bash

if [[ $# != 2 ]]; then
	echo "please pass 2 arguments, DEVICE and NAME"
	exit 0
fi

wandb_key=$(cat WANDB_API_KEY.txt) 

docker run -it \
	-v "$PWD":/src \
	-e WANDB_API_KEY=$wandb_key \
	-e DEVICE=$1 \
        -e NAME=$2 \
	--gpus all \
	car-racing:latest bash -c \
	  "echo 'Use python3.6 version:' && \
	  python3.6 --version && \
	  cd rnd_continues && python3.6 train.py"
