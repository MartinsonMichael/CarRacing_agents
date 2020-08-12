#!/bin/bash

if [[ $# != 2 ]]; then
	echo "please pass 2 arguments, DEVICE and NAME"
	exit 0
fi

wandb_key=$(cat WANDB_API_KEY.txt) 

if [[ $1 = 'cpu' ]]; then
  echo "Run CPU version of DRQ"

  docker run -it \
    -v "$PWD":/src \
    -e WANDB_API_KEY=$wandb_key \
    -e DEVICE=$1 \
    -e NAME=$2 \
    --user "$(id -u):$(id -g)" \
    car-racing:latest bash -c \
      "echo 'Use python3.6 version:' && \
      python3.6 --version && \
      cd drq && python3.6 train.py"
else
  echo "Run GPU version of DRQ"

  docker run -it \
    --gpus all \
    -v "$PWD":/src \
    -e WANDB_API_KEY=$wandb_key \
    -e DEVICE=$1 \
    -e NAME=$2 \
    --user "$(id -u):$(id -g)" \
    car-racing:latest bash -c \
      "echo 'Use python3.6 version:' && \
      python3.6 --version && \
      cd drq && python3.6 train.py"
fi