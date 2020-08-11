#!/bin/bash

if [[ $# -ne 2]]; then
	echo 'please pass 2 arguments, DEVICE and NAME'
	exit 0
fi

wandb_key=$(cat WANDB_API_KEY.txt) 

echo $wandb_key

docker run -it \
	-v .:/src \
	-e WANDB_API_KEY=$wandb_key \
	-e DEVICE=$1 \
        -e NAME=$2 \	
	car-racing bash \
		cd /src/rnd_continues &&
		python train.py
