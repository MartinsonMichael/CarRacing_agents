## Experiments with DRQ and RND
Goal was to train agent on Full Rotate track, on pure image, without any vector features.
In order to did that we try to increase exploration with help of RND. That failed.

Also we try DRQ. This was mush more better then RND or pure algorithms. But car didn't rotate.

As environment extension was added:
* settings with visual track points, 
it can be found for example in [CarIntersect settings for DRQ](drq/drq_full_config.yaml).
This settings placed in `state` part and looks like:
```
checkpoints:
    show: true
    size: 5.0
    hide_on_reach: true
```
        
* random initial position - train ability to start from any check point, 
by default starting point is choose uniformly. This settings is also placed in `state`. 
For example in [CarIntersect settings for DRQ](drq/drq_full_config.yaml). 
```
partrandom_start_position: true
random_start_position_bots: false
```

## How to launch
Almost all agents and CarIntersect Env configs can be launched in Docker.

### Pre requirements
1. If you haven't docker on you machine, set up it. 
2. Create file `WANDB_API_KEY.txt` with you wandb api key as a pure string.
Actually, if you don't want to log info to wandb you can ignore it, 
or create file with fake key. But fails of logging can brake training process.  

### Create Docker image
For everything run in terminal (except openai rnd):
`./make_car-racing_docker.sh`

For OpenAI RND run (cause them use old version of tf):
`./make_car-racing_docker_tf1X.sh` 

### Run RND experiments
Not OpenAI: `./run_rnd_exp.sh <arg1 - device> <arg2 - wandb name>`

OpenAI RND: `./run_oprn-ai_rnd.sh <arg1 - device> <arg2 - wandb name>`

where `device` is `cpu` or `cuda:<number of card>`, `wandb name` is name of experiment displayed in wandb

### Run DRQ experiments
`./run_drq_exp.sh <arg1 - device> <arg2 - wandb name>`

where `device` is `cpu` or `cuda:<number of card>`, `wandb name` is name of experiment displayed in wandb

### Run pure agent 
No drq, no rnd, no icm. As RL algorithm any: PPO, Rainbow, TD3, SAC (old version may be incompatible)

To launch:
1. create python environment (requirements in file `requirements.txt`) or use docker image `car-racing`
2. `python experiment_entrypoint.py <args...>`

where `<args...>` list see in `experiment_entrypoint.py` itself. Some important:
* `--name <str>` - name for experiment
* `--exp-settings <str>` - path to experiment config
* `--device <str>` - `cpu` or `cuda:<number>`

