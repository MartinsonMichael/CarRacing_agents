import argparse
import json

import gym
import wandb

from common_agents_utils import Config
from envs import get_state_type_from_settings_path, get_EnvCreator_by_settings

from ppo.PPO_continuous import PPO


def create_config(args):
    config = Config()
    config.environment = None

    mode = get_state_type_from_settings_path(args.env_settings)
    env_creator = get_EnvCreator_by_settings(args.env_settings)
    config.environment = env_creator(args.env_settings)()
    config.name = args.name
    config.hyperparameters = {
        "agent_class": "PPO",
        "name": args.name,
        "mode": mode,
        "seed": 12,
        "env_settings_file_path": args.env_settings,
        "device": args.device,
        "env_settings": json.load(open(args.env_settings)),

        "num_episodes_to_run": 15 * 10 ** 3,

        "render": False,
        "solved_reward": 300,  # stop training if avg_reward > solved_reward
        "log_interval": 20,  # print avg reward in the interval
        "max_episodes": 10000,  # max training episodes
        "max_timesteps": 1500,  # max timesteps in one episode

        "update_timestep": 4000,  # update policy every n timesteps
        "action_std": 0.5,  # constant std for action distribution (Multivariate Normal)
        "K_epochs": 80,  # update policy for K epochs
        "eps_clip": 0.2,  # clip parameter for PPO
        "gamma": 0.99,  # discount factor
        "lr": 0.0003,  # parameters for Adam optimizer
        "betas": (0.9, 0.999),

    }
    return config


def main(args):
    config = create_config(args)

    # test
    config.environment = gym.make("BipedalWalker-v2")

    wandb.init(
        notes=args.note,
        project='PPO',
        name=config.name,
        config=config.hyperparameters,
    )
    ppo_agent = PPO(config)

    print('Start training of PPO...')
    ppo_agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        action='store_true',
        help='just add this flag and outputs becames much more interesting'
    )
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--note', type=str, help='provude note for wandb')
    parser.add_argument(
        '--env-settings',
        type=str,
        default='envs/gym_car_intersect_fixed/settings_sets/env_settings__basic_straight_line.json',
        help='path to CarRacing env settings',
    )
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    args = parser.parse_args()

    if not args.debug:
        if args.name is None:
            raise ValueError('set name')

        if args.note is None:
            raise ValueError('set note, it is used for wandb')
    else:
        args.name = 'test'
        args.note = 'just test'
    main(args)
