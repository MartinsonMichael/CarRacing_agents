import argparse
import json
import os
import tensorflow as tf

import wandb

from common_agents_utils import Config
from envs import get_state_type_from_settings_path, get_EnvCreator_by_settings

from ppo.PPO_continuous import PPO


def create_config(args):
    config = Config()
    config.environment = None

    mode = get_state_type_from_settings_path(args.env_settings)
    env_creator = get_EnvCreator_by_settings(args.env_settings)
    config.environment_make_function = env_creator(args.env_settings)
    config.test_environment_make_function = config.environment_make_function
    config.name = args.name
    config.debug = args.debug
    log_tb_path = os.path.join('logs', 'PPO', config.name)
    if not os.path.exists(log_tb_path):
        os.makedirs(log_tb_path)
    config.tf_writer = tf.summary.create_file_writer(log_tb_path)

    config.hyperparameters = {
        "agent_class": "PPO",
        "name": args.name,
        "mode": mode,
        "seed": 12,
        "env_settings_file_path": args.env_settings,
        "device": args.device,
        "env_settings": json.load(open(args.env_settings)),

        "save_frequency_episode": 500,
        "log_interval": 20,

        "num_episodes_to_run": 50 * 10 ** 3,
        "max_episode_len": 500,

        "update_every_n_steps": 5000,
        "learning_updates_per_learning_session": 80,

        "discount_rate": 0.99,
        "eps_clip": 0.2,  # clip parameter for PPO

        # parameters for Adam optimizer
        "lr": 0.0003,
        "gradient_clipping_norm": 1.0,
        "betas": (0.9, 0.999),
    }
    return config


def main(args):
    config = create_config(args)

    wandb.init(
        notes=args.note,
        project='PPO',
        name=config.name,
        config=config.hyperparameters,
    )
    ppo_agent = PPO(config)

    if args.load != 'none':
        print(f'load from {args.load}')
        ppo_agent.load(args.load)

    print('Start training of PPO...')
    ppo_agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        default=False,
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
    parser.add_argument('--load', type=str, default='none', help="path to load model")
    args = parser.parse_args()

    if not args.debug:
        if args.name is None:
            raise ValueError('set name')

        if args.note is None:
            args.note = ''
    else:
        args.name = 'test'
        args.note = 'just test'
    main(args)
