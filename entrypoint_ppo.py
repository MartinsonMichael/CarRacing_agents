import argparse
import json
import os
import tensorflow as tf

import gym
import gym.wrappers.monitor
import wandb

from common_agents_utils import Config
from envs import get_state_type_from_settings_path, get_EnvCreator_by_settings
from envs.common_envs_utils.extended_env_wrappers import ObservationToFloat32, RewardDivider, VisualizerWrapper

from ppo.PPO_continuous import PPO


def create_config(args):
    config = Config()
    config.environment = None

    mode = get_state_type_from_settings_path(args.env_settings)
    env_creator = get_EnvCreator_by_settings(args.env_settings)
    config.environment_make_function = env_creator(args.env_settings)
    config.name = args.name
    config.debug = args.debug
    log_tb_path = os.path.join('logs', config.agent_class, config.name)
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

        "num_envs": 4,
        "save_frequency_episode": 500,

        "num_episodes_to_run": 15 * 10 ** 3,
        "max_episode_len": 350,

        "update_every_n_steps": 3000,
        "learning_updates_per_learning_session": 60,

        "discount_rate": 0.99,
        "eps_clip": 0.05,

        # parameters for Adam optimizer
        "lr": 0.001,
        "gradient_clipping_norm": 0.2,
        "betas": (0.9, 0.999),
    }
    return config


def main(args):
    config = create_config(args)

    # if config.debug:
    # test
    config.environment_make_function = lambda: ObservationToFloat32(gym.make("LunarLanderContinuous-v2"))
    config.test_environment_make_function = lambda: VisualizerWrapper(
        ObservationToFloat32(gym.make("LunarLanderContinuous-v2"))
    )

    # config.test_environment_make_function = lambda: gym.wrappers.Monitor(
    #     ObservationToFloat32(gym.make("LunarLanderContinuous-v2")),
    #     directory='save_animation_folder',
    #     mode='evaluation',
    #     force=True,
    # )

    if not config.debug:
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
            raise ValueError('set note, it is used for wandb')
    else:
        args.name = 'test'
        args.note = 'just test'
    main(args)
