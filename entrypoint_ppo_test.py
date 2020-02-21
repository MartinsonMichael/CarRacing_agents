import argparse
import json
import os
import tensorflow as tf

import gym
import wandb

from common_agents_utils import Config
# from common_agents_utils.VisualWrapper import VisualWrapper
from envs import get_state_type_from_settings_path, get_EnvCreator_by_settings
from envs.common_envs_utils.extended_env_wrappers import ObservationToFloat32, RewardDivider
from ppo.PPO_ICM_continuous import PPO_ICM

from ppo.PPO_continuous import PPO


def create_config(args):
    config = Config()
    config.environment = None

    config.environment_make_function = lambda: ObservationToFloat32(gym.make("LunarLanderContinuous-v2"))
    config.test_environment_make_function = config.environment_make_function

    config.name = args.name
    config.debug = args.debug
    log_tb_path = os.path.join('logs', config.agent_class, config.name)
    if not os.path.exists(log_tb_path):
        os.makedirs(log_tb_path)
    config.tf_writer = tf.summary.create_file_writer(log_tb_path)

    config.hyperparameters = {
        "agent_class": "PPO",
        "name": args.name,
        "seed": 12,
        "device": args.device,

        "save_frequency_episode": 500,
        "use_eval": args.use_eval,
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

    if config.debug:
        print('use DEBUG MODE')

    if not config.debug:
        wandb.init(
            notes=args.note,
            project='PPO',
            name=config.name,
            config=config.hyperparameters,
        )

    ppo_agent = PPO_ICM(config)
    # if args.icm:
    #     print('USE ICM')
    #     ppo_agent = PPO_ICM(config)
    # else:
    #     ppo_agent = PPO(config)

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
    parser.add_argument('--icm', action='store_true', help='use new mode')
    parser.add_argument('--use-eval', action='store_true', help='use eval episode while training')
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--note', type=str, help='provude note for wandb')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    parser.add_argument('--load', type=str, default='none', help="path to load model")
    args = parser.parse_args()

    if args.name is None:
        args.name = 'test'

    if args.note is None:
        args.note = 'no note'

    main(args)
