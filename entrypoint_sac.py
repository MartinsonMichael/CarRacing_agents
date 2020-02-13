import argparse
import json
import os
import sys
import time
from os.path import dirname, abspath
import tensorflow as tf
import wandb

from envs.common_envs_utils.env_makers import \
    get_state_type_from_settings_path, \
    get_EnvCreator_by_settings
from common_agents_utils import Config
from sac.SAC import SAC

sys.path.append(dirname(dirname(abspath(__file__))))


def create_config(args):
    config = Config()
    config.environment = None

    mode = get_state_type_from_settings_path(args.env_settings)
    env_creator = get_EnvCreator_by_settings(args.env_settings)
    config.environment_make_function = env_creator(args.env_settings)
    config.name = args.name
    config.agent_class = "SAC"
    log_tb_path = os.path.join('logs', config.agent_class, config.name)
    if not os.path.exists(log_tb_path):
        os.makedirs(log_tb_path)
    config.tf_writer = tf.summary.create_file_writer(log_tb_path)

    config.hyperparameters = {
        "agent_class": "SAC",
        "name": args.name,
        "mode": mode,
        "seed": 12,
        "env_settings_file_path": args.env_settings,
        "high_temperature": args.high_temperature,
        "num_episodes_to_run": 15*10**3,
        "device": args.device,
        "max_episode_steps": 300,
        "random_replay_prefill_ratio": args.start_buffer_random_ratio,
        "env_settings": json.load(open(args.env_settings)),

        "Actor": {
            "learning_rate": 3e-4,
            "gradient_clipping_norm": 0.005,
        },
        "Critic": {
            "learning_rate": 3e-4,
            "gradient_clipping_norm": 0.01,
            "tau": 0.95
        },

        "buffer_size": 125000,
        "save_frequency_episode": 500,
        "min_steps_before_learning": 500,
        "batch_size": 64,
        "discount_rate": 0.99,

        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
    }
    return config


def main(args):
    config = create_config(args)

    agent = SAC(config)

    if args.load != 'none':
        agent.load(args.load)

    if args.eval:
        print('Eval mode [where wont be any Training!]')
        agent.step_with_huge_stats()
        return

    print()
    print('hyperparameters:\n', agent.hyperparameters, '\n\nStart in 5 seconds...')
    wandb.init(
        notes=args.note,
        config=config.hyperparameters,
        name=config.name,
        project='SAC'
    )
    time.sleep(5.0)

    time_taken = agent.run_n_episodes(visualize=False)
    print("Time taken: {}".format(time_taken), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--note', type=str, help='provude note for wandb')
    parser.add_argument(
        '--env-settings',
        type=str,
        default='envs/gym_car_intersect_fixed/settings_sets/env_settings__basic_straight_line.json',
        help='path to CarRacing env settings',
    )
    parser.add_argument('--device', type=str, default='cpu', help='path to CarRacing env settings')
    parser.add_argument('--load', type=str, default='none', help='path to load model')
    parser.add_argument('--high-temperature', action='store_true', help='use high temperature keeping')
    parser.add_argument('--eval', action='store_true', help='only evaluate model (useful with --load flag)')
    parser.add_argument(
        '--start-buffer-random-ratio', type=float, default=1.0,
        help='ratio of random action for replay buffer pre-fill, useful for loaded agents',
    )
    args = parser.parse_args()

    if args.name is None:
        raise ValueError('set name')

    if args.note is None:
        raise ValueError('set note!!!')

    main(args)
