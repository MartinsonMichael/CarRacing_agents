import argparse
import json
import os
from collections import defaultdict

import tensorflow as tf

import wandb

from common_agents_utils import Config
from env.common_envs_utils.env_makers import \
    get_EnvCreator_with_memory_safe_combiner, get_state_type_from_settings
from ppo.PPO_ICM_continuous import PPO_ICM


def create_config(_args):
    config = Config()
    config.environment = None

    env_settings_json = json.load(open(_args.env_settings))

    mode = get_state_type_from_settings(env_settings_json)
    config.environment_make_function, config.phi = get_EnvCreator_with_memory_safe_combiner(env_settings_json)
    config.test_environment_make_function = config.environment_make_function
    config.name = _args.name
    config.debug = _args.debug
    log_tb_path = os.path.join('logs', 'PPO', config.name)
    if not os.path.exists(log_tb_path):
        os.makedirs(log_tb_path)
    config.tf_writer = tf.summary.create_file_writer(log_tb_path)

    config.table_path = 'PPO_tables'
    if not os.path.exists(config.table_path):
        os.makedirs(config.table_path)
    print('MODE : ', mode)

    config.hyperparameters = {
        "agent_class": "PPO",
        "name": _args.name,
        "mode": mode,
        "seed": 12,
        "device": _args.device,
        "env_settings": env_settings_json,

        "use_icm": _args.icm,
        "icm_config": {
            "phi": 'config.phi',
            "icm_gradient_clipping": 1.5,
        },

        "save_frequency_episode": 500,
        "log_interval": 20,
        "animation_record_frequency": 100,
        "record_animation": _args.record_animation,

        "num_episodes_to_run": 50 * 10 ** 3,
        "max_episode_len": 500,

        "update_every_n_steps": 5000,
        "learning_updates_per_learning_session": 1,

        "discount_rate": 0.99,
        "eps_clip": 0.2,  # clip parameter for PPO

        # parameters for Adam optimizer
        "lr": 1e-4,
        "gradient_clipping_norm": 0.3,
        "betas": (0.9, 0.999),
    }

    print(f"config.hyperparameters['env_settings']['agent_tracks'][0] : "
          f"{config.hyperparameters['env_settings']['agent_tracks'][0]}")

    config.hyperparameters.update({
        "track_type": defaultdict(
            lambda: 'unknown', {
                0: 'line',
                1: 'rotate',
                2: 'rotate_over_line',
                4: 'small_rotation',
            })[config.hyperparameters['env_settings']['agent_tracks'][0]]
    })
    print(f"Set track type : {config.hyperparameters['track_type']}")

    return config


def main(_args):
    config = create_config(_args)

    wandb.init(
        notes=_args.note,
        project='PPO',
        name=config.name,
        config=config.hyperparameters,
    )

    ppo_agent = PPO_ICM(config)

    if _args.load != 'none':
        print(f'load from {_args.load}')
        ppo_agent.load(_args.load)

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
    parser.add_argument('--icm', default=False, action='store_true', help='use icm')
    parser.add_argument('--record-animation', default=False, action='store_true', help='use icm')
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--note', type=str, help='provude note for wandb')
    parser.add_argument(
        '--env-settings',
        type=str,
        default='env/settings_sets/env_settings__basic_straight_line.json',
        help='path to CarRacing env settings',
    )
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    parser.add_argument('--load', type=str, default='none', help="path to load model")
    _args = parser.parse_args()

    if not _args.debug:
        if _args.name is None:
            raise ValueError('set name')

        if _args.note is None:
            _args.note = ''
    else:
        _args.name = 'test'
        _args.note = 'just test'
    main(_args)
