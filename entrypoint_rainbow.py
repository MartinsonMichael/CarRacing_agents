import json
import os
from collections import defaultdict

import wandb

from common_agents_utils import Config
from env import DiscreteWrapper
from env.common_envs_utils.env_makers import get_EnvCreator_with_memory_safe_combiner, get_state_type_from_settings
from rainbow.rainbow import Rainbow

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import functools
from builtins import *  # NOQA

from future import standard_library

standard_library.install_aliases()  # NOQA
import argparse
import os


def create_config(_args):
    config = Config()
    config.environment = None

    env_settings_json = json.load(open(_args.env_settings))

    mode = get_state_type_from_settings(env_settings_json)
    env_creator, config.phi = get_EnvCreator_with_memory_safe_combiner(env_settings_json, DiscreteWrapper)

    config.environment_make_function = env_creator
    config.test_environment_make_function = config.environment_make_function
    config.name = _args.name
    config.debug = _args.debug

    config.table_path = 'Rainbow_tables'
    if not os.path.exists(config.table_path):
        os.makedirs(config.table_path)
    print('MODE : ', mode)

    config.hyperparameters = {
        "agent_class": "Rainbow",
        "name": _args.name,
        "mode": mode,
        "seed": 12,
        "device": _args.device,
        "env_settings": env_settings_json,

        "use_icm": False,

        "save_frequency_episode": 500,
        "log_interval": 20,
        "animation_record_step_frequency": 10,
        "record_animation": _args.record_animation,
        "use_parallel_envs": True,
        "parallel_env_num": 6,
        "track_progress_success_threshold": 0.95,

        "replay_start_size": 2 * 10**4,
        "num_steps_to_run": 50 * 10 ** 7,
        "max_episode_len": 500,

        "discount_rate": 0.99,
        "noisy_net_sigma": 0.5,

        # parameters for Adam optimizer
        "lr": 1e-4,
        "gradient_clipping_norm": 0.3,
        "betas": (0.9, 0.999),
    }

    if config.hyperparameters['device'] == 'cpu':
        config.hyperparameters['gpu'] = -1
    elif config.hyperparameters['device'].startswith('cuda'):
        config.hyperparameters['gpu'] = int(config.hyperparameters['device'].split(':')[1])
    else:
        raise ValueError("unknown device\n"
                         "[for chainerRL.rainbow 'gpu' params should be setted, "
                         "and it usually build from 'device' param]")

    print(f"config.hyperparameters['env_settings']['agent_tracks'][0] : "
          f"{config.hyperparameters['env_settings']['agent_tracks'][0]}")

    config.hyperparameters.update({
        "track_type": defaultdict(
            lambda: 'unknown',
            {
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
        project='Rainbow',
        name=config.name,
        config=config.hyperparameters,
    )

    rainbow_agent = Rainbow(config)

    # if _args.load != 'none':
    #     print(f'load from {_args.load}')
    #     rainbow_agent.load(_args.load)

    print('Start training of Rainbow...')
    rainbow_agent.train()


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
    parser.add_argument('--note', type=str, help='provide note for wandb')
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
