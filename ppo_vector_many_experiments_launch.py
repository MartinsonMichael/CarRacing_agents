import argparse
import json
import os

import chainerrl
import tensorflow as tf
from typing import Iterable, Tuple
import wandb

from common_agents_utils import Config
from envs import CarRacingHackatonContinuousFixed, OnlyVectorsTaker
from ppo.PPO_ICM_continuous import PPO_ICM


def iterate_over_configs(_args) -> Iterable[Tuple[Config, str]]:
    config = Config()
    mode = 'vector'
    settings = json.load(open(_args.env_settings))
    print('MODE : ', mode)

    config.hyperparameters = {
        "agent_class": "PPO",
        "name": None,
        "mode": mode,
        "track_type": _args.track,
        "seed": 12,
        "device": _args.device,
        "env_settings": settings,

        "use_icm": _args.icm,
        "icm_config": {
            "state_mode": mode,
            "state_image_channel_cnt":
                config.test_environment_make_function().state_image_channel_cnt
                if mode in {'image', 'both'} else None,
        },

        "save_frequency_episode": 1000,
        "log_interval": 20,
        "animation_record_frequency": 100,
        "record_animation": True,
        "track_progress_success_threshold": 0.85,

        "num_episodes_to_run": 10 * 10 ** 3,
        "max_episode_len": 500,

        "update_every_n_steps": 5000,
        "learning_updates_per_learning_session": 1,

        "discount_rate": 0.99,
        "eps_clip": 0.2,  # clip parameter for PPO

        # parameters for Adam optimizer
        "lr": 0.0005,
        "gradient_clipping_norm": 0.5,
        "betas": (0.9, 0.999),
    }

    config.environment = None

    for index, vector_set in enumerate([
        {"hull_position", "hull_angle", "cross_road_sensor", "collide_sensor"},
        {"wheels_positions", "hull_angle", "cross_road_sensor", "collide_sensor"},
        {"hull_position", "hull_angle", "cross_road_sensor", "collide_sensor", "car_radar_2"},
        {"wheels_positions", "hull_angle", "cross_road_sensor", "collide_sensor", "car_radar_2"},
    ]):
        config.name = f"exp_{_args.name}_{index}"
        log_tb_path = os.path.join('logs', 'PPO', config.name)
        if not os.path.exists(log_tb_path):
            os.makedirs(log_tb_path)
        config.tf_writer = tf.summary.create_file_writer(log_tb_path)

        settings['agent_tracks'] = {'line': [0], 'rotate': [1], 'rotate_over_line': [2]}[_args.track]
        settings['state_config']['vector_car_features'] = list(vector_set)
        config.hyperparameters['env_settings'] = settings

        def env_creator():
            env = CarRacingHackatonContinuousFixed(settings_file_path_or_settings=settings)
            env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=500)
            env = OnlyVectorsTaker(env)
            env._max_episode_steps = 500
            return env

        config.environment_make_function = env_creator
        config.test_environment_make_function = env_creator
        config.debug = False

        wandb_note = f"Track : {args.track}. Use " + ", ".join(vector_set)

        yield config, wandb_note


def main(_args):
    for config, wandb_note in iterate_over_configs(_args):

        wandb.init(
            reinit=True,
            project='PPO_series',
            name=config.name,
            notes=_args.note + ' -> ' + wandb_note if _args.note is not None else wandb_note,
            config=config.hyperparameters,
        )

        ppo_agent = PPO_ICM(config)

        print('Start training of PPO...')
        print(f'note : {wandb_note}')

        ppo_agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--icm', default=False, action='store_true', help='use icm')
    parser.add_argument('--record-animation', default=False, action='store_true', help='use icm')
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--env-settings', type=str, default=None, help='name for experiment')
    parser.add_argument('--note', type=str, default=None, help='name for experiment')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    parser.add_argument('--track', type=str, default=None, help='name for experiment')

    args = parser.parse_args()

    if args.name is None:
        raise ValueError('set name')

    if args.track not in {'line', 'rotate', 'rotate_over_line'}:
        raise ValueError("set track, it is one of {'line', 'rotate', 'rotate_over_line'}")

    if args.env_settings is None:
        raise ValueError('set env settings')

    main(args)
