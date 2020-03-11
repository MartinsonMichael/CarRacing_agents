import argparse
import json
import os

import chainerrl
import tensorflow as tf
from typing import Iterable, Tuple
import wandb

from common_agents_utils import Config
from envs import get_state_type_from_settings_path, get_EnvCreator_by_settings, CarRacingHackatonContinuousFixed, \
    OnlyVectorsTaker
from ppo.PPO_ICM_continuous import PPO_ICM


env_setting = '''
{
  "name": "Basic, straight line",
  "cars_path": "envs/gym_car_intersect_fixed/env_data/cars",
  "background_path": "envs/gym_car_intersect_fixed/env_data/tracks/background_image_1520_1520.jpg",
  "annotation_path": "envs/gym_car_intersect_fixed/env_data/tracks/CarRacing_sq_extended_v2.0.xml",
  "TRACK_USER_INFO_NOT_A_SETTINGS": {
    "agent_track": {
      "0": "line",
      "1": "small rotate",
      "2": "rotate over line"
    },
    "agent_image_indexes": "use just 0, it works fine",
    "bot_track": {
      "0": "up-down line, cross all agent tracks",
      "1": "left-right line, cross only '2' agent track",
      "bot_number": "just number of bot which will appear in a single moment"
    }
  },
  "agent_tracks" : [1],
  "agent_image_indexes": [0],
  "bot_number" : 0,
  "bots_tracks": [0, 1],
  "image_scale": {
    "back_image_scale_factor": 0.12,
    "car_image_scale_factor": 0.1
  },
  "steer_policy": {
    "angle_steer": false,
    "angle_steer_multiplication": 5.0
  },
  "state_config": {
    "picture": false,
    "vector_car_features": [],
    "vector_env_features": []
  },
  "reward": {
    "track_checkpoint_expanding": 50,

    "is_collided": 0.0,
    "is_finish": 10,
    "is_out_of_track": -1,
    "is_out_of_map": -1,
    "is_out_of_road": -1,

    "idleness__punish_if_action_radius_less_then": 0.0,
    "idleness__punish_value": 0.0,

    "new_tiles_count": 0.5,
    "speed_multiplication_bonus": 0.0,

    "speed_per_point": 0.0,
    "if_speed_more_then_threshold": 0.0,
    "speed_threshold": 0.0,
    "time_per_point": 0.0,
    "time_per_tick": 0.0
  },
  "done": {
    "true_flags_to_done": ["is_out_of_road", "is_out_of_map", "is_out_of_track", "is_finish"],
    "false_flags_to_done" : []
  }
}
'''


def iterate_over_configs(exp_name) -> Iterable[Tuple[Config, str]]:
    config = Config()
    mode = 'vector'
    settings = json.loads(env_setting)
    print('MODE : ', mode)

    config.hyperparameters = {
        "agent_class": "PPO",
        "name": None,
        "mode": mode,
        "seed": 12,
        "device": args.device,
        "env_settings": settings,

        "use_icm": args.icm,
        "icm_config": {
            "state_mode": mode,
            "state_image_channel_cnt":
                config.test_environment_make_function().state_image_channel_cnt
                if mode in {'image', 'both'} else None,
        },

        "save_frequency_episode": 1000,
        "log_interval": 20,
        "animation_record_frequency": 250,
        "record_animation": args.record_animation,
        "track_progress_success_threshold": 0.8,

        "num_episodes_to_run": 50 * 10 ** 3,
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
        {"hull_position"},
        {"hull_position", "hull_angle"},
        {"hull_position", "hull_angle", "car_speed"},
        {"wheels_positions"},
        {"wheels_positions", "hull_angle"},
        {"wheels_positions", "hull_angle", "car_speed"},
    ]):
        config.name = f"exp_{exp_name}_{index}"
        log_tb_path = os.path.join('logs', 'PPO', config.name)
        if not os.path.exists(log_tb_path):
            os.makedirs(log_tb_path)
        config.tf_writer = tf.summary.create_file_writer(log_tb_path)

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

        wandb_note = "Use " + ", ".join(vector_set)

        yield config, wandb_note


def main(args):
    for config, wandb_note in iterate_over_configs(args.name):

        wandb.init(
            project='PPO',
            name=config.name,
            notes=args.note + ' -> ' + wandb_note if args.note is not None else wandb_note,
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
    parser.add_argument('--note', type=str, default=None, help='name for experiment')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    args = parser.parse_args()

    if args.name is None:
        raise ValueError('set name')

    main(args)
