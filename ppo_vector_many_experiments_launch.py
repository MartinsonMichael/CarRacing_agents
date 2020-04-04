import argparse
import collections
import os
from threading import Thread

import chainerrl
import tensorflow as tf
from typing import Iterable, Tuple
import wandb
import numpy as np

from common_agents_utils import Config
from env import CarRacingEnv, OnlyVectorsTaker
from ppo.PPO_ICM_continuous import PPO_ICM


env_settings = {
  "name": "Basic, straight line",
  "cars_path": "env/env_data/cars",
  "background_path": "env/env_data/tracks/background_image_1520_1520.jpg",
  "annotation_path": "env/env_data/tracks/CarRacing_sq_extended_v2.0.xml",
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
  "agent_tracks": None,
  "agent_image_indexes": [0],
  "bot_number": 0,
  "bots_tracks": [0],
  "image_scale": {
    "back_image_scale_factor": 0.12,
    "car_image_scale_factor": 0.1
  },
  "steer_policy": {
    "angle_steer": False,
    "angle_steer_multiplication": 5.0
  },
  "state_config": {
    "picture": False,
    "vector_car_features": None,
    "vector_env_features": []
  },
  "reward": {
    "track_checkpoint_expanding": 50,

    "is_collided": 0.0,
    "is_finish": 5,
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
    "true_flags_to_done": ["is_out_of_road", "is_out_of_map", "is_out_of_track", "is_finish", "is_collided"],
    "false_flags_to_done": []
  }
}


def iterate_over_configs(_args) -> Iterable[Tuple[Config, str]]:
    config = Config()
    mode = 'vector'
    print('MODE : ', mode)

    config.hyperparameters = {
        "agent_class": "PPO",
        "name": None,
        "mode": mode,
        "track_type": None,
        "seed": 12,
        "device": _args.device,
        "env_settings": env_settings,

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
        "track_progress_success_threshold": 0.92,

        "num_episodes_to_run": 15 * 10**3,
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

    def deep_dict_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = deep_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # track type 'line': [0] 'rotate': [1], 'rotate_over_line': [2]

    for exp_cycle_index in range(_args.exp_num):
        for index, params in enumerate([
            # Example:
            # {
            #     "track_type": "",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": []},
            #         "bot_number": 0,
            #     },
            # },

            # why angle is important?
            # {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position"
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "wheels_positions"
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "wheels_positions", "hull_angle",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle",
            #         ]},
            #         "bot_number": 0,
            #     },
            # },
            # ##
            #
            #
            # # what can help to converge faster?
            # {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "car_speed",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "track_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "wheels_positions",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "finish_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "line",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "time",
            #         ]},
            #         "bot_number": 0,
            #     },
            # },
            # ##
            #
            # # with track = rotate
            # {
            #     "track_type": "rotate",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "car_speed", "cross_road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "rotate",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "track_sensor", "cross_road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "rotate",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "wheels_positions", "cross_road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "rotate",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "finish_sensor", "cross_road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "rotate",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "road_sensor", "cross_road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # }, {
            #     "track_type": "rotate",
            #     "env_settings": {
            #         "state_config": {"picture": False, "vector_car_features": [
            #             "hull_position", "hull_angle", "time", "cross_road_sensor",
            #         ]},
            #         "bot_number": 0,
            #     },
            # },


            # Line with Bots
            {
                "track_type": "line",
                "env_settings": {
                    "state_config": {"picture": False, "vector_car_features": [
                        "hull_position", "hull_angle",
                    ]},
                    "bot_number": 4, "bot_tracks": [0],
                },
            },
            {
                "track_type": "line",
                "env_settings": {
                    "state_config": {"picture": False, "vector_car_features": [
                        "hull_position", "hull_angle", "cross_road_sensor",
                    ]},
                    "bot_number": 4, "bot_tracks": [0],
                },
            },
            {
                "track_type": "line",
                "env_settings": {
                    "state_config": {"picture": False, "vector_car_features": [
                        "hull_position", "hull_angle", "car_radar_2",
                    ]},
                    "bot_number": 4, "bot_tracks": [0],
                },
            },
            {
                "track_type": "line",
                "env_settings": {
                    "state_config": {"picture": False, "vector_car_features": [
                        "hull_position", "hull_angle", "car_radar_2", "cross_road_sensor",
                    ]},
                    "bot_number": 4, "bot_tracks": [0],
                },
            },

        ]):
            config.hyperparameters = deep_dict_update(
                config.hyperparameters,
                params,
            )
            config.hyperparameters['env_settings']['agent_tracks'] = {'line': [0], 'rotate': [1]}[
                config.hyperparameters['track_type']
            ]
            config.hyperparameters['seed'] = int(np.random.random_integers(1, 10**4))

            config.name = f"exp_{_args.name}_{exp_cycle_index}_{index}"
            log_tb_path = os.path.join('logs', 'PPO', config.name)
            if not os.path.exists(log_tb_path):
                os.makedirs(log_tb_path)
            config.tf_writer = tf.summary.create_file_writer(log_tb_path)

            def env_creator():
                env = CarRacingEnv(settings_file_path_or_settings=env_settings)
                env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=500)
                env = OnlyVectorsTaker(env)
                env._max_episode_steps = 500
                return env

            config.environment_make_function = env_creator
            config.test_environment_make_function = env_creator
            config.debug = False
            config.table_path = _args.table_path
            if not os.path.exists(config.table_path):
                os.makedirs(config.table_path)

            wandb_note = f"Track : {config.hyperparameters['track_type']}. Use " + ", ".join(
                config.hyperparameters['env_settings']['state_config']['vector_car_features']
            )

            yield config, wandb_note


def main(_args):
    for config, wandb_note in iterate_over_configs(_args):

        wandb.init(
            reinit=True,
            project='PPO_series',
            name=config.name,
            notes=wandb_note,
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
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    parser.add_argument('--table-path', type=str, help='path to table records')
    parser.add_argument('--exp-num', type=int, default=1)

    args = parser.parse_args()

    if args.name is None:
        raise ValueError('set name')

    if args.table_path is None:
        raise ValueError('set table path')

    main(args)
