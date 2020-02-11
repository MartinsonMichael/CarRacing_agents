import argparse
import os
import sys
from os.path import dirname, abspath
import tensorflow as tf

from envs.common_envs_utils.env_makers import \
    get_state_type_from_settings_path, \
    get_EnvCreator_by_settings
from sac.Config import Config
from sac.SAC import SAC

sys.path.append(dirname(dirname(abspath(__file__))))


def create_config(args):
    config = Config()
    config.seed = 1
    config.environment = None

    mode = get_state_type_from_settings_path(args.env_settings)
    env_creator = get_EnvCreator_by_settings(args.env_settings)
    config.environment = env_creator(args.env_settings)()
    config.env_settings = args.env_settings
    config.mode = mode

    config.high_temperature = args.high_temperature
    config.num_episodes_to_run = 15000
    config.file_to_save_data_results = 'result_cars'
    config.file_to_save_results_graph = 'graph_cars'
    config.show_solution_score = True
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0
    config.runs_per_agent = 1
    config.device = args.device
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = True
    config.max_episode_steps = 300
    config.random_replay_prefill_ration = args.start_buffer_random_ratio

    config.hyperparameters = {
        "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 3e-4,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 0.005,
                "initialiser": "Xavier"
            },
            "Critic": {
                "learning_rate": 3e-4,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 125000,
                "tau": 0.005,
                "gradient_clipping_norm": 0.01,
                "initialiser": "Xavier"
            },

            "save_frequency_episode": 500,
            "min_steps_before_learning": 50000,
            "batch_size": 64,
            "discount_rate": 0.99,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": None,
            "add_extra_noise": True,
            "do_evaluation_iterations": False,
            "clip_rewards": False,

            # "mode_to_use": "normal",
            # "rlkit_mode_parameters": {
            #     "explanation_steps_per_step": 1000,
            #     "update_steps_per_step": 1000,
            # }
        }
    }
    return config


def main(args):
    agent_title = args.name
    if not os.path.exists(os.path.join('logs', agent_title)):
        os.makedirs(os.path.join('logs', agent_title))
    tf_writer = tf.summary.create_file_writer(os.path.join('logs', agent_title))
    agent_config = create_config(args)
    agent_config.name = agent_title

    # random.randint(0, 2 ** 32 - 2)
    agent_config.seed = 42

    agent_config.hyperparameters = agent_config.hyperparameters['Actor_Critic_Agents']
    print("AGENT NAME: {}".format('SAC'))

    agent = SAC(agent_config, name=args.name, tf_writer=tf_writer)

    if args.load != 'none':
        agent.load(args.load)

    print(agent.hyperparameters)
    print("RANDOM SEED ", agent_config.seed)

    if args.eval:
        print('Eval mode [where wont be any Training!]')

        agent.step_with_huge_stats()

        return

    game_scores, rolling_scores, time_taken = agent.run_n_episodes(visualize=False)
    print("Time taken: {}".format(time_taken), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name for experiment')
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

    main(args)
