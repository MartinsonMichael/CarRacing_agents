from collections import defaultdict
from multiprocessing import Process

from common_agents_utils import Config
import time
import argparse
import json
import os

import chainer
import numpy as np

import chainerrl
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DistributionalDuelingDQN
from chainerrl import replay_buffer


# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
# parser.add_argument('--outdir', type=str, default='results',
#                     help='Directory path to save output files.'
#                          ' If it does not exist, it will be created.')
# parser.add_argument('--seed', type=int, default=0,
#                     help='Random seed [0, 2 ** 31)')
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--demo', action='store_true', default=False)
# parser.add_argument('--load-pretrained', action='store_true',
#                     default=False)
# parser.add_argument('--pretrained-type', type=str, default="best",
#                     choices=['best', 'final'])
# parser.add_argument('--load', type=str, default=None)
# parser.add_argument('--use-sdl', action='store_true', default=False)
# parser.add_argument('--eval-epsilon', type=float, default=0.0)
# parser.add_argument('--noisy-net-sigma', type=float, default=0.5)
# parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
# parser.add_argument('--max-frames', type=int,
#                     default=30 * 60 * 60,  # 30 minutes with 60 fps
#                     help='Maximum number of frames for each episode.')
# parser.add_argument('--replay-start-size', type=int, default=2 * 10 ** 4)
# parser.add_argument('--eval-n-steps', type=int, default=125000)
# parser.add_argument('--eval-interval', type=int, default=250000)
# parser.add_argument('--logging-level', type=int, default=20,
#                     help='Logging level. 10:DEBUG, 20:INFO etc.')
# parser.add_argument('--render', action='store_true', default=False,
#                     help='Render env states in a GUI window.')
# parser.add_argument('--monitor', action='store_true', default=False,
#                     help='Monitor env. Videos and additional information'
#                          ' are saved as output files.')
# parser.add_argument('--n-best-episodes', type=int, default=200)
# args = parser.parse_args()
from common_agents_utils.logger import Logger
from env.common_envs_utils.visualizer import save_as_mp4
from rainbow.rainbow_tools import DistributionalDuelingDQN_VectorPicture


class Rainbow:

    def __init__(self, config: Config):
        print('start to init rainbow')
        self.config = config
        self.name = config.name
        self.stat_logger: Logger = Logger(
            config,
            log_interval=config.hyperparameters.get('log_interval', 20),
        )
        self.hyperparameters = config.hyperparameters
        self.env = config.environment_make_function()
        self.test_env = config.test_environment_make_function()

        n_actions = self.test_env.action_space.n

        n_atoms = 51
        v_max = 10
        v_min = -10
        q_func = DistributionalDuelingDQN_VectorPicture(
            self.test_env.observation_space,
            n_actions, n_atoms, v_min, v_max,
        )

        # Noisy nets
        links.to_factorized_noisy(q_func, sigma_scale=self.hyperparameters['noisy_net_sigma'])
        # Turn off explorer
        explorer = explorers.Greedy()

        # Draw the computational graph and save it in the output directory.
        # chainerrl.misc.draw_computational_graph(
        #     [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        #     os.path.join(args.outdir, 'model'))

        # Use the same hyper parameters as https://arxiv.org/abs/1707.06887
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        opt.setup(q_func)

        # Prioritized Replay
        # Anneal beta from beta0 to 1 throughout training
        update_interval = 4
        betasteps = self.hyperparameters['num_steps_to_run'] / update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.5, beta0=0.4, betasteps=betasteps,
            num_steps=3,
            normalize_by_max='memory',
        )

        def phi(x):
            # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255

        self.agent = agents.CategoricalDoubleDQN(
            q_func, opt, rbuf, gpu=self.hyperparameters['gpu'], gamma=0.99,
            explorer=explorer, minibatch_size=32,
            replay_start_size=self.hyperparameters['replay_start_size'],
            target_update_interval=32000,
            update_interval=update_interval,
            batch_accumulator='mean',
            phi=phi,
        )

        self.folder_save_path = os.path.join('model_saves', 'Rainbow', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self._total_grad_steps = 0
        self.current_game_stats = None
        self.flush_stats()
        self.tf_writer = config.tf_writer

        self.accumulated_reward_mean = None
        self.accumulated_reward_std = None

        self._exp_moving_track_progress = 0.0

    def update_current_game_stats(self, reward, done, info):
        self.current_game_stats['reward'] += reward
        self.current_game_stats['env_steps'] += 1.0

        if 'is_finish' in info.keys():
            self.current_game_stats['finish'] = info['is_finish']
        if 'track_progress' in info.keys():
            self.current_game_stats['track_progress'] = info['track_progress']
        if 'is_collided' in info.keys():
            self.current_game_stats['is_collided'] = info['is_collided']

    def flush_stats(self):
        self.current_game_stats = defaultdict(float, {})

    def train(self) -> None:
        try:
            for index in range(self.hyperparameters['num_episodes_to_run']):

                self.run_one_episode()

                self._exp_moving_track_progress = (
                        0.98 * self._exp_moving_track_progress +
                        0.02 * self.current_game_stats.get('track_progress', 0)
                )
                self.current_game_stats.update({
                    'moving_track_progress': self._exp_moving_track_progress,
                    'total_env_episode': self.episode_number,
                    'total_grad_steps': self._total_grad_steps,
                })
                self.stat_logger.log_it(self.current_game_stats)
                if self._exp_moving_track_progress >= self.hyperparameters.get('track_progress_success_threshold', 10):
                    break

                self.flush_stats()

                if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                    self.save()
        finally:
            self.stat_logger.on_training_end()
            self.save(suffix='final')
            pass

    def save(self, suffix=None):
        pass

    def run_one_episode(self):
        record_anim = (
                self.episode_number % self.hyperparameters.get('animation_record_frequency', 1e6) == 0 and
                self.hyperparameters.get('record_animation', False)
        )

        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        images = []

        state = self.env.reset()
        reward = 0

        while not done:

            action = self.agent.act_and_train(state, reward)
            next_state, reward, done, info = self.env.step(action)

            self.global_step_number += 1
            self.update_current_game_stats(reward, done, info)

            if record_anim:
                images.append(self.env.render(full_image=True))

            total_reward += reward
            episode_len += 1

            state = next_state

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len'] \
                    or info.get('needs_reset', False):

                self.agent.stop_episode_and_train(state, reward, done=done)

                if record_anim:
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f'animation/Rainbow/{self.name}/_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4',
                            self.stat_logger
                        ),
                    ).start()

                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                break
