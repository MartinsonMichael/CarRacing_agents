from collections import defaultdict
from multiprocessing import Process

from common_agents_utils import Config, SubprocVecEnv_tf2
import time
import argparse
import json
import os

import chainer
import numpy as np

from chainerrl import agents
from chainerrl import explorers
from chainerrl import links
from chainerrl import replay_buffer


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
        if self.hyperparameters.get('use_parallel_envs', False):
            self.env = SubprocVecEnv_tf2(
                [
                    config.environment_make_function
                    for _ in range(self.hyperparameters['parallel_env_num'])
                ],
                state_flatter=None,
            )
        else:
            self.env = config.environment_make_function()

        self.test_env = config.test_environment_make_function()

        # function to prepare row observation to chainer format
        print(f"rainbow mode : {self.hyperparameters['mode']}")

        n_actions = self.test_env.action_space.n

        n_atoms = 51
        v_max = 10
        v_min = -10
        q_func = DistributionalDuelingDQN_VectorPicture(
            config.phi(self.test_env.reset()).shape,
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

        self.agent = agents.CategoricalDoubleDQN(
            q_func, opt, rbuf, gpu=self.hyperparameters['gpu'], gamma=0.99,
            explorer=explorer, minibatch_size=32,
            replay_start_size=self.hyperparameters['replay_start_size'],
            target_update_interval=32000,
            update_interval=update_interval,
            batch_accumulator='mean',
            phi=config.phi,
        )

        self.folder_save_path = os.path.join('model_saves', 'Rainbow', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self.batch_step_number = 0
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
        if self.hyperparameters['use_parallel_envs']:
            self._batch_train()
        else:
            raise NotImplemented

    def _batch_train(self) -> None:
        num_env = self.hyperparameters['parallel_env_num']
        total_reward = np.zeros(num_env, dtype=np.float32)
        episode_len = np.zeros(num_env, dtype=np.int32)
        state = self.env.reset()
        try:
            while True:
                self.global_step_number += num_env
                self.batch_step_number += 1

                actions = self.agent.batch_act_and_train(state)
                state, reward, dones, infos = self.env.step(actions)

                total_reward += reward
                episode_len += 1

                # Compute mask for done and reset
                resets = np.logical_or(
                    episode_len == self.hyperparameters['max_episode_len'],
                    np.logical_or(
                        [info.get('needs_reset', False) for info in infos],
                        [info.get('need_reset', False) for info in infos],
                    ),
                )
                # Agent observes the consequences
                self.agent.batch_observe_and_train(state, reward, dones, resets)

                self.update_current_game_stats(reward, dones[0], infos[0])
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
                if self.global_step_number > self.hyperparameters['num_steps_to_run']:
                    break

                self.flush_stats()

                if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                    self.save()

                # Make mask. 0 if done/reset, 1 if pass
                end = np.logical_or(resets, dones)
                not_end = np.logical_not(end)

                self.episode_number += int(end.sum())

                # For episodes that ends, do the following:
                #   1. increment the episode count
                #   2. record the return
                #   3. clear the record of rewards
                #   4. clear the record of the number of steps
                #   5. reset the env to start a new episode
                # 3-5 are skipped when training is already finished.
                # episode_idx += end
                # recent_returns.extend(episode_r[end])

                total_reward[end] = 0
                episode_len[end] = 0
                if np.any(end):
                    state[end] = self.env.force_reset(np.arange(num_env)[end])

                if self.batch_step_number % self.hyperparameters['animation_record_step_frequency'] == 0:
                    self._run_eval_episode()

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

    def _run_eval_episode(self):
        # record_anim = (
        #         self.batch_step_number % self.hyperparameters.get('animation_record_frequency', 1e6) == 0 and
        #         self.hyperparameters.get('record_animation', False)
        # )
        record_anim = True

        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        images = []

        state = self.test_env.reset()

        while not done:

            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)

            if record_anim:
                images.append(self.env.render(full_image=True))

            total_reward += reward
            episode_len += 1

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len'] \
                    or info.get('needs_reset', False):

                if record_anim:
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f'animation/Rainbow/{self.name}/_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4',
                            self.stat_logger
                        ),
                    ).start()
                break
