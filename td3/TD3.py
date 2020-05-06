import copy
from collections import defaultdict
from multiprocessing import Process

import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_agents_utils.typingTypes import *
from common_agents_utils import Config, Torch_Arbitrary_Replay_Buffer
from common_agents_utils.logger import Logger
from deep_utils.simple_adaptive_actor import StateAdaptiveActor
from deep_utils.simple_state_adaptive_q_value_ import DoubleStateAdaptiveCritic
from env.common_envs_utils.visualizer import save_as_mp4
from stable_baseline_replay_buffer.torch_replay_buffer import TorchReplayBuffer


class TD3:
    def __init__(self, config: Config):
        self.config = config
        self.hyperparameters = config.hyperparameters
        self.env = config.environment_make_function()

        self.memory = TorchReplayBuffer(
            size=10 ** 6,
            phi=config.phi,
            device=self.config.device,
        )
        state_shape = config.phi(self.env.reset()).shape
        self.action_size = self.env.action_space.shape[0]

        self.actor = StateAdaptiveActor(state_shape, self.action_size, self.config.device).to(self.config.device)
        self.actor_target: nn.Module = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hyperparameters['lr'])

        self.critic = DoubleStateAdaptiveCritic(state_shape, self.action_size, self.config.device).to(self.config.device)
        self.critic_target: nn.Module = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyperparameters['lr'])

        self.total_it = 0

        self.stat_logger: Logger = Logger(config, log_interval=self.config.log_interval)

        self.episode_number = 0
        self.global_step_number = 0
        self._total_grad_steps = 0
        self.current_game_stats = None
        self.flush_stats()

        self.accumulated_reward_mean = None
        self.accumulated_reward_std = None

        self._exp_moving_track_progress = 0.0

    def flush_stats(self):
        self.current_game_stats = defaultdict(float, {})

    def select_action(self, state: NpA):
        if self.global_step_number < self.hyperparameters['start_to_learn_time_point']:
            return np.random.uniform(-1, 1, size=self.action_size)

        action = self.actor(self.config.phi(state)).cpu().data.numpy().flatten()

        if self.global_step_number < self.hyperparameters['expl_noise_linear_decay_steps']:
            noise_variance = self.hyperparameters['expl_noise'] * \
                             (1 - self.episode_number / self.hyperparameters['expl_noise_linear_decay_steps'])
            action = (
                action
                + np.random.normal(0, noise_variance, size=self.action_size)
            )

        action = np.clip(action, -1, 1)

        return action

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = self.memory.sample(256)
        done = done.reshape(-1, 1)
        reward = reward.reshape(-1, 1)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.hyperparameters['policy_noise']
            ).clamp(-self.hyperparameters['noise_clip'], self.hyperparameters['noise_clip'])

            next_action = torch.clamp(self.actor_target(next_state) + noise, -1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # print(f'target_Q shape : {target_Q.shape}, done shape : {done.shape}, reward shape : {reward.shape}')

            target_Q = reward + (1 - done) * self.hyperparameters['discount_rate'] * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyperparameters['gradient_clipping_norm'])
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.hyperparameters['policy_update_freq'] == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparameters['gradient_clipping_norm'])
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.hyperparameters['tau'] * param.data +
                    (1 - self.hyperparameters['tau']) * target_param.data
                )

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.hyperparameters['tau'] * param.data +
                    (1 - self.hyperparameters['tau']) * target_param.data
                )

    def train(self) -> None:
        print('Start to train TD3')
        try:
            while True:

                self.run_one_episode()

                self._exp_moving_track_progress = (
                        0.98 * self._exp_moving_track_progress +
                        0.02 * self.current_game_stats.get('track_progress', 0)
                )
                self.current_game_stats.update({
                    'moving_track_progress': self._exp_moving_track_progress,
                    'total_env_episode': self.episode_number,
                    'total_env_steps': self.global_step_number,
                    'total_grad_steps': self._total_grad_steps,
                })
                self.stat_logger.log_it(self.current_game_stats)
                if self._exp_moving_track_progress >= self.hyperparameters.get('track_progress_success_threshold', 10):
                    break

                self.flush_stats()

                if self.global_step_number >= self.config.env_steps_to_run:
                    break

                # if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                #     self.save()
        finally:
            self.stat_logger.on_training_end()
            # self.save(suffix='final')
            pass

    def update_current_game_stats(self, reward, done, info):
        self.current_game_stats['reward'] += reward
        self.current_game_stats['env_steps'] += 1.0

        if 'is_finish' in info.keys():
            self.current_game_stats['finish'] = info['is_finish']
        if 'track_progress' in info.keys():
            self.current_game_stats['track_progress'] = info['track_progress']
        if 'is_collided' in info.keys():
            self.current_game_stats['is_collided'] = info['is_collided']

    def run_one_episode(self):
        state = self.env.reset()
        record_anim = (
            self.episode_number % self.config.animation_record_frequency == 0 and self.config.record_animation
        )

        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        images = []

        while not done:
            # Running policy_old:
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.global_step_number += 1
            self.update_current_game_stats(reward, done, info)

            if record_anim:
                images.append(self.env.render(full_image=True))

            total_reward += reward
            episode_len += 1

            self.memory.add(
                obs_t=state, action=action, reward=reward, obs_tp1=next_state, done=done,
            )
            state = next_state

            # update if its time
            if self.global_step_number > self.hyperparameters['start_to_learn_time_point']:
                self.update()

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.config.max_episode_len \
                    or self.global_step_number >= self.config.env_steps_to_run:
                if record_anim:
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f'animation/TD3/{self.config.name}/_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4',
                            self.stat_logger
                        ),
                    ).start()
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                break
