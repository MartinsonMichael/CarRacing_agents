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
from env.common_envs_utils.visualizer import save_as_mp4


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:
    def __init__(
            self,
            config: Config,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
    ):
        self.config = config
        self.hyperparameters = config.hyperparameters
        self.env = config.environment_make_function()
        self.device = self.hyperparameters['device']
        self.name = config.name

        self.memory = Torch_Arbitrary_Replay_Buffer(
            buffer_size=20 ** 5,
            batch_size=256,
            seed=0,
            device=self.device,
            sample_order=['state', 'action', 'reward', 'done', 'next_state'],
            do_it_auto=False,
        )
        state_description = self.env.observation_space
        self.action_size = self.env.action_space.shape[0]

        assert isinstance(state_description, gym.spaces.Box)
        assert len(state_description.shape) == 1
        state_size = state_description.shape[0]

        self.actor: nn.Module = Actor(state_size, self.action_size).to(self.device)
        self.actor_target: nn.Module = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic: nn.Module = Critic(state_size, self.action_size).to(self.device)
        self.critic_target: nn.Module = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.stat_logger: Logger = Logger(
            config,
            log_interval=config.hyperparameters.get('log_interval', 20),
        )

        self.episode_number = 0
        self.global_step_number = 0
        self._total_grad_steps = 0
        self.current_game_stats = None
        self.flush_stats()
        self.tf_writer = config.tf_writer

        self.accumulated_reward_mean = None
        self.accumulated_reward_std = None

        self._exp_moving_track_progress = 0.0

    def flush_stats(self):
        self.current_game_stats = defaultdict(float, {})

    def select_action(self, state: NpA):
        if self.global_step_number < self.hyperparameters['start_to_learn_time_point']:
            return np.random.uniform(-1, 1, size=self.action_size)

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        not_noisy_action = self.actor(state).cpu().data.numpy().flatten()

        action = (
                not_noisy_action
                + np.random.normal(0, 1 * self.hyperparameters['expl_noise'], size=self.action_size)
        )

        return action

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, done, next_state = self.memory.sample()
        done = done.reshape(-1, 1)
        reward = reward.reshape(-1, 1)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise

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
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparameters['gradient_clipping_norm'])
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
                self.episode_number % self.hyperparameters.get('animation_record_frequency', 1e6) == 0 and
                self.hyperparameters.get('record_animation', False)
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

            self.memory.add_experience(
                is_single=True,
                state=state, action=action, reward=reward, next_state=next_state, done=done,
            )
            state = next_state

            # update if its time
            if self.global_step_number > self.hyperparameters['start_to_learn_time_point']:
                self.update()

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len']:
                if record_anim:
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f'animation_PPO/{self.name}/_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4',
                            self.stat_logger
                        ),
                    ).start()
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                break
