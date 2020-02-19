import itertools
import os
import pickle
from collections import defaultdict
from typing import Dict, Union, Tuple
import time

import tensorflow as tf
import torch
import torch.nn as nn
import wandb
from torch.distributions import MultivariateNormal, Normal
import numpy as np

from common_agents_utils import Policy, ValueNet, Config, SubprocVecEnv_tf2, Torch_Separated_Replay_Buffer, \
    StateEncoder, EncodedForwardDynamicModel, InverseDynamicModel
from common_agents_utils.torch_gym_modules import _make_it_batched_torch_tensor
from envs.common_envs_utils.env_state_utils import get_state_combiner_by_settings_file, \
    from_image_vector_to_combined_state


class PPO:
    def __init__(self, config: Config):
        self.name = config.name
        self.debug = config.debug
        self.hyperparameters = config.hyperparameters
        self.eps_clip = config.hyperparameters['eps_clip']
        self.device = config.hyperparameters['device']

        self.test_env = config.test_environment_make_function()
        self.action_size = self.test_env.action_space.shape[0]
        # self.env = SubprocVecEnv_tf2([
        #     config.environment_make_function for _ in range(config.hyperparameters['num_envs'])
        # ])
        self._config = config
        self.create_env(config)

        self.memory = Torch_Separated_Replay_Buffer(
            buffer_size=10 ** 4,  # useless, it will be flushed frequently
            batch_size=10 ** 4,  # useless, it will be flushed frequently
            seed=0,
            device=self.device,
            state_extractor=lambda x: (None, x),
            state_producer=lambda x, y: y,
            sample_order=['state', 'action', 'reward', 'log_prob', 'done', 'next_state'],
        )

        state_description = self.test_env.observation_space
        action_size = self.test_env.action_space.shape[0]

        self.actor = Policy(
            state_description=state_description,
            action_size=action_size,
            hidden_size=128,
            device=self.device,
            double_action_size_on_output=True
        )
        self.critic = ValueNet(
            state_description=state_description,
            action_size=action_size,
            hidden_size=128,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor.parameters(), self.critic.parameters()),
            lr=config.hyperparameters['lr'],
            betas=config.hyperparameters['betas'],
        )

        self.actor_old = Policy(
            state_description=state_description,
            action_size=action_size,
            hidden_size=128,
            device=self.device,
            double_action_size_on_output=True
        )
        self.critic_old = ValueNet(
            state_description=state_description,
            action_size=action_size,
            hidden_size=128,
            device=self.device,
        )
        self.update_old_policy()
        self.mse = nn.MSELoss()

        self.MseLoss = nn.MSELoss()

        self.folder_save_path = os.path.join('model_saves', 'PPO', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self.current_game_stats = None
        self.mean_game_stats = None
        self.flush_stats()
        self.tf_writer = config.tf_writer

        self.accumulated_reward_mean = None
        self.accumulated_reward_std = None

    def create_env(self, config):
        self.env = config.environment_make_function()

        if config.hyperparameters.get('seed', None) is not None:
            print("Random Seed: {}".format(config.hyperparameters['seed']))
            torch.manual_seed(config.hyperparameters['seed'])
            self.env.seed(config.hyperparameters['seed'])
            self.test_env.seed(config.hyperparameters['seed'])
            np.random.seed(config.hyperparameters['seed'])

    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def update_current_game_stats(self, reward, done, info):
        self.current_game_stats['reward'] += reward
        self.current_game_stats['env_steps'] += 1.0

        if not done:
            return

        if 'is_finish' in info.keys():
            self.current_game_stats['finish'] = info['is_finish']
        if 'track_progress' in info.keys():
            self.current_game_stats['track_progress'] = info['track_progress']

    def flush_stats(self):
        self.current_game_stats = defaultdict(float, {})
        self.mean_game_stats = defaultdict(float, {})

    def save(self):
        current_save_path = os.path.join(
            self.folder_save_path,
            f"{self.name}_episode_{self.episode_number}_time_{time.time()}"
        )
        os.makedirs(current_save_path)

        torch.save(self.actor.state_dict(), os.path.join(current_save_path, 'actor'))
        torch.save(self.critic.state_dict(), os.path.join(current_save_path, 'critic'))
        torch.save(self.optimizer.state_dict(), os.path.join(current_save_path, 'optimizer'))

        pickle.dump((
            self.global_step_number,
            self.episode_number,

        ), open(os.path.join(current_save_path, 'stats.pkl'), 'wb'))

    def load(self, folder):

        self.actor.load_state_dict(torch.load(os.path.join(folder, 'actor')))
        self.critic.load_state_dict(torch.load(os.path.join(folder, 'critic')))
        self.optimizer.load_state_dict(torch.load(os.path.join(folder, 'optimizer')))
        self.update_old_policy()

        self.global_step_number, self.episode_number = pickle.load(
            open(os.path.join(folder, 'stats.pkl'), 'rb')
        )

    def get_action(self, state, evaluate=False) -> np.ndarray:
        actor_out = self.actor(state)
        action_mean, action_std = actor_out[:, self.action_size:], actor_out[:, :self.action_size]

        if evaluate:
            return torch.tanh(action_mean).detach().cpu().numpy()

        normal = Normal(action_mean, action_std.exp())

        return torch.tanh(normal.sample()).detach().cpu().numpy()

    def estimate_action(self, state: np.ndarray, action: Union[torch.FloatTensor, np.ndarray]) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute logprobs of action and entropy"""
        action = _make_it_batched_torch_tensor(action, self.device).detach()
        actor_out = self.actor(state)
        action_mean, action_std = actor_out[:, self.action_size:], actor_out[:, :self.action_size]

        normal = Normal(action_mean, action_std.exp())
        log_prob = normal.log_prob(torch.tanh(action.detach()))
        log_prob -= torch.log(1 + 1e-6 - action.pow(2))
        log_prob = log_prob.sum(1, keepdim=True)

        return log_prob, normal.entropy()

    def update(self):
        print('update')
        states, actions, rewards, log_probs, dones, next_states = self.memory.get_all()

        buffer_reward = 0
        discount_reward = []
        for cur_reward, cur_done in zip(reversed(rewards), reversed(dones)):
            if cur_done[0] == 1:
                buffer_reward = 0
            buffer_reward = float(cur_reward[0] + buffer_reward * self.hyperparameters['discount_rate'])
            discount_reward.append([buffer_reward])
        discount_reward = torch.from_numpy(
            np.array(discount_reward[::-1], dtype=np.float32)
        ).to(self.device).detach()

        # if self.accumulated_reward_mean is None:
        #     self.accumulated_reward_mean = discount_reward.mean()
        #     self.accumulated_reward_std = discount_reward.std() + 1e-5
        #     print(f'set reward mean / std : {self.accumulated_reward_mean} / {self.accumulated_reward_std}')
        #
        # discount_reward = (discount_reward - self.accumulated_reward_mean) / self.accumulated_reward_std

        for _ in range(self.hyperparameters['learning_updates_per_learning_session']):
            new_log_probs, new_entropy = self.estimate_action(states, actions)

            state_value = self.critic(states)
            # next_state_value = self.critic(next_states)
            # advantage = (rewards + self.hyperparameters['discount_rate'] * next_state_value - state_value).detach()
            advantage = discount_reward - state_value.detach()
            policy_ratio = torch.exp(new_log_probs - log_probs.detach())

            term_1 = policy_ratio * advantage
            term_2 = torch.clamp(policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # actor_loss = -1 * torch.min(term_1, term_2) - 0.01 * new_entropy
            # self.mean_game_stats['actor_loss'] += actor_loss.detach().cpu().numpy().mean()

            # critic_loss = torch.pow(state_value - discount_reward, 2)
            # self.mean_game_stats['critic_loss'] += critic_loss.detach().cpu().numpy().mean()

            loss = -1 * torch.min(term_1, term_2) - 0.01 * new_entropy + 0.5 * self.mse(discount_reward, state_value)
            # loss = (actor_loss + 0.5 * critic_loss).mean()
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.hyperparameters['gradient_clipping_norm'],
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.hyperparameters['gradient_clipping_norm'],
            )
            self.optimizer.step()

        # update old policy
        self.update_old_policy()

    def eval(self):
        print('Start eval...')
        state = self.test_env.reset()
        done = False
        total_reward = 0
        episode_len = 0
        while not done:
            # Running policy_old:
            action = self.get_action(state)[0]

            next_state, reward, done, info = self.test_env.step(action)
            self.global_step_number += 1

            total_reward += reward
            episode_len += 1

            state = next_state

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len']:
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                print(f"EVAL :{self.episode_number} R : {round(total_reward, 4)}\tTime : {episode_len}")
                break

    def train(self):
        # training loop
        for index in range(self.hyperparameters['num_episodes_to_run']):

            if self.hyperparameters.get('use_eval', False):
                if index % 10 == 0:
                    self.eval()

            attempt = 0
            while True:
                try:
                    # try На случай подения среды, у меня стандартный bipedal walker падает переодически :(
                    self.flush_stats()
                    self.run_one_episode()

                    self.log_it()
                    break
                except:
                    self.memory.clean_all_buffer()
                    print(f'\nenv fail\nrestart...   attempt {attempt}')
                    attempt += 1
                    if attempt >= 5:
                        print(f'khm, bad\nrecreating env...')
                        self._config.hyperparameters['seed'] = self._config.hyperparameters['seed'] + 1
                        self.create_env(self._config)
                    if attempt >= 10:
                        print(f'actually, it useless :(\nend training...\nsave...')
                        self.save()
                        print(f'save done.\nexiting...')
                        exit(1)

            if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                self.save()

    def run_one_episode(self):
        state = self.env.reset()
        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        while not done:
            # Running policy_old:
            action = self.get_action(state)[0]
            log_prob, _ = self.estimate_action(state, action)
            log_prob = log_prob.detach().cpu().numpy()[0]
            next_state, reward, done, info = self.env.step(action)
            self.global_step_number += 1
            self.update_current_game_stats(reward, done, info)

            total_reward += reward
            episode_len += 1

            self.memory.add_experience(
                state, action, reward, next_state, done, log_prob
            )
            state = next_state

            # update if its time
            if self.global_step_number % self.hyperparameters['update_every_n_steps'] == 0:
                self.update()
                self.memory.clean_all_buffer()


            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len']:
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                # print(f"Episode :{self.episode_number} R : {round(total_reward, 4)}\tTime : {episode_len}")
                break

    def log_it(self):

        stats = self.current_game_stats
        if self.current_game_stats.get('env_steps', 0) != 0:
            stats.update({
                name: value / self.current_game_stats['env_steps']
                for name, value in self.mean_game_stats.items()
            })

        if self.tf_writer is not None:
            with self.tf_writer.as_default():
                for name, value in stats.items():
                    tf.summary.scalar(name=name, data=value, step=self.episode_number)

        # WanDB logging:
        if not self.debug:
            wandb.log(row=stats, step=self.episode_number)
