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

from common_agents_utils import Policy, ValueNet, Config, SubprocVecEnv_tf2, Torch_Separated_Replay_Buffer
from envs.common_envs_utils.env_state_utils import get_state_combiner_by_settings_file, \
    from_image_vector_to_combined_state


class PPO:
    def __init__(self, config: Config):
        self.name = config.name
        self.debug = config.debug
        self.hyperparameters = config.hyperparameters
        self.eps_clip = config.hyperparameters['eps_clip']
        self.device = config.hyperparameters['device']

        self.test_env = config.environment_make_function()
        self.action_size = self.test_env.action_space.shape[0]
        # self.env = SubprocVecEnv_tf2([
        #     config.environment_make_function for _ in range(config.hyperparameters['num_envs'])
        # ])
        self.env = config.environment_make_function()

        if config.hyperparameters.get('seed', None) is not None:
            print("Random Seed: {}".format(config.hyperparameters['seed']))
            torch.manual_seed(config.hyperparameters['seed'])
            self.env.seed(config.hyperparameters['seed'])
            self.test_env.seed(config.hyperparameters['seed'])
            np.random.seed(config.hyperparameters['seed'])

        self.memory = Torch_Separated_Replay_Buffer(
            buffer_size=10 ** 4,  # useless, it will be flushed frequently
            batch_size=10 ** 4,  # useless, it will be flushed frequently
            seed=0,
            device=self.device,
            state_extractor=get_state_combiner_by_settings_file(self.hyperparameters['env_settings_file_path']),
            state_producer=from_image_vector_to_combined_state,
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
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.hyperparameters['lr'],
            betas=config.hyperparameters['betas'],
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
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


        self.MseLoss = nn.MSELoss()

        self.folder_save_path = os.path.join('model_saves', 'PPO', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self.current_game_stats = None
        self.mean_game_stats = None
        self.flush_stats()
        self.tf_writer = config.tf_writer

    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def update_current_game_stats(self, reward, done, info):
        self.current_game_stats['reward'] += reward
        self.current_game_stats['env_steps'] += 1.0

        if not done:
            return

        self.current_game_stats['finish'] = int(info.get('is_finish', -1))
        self.current_game_stats['track_progress'] = info.get('track_progress', -1)

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
        torch.save(self.actor_optimizer.state_dict(), os.path.join(current_save_path, 'actor_optimizer'))
        torch.save(self.critic.state_dict(), os.path.join(current_save_path, 'critic'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(current_save_path, 'critic_optimizer'))

        pickle.dump((
            self.global_step_number,
            self.episode_number,

        ), open(os.path.join(current_save_path, 'stats.pkl'), 'wb'))

    def load(self, folder):

        self.actor.load_state_dict(torch.load(os.path.join(folder, 'actor')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder, 'actor_optimizer')))
        self.critic.load_state_dict(torch.load(os.path.join(folder, 'critic')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder, 'critic_optimizer')))
        self.update_old_policy()

        self.global_step_number, self.episode_number = pickle.load(
            open(os.path.join(folder, 'stats.pkl'), 'rb')
        )

    def get_action(self, state):
        actor_out = self.actor(state)
        action_mean, action_std = actor_out[:, self.action_size:], actor_out[:, :self.action_size]

        normal = Normal(action_mean, action_std.exp())
        x_t = normal.rsample()
        # print('x_t', x_t)
        action = torch.tanh(x_t)
        # print(f'action : {action}')
        log_prob = normal.log_prob(x_t)
        # print(f'1 log_prob : {log_prob}')
        log_prob -= torch.log(1 - action.pow(2.0) + 1e-6)
        # print(f'2 log_prob : {log_prob}')
        log_prob = log_prob.sum(1, keepdim=True)
        # print(f'3 log_prob : {log_prob}')

        return action.detach(), log_prob, normal.entropy(), torch.tanh(action_mean).detach()

    def train(self):
        # training loop
        for _ in range(self.hyperparameters['num_episodes_to_run']):

            self.flush_stats()
            self.run_one_episode()
            self.log_it()

            if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                self.save()

    def update(self):
        states, actions, rewards, log_probs, dones, next_states = self.memory.get_all()
        # print(f'states shape : {states.shape}')
        # print(f'action shape : {actions.shape}')
        # print(f'rewards shape : {rewards.shape}')
        # print(f'log_probs shape : {log_probs.shape}')
        # print(f'dones shape : {dones.shape}')
        # print(f'next_state shape : {next_states.shape}')

        discount_reward = 0
        discount_reward_batch = []
        for cur_reward, cur_done in zip(reversed(rewards), reversed(dones)):
            if cur_done:
                discount_reward = 0
            discount_reward = float(cur_reward[0] + discount_reward * self.hyperparameters['discount_rate'])
            discount_reward_batch.append([discount_reward])
        discount_reward = torch.from_numpy(
            np.array(discount_reward_batch[::-1], dtype=np.float32)
        ).to(self.device).detach()
        del discount_reward_batch

        advantage = (
            rewards
            + self.hyperparameters['discount_rate'] * self.critic_old(next_states)
            - self.critic_old(states)
        ).detach()

        for _ in range(self.hyperparameters['learning_updates_per_learning_session']):
            new_action, new_log_probs, new_entropy, _ = self.get_action(states)

            policy_ratio = torch.exp(new_log_probs - log_probs)

            actor_loss = -1 * torch.min(
                policy_ratio * advantage,
                torch.clamp(policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            ) - 1e-4 * new_entropy
            actor_loss = actor_loss.mean()
            self.mean_game_stats['actor_loss'] += actor_loss.detach().cpu().numpy()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.hyperparameters['gradient_clipping_norm'],
            )
            self.actor_optimizer.step()

            critic_loss = torch.nn.MSELoss()(self.critic(states), discount_reward)
            self.critic_optimizer.zero_grad()
            self.mean_game_stats['critic_loss'] += critic_loss.detach().cpu().numpy()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.hyperparameters['gradient_clipping_norm'],
            )
            self.critic_optimizer.step()

        # update old policy
        self.update_old_policy()

    def run_one_episode(self):
        state = self.env.reset()
        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        while not done:
            # Running policy_old:
            action, log_prob, _, _ = self.get_action(state)
            action = action.detach().cpu().numpy()[0]
            log_prob = log_prob.detach().cpu().numpy()[0]
            # print(f'action : {action}')
            next_state, reward, done, info = self.env.step(action)
            self.global_step_number += 1
            self.update_current_game_stats(reward, done, info)

            total_reward += reward
            episode_len += 1

            # print(f'log_prob : {log_prob}')

            self.memory.add_experience(
                state, action, reward, next_state, done, log_prob
            )
            state = next_state

            # update if its time
            if self.global_step_number % self.hyperparameters['update_every_n_steps'] == 0:
                self.update()
                self.memory.clean_all_buffer()

            if done \
                    or info.get('was_reset', False) \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len']:
                print(f"Episode :{self.episode_number} R : {round(total_reward, 4)}\tTime : {episode_len}")
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
