import os
import pickle
from collections import defaultdict
from typing import Dict, Union
import time

import tensorflow as tf
import torch
import torch.nn as nn
import wandb
from torch.distributions import MultivariateNormal
import numpy as np

from common_agents_utils import Policy, AdvantageNet, Config


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_description, action_dim, action_std, hidden_size, device):
        super(ActorCritic, self).__init__()
        self.device = device
        # action mean range -1 to 1
        self.actor = Policy(
            action_size=action_dim,
            state_description=state_description,
            hidden_size=hidden_size,
            device=device,
            double_action_size_on_output=False,
        )
        # critic
        self.critic = AdvantageNet(
            action_size=action_dim,
            state_description=state_description,
            hidden_size=hidden_size,
            device=device,
        )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, config: Config):
        self.name = config.name
        self.hyperparameters = config.hyperparameters
        self.lr = config.hyperparameters['lr']
        self.betas = config.hyperparameters['betas']
        self.gamma = config.hyperparameters['gamma']
        self.eps_clip = config.hyperparameters['eps_clip']
        self.device = config.hyperparameters['device']

        self.env = config.environment
        if config.hyperparameters.get('seed', None) is not None:
            print("Random Seed: {}".format(config.hyperparameters['seed']))
            torch.manual_seed(config.hyperparameters['seed'])
            self.env.seed(config.hyperparameters['seed'])
            np.random.seed(config.hyperparameters['seed'])
        self.memory = Memory()

        state_description = config.environment.observation_space
        action_dim = config.environment.action_space.shape[0]
        
        self.policy = ActorCritic(
            state_description=state_description,
            action_dim=action_dim,
            hidden_size=128,
            action_std=config.hyperparameters['action_std'],
            device=self.device,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        self.policy_old = ActorCritic(
            state_description=state_description,
            action_dim=action_dim,
            hidden_size=128,
            action_std=config.hyperparameters['action_std'],
            device=self.device,
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.folder_save_path = os.path.join('model_saves', 'PPO', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self.current_game_stats = None
        self.mean_game_stats = None
        self.flush_stats()
        self.tf_writer = config.tf_writer
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.hyperparameters['learning_updates_per_learning_session']):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            self.mean_game_stats['mean_advantage'] += float(advantages.detach().cpu().numpy().mean())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.mean_game_stats['mean_loss'] += float(loss.detach().cpu().numpy().mean())

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

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

        torch.save(self.policy.state_dict(), os.path.join(current_save_path, 'policy'))
        torch.save(self.policy_old.state_dict(), os.path.join(current_save_path, 'policy_old'))

        torch.save(self.optimizer.state_dict(), os.path.join(current_save_path, 'optimizer'))

        pickle.dump((
            self.global_step_number,
            self.episode_number,

        ), open(os.path.join(current_save_path, 'stats.pkl'), 'wb'))

    def load(self, folder):

        self.policy.load_state_dict(torch.load(os.path.join(folder, 'policy')))
        self.policy_old.load_state_dict(torch.load(os.path.join(folder, 'policy_old')))
        self.optimizer.load_state_dict(torch.load(os.path.join(folder, 'optimizer')))

        self.global_step_number, self.episode_number = pickle.load(
            open(os.path.join(folder, 'stats.pkl'), 'rb')
        )

    def train(self):
        # training loop
        for _ in range(self.hyperparameters['num_episodes_to_run']):

            self.run_one_episode()
            self.log_it()

            if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                self.save()

    def run_one_episode(self):
        self.flush_stats()
        state = self.env.reset()
        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        while not done:
            # Running policy_old:
            action = self.select_action(state, self.memory)
            state, reward, done, info = self.env.step(action)
            self.global_step_number += 1
            self.update_current_game_stats(reward, done, info)

            total_reward += reward
            episode_len += 1

            # Saving reward and is_terminals:
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)

            # update if its time
            if self.global_step_number % self.hyperparameters['update_every_n_steps'] == 0:
                self.update(self.memory)
                self.memory.clear_memory()

            if done \
                    or info.get('was_reset', False) \
                    or info.get('need_reset', False) \
                    or episode_len > self.hyperparameters['max_episode_len']:
                print(f"R : {round(total_reward, 4)}\tTime : {episode_len}")
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
        wandb.log(row=stats, step=self.episode_number)
