import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import tensorflow as tf

from common_agents_utils import Policy, ValueNet, Torch_Separated_Replay_Buffer, Torch_Arbitrary_Replay_Buffer

from env.common_envs_utils.extended_env_wrappers import ObservationToFloat32

device = 'cuda:2'


# class Memory:
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.is_terminals = []
#
#     def clear_memory(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = Policy(
            state_description=state_dim,
            action_size=action_dim,
            hidden_size=128,
            device=device,
            double_action_size_on_output=False,
        )
        # critic
        self.critic = ValueNet(
            state_description=state_dim,
            action_size=action_dim,
            hidden_size=128,
            device=device,
        )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # memory.states.append(state)
        # memory.actions.append(action)
        # memory.logprobs.append(action_logprob)
        
        return action.detach().cpu().data.numpy().flatten(), action_logprob.detach().cpu().data.numpy().flatten()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state)
    
    def update(self, memory: Torch_Arbitrary_Replay_Buffer):
        # Monte Carlo estimate of rewards:
        old_states, old_actions, old_rewards, old_done, old_logprobs = memory.sample(get_all=True)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(old_rewards), reversed(old_done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.remove_all()


def main(args):
    ############## Hyperparameters ##############
    # env_name = "BipedalWalker-v2"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 2000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################

    memory = Torch_Arbitrary_Replay_Buffer(
        buffer_size=10 ** 4,
        device=device,
        sample_order=['state', 'action', 'reward', 'done', 'log_prob'],
    )

    wandb.init(
        project='PPO',
        name=f'origin_{args.name}',
    )

    log_tb_path = os.path.join('logs', 'PPO_Origin', args.name)
    if not os.path.exists(log_tb_path):
        os.makedirs(log_tb_path)
    tf_writer = tf.summary.create_file_writer(log_tb_path)
    
    # creating environment
    # env = gym.make(env_name)
    env = ObservationToFloat32(gym.make("LunarLanderContinuous-v2"))
    # state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if isinstance(random_seed, int):
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    ppo = PPO(env.observation_space, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action, log_prob = ppo.select_action(state)
            new_state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            # memory.rewards.append(reward)
            # memory.is_terminals.append(done)
            memory.add_experience(
                is_single=True,
                state=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
            )
            state = new_state
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            log_it({'reward': running_reward}, tf_writer, i_episode)
            running_reward = 0
            avg_length = 0


def log_it(current_game_stats, tf_writer, episode_number):

    with tf_writer.as_default():
        for name, value in current_game_stats.items():
            tf.summary.scalar(name=name, data=value, step=episode_number)

    # WanDB logging:
    wandb.log(row=current_game_stats, step=episode_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' - [default] or 'cuda:{number}'")
    args = parser.parse_args()

    device = args.device

    if args.name is None:
        raise ValueError('set name')

    main(args)
