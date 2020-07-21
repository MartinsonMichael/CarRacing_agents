import os
import pickle
from collections import defaultdict
import time
from multiprocessing import Process

import torch
import torch.nn as nn
import numpy as np

from common_agents_utils import Config, ActorCritic, Torch_Arbitrary_Replay_Buffer
from common_agents_utils.logger import Logger
from env.common_envs_utils.visualizer import save_as_mp4


class PPO:
    def __init__(self, config: Config):
        self.name = config.name
        self.stat_logger: Logger = Logger(config, log_interval=config.log_interval)
        self.config: Config = config
        self.hyperparameters = config.hyperparameters
        self.eps_clip = config.hyperparameters['eps_clip']

        self.test_env = config.test_environment_make_function()
        self.action_size = self.test_env.action_space.shape[0]
        self.env = None
        self.create_env()

        self.memory = Torch_Arbitrary_Replay_Buffer(
            buffer_size=10 ** 4,
            batch_size=10 ** 4,
            phi=config.phi,
            seed=0,
            device=self.config.device,
            sample_order=['state', 'action', 'reward', 'log_prob', 'done', 'next_state'],
            do_it_auto=False,
        )

        state_shape = config.phi(self.test_env.reset()).shape
        action_size = self.test_env.action_space.shape[0]

        self.ac: ActorCritic = ActorCritic(
            state_shape=state_shape,
            action_size=action_size,
            hidden_size=128,
            device=self.config.device,
            action_std=0.5,
            double_action_size_on_output=False,
        )
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr=config.hyperparameters['lr'],
            betas=config.hyperparameters['betas'],
        )

        self.ac_old: ActorCritic = ActorCritic(
            state_shape=state_shape,
            action_size=action_size,
            hidden_size=128,
            device=self.config.device,
            action_std=0.5,
            double_action_size_on_output=False,
        )
        self.update_old_policy()
        self.mse = nn.MSELoss()

        self.folder_save_path = os.path.join('model_saves', 'PPO', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self._total_grad_steps = 0
        self._wandb_anim_save = 0

        self.accumulated_reward_mean = None
        self.accumulated_reward_std = None

        self._exp_moving_track_progress = 0.0

    def create_env(self):
        self.env = self.config.environment_make_function()

        torch.manual_seed(self.config.seed)
        self.env.seed(self.config.seed)
        self.test_env.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def update_old_policy(self):
        self.ac_old.load_state_dict(self.ac.state_dict())

    def save(self, suffix=None):
        current_save_path = os.path.join(
            self.folder_save_path,
            f"{suffix}_{self.name}_episode_{self.episode_number}_time_{time.time()}"
            if suffix is not None else
            f"{self.name}_episode_{self.episode_number}_time_{time.time()}"
        )
        os.makedirs(current_save_path)

        torch.save(self.ac.state_dict(), os.path.join(current_save_path, 'ac'))
        torch.save(self.optimizer.state_dict(), os.path.join(current_save_path, 'optimizer'))

        pickle.dump((
            self.global_step_number,
            self.episode_number,

        ), open(os.path.join(current_save_path, 'stats.pkl'), 'wb'))

    def load(self, folder):

        self.ac.load_state_dict(torch.load(os.path.join(folder, 'ac')))
        self.optimizer.load_state_dict(torch.load(os.path.join(folder, 'optimizer')))
        self.update_old_policy()

        self.global_step_number, self.episode_number = pickle.load(
            open(os.path.join(folder, 'stats.pkl'), 'rb')
        )

    def update(self):
        states, actions, rewards, log_probs, dones, next_states = self.memory.get_all()

        buffer_reward = 0
        discount_reward = []

        if self.global_step_number < self.hyperparameters.get('linear_negative_reward_drop_steps', -1):
            neg_reward_coef = self.global_step_number / self.hyperparameters.get('linear_negative_reward_drop_steps', -1)
        else:
            neg_reward_coef = 1

        for cur_reward, cur_done in zip(reversed(rewards), reversed(dones)):
            if cur_done == 1:
                buffer_reward = 0
            if cur_reward < 0:
                cur_reward *= neg_reward_coef
            buffer_reward = float(cur_reward + buffer_reward * self.hyperparameters['discount_rate'])
            # discount_reward.insert(0, [buffer_reward])
            discount_reward.append([buffer_reward])
        discount_reward = torch.from_numpy(
            np.array(discount_reward[::-1], dtype=np.float32)
        ).to(self.config.device).detach()

        if self.hyperparameters.get('use_reward_discount', True):
            discount_reward = (discount_reward - discount_reward.mean()) / (discount_reward.std() + 1e-5)

        self.stat_logger.log_it({
            'discount_reward MEAN': float(discount_reward.detach().cpu().numpy().mean()),
            'discount_reward MAX': float(discount_reward.detach().cpu().numpy().max()),
        })

        sum_ppo_loss = 0.0
        sum_ppo_critic_loss = 0.0
        sum_ppo_actor_loss = 0.0
        for _ in range(self.hyperparameters['learning_updates_per_learning_session']):
            self._total_grad_steps += 1
            new_log_probs, new_entropy = self.ac.estimate_action(states, actions)

            state_value = self.ac.value(states)
            state_value_old = self.ac_old.value(states).detach()
            critic_loss = torch.min(
                self.mse(discount_reward, state_value),
                self.mse(
                    discount_reward,
                    # кустарный clamp
                    torch.min(
                        torch.max(state_value, state_value_old - self.eps_clip),
                        state_value_old + self.eps_clip
                    ),
                ),
            )

            if self.hyperparameters.get('loss_critic_clip_value', None) is not None:
                value = self.hyperparameters['loss_critic_clip_value']
                critic_loss = torch.clamp(critic_loss, -value, +value)

            advantage = discount_reward - state_value.detach()

            print('log_probs shape', log_probs.shape)

            policy_ratio = torch.exp(new_log_probs - log_probs.detach())

            term_1 = policy_ratio * advantage
            term_2 = torch.clamp(policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            actor_loss = -1 * torch.min(term_1, term_2)

            if self.hyperparameters.get('loss_actor_clip_value', None) is not None:
                value = self.hyperparameters['loss_actor_clip_value']
                actor_loss = torch.clamp(actor_loss, -value, +value)

            if self.hyperparameters.get('entropy_clip_value', None) is not None:
                value = self.hyperparameters['entropy_clip_value']
                new_entropy = torch.clamp(new_entropy, -value, +value)

            loss = (
                actor_loss
                - 0.01 * new_entropy
                + 0.5 * critic_loss
            )

            if self.hyperparameters.get('loss_clip_value', None) is not None:
                value = self.hyperparameters['loss_clip_value']
                loss = torch.clamp(loss, -value, +value)

            sum_ppo_loss += float(loss.mean().detach().cpu().numpy())
            sum_ppo_critic_loss += float(critic_loss.mean().detach().cpu().numpy())
            sum_ppo_actor_loss += float(actor_loss.mean().detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.hyperparameters['gradient_clipping_norm'])
            self.optimizer.step()

        self.stat_logger.log_it({
            'ppo_loss': float(sum_ppo_loss) / self.hyperparameters['learning_updates_per_learning_session'],
            'ppo_actor_loss': float(sum_ppo_actor_loss) / self.hyperparameters['learning_updates_per_learning_session'],
            'ppo_critic_loss': float(sum_ppo_critic_loss) / self.hyperparameters['learning_updates_per_learning_session'],
        })

        # update old policy
        self.update_old_policy()

    def train(self):
        print('Start to train PPO')
        while True:

            self.run_one_episode()

            self.stat_logger.log_it({
                'total_env_episode': self.episode_number,
                'total_env_steps': self.global_step_number,
                'total_grad_steps': self._total_grad_steps,
            })

            if self.global_step_number % self.config.eval_episode_freq == 0:
                self.run_one_episode(eval=True)

            # print(f"{self.global_step_number} / {self.config.env_steps_to_run}")
            if self.global_step_number >= self.config.env_steps_to_run:
                break

    def run_one_episode(self, eval: bool = False):
        if eval:
            self.env.make_next_run_eval()
        state = self.env.reset()
        record_anim = eval or (
            self.episode_number % self.config.animation_record_frequency == 0 and
            self.config.record_animation
        )

        done = False
        if not eval:
            self.episode_number += 1
        total_reward = 0
        episode_len = 0
        images = []

        while not done:
            # Running policy_old:
            action, log_prob, _ = self.ac.sample_action(
                self.config.phi(state),
                to_numpy=True,
                remove_batch=True,
                eval=eval,
            )
            next_state, reward, done, info = self.env.step(action)

            if record_anim:
                images.append(self.env.render(full_image=True))

            total_reward += reward
            episode_len += 1

            if not eval:
                self.global_step_number += 1

                self.memory.add_experience(
                    is_single=True,
                    state=state, action=action, reward=reward, next_state=next_state, done=done, log_prob=log_prob,
                )

                # update if its time
                if self.global_step_number % self.hyperparameters['update_every_n_steps'] == 0:
                    self.update()
                    self.memory.remove_all()

            state = next_state

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.config.max_episode_len \
                    or self.global_step_number >= self.config.env_steps_to_run:
                stats = {
                        'reward': total_reward,
                        'track_progress': info.get('track_progress', -1),
                        'env_steps': episode_len,
                    }
                if eval:
                    print("\nEval episode:\nR %.4f\tTime %.1f\tTrack %.2f\n" % (
                        total_reward,
                        episode_len,
                        info.get('track_progress', -1),
                    ))
                    stats = {'EVAL ' + key: value for key, value in stats.items()}
                    self.stat_logger.log_and_publish(stats)
                else:
                    self.stat_logger.log_it(stats)
                    self.stat_logger.on_episode_end()

                if record_anim:
                    wandb_anim_record: bool = False
                    if self._wandb_anim_save % self.config.wandb_animation_frequency == 0:
                        wandb_anim_record = True
                    self._wandb_anim_save += 1
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f"animation/PPO/{self.name}/{'EVAL' if eval else ''}_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4",
                            wandb_anim_record
                        ),
                    ).start()
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                break
