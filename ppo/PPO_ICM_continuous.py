import os
import pickle
from collections import defaultdict
import time
from multiprocessing import Process

import torch
import torch.nn as nn
import numpy as np

from common_agents_utils import Config, ActorCritic, Torch_Arbitrary_Replay_Buffer, ICM
from common_agents_utils.logger import Logger
from env.common_envs_utils.visualizer import save_as_mp4


class PPO_ICM:
    def __init__(self, config: Config):
        raise ValueError("class under rewriting")
        self.name = config.name
        self.stat_logger: Logger = Logger(
            config,
            log_interval=config.log_interval,
        )
        self.config = config
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

        if self.hyperparameters['use_icm']:
            self._icm: ICM = ICM(
                state_description=state_description,
                action_size=action_size,
                encoded_state_size=6,
                device=self.device,
                batch_size=256,
                buffer_size=10**5,
                update_per_step=1,
                config=config.hyperparameters['icm_config']
            )
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
            # chain(
            #     self.ac.parameters(),
            #     self._icm.parameters()
            # ) if self.hyperparameters['use_icm'] else self.ac.parameters(),
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
        self.current_game_stats = None
        self.flush_stats()

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
        for cur_reward, cur_done in zip(reversed(rewards), reversed(dones)):
            if cur_done == 1:
                buffer_reward = 0
            buffer_reward = float(cur_reward + buffer_reward * self.hyperparameters['discount_rate'])
            discount_reward.insert(0, [buffer_reward])
        discount_reward = torch.from_numpy(
            np.array(discount_reward, dtype=np.float32)
        ).to(self.config.device).detach()
        discount_reward = (discount_reward - discount_reward.mean()) / (discount_reward.std() + 1e-5)

        self.current_game_stats.update({
            'discount_reward MEAN': float(discount_reward.detach().cpu().numpy().mean()),
            'discount_reward MAX': float(discount_reward.detach().cpu().numpy().max()),
        })

        # intrinsic_loss = 0
        # if self.hyperparameters['use_icm']:
        #     self._icm.add_experience(
        #         is_single=False,
        #         state=states.detach().cpu().numpy(),
        #         action=actions.detach().cpu().numpy(),
        #         next_state=next_states.detach().cpu().numpy(),
        #     )
        #     intrinsic_reward, intrinsic_loss, intrinsic_stats = self._icm.get_intrinsic_reward_with_loss(
        #         state=states, action=actions, next_state=next_states, return_stats=True, print_debug=True,
        #     )
        #     self.current_game_stats.update(intrinsic_stats)
        #     discount_reward += torch.from_numpy(np.clip(intrinsic_reward, -3, 3)).to(self.device)
        #
        #     icm_update_stat = self._icm.update(return_stat=True)
        #     self.current_game_stats.update(icm_update_stat)
        #
        #     self.current_game_stats.update({
        #         'intrinsic_reward MEAN': intrinsic_reward.mean(),
        #         'intrinsic_reward MAX': intrinsic_reward.max(),
        #         'total_reward MEAN': float(discount_reward.detach().cpu().numpy().mean()),
        #         'total_reward MAX': float(discount_reward.detach().cpu().numpy().max()),
        #     })

        sum_ppo_loss = 0.0
        sum_ppo_critic_loss = 0.0
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

            advantage = discount_reward - state_value.detach()
            policy_ratio = torch.exp(new_log_probs - log_probs.detach())

            term_1 = policy_ratio * advantage
            term_2 = torch.clamp(policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            loss = -1 * torch.min(term_1, term_2) - 0.0 * new_entropy + 0.5 * critic_loss

            sum_ppo_loss += float(loss.mean().detach().cpu().numpy())
            sum_ppo_critic_loss += float(critic_loss.mean().detach().cpu().numpy())

            self.optimizer.zero_grad()
            if self.hyperparameters['use_icm']:
                pass
                # (loss.mean() + 0.5 * intrinsic_loss.mean()).backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(
                #     self._icm.parameters(), self.hyperparameters['icm_config'].get('icm_gradient_clipping', 1.0)
                # )
            else:
                loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.hyperparameters['gradient_clipping_norm'])
            self.optimizer.step()

        self.current_game_stats.update({
            'ppo_loss': float(sum_ppo_loss) / self.hyperparameters['learning_updates_per_learning_session'],
            'ppo_critic_loss': float(sum_ppo_critic_loss) / self.hyperparameters['learning_updates_per_learning_session'],
        })

        # update old policy
        self.update_old_policy()

    def train(self):
        print('Start to train PPO')
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

                # print(f"{self.global_step_number} / {self.config.env_steps_to_run}")
                if self.global_step_number >= self.config.env_steps_to_run:
                    break

                # if self.episode_number % self.hyperparameters['save_frequency_episode'] == 0:
                #     self.save()
        finally:
            self.stat_logger.on_training_end()
            # self.save(suffix='final')
            pass

    def run_one_episode(self):
        state = self.env.reset()
        record_anim = (
            self.episode_number % self.config.animation_record_frequency == 0 and
            self.config.record_animation
        )

        done = False
        self.episode_number += 1
        total_reward = 0
        episode_len = 0
        images = []

        while not done:
            # Running policy_old:
            action, log_prob, _ = self.ac.sample_action(self.config.phi(state), to_numpy=True, remove_batch=True)
            next_state, reward, done, info = self.env.step(action)
            self.global_step_number += 1
            self.update_current_game_stats(reward, done, info)

            if record_anim:
                images.append(self.env.render(full_image=True))

            total_reward += reward
            episode_len += 1

            self.memory.add_experience(
                is_single=True,
                state=state, action=action, reward=reward, next_state=next_state, done=done, log_prob=log_prob,
            )
            state = next_state

            # update if its time
            if self.global_step_number % self.hyperparameters['update_every_n_steps'] == 0:
                self.update()
                self.memory.remove_all()

            if done \
                    or info.get('need_reset', False) \
                    or episode_len > self.config.max_episode_len \
                    or self.global_step_number >= self.config.env_steps_to_run:
                if record_anim:
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f'animation/PPO/{self.name}/_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4',
                        ),
                    ).start()
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                break
