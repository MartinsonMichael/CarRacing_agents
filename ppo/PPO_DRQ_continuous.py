import os
import pickle
from collections import defaultdict
import time
from multiprocessing import Process

import kornia
import torch
import torch.nn as nn
import numpy as np
import wandb
from torchvision import transforms

from common_agents_utils import Config, ActorCritic, Torch_Arbitrary_Replay_Buffer
from common_agents_utils.typingTypes import TT, NpA, Tuple, Iterable
from common_agents_utils.logger import Logger
from env.common_envs_utils.visualizer import save_as_mp4


class PPO_DRQ:
    def __init__(self, config: Config):
        self.name = config.name
        self.stat_logger: Logger = Logger(
            config,
            log_interval=config.log_interval,
        )
        self.config = config
        self.hyperparameters = config.hyperparameters
        self.eps_clip = config.hyperparameters['eps_clip']

        self.test_env = config.test_environment_make_function()
        self.env = None
        self.create_env()

        self.memory = Torch_Arbitrary_Replay_Buffer(
            buffer_size=10 ** 4,
            batch_size=10 ** 4,
            phi=None,
            seed=0,
            device=self.config.device,
            sample_order=['state', 'action', 'reward', 'log_prob', 'done', 'next_state'],
            do_it_auto=False,
            convert_to_torch=False,
        )

        state_shape = config.phi(self.test_env.reset()).shape
        print(f'state shape : {state_shape}')
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

        # self.image_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomCrop(
        #         (84, 84),
        #         padding=self.hyperparameters['drq_padding'],
        #         pad_if_needed=True,
        #         padding_mode='edge',
        #     ),
        #     transforms.ToTensor(),
        # ])
        self.image_transform = nn.Sequential(
            nn.ReplicationPad2d(self.hyperparameters['drq_padding']),
            kornia.augmentation.RandomCrop((84, 84)),
        )

        self.folder_save_path = os.path.join('model_saves', 'PPO', self.name)
        self.episode_number = 0
        self.global_step_number = 0
        self._total_grad_steps = 0
        self.current_game_stats = None
        self._wandb_anim_save = 0
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

    def augment_state(self, state_batch) -> NpA:
        return np.array([
                self.config.phi((self.image_transform(x[0]), x[1]))
                for x in state_batch
            ],
            dtype=np.float32,
        )

    def update(self):
        states, actions, rewards, log_probs, dones, _ = self.memory.get_all()

        actions = torch.from_numpy(actions).to(self.config.device)
        log_probs = torch.from_numpy(log_probs).to(self.config.device)

        buffer_reward = 0
        discount_reward = []
        for cur_reward, cur_done in zip(reversed(rewards), reversed(dones)):
            if cur_done == 1:
                buffer_reward = 0
            buffer_reward = float(cur_reward + buffer_reward * self.hyperparameters['discount_rate'])
            discount_reward.append([buffer_reward])
        discount_reward = torch.from_numpy(
            np.array(discount_reward[::-1], dtype=np.float32)
        ).to(self.config.device).detach()
        discount_reward = (discount_reward - discount_reward.mean()) / (discount_reward.std() + 1e-5)

        self.current_game_stats.update({
            'discount_reward MEAN': float(discount_reward.detach().cpu().numpy().mean()),
            'discount_reward MAX': float(discount_reward.detach().cpu().numpy().max()),
        })

        sum_ppo_loss = 0.0
        sum_ppo_critic_loss = 0.0
        sum_ppo_actor_loss = 0.0

        states = torch.from_numpy(
                    np.array([self.config.phi(x) for x in states], dtype=np.float32)
                ).to(self.config.device)

        for _ in range(self.hyperparameters['learning_updates_per_learning_session']):
            self._total_grad_steps += 1

            actor_loss, critic_loss, entropy_loss = self.get_loss(
                states,
                actions,
                discount_reward,
                log_probs,
            )

            for _ in range(self.hyperparameters['drq_augment_num']):
                actor_loss_aug, critic_loss_aug, entropy_loss_aug = self.get_loss(
                    self.image_transform(states),
                    actions,
                    discount_reward,
                    log_probs,
                )

                actor_loss += actor_loss_aug
                critic_loss += critic_loss_aug
                entropy_loss += entropy_loss_aug

            actor_loss /= 1 + int(self.hyperparameters['drq_augment_num'])
            critic_loss /= 1 + int(self.hyperparameters['drq_augment_num'])
            entropy_loss /= 1 + int(self.hyperparameters['drq_augment_num'])

            loss = 0.5 * critic_loss + actor_loss + 0.001 * entropy_loss

            sum_ppo_loss += float(loss.mean().detach().cpu().numpy())
            sum_ppo_critic_loss += float(critic_loss.mean().detach().cpu().numpy())
            sum_ppo_actor_loss += float(actor_loss.mean().detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.hyperparameters['gradient_clipping_norm'])
            self.optimizer.step()

        self.current_game_stats.update({
            'ppo_loss': float(sum_ppo_loss) / self.hyperparameters['learning_updates_per_learning_session'],
            'ppo_critic_loss': float(sum_ppo_critic_loss) / self.hyperparameters['learning_updates_per_learning_session'],
            'ppo_actor_loss': float(sum_ppo_actor_loss) / self.hyperparameters['learning_updates_per_learning_session'],
        })

        # update old policy
        self.update_old_policy()

    def get_loss(self, state_batch, action_batch, discount_reward, log_prob) -> Tuple[TT, TT, TT]:
        new_log_probs, new_entropy = self.ac.estimate_action(state_batch, action_batch)

        state_value = self.ac.value(state_batch)

        state_value_old = self.ac_old.value(state_batch).detach()

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
        policy_ratio = torch.exp(new_log_probs - log_prob.detach())

        term_1 = policy_ratio * advantage
        term_2 = torch.clamp(policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

        return -1 * torch.min(term_1, term_2), critic_loss, new_entropy

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
                    wandb_anim_record: bool = False
                    if self._wandb_anim_save % self.config.wandb_animation_frequency == 0:
                        wandb_anim_record = True
                        print('save animation to wandb')
                        wandb.log({'animation': wandb.Video(np.transpose(np.array(images)[::3, ::2, ::2, :], (0, 3, 1, 2)), fps=4, format="gif")})
                    self._wandb_anim_save += 1
                    Process(
                        target=save_as_mp4,
                        args=(
                            images,
                            f'animation/PPO-DQR/{self.name}/_R:_{total_reward}_Time:_{episode_len}_{time.time()}.mp4',
                            self.stat_logger,
                            wandb_anim_record,
                        ),
                    ).start()
                if info.get('need_reset', False):
                    print('Was made panic env reset...')
                    raise ValueError
                break
