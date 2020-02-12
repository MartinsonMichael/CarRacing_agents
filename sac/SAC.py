import os
import pickle
import time
from typing import Dict, Union, Any

import wandb
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import tensorflow as tf

from common_agents_utils.Config import config_to_key_value
from envs.common_envs_utils.env_state_utils import \
    get_state_combiner_by_settings_file, \
    from_image_vector_to_combined_state

from common_agents_utils import Torch_Separated_Replay_Buffer, Config
from common_agents_utils import QNet, Policy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC:
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config: Config):
        self.name = config.name
        self.tf_writer = config.tf_writer
        self.environment = config.environment
        self.action_size = config.environment.action_space.shape[0]
        self.device = config.hyperparameters['device']

        self.hyperparameters = config.hyperparameters

        self.folder_save_path = os.path.join('model_saves', self.name)

        self.critic_local = QNet(
            state_description=config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        self.critic_local_2 = QNet(
            state_description=config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4,
        )
        self.critic_optimizer_2 = torch.optim.Adam(
            self.critic_local_2.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4,
        )

        self.critic_target = QNet(
            state_description=config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        self.critic_target_2 = QNet(
            state_description=config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        SAC.copy_model_over(self.critic_local, self.critic_target)
        SAC.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.memory = Torch_Separated_Replay_Buffer(
            self.hyperparameters["buffer_size"],
            self.hyperparameters["batch_size"],
            self.hyperparameters["seed"],
            device=self.device,
            state_extractor=get_state_combiner_by_settings_file(self.hyperparameters['env_settings_file_path']),
            state_producer=from_image_vector_to_combined_state,
        )

        self.actor_local = Policy(
            state_description=config.environment.observation_space,
            action_size=self.action_size,
            hidden_size=256,
            device=self.device,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(),
            lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4
        )

        self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item()  # heuristic value from the paper
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        self._game_stats = {}
        self._last_episode_save_count = 0
        self._current_run_global_steps = 0
        self.episode_number = 0
        self.global_step_number = 0

    def run_n_episodes(self, visualize=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        num_episodes = self.hyperparameters['num_episodes_to_run']
        start = time.time()

        for _ in range(num_episodes):
            self.reset_game()
            self.step(visualize if self.episode_number % 5 == 0 else False)
            self.episode_number += 1

        time_taken = time.time() - start
        print(f'Time taken : {time_taken}')
        return time_taken

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        # np.random.seed(int(round(time.time() * 1000)) % 2**31)
        """Resets the game informatenvsion so we are ready to play a new episode"""

        self.environment.seed(
            (int(time.time() * 10 ** 5) % 10 ** 5) * 10 ** 5 + int(time.time() * 10 ** 5) % 10 ** 5
        )
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        self._game_stats = {}

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, self.info = self.environment.step(action)
        self.update_stats_due_to_step_info(self.info, self.reward, self.done)
        self.total_episode_score_so_far += self.reward

    def step(self, visualize=False):
        self._last_episode_save_count += 1
        if self._last_episode_save_count % 15 == 1:
            self.step_with_huge_stats()

        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(False)
            self.conduct_action(self.action)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()

            mask = self.done
            self.memory.add_experience(
                self.state, self.action, self.reward, self.next_state, mask
            )
            self.state = self.next_state
            self.global_step_number += 1
            self._current_run_global_steps += 1
            print(f'global steps : {self.global_step_number}')

            if self.episode_step_number_val > self.hyperparameters['max_episode_steps'] + 10:
                break
            if self.info.get('need_reset', False) or self.info.get('was_reset', False):
                break

        self.log_it()

        print(f"score : {self.total_episode_score_so_far}")
        self.episode_number += 1

        if visualize and self._current_run_global_steps > self.hyperparameters["min_steps_before_learning"]:
            from envs.common_envs_utils import episode_visualizer
            episode_visualizer(
                env=self.environment,
                action_picker=lambda state: self.actor_pick_action(state, eval=True),
                name=self.name,
            )

        if self._last_episode_save_count >= self.hyperparameters['save_frequency_episode']:
            self._last_episode_save_count = 0
            self.save()

    def step_with_huge_stats(self):
        done = False
        state = self.environment.reset()
        total_reward = 0.0
        local_step_number = 0
        print('Huge Eval Start')
        mean_stats = {}

        def stats_extractor(x: Dict[str, Union[float, Any]], prefix='') -> Dict[str, float]:
            ret = {}
            for key, value in x.items():
                if isinstance(value, dict):
                    ret.update(stats_extractor(value, prefix=prefix + ' -> ' + key))
                else:
                    ret[prefix + ' -> ' + key] = value
            return ret

        while not done:
            local_step_number += 1
            print('\n***')
            print(f'step : {local_step_number}')
            full_stats = {}

            action_train, log_prob, action_eval, actor_step_stats = self.produce_action_and_action_info(state, return_stats=True)
            action_train = action_train.detach().cpu().numpy()
            action_eval = action_eval.detach().cpu().numpy()
            state, reward, done, info = self.environment.step(action_train[0])
            total_reward += reward
            print(f'reward : {reward}')
            # print(f'action train: {action_train}\t  Radius : {round(float((action_train ** 2).sum()), 2)}')
            full_stats['action_train_radius'] = (action_train ** 2).sum()
            # print(f'action eval : {action_eval}\t  Radius : {round(float((action_eval ** 2).sum()), 2)}')
            full_stats['action_eval_radius'] = (action_eval ** 2).sum()
            print('env info:')
            print(info)
            # print()

            q1_value, critic1_step_stats = self.critic_local(state, torch.from_numpy(action_train).to(self.device), return_stats=True)
            q2_value, critic2_step_stats = self.critic_local_2(state, torch.from_numpy(action_train).to(self.device), return_stats=True)
            # print(f'Q local value 1 : {q1_value.detach().cpu().numpy()[0]}')
            full_stats['q_local_value_1'] = q1_value.detach().cpu().numpy()[0]
            # print(f'Q local value 2 : {q2_value.detach().cpu().numpy()[0]}')
            full_stats['q_local_value_2'] = q2_value.detach().cpu().numpy()[0]

            q1_target_value, critic_target1_step_stats = self.critic_target(state, torch.from_numpy(action_train).to(self.device), return_stats=True)
            q2_target_value, critic_target2_step_stats = self.critic_target_2(state, torch.from_numpy(action_train).to(self.device), return_stats=True)
            # print(f'Q target value 1 : {q1_target_value.detach().cpu().numpy()[0]}')
            full_stats['q_target_value_1'] = q1_target_value.detach().cpu().numpy()[0]
            # print(f'Q target value 2 : {q2_target_value.detach().cpu().numpy()[0]}')
            full_stats['q_target_value_2'] = q2_target_value.detach().cpu().numpy()[0]

            # print('\nActor stats:')
            # print(actor_step_stats)
            #
            # print('\nCritic stats:')
            # print(critic1_step_stats)

            full_stats['actor'] = actor_step_stats
            full_stats['critic'] = critic1_step_stats
            full_stats = stats_extractor(full_stats)

            for key, value in full_stats.items():
                if key not in mean_stats.keys():
                    mean_stats[key] = []
                mean_stats[key].append(value)

            if 'was_reset' in info.keys():
                print('Was Reset')
                break

            if info.get('need_reset', False) or info.get('was_reset', False):
                break
            if local_step_number > self.hyperparameters['max_episode_steps'] + 10:
                break

        with self.tf_writer.as_default():
            tf.summary.scalar(
                name='huge eval reward',
                data=total_reward,
                step=self.episode_number,
            )
            tf.summary.scalar(
                name='huge eval steps cnt',
                data=local_step_number,
                step=self.episode_number,
            )
            for name, value in mean_stats.items():
                print(name)
                tf.summary.scalar(
                    name='MEAN' + ' ' + name,
                    data=np.array(value).mean(),
                    step=self.episode_number,
                )
                tf.summary.scalar(
                    name='MAX' + ' ' + name,
                    data=np.array(value).max(),
                    step=self.episode_number,
                )
                tf.summary.scalar(
                    name='MIN' + ' ' + name,
                    data=np.array(value).min(),
                    step=self.episode_number,
                )
                tf.summary.scalar(
                    name='MEDIAN' + ' ' + name,
                    data=np.median(np.array(value)),
                    step=self.episode_number,
                )

        print(f'Total reward : {total_reward}')
        print('Huge Eval End.')

    def save(self):
        current_save_path = os.path.join(
            self.folder_save_path,
            f"{self.name}_episode_{self.episode_number}_time_{time.time()}"
        )
        os.makedirs(current_save_path)

        torch.save(self.critic_local.state_dict(), os.path.join(current_save_path, 'critic_local'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(current_save_path, 'critic_optimizer'))
        torch.save(self.critic_target.state_dict(), os.path.join(current_save_path, 'critic_target'))

        torch.save(self.critic_local_2.state_dict(), os.path.join(current_save_path, 'critic_local_2'))
        torch.save(self.critic_optimizer_2.state_dict(), os.path.join(current_save_path, 'critic_optimizer_2'))
        torch.save(self.critic_target_2.state_dict(), os.path.join(current_save_path, 'critic_target_2'))

        torch.save(self.actor_local.state_dict(), os.path.join(current_save_path, 'actor_local'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(current_save_path, 'actor_optimizer'))

        torch.save(self.alpha, os.path.join(current_save_path, 'alpha'))
        torch.save(self.alpha_optim.state_dict(), os.path.join(current_save_path, 'alpha_optim'))

        pickle.dump((
            self.global_step_number,
            self.episode_number,

        ), open(os.path.join(current_save_path, 'stats.pkl'), 'wb'))

        return current_save_path

    def load(self, folder):

        for model_name in ['critic_local', 'critic_target', 'critic_optimizer',
                           'critic_local_2', 'critic_target_2', 'critic_optimizer_2',
                           'actor_local', 'actor_optimizer',
                           'alpha_optim']:
            self.__getattribute__(model_name).load_state_dict(torch.load(os.path.join(folder, model_name)))

        self.alpha = torch.load(os.path.join(folder, 'alpha'))

        self.global_step_number, self.episode_number = pickle.load(
            open(os.path.join(folder, 'stats.pkl'), 'rb')
        )

    def update_stats_due_to_step_info(self, info, reward, done):
        if not done:
            return
        if 'is_finish' in info.keys():
            if info['is_finish']:
                self._game_stats['finished'] = 1
            else:
                self._game_stats['finished'] = 0
        if 'time' in info.keys():
            self._game_stats['env steps taken'] = info['time']
        if 'track_progress' in info.keys():
            self._game_stats['track_progress'] = info['track_progress']

        # self._game_stats['temperature'] = self.alpha.cpu().detach().numpy()[0]

    def log_it(self):
        if self._current_run_global_steps < self.hyperparameters['min_steps_before_learning']:
            return

        log_stats = {}

        for name, value in self._game_stats.items():
            if 'loss' in name and self.episode_step_number_val > 0:
                value /= self.episode_step_number_val
                name = 'average ' + name
            log_stats[name] = value

        if self.tf_writer is not None:
            with self.tf_writer.as_default():
                for name, value in log_stats.items():
                    tf.summary.scalar(name=name, data=value, step=self.episode_number)

        # WanDB logging:
        wandb.log(row=log_stats, step=self.episode_number)

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None:
            state = self.state

        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        else:
            if self._current_run_global_steps < self.hyperparameters["min_steps_before_learning"]:
                if self._current_run_global_steps < self.hyperparameters['random_replay_prefill_ration'] * self.hyperparameters["min_steps_before_learning"]:
                    action = np.random.uniform(-1, 1, self.environment.action_space.shape)
                else:
                    action = self.actor_pick_action(state=state, eval=False)
                print("Picking random action ", action)
            else:
                action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state

        if not eval:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, return_stats: bool = False):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        if return_stats:
            actor_output, actor_stats = self.actor_local(state, return_stats)
        else:
            actor_output = self.actor_local(state)

        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)

        if return_stats:
            actor_stats['action'] = {
                'mean': mean.detach().cpu().numpy(),
                'std': std.detach().cpu().numpy(),
            }
            return action, log_prob, torch.tanh(mean), actor_stats
        else:
            return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return (
                self._current_run_global_steps > self.hyperparameters["min_steps_before_learning"] and
                self.enough_experiences_to_learn_from() and
                self._current_run_global_steps % self.hyperparameters["update_every_n_steps"] == 0
        )

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)

            # print(f'next_state_batch : {next_state_batch.shape}')
            # print(f'next_state_action : {next_state_action.shape}')

            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

            # print(f'reward : {reward_batch.shape}')
            # print(f'mask : {mask_batch.shape}')
            # print(f'min_qf_next_target : {min_qf_next_target.shape}')

            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)

            # print(f'next_q_value : {next_q_value.shape}')

        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)

        # print(f'state batch : {state_batch.shape}')
        # print(f'qf1 : {qf1.shape}')

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('test_log_dir')
        # writer.add_graph(critic_loss_1)
        # writer.add_graph(actor_loss)

        for loss_name, loss_value in [
            ('q1 loss', critic_loss_1),
            ('q2 loss', critic_loss_2),
            ('policy loss', actor_loss),
            ('temperature loss', alpha_loss),
            ('temperature', self.alpha[0]),
        ]:
            if loss_name not in self._game_stats.keys():
                self._game_stats[loss_name] = 0.0
            self._game_stats[loss_name] += loss_value.cpu().detach().numpy()

        self.take_optimisation_step(
            self.critic_optimizer,
            self.critic_local,
            critic_loss_1,
            self.hyperparameters["Critic"]["gradient_clipping_norm"]
        )
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])
        if alpha_loss is not None:
            if self.hyperparameters['high_temperature']:
                if self.global_step_number < 500:
                    self.alpha = torch.from_numpy(np.array([1.0], dtype=np.float32)).to(self.device)
                elif self.global_step_number < 10000:
                    self.alpha = torch.from_numpy(np.array([
                        1.0 - self.global_step_number / 11000.0
                    ], dtype=np.float32)).to(self.device)
                else:
                    self.alpha = torch.from_numpy(np.array([0.05], dtype=np.float32)).to(self.device)
            else:
                self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
                # self.take_optimisation_step(
                #     optimizer=self.alpha_optim,
                #     network=None,
                #     loss=alpha_loss,
                #     clipping_norm=0.1,
                #     retain_graph=False,
                # )
                self.alpha = self.log_alpha.exp()
        self.alpha = torch.clamp(self.alpha, 0.0, 4.0)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")


    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())