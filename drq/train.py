import argparse

import wandb
import yaml

if __name__ == '__main__':
    try:
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.pardir))

        from env import OnlyImageTaker, DictToTupleWrapper, ChannelSwapper, ImageStackWrapper
        from env.CarIntersect import CarIntersect
        from common_agents_utils.logger import Logger
    except:
        print("If you launch this from env folder, you probably will have some import problems.")


import os
import time

import hydra
import torch

import utils
from logger import Logger as original_Logger
from replay_buffer import ReplayBuffer
# from video import VideoRecorder

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg, env):
        self.env = env

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = original_Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency_step,
            agent=cfg.agent.name,
            action_repeat=cfg.action_repeat,
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            -1,  # float(self.env.action_space.low.min()),
            +1,  # float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        # self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    # def evaluate(self):
    #     average_episode_reward = 0
    #     for episode in range(self.cfg.num_eval_episodes):
    #         obs = self.env.reset()
    #         # self.video_recorder.init(enabled=(episode == 0))
    #         done = False
    #         episode_reward = 0
    #         episode_step = 0
    #         while not done:
    #             with utils.eval_mode(self.agent):
    #                 action = self.agent.act(obs, sample=False)
    #             obs, reward, done, info = self.env.step(action)
    #             # self.video_recorder.record(self.env)
    #             episode_reward += reward
    #             episode_step += 1
    #
    #         average_episode_reward += episode_reward
    #         # self.video_recorder.save(f'{self.step}.mp4')
    #     average_episode_reward /= self.cfg.num_eval_episodes
    #     self.logger.log('eval/episode_reward', average_episode_reward,
    #                     self.step)
    #     self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        logger = Logger(
            model_config=None,
            use_wandb=True, use_console=True, use_tensorboard=False,
            log_interval=2,
        )

        info = dict()
        total_env_step = 0

        while self.step < self.cfg.num_train_steps:
            if done:
                # if self.step > 0:
                #     self.logger.log('train/duration',
                #                     time.time() - start_time, self.step)
                #     start_time = time.time()
                #     self.logger.dump(
                #         self.step, save=(self.step > self.cfg.num_seed_steps))
                #
                # # evaluate agent periodically
                # if self.step % self.cfg.eval_frequency == 0:
                #     self.logger.log('eval/episode', episode, self.step)
                #     self.evaluate()
                #
                # self.logger.log('train/episode_reward', episode_reward,
                #                 self.step)

                if total_env_step > 0:
                    logger.log_it({
                        'reward': episode_reward,
                        'total_env_steps': total_env_step,
                        'track_progress': info.get('track_progress', 0.0),
                        'env_steps': episode_step,
                    })

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps and self.step % 10 == 0:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, info = self.env.step(action)
            total_env_step += 1

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 >= 500 else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):

    print(f'cur dir : {os.path.abspath(os.path.curdir)}')
    ps = os.path.abspath(os.path.curdir)

    os.chdir('../../../../.')
    print(f"mid dir : {os.path.abspath(os.path.curdir)}")

    env = CarIntersect(settings_file_path_or_settings=cfg.car_intersect_config)
    env = DictToTupleWrapper(env)
    env = ChannelSwapper(env)
    env = ImageStackWrapper(env)
    env = OnlyImageTaker(env)

    os.chdir(ps)
    print(f'dir after changes: {os.path.abspath(os.path.curdir)}')

    workspace = Workspace(cfg, env=env)
    workspace.run()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default='cpu', help='name for experiment')
    # _args = parser.parse_args()

    wandb.init(
        project='CarRacing_MassExp_test',
        reinit=True,
        name='drq_original',
    )

    main()
