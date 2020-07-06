import json

import dmc2gym
import wandb
import numpy as np
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
        print("If you launch this from . folder, you probably will have some import problems.")


import os
import time

import hydra
import torch

import utils
from logger import Logger as original_Logger
from replay_buffer import ReplayBuffer


class Workspace(object):
    def __init__(self, cfg, env):
        self.env = env

        # self.env = dmc2gym.make(
        #     domain_name='CarIntersect',
        #     task_name=task_name,
        #     seed=cfg.seed,
        #     visualize_reward=False,
        #     from_pixels=True,
        #     frame_skip=cfg.action_repeat,
        # )

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

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [-1, +1]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        self.step = 0

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

        images = []
        record_cur_episode = False

        while self.step < self.cfg.num_train_steps:
            if done:

                if record_cur_episode:
                    wandb.log({
                        'animation': wandb.Video(
                            np.transpose(np.array(images)[::3, ::2, ::2, :], (0, 3, 1, 2)),
                            fps=4,
                            format="gif",
                        )
                    })
                    record_cur_episode = False
                    images = []

                if episode % 100 == 10:
                    record_cur_episode = True

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

            if record_cur_episode:
                images.append(self.env.render(full_image=True))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 >= 130 else done
            episode_reward += reward

            if episode_step > 200:
                done = True

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

    NAME = str(time.time())
    if 'NAME' in os.environ.keys():
        NAME = os.environ['NAME']

    try:
        env_settings = json.load(open(cfg.car_intersect_config, 'r'))
    except:
        env_settings = yaml.load(open(cfg.car_intersect_config, 'r'))

    wandb.init(
        project='CarRacing_MassExp_test',
        reinit=True,
        name=f'drq_original_{NAME}',
        config=env_settings,
    )

    env = CarIntersect(settings_file_path_or_settings=env_settings)
    env = DictToTupleWrapper(env)
    env = ChannelSwapper(env)
    env = ImageStackWrapper(env)
    env = OnlyImageTaker(env)

    os.chdir(ps)
    print(f'dir after changes: {os.path.abspath(os.path.curdir)}')

    if 'DEVICE' in os.environ.keys():
        cfg.device = os.environ['DEVICE']
    cfg.seed = np.random.randint(0, 2**16-1)
    print(f'use seed : {cfg.seed}')
    workspace = Workspace(cfg, env=env)
    workspace.run()


if __name__ == '__main__':

    main()
