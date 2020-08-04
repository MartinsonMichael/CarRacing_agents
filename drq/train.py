import json
import time

import hydra
import numpy as np
import torch
import wandb
import yaml

import utils
from replay_buffer import ReplayBuffer

if __name__ == '__main__':
    try:
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.pardir))

        from env import OnlyImageTaker, DictToTupleWrapper, ChannelSwapper, ImageStackWrapper
        from env.common_envs_utils.env_makers import get_EnvCreator_with_memory_safe_combiner
        from env.common_envs_utils.env_evaluater import evaluate_and_log, create_eval_env
        from env.common_envs_utils.batch_evaluater import BatchEvaluater
        from common_agents_utils import SubprocVecEnv_tf2

        from env.CarIntersect import CarIntersect
        from common_agents_utils.logger import Logger
    except:
        print("If you launch this from . folder, you probably will have some import problems.")


class Workspace(object):
    def __init__(self, cfg, env_maker, phi, work_dir):
        self.env_maker = env_maker
        self.env = env_maker()
        self.phi = phi

        # self.env = dmc2gym.make(
        #     domain_name='CarIntersect',
        #     task_name=task_name,
        #     seed=cfg.seed,
        #     visualize_reward=False,
        #     from_pixels=True,
        #     frame_skip=cfg.action_repeat,
        # )

        self.work_dir = work_dir
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg


        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        cfg.agent.params.obs_shape = self.phi(self.env.reset()).shape
        print(f'DRQ: get observation shape : {cfg.agent.params.obs_shape}')

        cfg.agent.params.action_shape = self.env.action_space.shape
        print(f'DRQ: get action shape : {cfg.agent.params.obs_shape}')

        cfg.agent.params.action_range = [-1, +1]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            obs_shape=cfg.agent.params.obs_shape,
            action_shape=self.env.action_space.shape,
            capacity=cfg.replay_buffer_capacity,
            image_pad=self.cfg.image_pad,
            device=self.device,
            phi=self.phi,
        )

        self.step = 0

    def run(self):
        NUM_ENVS = int(os.environ.get('NUM_ENVS', 16))

        logger = Logger(
            model_config=None,
            use_wandb=True,
            use_console=True,
            log_interval=20,
        )

        print('create evaluater')
        evaluater: BatchEvaluater = BatchEvaluater(
                env_create_method=self.env_maker,
                batch_action_get_method=lambda batch_eval_state: self.agent.act(
                    [self.phi(x) for x in batch_eval_state],
                    sample=False
                ),
                logger=logger,
                exp_class='DRQ_original',
                exp_name=self.cfg.name,
                max_episode_len=130,
                debug=True,
        )

        self.env = SubprocVecEnv_tf2([
                self.env_maker for _ in range(NUM_ENVS)
            ],
            state_flatter=lambda x: np.array(x, dtype=np.object),
        )

        os.chdir(self.work_dir)
        print(os.path.abspath(os.curdir))

        total_env_step = 0

        images = []
        record_cur_episode = False
        total_updates = 0
        total_episodes = 0

        total_reward = np.zeros(NUM_ENVS, dtype=np.float32)
        total_steps = np.zeros(NUM_ENVS, dtype=np.int16)

        zero_env_episodes = 0
        obs = self.env.reset()

        print('run test evaluate')
        evaluater.evaluate(log_animation=True)
        print('ee! test evaluate finish!')

        while self.step < self.cfg.num_train_steps:
            self.step += NUM_ENVS

            if self.step % 10000 == 100:
                with utils.eval_mode(self.agent):
                    evaluater.evaluate(log_animation=True)

            if self.step < self.cfg.num_seed_steps:
                action = np.random.uniform(-1, 1, size=(NUM_ENVS, 3))
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act([self.phi(obs_i) for obs_i in obs], sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    total_updates += 1
                    self.agent.update(self.replay_buffer, None, self.step)

            next_obs, reward, done, info = self.env.step(action)
            total_env_step += NUM_ENVS

            if record_cur_episode:
                images.append(self.env.render_zero())

            # allow infinite bootstrap
            done_no_max = np.array([
                0 if ep_len > 130 else int(ep_dn)
                for (ep_len, ep_dn) in zip(total_steps, done)
            ], dtype=np.int16)
            total_reward += reward
            total_steps += 1

            if np.any(done):
                for ind, done_i in enumerate(done):
                    if not done_i:
                        continue
                    logger.log_it({
                        'reward': total_reward[ind],
                        'track_progress': info[ind].get('track_progress', 0.0),
                        'env_steps': total_steps[ind],
                    })

            total_steps[done] = 0
            total_reward[done] = 0
            total_episodes += done.sum()

            for o, a, r, no, d, dnm in zip(obs, action, reward, next_obs, done, done_no_max):
                self.replay_buffer.add(o, a, r, no, d, dnm)

            if np.any(done):
                res_obs = self.env.reset(dones=done)
                next_obs[done] = res_obs
            obs = next_obs

            if done[0]:
                zero_env_episodes += 1
                if record_cur_episode:
                    wandb.log({
                        'train_animation': wandb.Video(
                            np.transpose(np.array(images), (0, 3, 1, 2)),
                            fps=16,
                            format="mp4",
                        )
                    })
                    record_cur_episode = False
                    images = []

                if zero_env_episodes % 100 == 10:
                    record_cur_episode = True

                logger.log_it({
                    'total_env_steps': total_env_step,
                    'total_updates': total_updates,
                })
                logger.publish_logs(step=total_env_step)


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    print(f'cur dir : {os.path.abspath(os.path.curdir)}')
    ps = os.path.abspath(os.path.curdir)

    # os.chdir('../../../../.')
    os.chdir('..')
    print(f"mid dir : {os.path.abspath(os.path.curdir)}")

    NAME = str(time.time())
    if 'NAME' in os.environ.keys():
        NAME = os.environ['NAME']
    cfg.name = NAME

    if 'ENV_CONFIG_PATH' in os.environ.keys():
        cfg.car_intersect_config = os.environ['ENV_CONFIG_PATH']

    if 'DEVICE' in os.environ.keys():
        cfg.device = os.environ['DEVICE']

    if 'TEST' in os.environ.keys():
        print('DRQ launcher: TEST settings is used')
        cfg.num_seed_steps = 1000

    try:
        env_settings = json.load(open(cfg.car_intersect_config, 'r'))
    except:
        env_settings = yaml.load(open(cfg.car_intersect_config, 'r'))

    if 'home-test' not in NAME:
        wandb.init(
            project='CarRacing_DRQ',
            reinit=True,
            name=f'drq_original_{NAME}',
            config=env_settings,
        )
    cfg.seed = np.random.randint(0, 2 ** 16 - 1)

    print(f'use name : {NAME}')
    print(f'use env config : {cfg.car_intersect_config}')
    print(f'use seed : {cfg.seed}')
    print(f'use device : {cfg.device}')

    env_maker, phi = get_EnvCreator_with_memory_safe_combiner(env_settings)

    # os.chdir(ps)
    print(f'dir after changes: {os.path.abspath(os.path.curdir)}')

    workspace = Workspace(cfg, env_maker=env_maker, phi=phi, work_dir=ps)
    workspace.run()


if __name__ == '__main__':
    main()
