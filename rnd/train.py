import json
import time

import gym
import torch
import wandb
import yaml
from agents import RNDAgent
from envs import AtariEnvironment

from car_intersect_env_maker import makeCarIntersect, evaluate_and_log, create_eval_env
from utils import RunningMeanStd, RewardForwardFilter, make_train_data
from config import default_config
from torch.multiprocessing import Pipe

# from tensorboardX import SummaryWriter

import numpy as np

if __name__ == '__main__':
    try:
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.pardir))

        from env.common_envs_utils.env_makers import get_EnvCreator_with_pure_vectors
        from env.common_envs_utils.action_wrappers import DiscreteWrapper
        from common_agents_utils.logger import Logger
    except:
        print("If you launch this from . folder, you probably will have some import problems.")


def main():
    NAME = str(time.time())
    if 'NAME' in os.environ.keys():
        NAME = os.environ['NAME']
    else:
        raise ValueError('set NAME via env variable')

    try:
        env_settings = json.load(open(default_config['CarIntersectConfigPath'], 'r'))
    except:
        env_settings = yaml.load(open(default_config['CarIntersectConfigPath'], 'r'))

    wandb.init(
        project='CarRacing_RND',
        reinit=True,
        name=f'drq_original_{NAME}',
        config={'env_config': env_settings, 'agent_config': default_config},
    )

    # print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']

    env_id = default_config['EnvID']
    # env_type = default_config['EnvType']

    # if env_type == 'mario':
    #     env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    # elif env_type == 'atari':
    #     env = gym.make(env_id)
    # else:
    #     raise NotImplementedError

    seed = np.random.randint(0, 2 ** 16 - 1)

    print(f'use name : {NAME}')
    print(f"use env config : {default_config['CarIntersectConfigPath']}")
    print(f'use seed : {seed}')
    print(f"use device : {os.environ['DEVICE']}")

    os.chdir('..')
    env = makeCarIntersect(env_settings)
    eval_env = create_eval_env(makeCarIntersect(env_settings))

    # input_size = env.observation_space.shape  # 4
    input_size = env.observation_space.shape
    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_size = env.action_space.n  # 2

    env.close()

    is_load_model = True
    is_render = False
    # model_path = 'models/{}.model'.format(NAME)
    # predictor_path = 'models/{}.pred'.format(NAME)
    # target_path = 'models/{}.target'.format(NAME)

    # writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    agent = RNDAgent(
        input_size,
        action_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        device=os.environ['DEVICE'],
    )

    # if is_load_model:
    #     print('load model...')
    #     if use_cuda:
    #         agent.model.load_state_dict(torch.load(model_path))
    #         agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
    #         agent.rnd.target.load_state_dict(torch.load(target_path))
    #     else:
    #         agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    #         agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
    #         agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    #     print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = AtariEnvironment(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, settings=env_settings)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    os.chdir('rnd')

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    logger = Logger(None, use_console=True, use_wandb=True, log_interval=1)

    print('Test evaluater:')
    evaluate_and_log(
        eval_env=eval_env,
        action_get_method=lambda eval_state: agent.get_action(
            np.tile(np.float32(eval_state), (1, 4, 1, 1)) / 255.
        )[0],
        logger=logger,
        log_animation=True,
        exp_class='RND',
        exp_name=NAME,
        debug=True,
    )
    print('end evaluater test.')

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, action_size, size=(num_worker, ))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                # writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                # writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                # writer.add_scalar('data/step', sample_step, sample_episode)
                logger.log_it({
                    'reward_per_episode': sample_rall,
                    'intrinsic_reward': sample_i_rall,
                    'episode_steps': sample_step,
                })
                logger.on_episode_end()
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        # writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

        # if global_step % (num_worker * num_step * 100) == 0:
        #     print('Now Global Step :{}'.format(global_step))
        #     torch.save(agent.model.state_dict(), model_path)
        #     torch.save(agent.rnd.predictor.state_dict(), predictor_path)
        #     torch.save(agent.rnd.target.state_dict(), target_path)

        if global_update % 30 == 0:
            evaluate_and_log(
                eval_env=eval_env,
                action_get_method=lambda eval_state: agent.get_action(
                    np.tile(np.float32(eval_state), (1, 4, 1, 1)) / 255.
                )[0],
                logger=logger,
                log_animation=True,
                exp_class='RND',
                exp_name=NAME,
            )
            logger.on_episode_end()


if __name__ == '__main__':
    main()
