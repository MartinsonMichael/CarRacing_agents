from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import os

from chainerrl.experiments import train_agent_batch

from env import DiscreteWrapper
from env.common_envs_utils.env_makers import make_CarRacing_fixed_vector_features, make_CarRacing_fixed_image_features, \
    make_CarRacing_fixed_combined_features, get_state_type_from_settings_path
from rainbow.extended_rainbow import ExtendedEvaluator, run_evaluation_episodes, DistributionalDuelingDQN_Vector, \
    QuadraticDecayEpsilonGreedy

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import functools
from builtins import *  # NOQA

from future import standard_library

standard_library.install_aliases()  # NOQA
import argparse
import os

import chainer
import numpy as np

import chainerrl
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DistributionalDuelingDQN
from chainerrl import replay_buffer


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='CarIntersect-v3')
    parser.add_argument('--outdir', type=str, default='rainbow_results', help='Directory path to save output files.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--eval-epsilon', type=float, default=0.0)
    parser.add_argument('--noisy-net-sigma', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=2 * 10 ** 6)
    parser.add_argument('--replay-start-size', type=int, default=2 * 10 ** 4)
    parser.add_argument('--eval-n-episodes', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=5 * 10 ** 3)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--monitor', action='store_true', default=False, help='Monitor env.')
    parser.add_argument('--num-env', type=int, default=40)
    parser.add_argument('--final-epsilon', type=float, default=0.01)
    parser.add_argument('--final-exploration-frames', type=int, default=2 * 10 ** 4)
    parser.add_argument(
        '--env-settings',
        type=str,
        default='env/settings_sets/env_settings__basic_straight_line.json',
        help='path to settings file for CarRacingFixed',
    )
    parser.add_argument('--name', type=str, help='name')
    args = parser.parse_args()

    env_creater = None
    mode = get_state_type_from_settings_path(args.env_settings)
    if mode not in ['both', 'image', 'vector']:
        raise ValueError('set correct mode')
    else:
        if mode == 'vector':
            env_creater = make_CarRacing_fixed_vector_features
        if mode == 'image':
            env_creater = make_CarRacing_fixed_image_features
        if mode == 'both':
            env_creater = make_CarRacing_fixed_combined_features

    if args.name is None:
        raise ValueError('need name')

    import logging
    logging.basicConfig(level=args.logging_level)

    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_batch_env(test):
        vec_env = chainerrl.envs.MultiprocessVectorEnv([
            functools.partial(env_creater(
                args.env_settings,
                name=args.name,
                discrete_wrapper=DiscreteWrapper,
            ))
            for _, _ in enumerate(range(args.num_envs))
        ])
        # vec_env = chainerrl.wrappers.VectorFrameStack(vec_env, 4)
        # print(vec_env.observation_space)
        return vec_env

    env = make_batch_env(test=False)

    n_actions = env.action_space.n

    n_atoms = 51
    v_max = 10
    v_min = -10
    test_env = env_creater(args.env_settings, name='test_env', discrete_wrapper=DiscreteWrapper)()

    if mode in ['both', 'image']:
        channels_num = test_env.observation_space.shape[0]
        q_func = DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=channels_num)
    if mode == 'vector':
        state_size = test_env.observation_space.shape[0]
        q_func = DistributionalDuelingDQN_Vector(n_actions, n_atoms, v_min, v_max, state_size=state_size)

    # Noisy nets
    links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
    # Turn off explorer
    # explorer = explorers.LinearDecayEpsilonGreedy(
    #     0.3, args.final_epsilon,
    #     args.final_exploration_frames,
    #     lambda: np.random.randint(n_actions))
    explorer = QuadraticDecayEpsilonGreedy(
        start_epsilon=0.4,
        end_epsilon=0.001,
        decay_steps=args.steps * 0.90,
        random_action_func=lambda: np.random.randint(n_actions),
    )


    # Draw the computational graph and save it in the output directory.
    # chainerrl.misc.draw_computational_graph(
    #     [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
    #     os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as https://arxiv.org/abs/1707.06887
    opt = chainer.optimizers.Adam(0.001, eps=1.5 * 10 ** -4)
    opt.setup(q_func)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 4
    betasteps = args.steps / update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(
        10 ** 5, alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=10)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    # Agent = agents.CategoricalDoubleDQN
    Agent = agents.CategoricalDoubleDQN
    print(args.replay_start_size)

    agent = Agent(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, minibatch_size=64,
        replay_start_size=args.replay_start_size,
        target_update_interval=3 * 10 ** 3,
        update_interval=update_interval,
        batch_accumulator='mean',
        phi=phi,
    )
    agent.name = args.name

    if args.load is not None:
        print('loading from')
        dir_of_best_network = os.path.join(args.load)
        agent.load(dir_of_best_network)

    if args.evaluate:
        for _ in range(10):
            stats = run_evaluation_episodes(
                env=env,
                agent=agent,
                n_episodes=10,
                logger=None,
            )
            print(stats)
    else:
        print('training started')

        logger = logging.getLogger(__name__)
        os.makedirs(args.outdir, exist_ok=True)

        evaluator = ExtendedEvaluator(
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_episodes,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            max_episode_len=300,
            env=test_env,
            step_offset=0,
            save_best_so_far_agent=True,
            logger=logger,
        )

        train_agent_batch(
            agent, env, args.steps, args.outdir,
            max_episode_len=300,
            step_offset=0,
            eval_interval=args.eval_interval,
            evaluator=evaluator,
            successful_score=0.15,
            return_window_size=100,
            log_interval=100,
            step_hooks=(),
            logger=logger)


if __name__ == '__main__':
    main()
