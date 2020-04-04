import logging
import os
import time
import datetime

import chainer
import numpy as np
import tensorflow as tf
from chainerrl import agents, action_value, explorer
from chainerrl.experiments.evaluator import record_stats, save_agent
from chainerrl.explorers.epsilon_greedy import select_action_epsilon_greedily
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainer import functions as F
from chainer import links as L

from env.common_envs_utils.visualizer import plot_sequence_images


def run_evaluation_episodes(env, agent, n_steps, n_episodes,
                            max_episode_len=None, logger=None, visualise=None):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    scores = []
    terminate = False
    timestep = 0

    statistic = {
        'score': 0.0,
        'finished': 0.0,
        'env steps taken': 0.0,
        'track %': 0.0,
    }

    reset = True
    images = []
    obs = None
    while not terminate:
        if reset:
            env.visualize_next_episode()
            obs = env.reset()
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        images.append(env.get_true_picture())
        timestep += 1
        reset = (done or statistic['env steps taken'] >= max_episode_len or info.get('needs_reset', False))

        statistic['score'] += r
        statistic['env steps taken'] += 1.0
        if info.get('is_finish', False):
            statistic['finished'] = 1.0


        if reset or terminate:
            agent.stop_episode()
            statistic['track_progress'] = info['track_progress']
            break

    # FIXME
    # if no ffmpeg it fails and whole training process fails
    if visualise is not None and False:
        folder_full_path = visualise
        plot_sequence_images(images, need_disaply=False, need_save=os.path.join(
            folder_full_path, f"R_{statistic['score']}__Step_{statistic['env steps taken']}__Time_{datetime.datetime.now()}_.mp4"
        ))

    return statistic


class ExtendedEvaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(self,
                 agent,
                 env,
                 n_steps,
                 n_episodes,
                 eval_interval,
                 outdir,
                 max_episode_len=None,
                 step_offset=0,
                 save_best_so_far_agent=True,
                 logger=None,
                 ):
        assert (n_steps is None) != (n_episodes is None), \
            ("One of n_steps or n_episodes must be None. " +
             "Either we evaluate for a specified number " +
             "of episodes or for a specified number of timesteps.")
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = (self.step_offset -
                            self.step_offset % self.eval_interval)
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        self._evaluated_overall_time = 0
        agent_name = self.agent.name
        self.folder_full_path = os.path.join('rainbow_save_animation_folder', agent_name)
        if not os.path.exists(self.folder_full_path):
            os.makedirs(self.folder_full_path)
        self.tf_writer = tf.summary.create_file_writer(os.path.join('rainbow_tb_logs', agent_name))

    def log_tf_records(self, statistics, step):
        with self.tf_writer.as_default():
            for name, value in statistics.items():
                tf.summary.scalar(name=name, data=value, step=step)

            for (name, value) in self.agent.get_statistics():
                tf.summary.scalar(name=name, data=value, step=step)

    def evaluate_and_update_max_score(self, t, episodes):
        self._evaluated_overall_time += 1
        sum_statistic = {}

        for index in range(5):
            print(f'Evaluation {index} : ', end='', flush=True)
            statistic = run_evaluation_episodes(
                self.env, self.agent, self.n_steps, self.n_episodes,
                max_episode_len=self.max_episode_len,
                logger=self.logger,
                visualise=self.folder_full_path if index == 0 else None,
            )
            print(f"score : {statistic['score']}", flush=True)
            for key, value in statistic.items():
                if key in sum_statistic.keys():
                    sum_statistic[key] += value
                else:
                    sum_statistic[key] = value

        statistic = {key: value / 5 for key, value in sum_statistic.items()}

        self.log_tf_records(statistic, episodes)

        mean = statistic['score']
        if mean > self.max_score:
            self.logger.info('The best score is updated %s -> %s',
                             self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            print('will evaluate')
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None


class QuadraticDecayEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with quadratic decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func, logger=logging.getLogger(__name__)):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        assert start_epsilon > end_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.logger = logger
        self.epsilon = start_epsilon

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            t0 = self.decay_steps
            x0 = self.start_epsilon - self.end_epsilon
            A = x0 / t0**2
            t1 = t0 - t
            return A * t1**2 + self.end_epsilon

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        self.logger.debug('t:%s a:%s %s', t, a, greedy_str)
        return a

    def __repr__(self):
        return 'LinearDecayEpsilonGreedy(epsilon={})'.format(self.epsilon)



class DistributionalDuelingDQN_Vector(chainer.Chain, StateQFunction, RecurrentChainMixin):
    """Distributional dueling fully-connected Q-function with discrete actions.

    """

    def __init__(self, n_actions, n_atoms, v_min, v_max,
                 state_size, activation=F.relu, bias=0.1):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.state_size = state_size
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        z_values = self.xp.linspace(v_min, v_max,
                                    num=n_atoms,
                                    dtype=np.float32)
        self.add_persistent('z_values', z_values)

        with self.init_scope():
            self.l1 = L.Linear(state_size, 256)
            self.l2 = L.Linear(256, 256)

            self.main_stream = L.Linear(256, 1024)
            self.a_stream = L.Linear(512, n_actions * n_atoms)
            self.v_stream = L.Linear(512, n_atoms)

    def __call__(self, x):
        # h = x
        # for l in self.conv_layers:
        #     h = self.activation(l(h))
        h = self.activation(self.l2(
                self.activation(self.l1(
                    x
                ))
            )
        )

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h))
        h_a, h_v = F.split_axis(h, 2, axis=-1)
        ya = F.reshape(self.a_stream(h_a),
                       (batch_size, self.n_actions, self.n_atoms))

        mean = F.sum(ya, axis=1, keepdims=True) / self.n_actions

        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = F.reshape(self.v_stream(h_v), (batch_size, 1, self.n_atoms))
        ya, ys = F.broadcast(ya, ys)
        q = F.softmax(ya + ys, axis=2)

        return action_value.DistributionalDiscreteActionValue(q, self.z_values)

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
            ('n_updates', self.optimizer.t),
        ]
