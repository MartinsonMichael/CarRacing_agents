import os
from multiprocessing import Process
from typing import Callable, Optional
import time
import numpy as np

from common_agents_utils.logger import Logger
from common_agents_utils.multiprocessor_env import SubprocVecEnv_tf2
from env import CarIntersectEvalWrapper, CarIntersect
from env.common_envs_utils.visualizer import save_as_mp4


class BatchEvaluater:

    def __init__(
        self,
        env_create_method: Callable[[], CarIntersect],
        logger: Logger,
        batch_action_get_method: Callable,
        batch_size: int = 4,
        exp_class: Optional[str] = None,
        exp_name: Optional[str] = None,
        max_episode_len: int = 500,
        debug: bool = False,
    ):
        self.logger = logger
        self.action_get = batch_action_get_method
        self.batch_size = batch_size
        self.exp_class = exp_class
        self.exp_name = exp_name
        self.max_episode_len = max_episode_len
        self._debug = debug

        self.envs = SubprocVecEnv_tf2([
                lambda: CarIntersectEvalWrapper(env_create_method()) for _ in range(self.batch_size)
            ],
            state_flatter=lambda x: np.array(x, dtype=np.object),
        )

        self.not_finished = np.ones(batch_size, dtype=np.int16)
        self.total_reward = np.zeros(batch_size, dtype=np.float32)
        self.total_steps = np.zeros(batch_size, dtype=np.int16)
        self.total_track_progress = np.zeros(batch_size, dtype=np.float32)

    def evaluate(self, log_animation: bool):
        batch_state = self.envs.reset()

        self.not_finished.fill(1)
        self.total_reward.fill(0.0)
        self.total_steps.fill(0)
        self.total_track_progress.fill(0.0)

        img = []

        while True:

            batch_action = self.action_get(batch_state)
            batch_state, batch_reward, batch_done, batch_info = self.envs.step(batch_action)

            self.total_steps += self.not_finished * 1
            self.total_reward += self.not_finished * batch_reward

            currently_done = np.logical_or(batch_done, self.total_steps > self.max_episode_len)

            self.not_finished *= 1 - currently_done.astype(np.int16)

            self.total_track_progress[currently_done] = np.array([
                x.get('track_progress', -1.0)
                for x in batch_info[currently_done]
            ])

            if log_animation and self.not_finished[0] == 1:
                img.append(self.envs.render_zero())

            if self.not_finished.sum() == 0:
                break

            if np.any(batch_done):
                batch_state[batch_done] = self.envs.reset(dones=batch_done)

        self.logger.log_and_publish({
            'EVAL reward': self.total_reward.mean(),
            'EVAL track_progress': self.total_track_progress.mean(),
            'EVAL env_steps': self.total_steps.mean(),
        })

        if log_animation:
            Process(
                target=save_as_mp4,
                args=(
                    img,
                    os.path.join(
                        'animation',
                        self.exp_class,
                        self.exp_name,
                        f"EVAL_R:_{self.total_reward.mean()}_Time:_{self.total_steps.mean()}_{time.time()}.mp4",
                    ),
                    True
                ),
            ).start()

