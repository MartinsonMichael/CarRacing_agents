from typing import Dict, Any
from collections import defaultdict

import wandb
import tensorflow as tf
import numpy as np

from common_agents_utils import Config


class Logger:

    def __init__(
            self,
            model_config: Config,
            use_wandb: bool = True,
            use_tensorboard: bool = True,
            use_console: bool = True,
            log_interval: int = 20
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.use_console = use_console

        self.model_config = model_config
        self.log_interval = log_interval

        self._stats = defaultdict(list, {})
        self.episode_number: int = 0

    def log_it(self, stats) -> None:
        self._accumulate_stats(stats)
        self.episode_number += 1
        if self.episode_number % self.log_interval == 0:
            self._publish_logs()

    def _accumulate_stats(self, stats: Dict[str, Any]) -> None:
        for key, value in stats.items():
            if isinstance(value, (list, np.ndarray)):
                _value = np.mean(value)
            elif isinstance(value, (int, float, bool, np.int32, np.float32, np.bool)):
                _value = value
            else:
                raise ValueError(f"Logger -> unknown type : {type(value)} with value : {value}")
            self._stats[key].append(float(_value))

    def _publish_logs(self) -> None:
        _stats = {}
        for key, value in self._stats.items():
            _stats[key] = np.mean(value)
        self._stats = _stats

        if self.use_console:
            self._publish_console()
        if self.use_wandb:
            self._publish_wandb()
        if self.use_tensorboard:
            self._publish_tensorboard()
        self._stats = defaultdict(list, {})

    def _publish_wandb(self) -> None:
        try:
            wandb.log(row=self._stats, step=self.episode_number)
        except:
            print('call wandb init, currently wandb log is disabled')

    def _publish_tensorboard(self) -> None:
        with self.model_config.tf_writer.as_default():
            for name, value in self._stats.items():
                tf.summary.scalar(name=name, data=value, step=self.episode_number)

    def _publish_console(self) -> None:
        print("Episode %d\tR %.3f\tTime %.1f" % (
                self.episode_number,
                self._stats.get('reward', None),
                self._stats.get('env_steps', None)
            )
        )
