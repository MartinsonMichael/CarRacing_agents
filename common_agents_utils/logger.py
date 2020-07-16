import os
from typing import Dict, Any, Union, Optional
from collections import defaultdict

import wandb
import numpy as np

from common_agents_utils import Config


class Logger:

    def __init__(
            self,
            model_config: Optional[Config] = None,
            use_wandb: bool = True,
            use_console: bool = True,
            log_interval: int = 20
    ):
        self.use_wandb = use_wandb
        self.use_console = use_console

        self.model_config = model_config
        self.log_interval = log_interval

        self._stats = defaultdict(list, {})
        self.episode_number: int = 0

        self._keeping_stats = None

    def log_it(self, stats: Dict) -> None:
        for key, value in Logger._accumulate_stats(stats).items():
            self._stats[key].append(value)

    def log_and_publish(self, stats: Dict) -> None:
        stats = Logger._accumulate_stats(stats)
        self._publish_logs(stats, self.episode_number)

    def on_episode_end(self) -> None:
        self.episode_number += 1
        if self.episode_number % self.log_interval == 0:
            _stats = Logger._get_mean_logs(self._stats)
            self._publish_logs(_stats, self.episode_number)

    def on_training_end(self) -> None:
        if self._keeping_stats is None:
            self._keeping_stats = self._get_mean_logs(self._stats)

        if self.model_config is not None:
            self._write_finals_to_csv()

    @staticmethod
    def _accumulate_stats(stats: Dict[str, Any]) -> Dict:
        accumulated = defaultdict(list, {})
        for key, value in stats.items():
            if isinstance(value, (list, np.ndarray)):
                _value = np.mean(value)
            elif isinstance(value, (int, float, bool, np.int32, np.float32, np.bool)):
                _value = value
            else:
                raise ValueError(f"Logger -> unknown type : {type(value)} with value : {value}")
            accumulated[key].append(float(_value))
        return accumulated

    @staticmethod
    def _get_mean_logs(stats) -> Dict[str, float]:
        _stats = {}
        for key, value in stats.items():
            _stats[key] = float(np.mean(value))
        return _stats

    def _publish_logs(self, stats, episode_number) -> None:
        _stats = Logger._get_mean_logs(stats)
        self._keeping_stats = self._stats

        if self.use_console:
            Logger._publish_console(_stats, episode_number)
        if self.use_wandb:
            Logger._publish_wandb(_stats, episode_number)
        self._stats = defaultdict(list, {})

    @staticmethod
    def _publish_wandb(stats: Dict, episode_number: int) -> None:
        try:
            wandb.log(row=stats, step=episode_number)
        except:
            print('call wandb init. Currently wandb log is disabled')

    @staticmethod
    def _publish_console(stats: Dict, episode_number: int) -> None:
        print("Episode %d\tR %.4f\tTime %.1f\tTrack %.2f" % (
                episode_number,
                stats.get('reward', stats.get('EVAL reward', -1)),
                stats.get('env_steps', stats.get('EVAL env_steps', -1)),
                stats.get('track_progress', stats.get('EVAL track_progress', -1)),
            )
        )

    def _create_final_stat_dict(self) -> Dict[str, Union[int, float, str, bool]]:

        # create record about env observation format
        state_config = self.model_config.env_config['state']
        state_record = 'Image' if state_config['picture'] is True else ""
        if len(state_config['vector_car_features']) != 0:
            if len(state_record) != 0:
                state_record += ", "
            state_record += "[" + ", ".join(sorted(state_config['vector_car_features'])) + "]"

        final_stats = {
            'agent_class': self.model_config.agent_class,
            'total_env_episode': int(self._keeping_stats.get('total_env_episode', '-1')),
            'total_env_steps': int(self._keeping_stats.get('total_env_steps', '-1')),
            'track_progress': float(self._keeping_stats.get('moving_track_progress', -1)),
            'total_grad_steps': int(self._keeping_stats.get('total_grad_steps', -1)),
            'env_state': state_record,
            'agent_track': ', '.join(self.model_config.env_config['agent_tracks']),
            'icm': self.model_config.hyperparameters.get('use_icm', False),
            'lr': self.model_config.hyperparameters.get('lr', '--'),
        }

        return final_stats

    def _write_finals_to_csv(self) -> None:
        final_stats = self._create_final_stat_dict()
        import pandas as pd

        ind_path_to_table = os.path.join(self.model_config.table_path, f'{self.model_config.name}_records.csv')
        data_self = pd.DataFrame(columns=final_stats.keys())
        data_self = data_self.append(final_stats, ignore_index=True)
        data_self.to_csv(ind_path_to_table, index=False)

        path_to_table = os.path.join(self.model_config.table_path, 'records.csv')
        if not os.path.exists(path_to_table):
            data = pd.DataFrame(columns=final_stats.keys())
        else:
            data = pd.read_csv(path_to_table)

        data = data.append(final_stats, ignore_index=True)
        data.to_csv(path_to_table, index=False, index_label=False)
