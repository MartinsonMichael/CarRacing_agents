import json
from typing import Union, Dict, Any
from collections import deque
import numpy as np


class Rewarder:
    """
    Class to define reward policy.
    """

    def __init__(self, settings):
        self._settings = settings
        self._settings_reward: Dict[str, Union[int, float, bool]] = settings['reward']
        self._settings_done: Dict[str, Any] = settings['done']
        self._finish_times: int = 0
        self._prev_coordinates = deque(maxlen=10)

    def get_step_reward(self, car_stats: Dict[str, Any]) -> float:
        """
        function to compute reward for current step
        :param car_stats: dist with car stats
        keys are:
        'new_tiles_count': integer, how many track point achieve agent at last step
        'is_finish': bool
        'is_collided': bool
        'is_out_of_track': bool, car not on chosen track
        'is_on_cross_road': bool, car is on cross road
        'is_out_of_map': bool
        'is_out_of_road': bool, car not on any road
        'speed': float, car linear velocity
        'time': integer, steps from car creating
        :return: reward for current step
        """
        step_reward = 0.0

        step_reward += car_stats['new_tiles_count'] * self._settings_reward['new_tiles_count']
        step_reward += car_stats['time'] * self._settings_reward['time_per_point']
        step_reward += self._settings_reward['time_per_tick']

        sumdist = 0
        cur_point = np.array(car_stats.get('coordinate_vector', [0, 0]))
        for prev_dot in self._prev_coordinates:
            sumdist += np.sqrt((prev_dot - cur_point)**2)
        sumdist = sumdist / len(self._prev_coordinates)
        step_reward += sumdist * self._settings_reward.get('displacement', 0.0)

        if np.sqrt((np.array(car_stats['last_action']) ** 2).sum()) < \
                self._settings_reward['idleness__punish_if_action_radius_less_then']:
            step_reward += self._settings_reward['idleness__punish_value']

        step_reward += self._settings_reward.get('speed_multiplication_bonus', 0) * car_stats['speed']

        for is_item in ['is_collided', 'is_finish', 'is_out_of_track', 'is_out_of_map', 'is_out_of_road']:
            if car_stats[is_item]:
                step_reward += self._settings_reward[is_item]

        self._prev_coordinates.append(cur_point)

        return step_reward

    def get_step_done(self, car_stats) -> bool:
        """
        function to compute done flag for current step
        :param car_stats: dist with car stats
        keys are:
        'new_tiles_count': integer, how many track point achieve agent at last step
        'is_finish': bool
        'is_collided': bool
        'is_out_of_track': bool, car not on chosen track
        'is_on_cross_road': bool, car is on cross road
        'is_out_of_map': bool
        'is_out_of_road': bool, car not on any road
        'speed': float, car linear velocity
        'time': integer, steps from car creating
        :return: bool, done flag for current step
        """
        done = False

        for item in self._settings_done['true_flags_to_done']:
            if item == 'is_collided' and car_stats['time'] < 15:
                continue
            if car_stats[item]:
                done = True
                break

        return done
