import gym
import numpy as np


class RewardClipperWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward / 100


class RewardDivider(gym.RewardWrapper):
    def __init__(self, env, ratio):
        super().__init__(env)
        self._reward_ratio = ratio

    def reward(self, reward):
        return reward / self._reward_ratio


class RewardNormalizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._r = 0
        self._running_statistic = []

    def reward(self, reward):
        self._r = 0.95 * self._r + reward
        self._running_statistic.append(self._r)
        std = np.std(self._running_statistic)
        if std < 1e-8:
            return reward
        return reward / std

    def reset(self):
        self._running_statistic = []
        self._r = 0
        return self.env.reset()
