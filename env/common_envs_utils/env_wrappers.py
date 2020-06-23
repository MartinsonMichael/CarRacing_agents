import cv2
import gym
import numpy as np
import collections
from gym import spaces

from common_agents_utils.typingTypes import NpA, TT
from typing import Dict, Tuple, List, Optional


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, channel_order='chw'):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        To use this wrapper, OpenCV-Python is required.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            'hwc': (self.height, self.width, 3),
            'chw': (3, self.height, self.width),
        }
        self.shape = shape[channel_order]
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=self.shape, dtype=np.uint8,
        )

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if frame.shape != self.shape:
            frame = np.transpose(frame, (2, 1, 0))
        return frame.astype(np.uint8)


class DictToTupleWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_dict_name='picture', vector_dict_name='car_vector'):
        super().__init__(env)

        self._image_dict_name = image_dict_name
        self._vector_dict_name = vector_dict_name
        self.observation_space = gym.spaces.Tuple((
            self.env.observation_space.spaces[self._image_dict_name],
            self.env.observation_space.spaces[self._vector_dict_name],
        ))

    def observation(self, obs):
        assert isinstance(obs, dict), "DictToTupleWrapper expect observation to be dict, " \
                                      f"but it has type : {type(obs)}"
        return tuple((obs[self._image_dict_name], obs[self._vector_dict_name]))


class ChannelSwapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        assert isinstance(self.observation_space, gym.spaces.Tuple)

        self.observation_space.spaces = gym.spaces.Tuple((
            gym.spaces.Box(
                low=ChannelSwapper._image_channel_transpose(self.observation_space.spaces[0].low),
                high=ChannelSwapper._image_channel_transpose(self.observation_space.spaces[0].high),
                dtype=self.observation_space.spaces[0].dtype,
            ),
            self.observation_space.spaces[1],
        ))

    @staticmethod
    def _image_channel_transpose(image):
        return np.transpose(image, (2, 1, 0))

    def observation(self, obs) -> Tuple[NpA, Optional[NpA]]:
        assert isinstance(obs, tuple), "ChannelSwapper wrapper expect observation to be tuple, " \
                                       f"but is has type {type(obs)}"
        return tuple((ChannelSwapper._image_channel_transpose(obs[0]), obs[1]))


class SkipWrapper(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipWrapper, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        obs = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        del self._obs_buffer
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


class ImageStackWrapper(gym.Wrapper):
    def __init__(self, env, neutral_action=None, frames_in_stack=4, channel_order=None):
        super(ImageStackWrapper, self).__init__(env)
        self._stack_len = frames_in_stack

        if neutral_action is not None:
            self._neutral_action = neutral_action
        else:
            self._neutral_action = [0, 0, 0]
        assert isinstance(self.observation_space, gym.spaces.Tuple), "work only with observation as tuple"
        assert channel_order in ['hwc', 'chw', None], "channel order mus be one of ['hwc', 'chw'], " \
                                                      "or None for autodetect"
        if channel_order is None:
            if self.observation_space.spaces[0].shape[0] == 3:
                self._channel_order = 'chw'
            elif self.observation_space.spaces[0].shape[2] == 3:
                self._channel_order = 'hwc'
            else:
                raise ValueError(f"can't determine channel order for shape {self.observation_space.spaces[0].shape}")
        else:
            self._channel_order = channel_order

        shape = self.observation_space.spaces[0].shape
        if channel_order == 'hwc':
            self._final_image_shape = tuple((shape[0], shape[1], shape[2] * self._stack_len))
            self._single_channel_num = shape[2]
        else:
            self._final_image_shape = tuple((shape[0] * self._stack_len, shape[1], shape[2]))
            self._single_channel_num = shape[0]
        self.observation_space.spaces = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=self._final_image_shape, dtype=np.uint8),
            self.observation_space.spaces[1],
        ))

    def _get_concatenate_axis(self) -> int:
        return 2 if self._channel_order == 'hwc' else 0

    def _get_zero_obs_for_steps(self, step_num: int) -> np.ndarray:
        if self._channel_order == 'hwc':
            return np.zeros((
                *self._final_image_shape[:2],
                step_num * self._single_channel_num,
            ), dtype=np.uint8)
        else:
            return np.zeros((
                step_num * self._single_channel_num,
                *self._final_image_shape[1:],
            ), dtype=np.uint8)

    def _get_stack_buffer(self, action, step_num):
        total_reward = 0.0
        done, info, last_obs = None, None, None
        final_image_obs = self._get_zero_obs_for_steps(step_num)
        for index in range(step_num):
            obs, reward, done, info = self.env.step(action)
            last_obs = obs
            total_reward += reward
            if self._channel_order == 'chw':
                final_image_obs[
                    index * self._single_channel_num : (index + 1) * self._single_channel_num, :, :,
                ] = obs[0]
            else:
                final_image_obs[
                    :, :, index * self._single_channel_num : (index + 1) * self._single_channel_num,
                ] = obs[0]
            if done:
                break
        return tuple((final_image_obs, *last_obs[1:])), total_reward, done, info

    def step(self, action):
        return self._get_stack_buffer(action, self._stack_len)

    def reset(self):
        initial_obs = self.env.reset()
        obs, _, _, _ = self._get_stack_buffer(self._neutral_action, self._stack_len - 1)
        image_obs = np.concatenate([initial_obs[0], obs[0]], axis=self._get_concatenate_axis())
        return tuple((image_obs, *obs[1:]))


class OnlyImageTaker(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces[0]

    def observation(self, obs):
        assert isinstance(obs, tuple), "OnlyImageTaker expect observation to be tuple, " \
                                      f"but it has type : {type(obs)}"
        return obs[0]
