from typing import Dict

import cv2
import gym
import numpy as np
import collections

import torch


class OriginalStateKeeper(gym.ObservationWrapper):
    """save state"""

    def __init__(self, env, state_save_name='original_state'):
        super().__init__(env)
        self._state_save_name = state_save_name
        self.__setattr__(state_save_name, None)

    def observation(self, observation):
        self.__setattr__(self._state_save_name, observation)
        return observation


class ImageToFloat(gym.ObservationWrapper):
    def __init__(self, env, image_dict_name='picture'):
        super().__init__(env)
        self._image_dict_name = image_dict_name
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space.spaces[self._image_dict_name] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                dtype=np.float32,
                shape=self.observation_space.spaces[self._image_dict_name].shape,
            )
        else:
            raise ValueError('')

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs.update({
            self._image_dict_name: np.array(obs[self._image_dict_name]).astype(np.float32) / 256
        })
        return obs


class OnlyVectorsTaker(gym.ObservationWrapper):
    def __init__(self, env, vector_car_name='car_vector', vector_env_name='env_vector'):
        super().__init__(env)
        self._vector_car_name = vector_car_name
        self._vector_env_name = vector_env_name
        self.observation_space = gym.spaces.Box(
            low=np.array(
                list(self.observation_space.spaces[self._vector_car_name].low) +
                list(self.observation_space.spaces[self._vector_env_name].low)
            ),
            high=np.array(
                list(self.observation_space.spaces[self._vector_car_name].high) +
                list(self.observation_space.spaces[self._vector_env_name].high)
            ),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.concatenate(
            (obs[self._vector_car_name], obs[self._vector_env_name]),
            axis=0,
        )


class OnlyImageTaker(gym.ObservationWrapper):
    def __init__(self, env, image_dict_name='picture'):
        super().__init__(env)
        self._image_dict_name = image_dict_name
        self.observation_space = self.observation_space.spaces[self._image_dict_name]

    def observation(self, obs):
        return obs[self._image_dict_name]


class ImageWithVectorCombiner(gym.ObservationWrapper):
    def __init__(self, env, image_dict_name='picture', vector_car_name='car_vector'):
        super().__init__(env)
        self._image_name = image_dict_name
        self._vector_car_name = vector_car_name

        image_space = self.env.observation_space.spaces[self._image_name]
        vector_space = self.env.observation_space.spaces[self._vector_car_name]

        low = self.observation({self._image_name: image_space.low, self._vector_car_name: vector_space.low})
        high = self.observation({self._image_name: image_space.high, self._vector_car_name: vector_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        image = observation[self._image_name]
        vector = observation[self._vector_car_name]
        vector_channel = np.ones(shape=list(image.shape[:-1]) + [len(vector)], dtype=np.float32) * vector
        return np.concatenate([image.astype(np.float32), vector_channel], axis=-1)


class ChannelSwapper(gym.ObservationWrapper):

    def __init__(self, env, image_dict_name='picture'):
        gym.ObservationWrapper.__init__(self, env)
        self._image_dict_name = image_dict_name

        if isinstance(self.observation_space, gym.spaces.Dict):
            # print(f"swap dict axis from : {self.observation_space.spaces['picture'].shape}", end=' ')
            self.observation_space.spaces[self._image_dict_name] = gym.spaces.Box(
                low=ChannelSwapper._image_channel_transpose(
                    self.observation_space.spaces[self._image_dict_name].low,
                ),
                high=ChannelSwapper._image_channel_transpose(
                    self.observation_space.spaces[self._image_dict_name].high,
                ),
                dtype=self.observation_space.spaces[self._image_dict_name].dtype,
            )
            # print(f"to : {self.observation_space.spaces['picture'].shape}")
        elif isinstance(self.observation_space, gym.spaces.Box):
            # print(f"swap box axis from {self.observation_space.shape}", end=' ')
            self.observation_space = gym.spaces.Box(
                low=ChannelSwapper._image_channel_transpose(self.observation_space.low),
                high=ChannelSwapper._image_channel_transpose(self.observation_space.high),
                dtype=self.observation_space.dtype,
            )
            # print(f"to : {self.observation_space.shape}")

    @staticmethod
    def _image_channel_transpose(image):
        return np.transpose(image, (2, 1, 0))

    def observation(self, observation):
        if isinstance(observation, dict):
            # print(f"swapper input image shape : {observation['picture'].shape}")
            observation.update({
                self._image_dict_name:
                    ChannelSwapper._image_channel_transpose(observation[self._image_dict_name])
            })
            return observation
        # print(f"swapper input image shape : {observation.shape}")
        return ChannelSwapper._image_channel_transpose(observation)


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
    def __init__(self, env, neutral_action, frames_in_stack=4, image_dict_name='picture', channel_order='hwc'):
        super(ImageStackWrapper, self).__init__(env)
        self._stack_len = frames_in_stack
        self._image_name = image_dict_name
        # action to perform after reset to make frames_in_stack frames in stack
        self._neutral_action = neutral_action
        assert isinstance(self.observation_space, gym.spaces.Dict), "work only with obs as dict"
        assert channel_order in ['hwc', 'chw'], "channel order mus be one of ['hwc', 'chw']"
        self._channel_order = channel_order

        if channel_order == 'hwc':
            self.observation_space.spaces[self._image_name] = gym.spaces.Box(
                low=self.observation_space.spaces[self._image_name].low[0, 0, 0],
                high=self.observation_space.spaces[self._image_name].low[0, 0, 0],
                shape=tuple([
                    self.observation_space.spaces[self._image_name].shape[0],
                    self.observation_space.spaces[self._image_name].shape[1],
                    self.observation_space.spaces[self._image_name].shape[2] * frames_in_stack,
                ])
            )
        else:
            self.observation_space.spaces[self._image_name] = gym.spaces.Box(
                low=self.observation_space.spaces[self._image_name].low[0, 0, 0],
                high=self.observation_space.spaces[self._image_name].low[0, 0, 0],
                shape=tuple([
                    self.observation_space.spaces[self._image_name].shape[0] * frames_in_stack,
                    self.observation_space.spaces[self._image_name].shape[1],
                    self.observation_space.spaces[self._image_name].shape[2],
                ])
            )

    def _get_concatenate_axis(self) -> int:
        return 2 if self._channel_order == 'hwc' else 0

    def _get_stack_buffer(self, action, step_num):
        total_reward = 0.0
        done = None
        info = None
        obs = None
        buf = []
        for index in range(step_num):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            buf.append(obs[self._image_name])
            if done:
                # make sure we have exactly stack_len frames
                if index + 1 != self._stack_len:
                    buf.extend([
                        np.zeros_like(obs[self._image_name])
                        for _ in range(self._stack_len - index - 1)
                    ])
                break
        obs.update({self._image_name: np.concatenate(buf, axis=self._get_concatenate_axis())})
        return obs, total_reward, done, info

    def step(self, action):
        return self._get_stack_buffer(action, self._stack_len)

    def reset(self):
        initial_obs = self.env.reset()
        obs, total_reward, done, info = self._get_stack_buffer(self._neutral_action, self._stack_len - 1)
        obs.update({self._image_name:
            np.concatenate([
                initial_obs[self._image_name],
                obs[self._image_name]
            ], axis=self._get_concatenate_axis())
        })
        return obs


class ExtendedMaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, image_dict_name='picture'):
        """Return only every `skip`-th frame"""
        super(ExtendedMaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
        self._image_dict_name = image_dict_name

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        obs = None
        self._obs_buffer = []
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs[self._image_dict_name])
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        del self._obs_buffer
        obs.update({self._image_dict_name: max_frame})
        return obs, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        del self._obs_buffer
        self._obs_buffer = []
        self._obs_buffer.append(obs)
        return obs


class FrameCompressor(gym.ObservationWrapper):
    def __init__(self, env, image_dict_name='picture'):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        To use this wrapper, OpenCV-Python is required.
        """
        gym.ObservationWrapper.__init__(self, env)
        self._image_dict_name = image_dict_name
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space.spaces[image_dict_name] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(84, 84, 3),
                dtype=np.uint8,
            )
        else:
            raise ValueError('ExtendedWarpFrame should wrap dict observations')

    def observation(self, obs: dict):
        frame = obs[self._image_dict_name]

        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.uint8)
        obs.update({self._image_dict_name: frame})

        # print(f"compressor, output image shape : {obs['picture'].shape}")

        return obs
