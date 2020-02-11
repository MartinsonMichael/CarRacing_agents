import json
from typing import Optional

import chainerrl
import numpy as np

from envs.common_envs_utils.env_wrappers import DiscreteWrapper
from envs.common_envs_utils.extended_env_wrappers import ExtendedMaxAndSkipEnv, FrameCompressor, \
    ImageWithVectorCombiner, ChannelSwapper, OnlyImageTaker, OnlyVectorsTaker, \
    ImageToFloat, ImageStackWrapper
from envs.gym_car_intersect_fixed import CarRacingHackatonContinuousFixed


def get_state_type_from_settings_path(settings_path: str) -> str:
    settings = json.load(open(settings_path))
    have_image = settings['state_config']['picture']
    have_vector = len(settings['state_config']['vector_car_features']) != 0
    if have_image and have_vector:
        return 'both'
    elif have_vector:
        return 'vector'
    elif have_image:
        return 'image'

    raise ValueError(f'unknown state type on path : {settings_path}')


def get_EnvCreator_by_settings(settings_path: str):
    expected_state_type = get_state_type_from_settings_path(settings_path)
    if expected_state_type == 'both':
        return make_CarRacing_fixed_combined_features
    if expected_state_type == 'image':
        return make_CarRacing_fixed_image_features
    if expected_state_type == 'vector':
        return make_CarRacing_fixed_vector_features

    raise ValueError(f'unknown state type on path : {settings_path}')


def make_CarRacing_fixed_combined_features(settings_path: str, name: Optional[str] = None, discrete_wrapper=None):
    def f():
        env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
        # env = FrameCompressor(env)
        # env = ImageStackWrapper(env, channel_order='hwc', neutral_action=np.array([0.0, 0.0, 0.0]))
        # -> dict[(.., .., 3), (16)]
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
        env = FrameCompressor(env)
        env = ImageToFloat(env)
        # -> dict[(84, 84, 3), (16)]
        env = ImageWithVectorCombiner(env)
        # -> Box(84, 84, 19)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)

        if discrete_wrapper is not None:
            env = discrete_wrapper(env)

        env._max_episode_steps = 250

        if name is not None:
            env.name = name

        return env
    return f


def make_CarRacing_fixed_image_features(settings_path: str, name: Optional[str] = None, discrete_wrapper=None):
    def f():
        env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
        # -> dict[(~106, ~106, 3), (~5-11)]
        env = FrameCompressor(env)
        env = ImageStackWrapper(env, neutral_action=np.array([0.0, 0.0, 0.0]), frames_in_stack=4)
        env = ImageToFloat(env)
        env = OnlyImageTaker(env)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)

        if discrete_wrapper is not None:
            env = discrete_wrapper(env)

        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=300)
        env._max_episode_steps = 300

        if name is not None:
            env.name = name

        return env
    return f


def make_CarRacing_fixed_vector_features(settings_path: str, name: Optional[str] = None, discrete_wrapper=None):
    def f():
        env = CarRacingHackatonContinuousFixed(settings_file_path=settings_path)
        # -> dict[(.., .., 3), (16)]
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=450)
        # -> dict[(84, 84, 3), (16)]
        env = OnlyVectorsTaker(env)
        # -> Box(16)
        env._max_episode_steps = 450

        if discrete_wrapper is not None:
            env = discrete_wrapper(env)

        if name is not None:
            env.name = name

        return env
    return f
