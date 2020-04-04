import json
from typing import Optional, Dict, Union, Any

import chainerrl
import numpy as np

from env.common_envs_utils.extended_env_wrappers import ExtendedMaxAndSkipEnv, FrameCompressor, \
    ImageWithVectorCombiner, ChannelSwapper, OnlyImageTaker, OnlyVectorsTaker, \
    ImageToFloat, ImageStackWrapper, ObservationPictureNormalizer, RewardNormalizer
from env.CarRacing_env import CarRacingEnv


def get_state_type_from_settings_path(settings_path: str) -> str:
    return get_state_type_from_settings(json.load(open(settings_path)))


def get_state_type_from_settings(settings: Dict[str, Any]) -> str:
    have_image = settings['state_config']['picture']
    have_vector = len(settings['state_config']['vector_car_features']) != 0
    if have_image and have_vector:
        return 'both'
    elif have_vector:
        return 'vector'
    elif have_image:
        return 'image'

    raise ValueError(f'unknown state type on settings : {settings}')


def get_EnvCreator_by_settings(settings_path: Union[str, Dict]):
    if isinstance(settings_path, str):
        expected_state_type = get_state_type_from_settings_path(settings_path)
    else:
        expected_state_type = get_state_type_from_settings(settings_path)

    if expected_state_type == 'both':
        return make_CarRacing_fixed_combined_features
    if expected_state_type == 'image':
        return make_CarRacing_fixed_image_features
    if expected_state_type == 'vector':
        return make_CarRacing_fixed_vector_features

    raise ValueError(f'unknown state type on path : {settings_path}')


def make_CarRacing_fixed_combined_features(settings_path: str, name: Optional[str] = None, discrete_wrapper=None):
    def f():
        env = CarRacingEnv(settings_file_path_or_settings=settings_path)
        env = FrameCompressor(env)

        FRAME_STACK = 4
        env = ImageStackWrapper(env, neutral_action=np.array([0.0, 0.0, 0.0]), frames_in_stack=FRAME_STACK)
        env.state_image_channel_cnt = FRAME_STACK

        env = ImageToFloat(env)
        # -> dict[(84, 84, 3), (16)]
        env = ImageWithVectorCombiner(env)
        # -> Box(84, 84, 19)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)

        if discrete_wrapper is not None:
            env = discrete_wrapper(env)

        if name is not None:
            env.name = name

        return env
    return f


def make_CarRacing_fixed_image_features(settings_path: str, name: Optional[str] = None, discrete_wrapper=None):
    def f():
        env = CarRacingEnv(settings_file_path_or_settings=settings_path)
        # -> dict[(~106, ~106, 3), (~5-11)]
        env = FrameCompressor(env)
        FRAME_STACK = 4
        env = ImageStackWrapper(env, neutral_action=np.array([0.0, 0.0, 0.0]), frames_in_stack=FRAME_STACK)
        env.state_image_channel_cnt = FRAME_STACK

        # env = ImageToFloat(env)
        env = ObservationPictureNormalizer(env)
        env = RewardNormalizer(env)
        env = OnlyImageTaker(env)
        env = ChannelSwapper(env)
        # -> Box(19, 84, 84)

        if discrete_wrapper is not None:
            env = discrete_wrapper(env)

        if name is not None:
            env.name = name

        return env
    return f


def make_CarRacing_fixed_vector_features(settings_path: str, name: Optional[str] = None, discrete_wrapper=None):
    def f():
        env = CarRacingEnv(settings_file_path_or_settings=settings_path)
        # -> dict[(.., .., 3), (16)]
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=500)
        # -> dict[(84, 84, 3), (16)]
        env = OnlyVectorsTaker(env)
        # -> Box(16)
        env._max_episode_steps = 500

        if discrete_wrapper is not None:
            env = discrete_wrapper(env)

        if name is not None:
            env.name = name

        return env
    return f
