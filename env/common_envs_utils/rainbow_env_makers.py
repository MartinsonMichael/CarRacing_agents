from typing import Dict, Type, Callable

from gym import ActionWrapper

from env.common_envs_utils import \
    get_state_type_from_settings, \
    CarRacingEnv, \
    OnlyVectorsTaker, OnlyImageTaker, ChannelSwapper, FrameCompressor
from env.common_envs_utils.extended_env_wrappers import MemorySafeFeatureCombiner


def get_EnvCreator_with_memory_safe_combiner(settings: Dict, discrete_wrapper: Type[ActionWrapper]) -> Callable:
    state = get_state_type_from_settings(settings)
    if state == 'image':
        return make_CarRacing_Rainbow_Picture(settings, discrete_wrapper)
    if state == 'vector':
        return make_CarRacing_Rainbow_Vector(settings, discrete_wrapper)
    if state == 'both':
        return make_CarRacing_Rainbow_Both(settings, discrete_wrapper)


def make_CarRacing_Rainbow_Both(settings: Dict, discrete_wrapper: Type[ActionWrapper]) -> Callable:
    def f():
        env = CarRacingEnv(settings_file_path_or_settings=settings)
        env = FrameCompressor(env)

        # we will swap channels in phi
        # env = ChannelSwapper(env)

        env = MemorySafeFeatureCombiner(env)
        # FRAME_STACK = 4
        # env = ImageStackWrapper(env, neutral_action=np.array([0.0, 0.0, 0.0]), frames_in_stack=FRAME_STACK)
        # env.state_image_channel_cnt = FRAME_STACK
        env = discrete_wrapper(env)
        return env
    return f


def make_CarRacing_Rainbow_Vector(settings: Dict, discrete_wrapper: Type[ActionWrapper]) -> Callable:
    def f():
        env = CarRacingEnv(settings_file_path_or_settings=settings)
        env = OnlyVectorsTaker(env)
        env = discrete_wrapper(env)
        return env
    return f


def make_CarRacing_Rainbow_Picture(settings: Dict, discrete_wrapper: Type[ActionWrapper]) -> Callable:
    def f():
        env = CarRacingEnv(settings_file_path_or_settings=settings)
        env = FrameCompressor(env)
        env = OnlyImageTaker(env)
        env = ChannelSwapper(env)
        env = discrete_wrapper(env)
        return env
    return f
