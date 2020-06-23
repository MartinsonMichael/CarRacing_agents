import json
from typing import Dict, Type, Callable, Tuple, Any

from gym import ActionWrapper
import numpy as np

from env.common_envs_utils import \
    ChannelSwapper, DictToTupleWrapper, ImageStackWrapper
from env.CarIntersect import CarIntersect


def get_state_type_from_settings_path(settings_path: str) -> str:
    return get_state_type_from_settings(json.load(open(settings_path)))


def get_state_type_from_settings(settings: Dict[str, Any]) -> str:
    have_image: bool = settings['state']['picture']
    have_vector: bool = len(settings['state']['vector_car_features']) != 0
    if have_image and have_vector:
        return 'both'
    elif have_vector:
        return 'vector'
    elif have_image:
        return 'image'

    raise ValueError(f'unknown state type on settings : {settings}')


def get_EnvCreator_with_memory_safe_combiner(
        settings: Dict,
        discrete_wrapper: Type[ActionWrapper] = None
) -> Tuple[Callable, Callable]:
    """
    Return function to make env and function to create np.ndarray from env returned state
    """
    state = get_state_type_from_settings(settings)
    if state == 'image':
        return (
            make_CarRacing_Picture(settings, discrete_wrapper),
            image_phi,
        )
    if state == 'vector':
        return (
            make_CarRacing_Vector(settings, discrete_wrapper),
            vector_phi,
        )
    if state == 'both':
        return (
            make_CarRacing_Both(settings, discrete_wrapper),
            both_phi,
        )


def image_phi(x):
    # image, vector = x
    return np.array(x[0]).astype(np.float32) / 255


def vector_phi(x):
    # image, vector = x
    return x[1]


def both_phi(x):
    image, vector = x
    vector_channel = np.ones(shape=list(image.shape[:-1]) + [len(vector)], dtype=np.float32) * vector
    combined = np.concatenate([image.astype(np.float32) / 255, vector_channel], axis=-1)
    return np.transpose(combined, (2, 1, 0))


def make_CarRacing_Both(settings: Dict, discrete_wrapper: Type[ActionWrapper] = None) -> Callable:
    def f():
        env = CarIntersect(settings_file_path_or_settings=settings)
        env = DictToTupleWrapper(env)
        if discrete_wrapper is not None:
            env = discrete_wrapper(env)
        return env
    return f


def make_CarRacing_Vector(settings: Dict, discrete_wrapper: Type[ActionWrapper] = None) -> Callable:
    def f():
        env = CarIntersect(settings_file_path_or_settings=settings)
        env = DictToTupleWrapper(env)
        # env = OnlyVectorsTaker(env)
        if discrete_wrapper is not None:
            env = discrete_wrapper(env)
        return env
    return f


def make_CarRacing_Picture(settings: Dict, discrete_wrapper: Type[ActionWrapper] = None) -> Callable:
    def f():
        env = CarIntersect(settings_file_path_or_settings=settings)
        env = DictToTupleWrapper(env)
        # env = OnlyImageTaker(env)
        env = ChannelSwapper(env)
        env = ImageStackWrapper(env)
        if discrete_wrapper is not None:
            env = discrete_wrapper(env)
        return env
    return f
