from typing import Tuple, Union
import numpy as np

from envs import get_state_type_from_settings_path


def get_state_combiner_by_settings_file(settings_file_path: str):
    """
    return function that
    Transform single state np.ndarray to
    0 - image np.ndarray with dtype np.uint8 and
    1 - vector np.ndarray with type np.float32
    """
    state_type = get_state_type_from_settings_path(settings_file_path)
    if state_type == 'both':
        return lambda state: _state_splitter__both(state)
    if state_type == 'vector':
        return lambda state: (None, state)
    if state_type == 'image':
        return lambda state: (_prepare_image_to_buffer(state), None)


def _state_splitter__both(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(state.shape) == 3, "state must have 3 dimensions"

    state_picture, state_vector_extended = np.split(state, [3], axis=0)
    state_vector = state_vector_extended[:, 0, 0]
    del state_vector_extended

    return _prepare_image_to_buffer(state_picture), state_vector


def from_image_vector_to_combined_state(image: Union[np.ndarray, None], vector: Union[np.ndarray, None]) -> np.ndarray:
    if image is not None and vector is not None:
        return np.concatenate([
                _prepare_image_to_model(image),
                np.transpose(np.tile(vector, (image.shape[1], image.shape[2], 1)), (2, 1, 0)),
            ],
            axis=0,
        )
    if image is not None:
        return _prepare_image_to_model(image)
    if vector is not None:
        return vector

    raise ValueError('both image and vector are none')


def _prepare_image_to_buffer(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(a=image * 255.0, a_min=0.0, a_max=255.0).astype(np.uint8)


def _prepare_image_to_model(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image
