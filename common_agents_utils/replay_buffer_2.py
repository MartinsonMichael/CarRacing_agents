from collections import namedtuple, deque
import random
from typing import Tuple
import torch
import numpy as np


class Torch_Arbitrary_Replay_Buffer(object):
    """
    Replay buffer to store past experiences that the agent can then use for training data

    INIT accept kwargs:
     - batch_size: int
     -
    """

    def __init__(
            self, buffer_size, device,
            sample_order=['state', 'action', 'reward', 'next_state', 'done'], **kwargs
    ):
        self.separate_state = kwargs.get('separate_state', False)
        if self.separate_state:
            raise NotImplemented
            # assert 'state_extractor' in kwargs.keys() and 'state_producer' in kwargs.keys(),\
            #     "if separate_state=True state_extractor and state_producer should be provided"
            # self._state_extractor = kwargs.get('state_extractor', lambda x: (None, x))
            # self._state_producer = kwargs.get('state_producer', lambda x, y: y)

        self.sample_order = sample_order

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = kwargs.get('batch_size', 64)
        self.experience = namedtuple(
            "Experience",
            field_names=self.sample_order
        )
        self.seed = random.seed(kwargs.get('seed', 42))
        self.device = device

    def add_experience(self, is_single=True, **kwargs):
        """Adds experience(s) into the replay buffer"""
        if not is_single:
            for values in zip(kwargs[name] for name in self.sample_order):
                self._add_single_experience(**{name: value for name, value in zip(self.sample_order, values)})
        else:
            self._add_single_experience(**kwargs)

    def _unwrap(self, item):
        if isinstance(item, (int, float, bool, np.float32, np.bool, np.int32)):
            return item
        if isinstance(item, list):
            if len(item) == 0:
                return self._unwrap(item[0])
            return item
        if isinstance(item, np.ndarray):
            if item.shape[0] == 1:
                return self._unwrap(item[0])
            return item

        raise ValueError(
            'add another type to replay_buffer\n'
            f'it was type : {type(item)}\n'
            f'content : {item}\n'
        )

    def _add_single_experience(self, **kwargs):
        self.memory.append(self.experience(*[
            self._unwrap(kwargs[name]) for name in self.sample_order
        ]))

    def _prepare_row_of_samples(self, experiences, attribute_name) -> torch.Tensor:
        return torch.from_numpy(
            np.array([e.__getattribute__(attribute_name) for e in experiences], dtype=np.float32)
        ).to(self.device)

    def sample(self, get_all=False, num_experiences=None, sample_order=None) -> Tuple[torch.Tensor]:
        """
        Sample experiences from replay buffer

        :param get_all: if True, return all buffer
        :param num_experiences: if provided return batch of size num_experiences
        :param sample_order: if provided return not standard order and set of storable experiences
        :return: Tuple[torch.Tensor]
        """
        if get_all and num_experiences is not None:
            raise ValueError('can not get_all and num_experiences simultaneously')
        if get_all:
            return self.get_all(sample_order)
        if num_experiences is None:
            num_experiences = self.batch_size
        if sample_order is None:
            sample_order = self.sample_order
        experiences = random.sample(self.memory, k=num_experiences)
        return tuple(
            self._prepare_row_of_samples(experiences, name)
            for name in sample_order
        )

    def __len__(self):
        return len(self.memory)

    def len(self):
        return self.__len__()

    def remove_all(self):
        self.memory.clear()

    def get_all(self, sample_order=None) -> Tuple[torch.Tensor]:
        if sample_order is None:
            sample_order = self.sample_order
        return tuple(
            self._prepare_row_of_samples(self.memory, name)
            for name in sample_order
        )
