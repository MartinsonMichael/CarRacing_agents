from collections import namedtuple, deque
import random
import torch
import numpy as np


class Torch_Separated_Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(
            self, buffer_size, batch_size, seed, device,
            sample_order=['state', 'action', 'reward', 'next_state', 'done'],
            **kwargs
    ):
        self.separate_state = kwargs.get('separate_state', False)
        if self.separate_state:
            raise NotImplemented
            assert 'state_extractor' in kwargs.keys() and 'state_producer' in kwargs.keys(),\
                "if separate_state=True state_extractor and state_producer should be provided"
            self._state_extractor = kwargs.get('state_extractor', lambda x: (None, x))
            self._state_producer = kwargs.get('state_producer', lambda x, y: y)

        self.sample_order = sample_order

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=sample_order
        )
        self.seed = random.seed(seed)
        self.device = device

    def add_experience(self, is_single=True, **kwargs):
        """Adds experience(s) into the replay buffer"""
        if is_single:
            for values in zip(kwargs[name] for name in self.sample_order):
                self._add_single_experience(**{name: value for name, value in zip(self.sample_order, values)})
        else:
            self._add_single_experience(**kwargs)

    def _unwrap(self, item):
        if isinstance(item, (int, float, bool)):
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
        self.memory.append(self.experience(
            self._unwrap(kwargs[name]) for name in self.sample_order
        ))

    def _prepare_row_of_samples(self, experiences, attribute_name) -> torch.Tensor:
        if attribute_name == 'state':
            return torch.from_numpy(np.array([
                self._state_producer(e.state_picture, e.state_vector)
                for e in experiences
            ], dtype=np.float32)).to(self.device)
        if attribute_name == 'next_state':
            return torch.from_numpy(np.array([
                self._state_producer(e.next_state_picture, e.next_state_vector)
                for e in experiences
            ], dtype=np.float32)).to(self.device)
        if attribute_name == 'action':
            return torch.from_numpy(np.array([e.action for e in experiences], dtype=np.float32)).to(self.device)
        if attribute_name == 'reward':
            return torch.from_numpy(np.array([[e.reward] for e in experiences], dtype=np.float32)).to(self.device)
        if attribute_name == 'done':
            return torch.from_numpy(np.array([[e.done] for e in experiences], dtype=np.float32)).to(self.device)
        if attribute_name == 'log_prob':
            return torch.from_numpy(np.array([e.log_prob for e in experiences], dtype=np.float32)).to(self.device)

    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        batch = (
            self._prepare_row_of_samples(experiences, name)
            for name in self._sample_order
        )

        # print(f"exp state_picture.shape: {experiences[0].state_picture.shape}")
        # print(f"exp state_vector.shape: {experiences[0].state_vector.shape}")
        # print('whole shape : ', batch[0][0].shape)

        return batch

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

    def clean_all_buffer(self):
        self.memory.clear()

    def get_all(self):
        batch = (
            self._prepare_row_of_samples(self.memory, name)
            for name in self._sample_order
        )
        return batch
