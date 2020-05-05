from collections import namedtuple, deque
import random
from common_agents_utils.typingTypes import *
import torch
import numpy as np


class TorchReplayBufferV2(object):
    """
    Replay buffer to store past experiences that the agent can then use for training data

    INIT accept kwargs:
     - batch_size: int
     -
    """

    def __init__(
            self,
            buffer_size: int,
            device: str,
            phi: Optional[Callable] = None,
            sample_order: Tuple[str] = ('state', 'action', 'reward', 'next_state', 'done'),
            keys_to_apply_phi: Union[List[str], str, None] = 'default',
            batch_size: int = 256,
            seed: int = 42,
            unwrap_cuda: bool = True,
            unwrap_shape: bool = False,
    ):
        if keys_to_apply_phi is None:
            self.keys_to_apply_phi = []
        else:
            if isinstance(keys_to_apply_phi, list):
                self.keys_to_apply_phi = keys_to_apply_phi
            elif isinstance(keys_to_apply_phi, str):
                assert keys_to_apply_phi == 'default'
                self.keys_to_apply_phi = [x for x in sample_order if 'state' in x]
            else:
                raise ValueError('unknown keys_to_apply_phi')

        if phi is None:
            assert not isinstance(keys_to_apply_phi, list), \
                "Set phi, or use keys_to_apply_phi as None (or 'default')"
            self.phi = None
        else:
            self.phi = phi

        self.sample_order = sample_order
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=self.sample_order)

        self.batch_size = batch_size
        self.seed = seed
        self.unwrap_cuda = unwrap_cuda
        self.unwrap_shape = unwrap_shape

        self.device = device

    def add_experience(self, is_single=True, **kwargs):
        """Adds experience(s) into the replay buffer"""
        if not is_single:
            self.add_batch_experience(**kwargs)
        else:
            self.add_single_experience(**kwargs)

    def add_batch_experience(self, **kwargs):
        for values in zip(*[kwargs[name] for name in self.sample_order]):
            self.add_single_experience(**{name: value for name, value in zip(self.sample_order, values)})

    def _unwrap_shape(self, item) -> Any:
        while True:
            if isinstance(item, (int, float, bool, np.float32, np.bool, np.int32)):
                return item
            if isinstance(item, (list, np.ndarray, tuple)):
                if len(item) == 0:
                    item = item[0]
                    continue
                return item

            raise ValueError(
                'add another type to replay_buffer\n'
                f'it was type : {type(item)}\n'
                f'content : {item}\n'
            )

    def _unwrap_cuda(self, item) -> NpA:
        if isinstance(item, (torch.FloatTensor, torch.Tensor, torch.cuda.FloatTensor)):
            return item.detach().cpu().numpy()
        return item

    def add_single_experience(self, **kwargs) -> None:
        exp = (kwargs[name] for name in self.sample_order)
        if self.unwrap_cuda and self.unwrap_shape:
            exp = (self._unwrap_shape(self._unwrap_cuda(item)) for item in exp)
        elif self.unwrap_cuda:
            exp = (self._unwrap_cuda(item) for item in exp)
        elif self.unwrap_shape:
            exp = (self._unwrap_shape(item) for item in exp)

        self.memory.append(self.experience(exp))

    def _prepare_row_of_samples(self, experiences, attribute_name: str) -> TT:
        if attribute_name is self.keys_to_apply_phi:
            return torch.from_numpy(
                np.array([self.phi(e.__getattribute__(attribute_name)) for e in experiences], dtype=np.float32)
            ).to(self.device)
        else:
            return torch.from_numpy(
                np.array([e.__getattribute__(attribute_name) for e in experiences], dtype=np.float32)
            ).to(self.device)

    def sample(self, num_experiences=None) -> Tuple[TT, ...]:
        """
        Sample experiences from replay buffer

        :param num_experiences: if provided return batch of size num_experiences
        :return: Tuple[torch.Tensor]
        """
        if num_experiences is None:
            num_experiences = self.batch_size
        experiences = random.sample(self.memory, k=num_experiences)
        return tuple(
            self._prepare_row_of_samples(experiences, name)
            for name in self.sample_order
        )

    def __len__(self) -> int:
        return len(self.memory)

    def len(self) -> int:
        return self.__len__()

    def remove_all(self) -> None:
        self.memory.clear()

    def get_all(self) -> Tuple[TT, ...]:
        return tuple(
            self._prepare_row_of_samples(self.memory, name)
            for name in self.sample_order
        )
