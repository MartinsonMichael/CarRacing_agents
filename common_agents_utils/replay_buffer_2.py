from collections import namedtuple, deque
import random
from common_agents_utils.typingTypes import *
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
        print('replay buffer -> kwargs')
        print(kwargs)
        # self.separate_state = kwargs.get('separate_state', False)
        # if self.separate_state:
        #     raise NotImplemented

        self.sample_order = sample_order
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=self.sample_order)

        self._auto = kwargs.get('do_it_auto', True)
        self.mode = kwargs.get('state_mode', None)
        self.state_split_channels = kwargs.get('state_channel_split', None)

        self._auto_was_inited: bool = False
        self.batch_size = kwargs.get('batch_size', 64)
        self.seed = random.seed(kwargs.get('seed', 42))

        # convert input values and deconvert them by passing rules, if necessary check types of stored items
        self._sample_converter: Dict[str, Callable[[Any], Any]] = kwargs.get('sample_converter', {})
        self._sample_deconverter: Dict[str, Callable[[Any], Any]] = kwargs.get('sample_deconverter', {})
        self._check_type_dict: Dict[str, Any] = kwargs.get('check_type_dict', {})

        self.device = device

    def add_experience(self, is_single=True, **kwargs):
        """Adds experience(s) into the replay buffer"""
        if not is_single:
            for values in zip(*[kwargs[name] for name in self.sample_order]):
                self._add_single_experience(**{name: value for name, value in zip(self.sample_order, values)})
        else:
            self._add_single_experience(**kwargs)

    @staticmethod
    def _state_splitter__both(state: NpA, channel_to_split=3) -> Tuple[NpA, NpA]:
        assert len(state.shape) == 3, "state must have 3 dimensions"

        state_picture, state_vector_extended = np.split(state, [channel_to_split], axis=0)
        state_vector = state_vector_extended[:, 0, 0]
        del state_vector_extended

        return (state_picture / 255).astype(np.uint8), state_vector

    @staticmethod
    def _from_image_vector_to_combined_state(compress_state: Tuple[NpA, NpA]) -> NpA:
        image, vector = compress_state
        return np.concatenate([
                image.astype(np.float32) * 255,
                np.transpose(np.tile(vector, (image.shape[1], image.shape[2], 1)), (2, 1, 0)),
            ],
            axis=0,
        )

    def _init_auto(self, **kwargs) -> None:
        print(f"replay buffer auto detector:")
        print(f"use mode : {self.mode}")
        self._auto_was_inited = True
        item_auto_detected = 0
        for name, _value in kwargs.items():
            value = self._unwrap(_value)
            if name.startswith('state') or name.endswith('state'):
                print(f"{item_auto_detected} -> {name}")
                print(f"{name} is state container")
                print(f"input shape : {value.shape}")
                print(f"min max values -> {np.min(value)}, {np.max(value)}")
                if len(value.shape) == 3:
                    if self.mode == 'vector':
                        raise ValueError(f'mode {self.mode} is incompatible with 3 size state dimension')

                    if self.mode == 'image' or self.mode is None:
                        print(f"state treat as image (convert to uint8 and deconvert to float32)")
                        self._sample_converter[name] = lambda x: np.array(x * 255).astype(np.uint8)
                        self._sample_deconverter[name] = lambda x: np.array(x).astype(np.float32) / 255

                    if self.mode == 'both':
                        print(f"it treat as BOTH (convert to uint8+float32 and deconvert to float32)")

                        self._sample_converter[name] = \
                            lambda x: Torch_Arbitrary_Replay_Buffer._state_splitter__both(x, self.state_split_channels)
                        self._sample_deconverter[name] = \
                            lambda x: Torch_Arbitrary_Replay_Buffer._from_image_vector_to_combined_state(x)

                if len(value.shape) == 1:
                    if self.mode in {'image', 'both'}:
                        raise ValueError(f'mode {self.mode} is incompatible with 1 size state dimension')

                    print(f"state treat as vector (no convert, no deconvert, type float32)")
                    # self._check_type_dict[name] = np.float32
                print('***')
                item_auto_detected += 1
        if item_auto_detected == 0:
            print("No item was auto detected.")
        print("END of 'replay buffer auto detector'")

    def _unwrap(self, item):
        if isinstance(item, (torch.FloatTensor, torch.Tensor, torch.cuda.FloatTensor)):
            item = item.detach().cpu().numpy()
        while True:
            if isinstance(item, (int, float, bool, np.float32, np.bool, np.int32)):
                return item
            if isinstance(item, list):
                if len(item) == 0:
                    item = item[0]
                    continue
                return item
            if isinstance(item, np.ndarray):
                if item.shape[0] == 1:
                    item = item[0]
                    continue
                return item
            raise ValueError(
                'add another type to replay_buffer\n'
                f'it was type : {type(item)}\n'
                f'content : {item}\n'
            )

    def _type_checker(self, value, name: str):
        if name in self._check_type_dict.keys():
            assert type(value) == self._check_type_dict[name], \
                f"You pass value check for {name}, BUT!\n"\
                f"it must be {self._check_type_dict[name]}\n"\
                f"you pass {type(value)}"

    def _converter(self, value, name: str):
        if name in self._sample_converter.keys():
            return self._sample_converter[name](value)
        return value

    def _deconverter(self, value: NpA, name: str) -> NpA:
        if name in self._sample_deconverter.keys():
            return self._sample_deconverter[name](value)
        return value

    def _add_single_experience(self, **kwargs) -> None:
        if self._auto and not self._auto_was_inited:
            self._init_auto(**kwargs)

        exp = [self._unwrap(kwargs[name]) for name in self.sample_order]

        if self._sample_converter.__len__() != 0:
            exp = [self._converter(value, name) for name, value in zip(self.sample_order, exp)]

        if self._check_type_dict.__len__() == 0:
            for name, value in zip(self.sample_order, exp):
                self._type_checker(value, name)

        self.memory.append(self.experience(*exp))

    def _prepare_row_of_samples(self, experiences, attribute_name: str) -> TT:
        return torch.from_numpy(
            np.array([
                self._deconverter(e.__getattribute__(attribute_name), attribute_name)
                for e in experiences
            ], dtype=np.float32)
        ).to(self.device)

    def sample(self, get_all=False, num_experiences=None, sample_order=None) -> Tuple[TT]:
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

    def __len__(self) -> int:
        return len(self.memory)

    def len(self) -> int:
        return self.__len__()

    def remove_all(self) -> None:
        self.memory.clear()

    def get_all(self, sample_order=None) -> Tuple[TT]:
        if sample_order is None:
            sample_order = self.sample_order
        return tuple(
            self._prepare_row_of_samples(self.memory, name)
            for name in sample_order
        )

    def iterate_over_N_batch(self, N: int, batch_size=None, sample_order=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sample_order is None:
            sample_order = self.sample_order
        indexes = np.arange(len(self))
        np.random.shuffle(indexes)
        experiences = random.sample(self.memory, k=batch_size * N)
        for index in range(0, N, batch_size):
            yield tuple(
                self._prepare_row_of_samples(experiences[index:index+batch_size], name)
                for name in sample_order
            )
