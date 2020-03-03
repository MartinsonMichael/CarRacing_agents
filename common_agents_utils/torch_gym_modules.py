from common_agents_utils.typingTypes import *

import numpy as np
import torch
import torch.nn as nn
from gym import spaces


def process_kwargs(tensor: TT, **kwargs) -> npTT:
    POSSIBLE_KEYS = ['to_numpy', 'remove_batch']
    if len(set(kwargs.keys()) - set(POSSIBLE_KEYS)) != 0:
        raise ValueError(
            f'process_kwargs for kwargs {kwargs} fails\n'
            f'function expect only keys : {POSSIBLE_KEYS}'
        )
    if kwargs.get('remove_batch', False):
        if tensor.shape[0] != 1:
            raise ValueError(f'can remove batch only from tensor with shape (1, ...), you have {tensor.shape}')
        tensor = tensor[0]
    if kwargs.get('to_numpy', False):
        tensor = tensor.detach().cpu().numpy()

    return tensor


def get_activated_ratio(x: Union[torch.Tensor, torch.FloatTensor]) -> float:
    if torch.prod(torch.tensor(x.size())).cpu().numpy() == 0:
        return 0.0
    return (x.detach() > 0).cpu().numpy().sum() / torch.prod(torch.tensor(x.size())).cpu().numpy()


def make_it_batched_torch_tensor(x, device) -> Union[torch.FloatTensor, torch.Tensor]:
    if isinstance(x, (torch.FloatTensor, torch.Tensor, torch.cuda.FloatTensor)):
        if len(x.shape) == 2 or len(x.shape) == 4:
            return x.to(device)
        else:
            return x.unsqueeze_(0).to(device)
    if isinstance(x, np.ndarray):
        if len(x.shape) == 2 or len(x.shape) == 4:
            return torch.from_numpy(x.astype(np.float32)).to(device)
        else:
            return torch.from_numpy(np.array([x]).astype(np.float32)).to(device)

    print('state trouble')
    print(f'state type: {type(x)}')
    print(x)

    raise ValueError('add dict!')


class PictureProcessor(nn.Module):
    def __init__(self, in_channels=3, device='cpu'):
        super(PictureProcessor, self).__init__()
        self._device = device

        self._conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(8, 8),
            stride=(4, 4),
        ).to(device)
        torch.nn.init.xavier_uniform_(self._conv1.weight)
        torch.nn.init.constant_(self._conv1.bias, 0)

        self._conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=(2, 2),
        ).to(device)
        torch.nn.init.xavier_uniform_(self._conv2.weight)
        torch.nn.init.constant_(self._conv2.bias, 0)

        self._conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
        ).to(device)
        torch.nn.init.xavier_uniform_(self._conv3.weight)
        torch.nn.init.constant_(self._conv3.bias, 0)

    def forward(self, state, return_stats: bool = False):
        if not return_stats:
            x = self._conv1(state)
            x = self._conv2(x)
            x = self._conv3(x)
            return x.view(x.size(0), -1)
        else:
            stats = {}
            x = self._conv1(state)
            stats['conv1'] = {
                'was activated': get_activated_ratio(x),
            }
            x = self._conv2(x)
            stats['conv2'] = {
                'was activated': get_activated_ratio(x),
            }
            x = self._conv3(x)
            stats['conv3'] = {
                'was activated': get_activated_ratio(x),
            }
            return x.view(x.size(0), -1), stats

    def get_out_shape_for_in(self, input_shape):
        return self.forward(torch.from_numpy(np.zeros(
            shape=(1, *tuple(input_shape)),
            dtype=np.float32)).to(self._device)
        ).shape[1]


class StateLayer(nn.Module):
    def __init__(
            self, state_description: Union[spaces.Dict, spaces.Box],
            hidden_size: int,
            device: str,
            activation_for_picture: Optional = None,
    ):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(StateLayer, self).__init__()
        self._device = device

        self.hidden_size = hidden_size
        self.activation_for_picture = activation_for_picture

        self._picture_layer = None
        self._picture_head = None
        self._vector_layer = None

        print(f'StateLayer -> state_description : {state_description}')

        if isinstance(state_description, (spaces.Dict, dict)):
            raise ValueError('State layer dont support Dict now :(')

        if isinstance(state_description, spaces.Box):
            if len(state_description.shape) == 3:
                self._init_picture_layers(state_description.shape)

            if len(state_description.shape) == 1:
                self._init_vector_layer(state_description.shape[0])

    def _init_picture_layers(self, input_shape):
        self._picture_layer: PictureProcessor = PictureProcessor(input_shape[0], device=self._device)
        self._picture_head = nn.Linear(
            in_features=self._picture_layer.get_out_shape_for_in(input_shape),
            out_features=self.hidden_size,
        )
        self._picture_head.to(self._device)
        torch.nn.init.xavier_uniform_(self._picture_head.weight)
        torch.nn.init.constant_(self._picture_head.bias, 0)

    def _init_vector_layer(self, input_size):
        self._vector_layer = nn.Linear(in_features=input_size, out_features=self.hidden_size).to(self._device)
        torch.nn.init.xavier_uniform_(self._vector_layer.weight)
        torch.nn.init.constant_(self._vector_layer.bias, 0)

    def get_out_shape_for_in(self):
        return self.hidden_size

    def forward_picture(self, state: torch.Tensor, return_stats: bool = False) -> TTOrTTStat:
        stat = {}
        if return_stats:
            x, stat = self._picture_layer(state, return_stats=True)
        else:
            x = self._picture_layer(state, return_stats=False)

        if self.activation_for_picture is not None:
            x = self.activation_for_picture(x)
        x = self._picture_head(x)

        if return_stats:
            return x, stat
        return x

    def forward_vector(self, state: torch.Tensor, return_stats: bool = False) -> TTOrTTStat:
        if return_stats:
            return self._vector_layer(state), {}
        else:
            return self._vector_layer(state)

    def forward(self, state: npTT, return_stats: bool = False) -> TTOrTTStat:
        state = make_it_batched_torch_tensor(state, self._device)

        if len(state.shape) == 4:
            return self.forward_picture(state, return_stats)
        if len(state.shape) == 2:
            return self.forward_vector(state, return_stats)

        print('state')
        print(f'state type : {type(state)}')
        print(f'state shape : {state.shape}')
        print(state)

        raise ValueError()


class ActionLayer(nn.Module):
    def __init__(self, action_size: int, hidden_size: int, device: str):
        super(ActionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.device = device

        self._dense = nn.Linear(action_size, hidden_size).to(device)
        torch.nn.init.xavier_uniform_(self._dense.weight)
        torch.nn.init.constant_(self._dense.bias, 0)

    def get_out_shape_for_in(self):
        return self.hidden_size

    def forward(self, action, return_stat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x = self._dense(make_it_batched_torch_tensor(action, self.device))
        if return_stat:
            return x, {}
        return x
