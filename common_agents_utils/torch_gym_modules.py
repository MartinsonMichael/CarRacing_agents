from typing import Dict, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces


def get_activated_ratio(x: Union[torch.Tensor, torch.FloatTensor]) -> float:
    if torch.prod(torch.tensor(x.size())).cpu().numpy() == 0:
        return 0.0
    return (x.detach() > 0).cpu().numpy().sum() / torch.prod(torch.tensor(x.size())).cpu().numpy()


def _make_it_batched_torch_tensor(x, device) -> Union[torch.FloatTensor, torch.Tensor]:
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
    def __init__(self, state_description: Union[spaces.Dict, spaces.Box], hidden_size, device):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(StateLayer, self).__init__()
        self._device = device

        self._state_layer_out_size = 0

        self._picture_layer = None
        self._vector_layer = None

        print(f'StateLayer -> state_description : {state_description}')

        if isinstance(state_description, (spaces.Dict, dict)):
            if 'picture' in state_description.keys() and state_description['picture'] is not None:
                self._picture_layer = PictureProcessor(device=self._device)
                self._state_layer_out_size += self._picture_layer.get_out_shape_for_in(
                    state_description['picture']
                )

            if 'vector' in state_description.keys() and state_description['vector'] is not None:
                self._vector_layer = nn.Linear(in_features=state_description['vector'], out_features=hidden_size)
                self._vector_layer.to(self._device)
                torch.nn.init.xavier_uniform_(self._vector_layer.weight)
                torch.nn.init.constant_(self._vector_layer.bias, 0)
                self._state_layer_out_size += hidden_size

        if isinstance(state_description, spaces.Box):
            if len(state_description.shape) == 3:
                self._picture_layer = PictureProcessor(state_description.shape[0], device=self._device)
                self._state_layer_out_size = self._picture_layer.get_out_shape_for_in(
                    state_description.shape
                )
            if len(state_description.shape) == 1:
                self._vector_layer = nn.Linear(in_features=state_description.shape[0], out_features=hidden_size)
                self._vector_layer.to(self._device)
                torch.nn.init.xavier_uniform_(self._vector_layer.weight)
                torch.nn.init.constant_(self._vector_layer.bias, 0)
                self._state_layer_out_size += hidden_size

    def get_out_shape_for_in(self):
        return self._state_layer_out_size

    def forward_picture(self, state: torch.Tensor, return_stats: bool = False):
        return self._picture_layer(state, return_stats)

    def forward_vector(self, state: torch.Tensor, return_stats: bool = False):
        if return_stats:
            return self._vector_layer(state), {}
        else:
            return self._vector_layer(state)

    def forward(
            self,
            state: Union[Dict[str, torch.FloatTensor], torch.FloatTensor],
            return_stats: bool = False,
    ):
        state = _make_it_batched_torch_tensor(state, self._device)

        if isinstance(state, dict):
            raise ValueError('add dict to StateLayer!')
            # return self.forward_dict(state)

        if isinstance(state, (torch.FloatTensor, torch.Tensor)):
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
        self._state_layer_out_size = hidden_size
        self.device = device
        self._dense = nn.Linear(action_size, hidden_size)

    def get_out_shape_for_in(self):
        return self._state_layer_out_size

    def forward(self, action, return_stat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x = self._dense(_make_it_batched_torch_tensor(action, self.devic))
        if return_stat:
            return x, {}
        return x