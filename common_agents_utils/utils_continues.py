from typing import Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch.distributions import Normal


def get_activated_ratio(x: Union[torch.Tensor, torch.FloatTensor]) -> float:
    if torch.prod(torch.tensor(x.size())).cpu().numpy() == 0:
        return 0.0
    return (x.detach() > 0).cpu().numpy().sum() / torch.prod(torch.tensor(x.size())).cpu().numpy()


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

    # def forward_dict(self, state: Dict[str, torch.FloatTensor]):
    #     state_pic = None
    #     if self._picture_layer is not None:
    #         state_pic = self._picture_layer(state['picture'])
    #
    #     state_vec = None
    #     if self._vector_layer is not None:
    #         state_vec = self._vector_layer(state['vector'])
    #
    #     if state_vec is not None and state_pic is not None:
    #         return torch.cat((state_pic, state_vec), dim=1)
    #
    #     if state_pic is not None:
    #         return state_pic
    #
    #     if state_vec is not None:
    #         return state_vec
    #
    #     raise ValueError("state should be Dict['picture' : tensor or None, 'vector' : tensor or None]")

    def forward_picture(self, state: torch.Tensor, return_stats: bool = False):
        return self._picture_layer(state, return_stats)

    def forward_vector(self, state: torch.Tensor, return_stats: bool = False):
        if return_stats:
            return self._vector_layer(state), {}
        else:
            return self._vector_layer(state)

    def _make_it_torch_tensor(self, x):
        if isinstance(x, (torch.FloatTensor, torch.Tensor, torch.cuda.FloatTensor)):
            if len(x.shape) == 2 or len(x.shape) == 4:
                return x.to(self._device)
            else:
                return x.unsqueeze_(0).to(self._device)
        if isinstance(x, np.ndarray):
            if len(x.shape) == 2 or len(x.shape) == 4:
                return torch.from_numpy(x.astype(np.float32)).to(self._device)
            else:
                return torch.from_numpy(np.array([x]).astype(np.float32)).to(self._device)

        print('state trouble')
        print(f'state type: {type(x)}')
        print(x)

        raise ValueError('add dict!')

    def forward(
            self,
            state: Union[Dict[str, torch.FloatTensor], torch.FloatTensor],
            return_stats: bool = False,
    ):
        state = self._make_it_torch_tensor(state)

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


class QNet(nn.Module):
    """
        Torch architecture, predict q-value from state and taken action.

        State[vector with shape of len 1 or 3] -> Q-Value[single number]
    """
    def __init__(self, state_description: Union[spaces.Dict, spaces.Box], action_size, hidden_size, device):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(QNet, self).__init__()
        self._device = device

        self._state_layer = StateLayer(state_description, hidden_size, device)

        self._dense_a = nn.Linear(in_features=action_size, out_features=hidden_size).to(self._device)
        torch.nn.init.xavier_uniform_(self._dense_a.weight)
        torch.nn.init.constant_(self._dense_a.bias, 0)

        self._dense2 = nn.Linear(
            in_features=hidden_size + self._state_layer.get_out_shape_for_in(),
            out_features=hidden_size,
        ).to(self._device)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1).to(self._device)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, state, action, return_stats: bool = False):
        if not return_stats:
            s = self._state_layer(state)
            a = F.leaky_relu(self._dense_a(action))
            x = torch.cat((s, a), 1)
            x = F.leaky_relu(self._dense2(x))
            x = self._head1(x)
            return x
        else:
            stats = {}
            s, state_stats = self._state_layer(state, True)
            stats['state_proc'] = state_stats
            a = F.leaky_relu(self._dense_a(action))
            stats['action_proc'] = {
                'was activated': get_activated_ratio(a),
            }
            x = torch.cat((s, a), 1)
            x = F.leaky_relu(self._dense2(x))
            stats['dense2'] = {
                'was activated': get_activated_ratio(x),
            }
            x = self._head1(x)
            return x, stats


class ValueNet(nn.Module):
    """
        Torch architecture, predict advantage from state.

        State[vector with shape of len 1 or 3] -> Advantage[single number]
    """
    def __init__(self, state_description: Union[spaces.Dict, spaces.Box], action_size, hidden_size, device):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(ValueNet, self).__init__()
        self._device = device

        self._state_layer = StateLayer(state_description, hidden_size, device)

        self._dense2 = nn.Linear(in_features=hidden_size,out_features=hidden_size).to(self._device)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1).to(self._device)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, state, return_stats: bool = False):
        if not return_stats:
            x = self._state_layer(state)
            x = F.leaky_relu(self._dense2(x))
            x = self._head1(x)
            return x
        else:
            stats = {}
            x, state_stats = self._state_layer(state, True)
            stats['state_proc'] = state_stats
            x = F.leaky_relu(self._dense2(x))
            stats['dense2'] = {
                'was activated': get_activated_ratio(x),
            }
            x = self._head1(x)
            return x, stats


class Policy(nn.Module):
    """
        Torch architecture, predict action distribution from state.
        Can be used with flag 'double_action_size_on_output=True' (default) to predict 2*actoin_size numbers or
        with 'double_action_size_on_output=False' to predict actoin_size numbers.

        State[vector with shape of len 1 or 3] -> Policy[2 * actoin_size numbers (or just action_size numbers)]
    """
    def __init__(
            self, state_description: Union[spaces.Dict, spaces.Box], action_size, hidden_size, device,
            double_action_size_on_output=True
    ):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(Policy, self).__init__()
        self._device = device
        self._action_size = action_size

        self._state_layer = StateLayer(state_description, hidden_size, device)

        self._dense2 = nn.Linear(in_features=self._state_layer.get_out_shape_for_in(), out_features=hidden_size).to(self._device)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head = nn.Linear(
            in_features=hidden_size,
            out_features=2 * action_size if double_action_size_on_output else action_size
        ).to(self._device)
        torch.nn.init.xavier_uniform_(self._head.weight)
        torch.nn.init.constant_(self._head.bias, 0)

    def forward(self, state, return_stats: bool = False):
        if not return_stats:
            x = self._state_layer(state)
            x = F.leaky_relu(self._dense2(x))
            x = self._head(x)
            return x
        else:
            stats = {}
            x, state_stats = self._state_layer(state, True)
            stats['state_proc'] = state_stats
            x = F.leaky_relu(self._dense2(x))
            stats['dense2'] = {
                'was activated': get_activated_ratio(x)
            }
            x = self._head(x)
            return x, stats
