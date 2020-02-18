from typing import Dict, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch.distributions import Normal

from common_agents_utils.torch_gym_modules import StateLayer, get_activated_ratio


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


