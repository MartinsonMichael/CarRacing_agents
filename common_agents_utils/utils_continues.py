from common_agents_utils.typingTypes import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch.distributions import Normal

from common_agents_utils.torch_gym_modules import \
    StateLayer, \
    get_activated_ratio, make_it_batched_torch_tensor, process_kwargs


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

        self._dense2 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size).to(self._device)
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
    def __init__(
            self,
            state_description: Union[spaces.Dict, spaces.Box],
            hidden_size: int,
            device: str,
    ):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(ValueNet, self).__init__()
        self._device = device

        self._state_layer = StateLayer(
            state_description=state_description,
            hidden_size=hidden_size,
            device=device,
            activation_for_picture=F.leaky_relu
        )

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1).to(self._device)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, state, return_stats: bool = False) -> TTOrTTStat:
        if not return_stats:
            x = self._state_layer(state)
            x = self._head1(x)
            return x
        else:
            stats = {}
            x, state_stats = self._state_layer(state, True)
            stats['state_proc'] = state_stats
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
            self,
            state_description: Union[spaces.Dict, spaces.Box],
            action_size: int,
            hidden_size: int,
            device: str,
            double_action_size_on_output: bool = True,
    ):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(Policy, self).__init__()
        self._device = device
        self._action_size = action_size

        self._state_layer: StateLayer = StateLayer(
            state_description,
            hidden_size,
            device,
            activation_for_picture=F.leaky_relu
        )

        self._head = nn.Linear(
            in_features=hidden_size,
            out_features=2 * action_size if double_action_size_on_output else action_size
        ).to(self._device)
        torch.nn.init.xavier_uniform_(self._head.weight)
        torch.nn.init.constant_(self._head.bias, 0)

    def forward(self, state, return_stats: bool = False):
        if not return_stats:
            x = self._state_layer(state)
            x = self._head(x)
            return x
        else:
            stats = {}
            x, state_stats = self._state_layer(state, True)
            stats['state_proc'] = state_stats
            x = self._head(x)
            return x, stats


class ActorCritic(nn.Module):
    def __init__(
            self,
            state_description: Union[spaces.Dict, spaces.Box],
            action_size: int,
            hidden_size: int,
            device: str,
            double_action_size_on_output: bool = True,
            action_std=Optional[float],
    ):
        assert isinstance(state_description, (spaces.Dict, spaces.Box)), \
            "state_description must be spaces.Dict or spaces.Box"
        super(ActorCritic, self).__init__()
        self.double_action_size_on_output = double_action_size_on_output
        self.action_size = action_size
        self.action_std = action_std
        self.device = device

        if self.action_std is None and not self.double_action_size_on_output:
            raise ValueError("provide one of 'action_std', 'double_action_size_on_output'")

        self.actor: Policy = Policy(
            state_description=state_description,
            action_size=self.action_size,
            device=self.device,
            hidden_size=hidden_size,
            double_action_size_on_output=self.double_action_size_on_output,
        )
        self.critic: ValueNet = ValueNet(
            state_description=state_description,
            device=self.device,
            hidden_size=hidden_size,
        )

    def value(self, state) -> TT:
        return self.critic(state)

    def _get_mean_std(self, state) -> Tuple[TT, TT]:
        action_out = self.actor(state)
        if self.double_action_size_on_output:
            return action_out[:, self.action_size:], action_out[:, :self.action_size]
        else:
            return action_out, self.action_std

    def sample_action(self, state, **kwargs) -> Tuple[npTT, npTT, npTT]:
        """
        Sample action from gauss distribution
        Return: action, log_prob, entropy
        """
        distribution = Normal(*self._get_mean_std(state))
        action = distribution.sample()
        return \
            process_kwargs(action.detach(), **kwargs), \
            process_kwargs(distribution.log_prob(action), **kwargs), \
            process_kwargs(distribution.entropy(), **kwargs),

    def estimate_action(self, state, action) -> Tuple[TT, TT]:
        """
        Create distribution via state, and compute given action log_prob.
        Return: action, log_prob, entropy
        """
        _action = make_it_batched_torch_tensor(action, self.device)
        distribution = Normal(*self._get_mean_std(state))
        return distribution.log_prob(_action), distribution.entropy()

    def forward(self, **kwargs: Any):
        raise NotImplemented
