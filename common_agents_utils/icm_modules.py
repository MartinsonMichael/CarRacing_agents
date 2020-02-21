from typing import Dict, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from common_agents_utils.torch_gym_modules import ActionLayer, StateLayer


class ICM:
    def __init__(
            self,
            state_description: Union[spaces.Box, spaces.Dict],
            encoded_state_size: int,
            action_size: int,
            device: str,
            buffer_size: int = 10**6,
            batch_size: int = 256,
            update_per_step: int = 10,
            hidden_size: int = 40,
    ):
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_per_step = update_per_step
        self._encoded_state_size = encoded_state_size

        self._encoder: StateEncoder = StateEncoder(
            state_description=state_description,
            encoded_size=self._encoded_state_size,
            hidden_size=hidden_size,
            device=self.device,
        )
        self._inverse: InverseDynamicModel = InverseDynamicModel(
            state_size=self._encoded_state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            device=self.device,
        )
        self._forward: EncodedForwardDynamicModel = EncodedForwardDynamicModel(

        )




class InverseDynamicModel(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int, device: str):
        super(InverseDynamicModel, self).__init__()

        self._state = nn.Linear(state_size, hidden_size).to(device)
        self._next_state = nn.Linear(state_size, hidden_size).to(device)
        self._dense_1 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
        ).to(device)
        self.head = nn.Linear(hidden_size, action_size).to(device)

    def forward(self, state, next_state, return_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        s = F.relu(self._state(state))
        ns = F.relu(self._next_state(next_state))
        x = F.relu(self._dense_1(torch.cat((s, ns), dim=1)))
        x = self.head(x)

        if return_stats:
            return x, {}
        return x


class StateEncoder(nn.Module):
    def __init__(self,
                 state_description: Union[spaces.Dict, spaces.Box],
                 encoded_size: int,
                 hidden_size: int,
                 device: str
                 ):
        super(StateEncoder, self).__init__()

        self._state = StateLayer(state_description, hidden_size, device).to(device)
        hidden_max = self._state.get_out_shape_for_in()
        self._dense_1 = nn.Linear(
            in_features=hidden_max,
            out_features=int(hidden_max / 2),
        ).to(device)
        self._dense_2 = nn.Linear(
            in_features=int(hidden_max / 2),
            out_features=int(hidden_max / 4),
        ).to(device)
        self.head = nn.Linear(int(hidden_max / 4), encoded_size).to(device)

    def forward(self, state, return_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x = F.relu(self._state(state))
        x = F.relu(self._dense_1(x))
        x = F.relu(self._dense_2(x))
        x = self.head(x)

        if return_stats:
            return x, {}
        return x


class EncodedForwardDynamicModel(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_size: int,
                 device: str
                 ):
        super(EncodedForwardDynamicModel, self).__init__()

        self._state = nn.Linear(state_size, hidden_size).to(device)
        self._action = ActionLayer(action_size, hidden_size, device).to(device)

        self._dense_1 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
        ).to(device)
        self.head = nn.Linear(hidden_size, state_size).to(device)

    def forward(self, state, action, return_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        s = F.relu(self._state(state))
        a = F.relu(self._action(action))
        x = F.relu(self._dense_1(torch.cat((s, a), dim=1)))
        x = self.head(x)

        if return_stats:
            return x, {}
        return x


# class EncodedValueNet(nn.Module):
#     def __init__(self, state_size: int, device: str):
#         super(EncodedValueNet, self).__init__()
#         self._head = nn.Linear(state_size, 1).to(device)
#
#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         return self._head(state)
#
#
# class EncodedPolicyNet(nn.Module):
#     def __init__(self, state_size: int, action_size: int, device: str, double_action_size_on_output=False):
#         super(EncodedPolicyNet, self).__init__()
#         self._head = nn.Linear(
#             in_features=state_size,
#             out_features=action_size*2 if double_action_size_on_output else action_size
#         ).to(device)
#
#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         return self._head(state)
