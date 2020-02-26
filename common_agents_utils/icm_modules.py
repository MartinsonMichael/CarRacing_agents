import itertools
from collections import defaultdict
from typing import Dict, Any, Union, Tuple, Iterable, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from common_agents_utils.replay_buffer_2 import Torch_Arbitrary_Replay_Buffer
from common_agents_utils.torch_gym_modules import ActionLayer, StateLayer, make_it_batched_torch_tensor

TT = torch.Tensor
NpA = np.ndarray
npTT = Union[TT, NpA]
StatType = Dict[str, Any]
TTStat = Tuple[TT, StatType]
NpAStat = Tuple[NpA, StatType]
NpAOrNpAStat = Union[NpA, NpAStat]
TTOrTTStat = Union[TT, TTStat]


class ICM:
    def __init__(
            self,
            state_description: Union[spaces.Box, spaces.Dict],
            encoded_state_size: int,
            action_size: int,
            device: str,
            buffer_size: int = 10**6,
            batch_size: int = 256,
            update_per_step: int = 150,
            hidden_size: int = 40,
            clipping_gradient_norm: float = 1.0,
    ):
        self.device: str = device
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.update_per_step: int = update_per_step
        self._encoded_state_size: int = encoded_state_size
        self._clipping_gradient_norm: float = clipping_gradient_norm

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
        self._forward: ForwardDynamicModel = ForwardDynamicModel(
            state_size=self._encoded_state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            device=self.device,
        )
        self._optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=3e-4,
        )
        self.replay_buffer: Torch_Arbitrary_Replay_Buffer = Torch_Arbitrary_Replay_Buffer(
            buffer_size=buffer_size,
            device=self.device,
            batch_size=batch_size,
            sample_order=['state', 'action', 'next_state'],
        )
        self.mse = nn.MSELoss()

    def parameters(self) -> Iterable[TT]:
        return itertools.chain(self._encoder.parameters(), self._forward.parameters(), self._inverse.parameters())

    def add_experience(self, is_single=True, **kwargs) -> None:
        if set(kwargs.keys()) != {'state', 'action', 'next_state'}:
            raise ValueError("you can save to ICM replay buffer only {'state', 'action', 'next_state'}")
        self.replay_buffer.add_experience(is_single=is_single, **kwargs)

    def update(self, return_stat=False) -> Optional[StatType]:
        stat = defaultdict(list)
        stat['buffer_len'].append(self.replay_buffer.len())

        for _ in range(self.update_per_step):
            state_batch, action_batch, next_state_batch = self.replay_buffer.sample()
            cur_stat = self._update_on_batch(state_batch, action_batch, next_state_batch, return_stats=return_stat)
            if return_stat:
                for key, value in cur_stat.items():
                    stat[key].append(value)

        if return_stat:
            if len(stat) == 0:
                return {}
            return {key: float(np.mean(value)) for key, value in stat.items()}

    def _update_on_batch(
            self, state_batch, _action_batch, next_state_batch,
            return_reward=False, return_stats=False,
    ) -> Optional[Union[StatType, NpAOrNpAStat]]:
        stat = {}
        action_batch = make_it_batched_torch_tensor(_action_batch, self.device).detach()
        encoded_state = self._encoder(state_batch)
        encoded_next_state = self._encoder(next_state_batch)
        predicted_encoded_next_state = self._forward(encoded_state, action_batch)

        forward_loss = ((predicted_encoded_next_state - encoded_next_state)**2).mean(dim=1)

        predicted_action = self._inverse(encoded_state, encoded_next_state)
        inverse_loss = ((predicted_action - action_batch)**2).mean(dim=1)

        loss = (forward_loss + inverse_loss).mean()
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self._clipping_gradient_norm)
        self._optimizer.step()

        if return_stats:
            stat['icm_forward_loss'] = float(forward_loss.detach().cpu().numpy().mean())
            stat['icm_inverse_loss'] = float(inverse_loss.detach().cpu().numpy().mean())
            stat['icm_loss'] = float(loss.detach().cpu().numpy().mean())

        if return_reward:
            if return_stats:
                return inverse_loss.view(-1, 1).detach().cpu().numpy(), stat
            return inverse_loss.view(-1, 1).detach().cpu().numpy()
        if return_stats:
            return stat

    def get_intrinsic_reward(
            self, state: npTT, action: npTT, next_state: npTT, return_stats=False
    ) -> Union[Tuple[NpA, StatType], NpA]:

        encoded_state = self._encoder(state)
        encoded_next_state = self._encoder(next_state)

        predicted_action = self._inverse(encoded_state, encoded_next_state)
        inverse_loss = ((action.detach() - predicted_action)**2).mean(dim=1)

        if return_stats:
            return inverse_loss.view(-1, 1).detach().cpu().numpy(), {}
        else:
            return inverse_loss.view(-1, 1).detach().cpu().numpy()

    def get_intrinsic_reward_with_loss(
            self, state: npTT, action: npTT, next_state: npTT, return_stats=False
    ) -> Union[Tuple[NpA, TT, StatType], Tuple[NpA, TT]]:

        encoded_state = self._encoder(state)
        encoded_next_state = self._encoder(next_state)

        predicted_action = self._inverse(encoded_state, encoded_next_state)
        inverse_loss = ((action.detach() - predicted_action)**2).mean(dim=1)

        predicted_encoded_next_state = self._forward(encoded_state, action)
        forward_loss = ((encoded_next_state - predicted_encoded_next_state)**2).mean(dim=1)
        loss = (forward_loss + inverse_loss).mean()

        if return_stats:
            stat = {
                'icm_inverse_loss': inverse_loss.detach().cpu().numpy().mean(),
                'icm_forward_loss': forward_loss.detach().cpu().numpy().mean(),
                'icm_full_loss': loss.detach().cpu().numpy().mean(),
            }
            return inverse_loss.view(-1, 1).detach().cpu().numpy(), loss, {}
        else:
            return inverse_loss.view(-1, 1).detach().cpu().numpy(), loss


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

    def forward(self, state: npTT, next_state: npTT, return_stats: bool = False) -> TTOrTTStat:
        assert state.shape == next_state.shape
        s = F.relu(self._state(state))
        ns = F.relu(self._next_state(next_state))
        x = F.relu(self._dense_1(torch.cat((s, ns), dim=1)))
        x = self.head(x)

        if return_stats:
            return x, {}
        return x


class StateEncoder(nn.Module):
    def __init__(
        self,
        state_description: Union[spaces.Dict, spaces.Box],
        encoded_size: int,
        hidden_size: int,
        device: str,
        use_batch_normalize: bool = True,
    ):
        super(StateEncoder, self).__init__()
        self._use_batch_normalize = use_batch_normalize
        self.device = device

        self._state: StateLayer = StateLayer(state_description, hidden_size, device)
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

    def forward(self, state: npTT, return_stats: bool = False) -> TTOrTTStat:
        x = make_it_batched_torch_tensor(state, device=self.device)
        # if self._use_batch_normalize:
        #     x = nn.BatchNorm2d()(x)
        x = F.relu(self._state(x))
        x = F.relu(self._dense_1(x))
        x = F.relu(self._dense_2(x))
        x = self.head(x)

        if return_stats:
            return x, {}
        return x


class ForwardDynamicModel(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        device: str,
    ):
        super(ForwardDynamicModel, self).__init__()

        self._state = nn.Linear(state_size, hidden_size).to(device)
        self._action: ActionLayer = ActionLayer(action_size, hidden_size, device)

        self._dense_1 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
        ).to(device)
        self.head = nn.Linear(hidden_size, state_size).to(device)

    def forward(self, state: npTT, action: npTT, return_stats: bool = False) -> TTOrTTStat:
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
