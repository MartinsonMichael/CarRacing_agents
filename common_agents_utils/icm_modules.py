import itertools
from collections import defaultdict
from common_agents_utils.typingTypes import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from common_agents_utils.replay_buffer_2 import Torch_Arbitrary_Replay_Buffer
from common_agents_utils.torch_gym_modules import ActionLayer, StateLayer, make_it_batched_torch_tensor


class ICM:
    def __init__(
            self,
            state_description: Union[spaces.Box, spaces.Dict],
            encoded_state_size: int,
            action_size: int,
            device: str,
            buffer_size: int = 10**6,
            batch_size: int = 256,
            update_per_step: int = 50,
            hidden_size: int = 40,
            clipping_gradient_norm: float = 0.75,
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

            _, loss, cur_stat = self.get_intrinsic_reward_with_loss(
                state_batch,
                action_batch,
                next_state_batch,
                return_stats=True,
            )

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self._clipping_gradient_norm)
            self._optimizer.step()

            if return_stat:
                for key, value in cur_stat.items():
                    stat[key].append(value)

        if return_stat:
            if len(stat) == 0:
                return {}
            return {key: float(np.mean(value)) for key, value in stat.items()}

    def get_intrinsic_reward_with_loss(
            self, state: npTT, action: npTT, next_state: npTT, return_stats=False
    ) -> Union[Tuple[NpA, TT, StatType], Tuple[NpA, TT]]:
        _action = make_it_batched_torch_tensor(action, device=self.device)

        encoded_state = self._encoder(state)
        encoded_next_state = self._encoder(next_state)

        predicted_action = self._inverse(encoded_state, encoded_next_state)

        print('predicted_action')
        print(predicted_action[:10, :])

        print('real actions')
        print(_action[:10, :])

        inverse_loss = ((_action.detach() - predicted_action)**2).mean(dim=1)

        predicted_encoded_next_state = self._forward(encoded_state, _action)
        forward_loss = ((encoded_next_state - predicted_encoded_next_state)**2).mean(dim=1)
        loss = (forward_loss + inverse_loss).mean()

        if return_stats:
            stat = {
                'icm_inverse_loss': inverse_loss.detach().cpu().numpy().mean(),
                'icm_forward_loss': forward_loss.detach().cpu().numpy().mean(),
                'icm_full_loss': loss.detach().cpu().numpy().mean(),
            }
            return inverse_loss.view(-1, 1).detach().cpu().numpy(), loss, stat
        else:
            return inverse_loss.view(-1, 1).detach().cpu().numpy(), loss


class InverseDynamicModel(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int, device: str):
        super(InverseDynamicModel, self).__init__()

        self._state = nn.Linear(state_size, hidden_size).to(device)
        self._next_state = nn.Linear(state_size, hidden_size).to(device)
        self._dense_1 = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.head = nn.Linear(hidden_size, action_size).to(device)

    def forward(self, state: npTT, next_state: npTT, return_stats: bool = False) -> TTOrTTStat:
        assert state.shape == next_state.shape
        s = F.relu(self._state(state))
        ns = F.relu(self._next_state(next_state))
        x = F.relu(self._dense_1(torch.cat((s, ns), dim=1)))
        x = F.tanh(self.head(x))

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
        self._dense_2 = nn.Linear(
            in_features=int(hidden_max),
            out_features=int(hidden_max / 2),
        ).to(device)
        self.head = nn.Linear(int(hidden_max / 2), encoded_size).to(device)

    def forward(self, state: npTT, return_stats: bool = False) -> TTOrTTStat:
        x = make_it_batched_torch_tensor(state, device=self.device)
        x = F.relu(self._state(x))
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
