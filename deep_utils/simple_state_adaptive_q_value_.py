import torch
from torch import nn

from common_agents_utils.typingTypes import *
from common_agents_utils import make_it_batched_torch_tensor


class StateAdaptiveQValue(nn.Module):
    def __init__(self, state_shape, action_size, device, hidden_size=256):
        super(StateAdaptiveQValue, self).__init__()
        self.device = device
        assert isinstance(state_shape, tuple)
        assert len(state_shape) == 3 or len(state_shape) == 1

        self.is_state_picture = len(state_shape) == 3
        print(f"init StateAdaptiveQValue, state shape : {state_shape}", end='')
        if self.is_state_picture:
            print(f", channels number : {state_shape[0]}")
        else:
            print()

        if self.is_state_picture:
            self._state_layers = nn.Sequential(
                nn.Conv2d(state_shape[0], 32, (8, 8), (4, 4)),
                torch.nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), (2, 2)),
                torch.nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), (1, 1)),
                torch.nn.ReLU(),
            ).to(self.device)
        else:
            self._state_layers = nn.Sequential(
                nn.Linear(state_shape[0], hidden_size)
            ).to(self.device)
        state_layer_out_shape = self._state_layers(
            torch.zeros((1, *state_shape)).to(self.device)
        ).view(-1).shape[0]
        if self.is_state_picture:
            print(f"out of state layer : {state_layer_out_shape}")

        self._h1_act = nn.Linear(action_size, hidden_size).to(self.device)

        self._h2 = nn.Linear(state_layer_out_shape + hidden_size, hidden_size).to(self.device)
        self._head = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, state, action) -> TT:
        x = make_it_batched_torch_tensor(state, self.device)
        x = torch.relu(self._state_layers(x))
        x = x.flatten(start_dim=1)

        act = make_it_batched_torch_tensor(action, self.device)
        act = torch.relu(self._h1_act(act))

        x = torch.cat([x, act], dim=1)
        x = torch.relu(self._h2(x))
        x = self._head(x)

        return x


class DoubleStateAdaptiveCritic(nn.Module):

    def __init__(self, state_shape, action_size, device, hidden_size=256):
        super(DoubleStateAdaptiveCritic, self).__init__()
        self.device = device
        self._q1: StateAdaptiveQValue = StateAdaptiveQValue(state_shape, action_size, device, hidden_size)
        self._q2: StateAdaptiveQValue = StateAdaptiveQValue(state_shape, action_size, device, hidden_size)

    def forward(self, state, action) -> Tuple:
        return tuple([
            self._q1(state, action),
            self._q2(state, action),
        ])

    def Q1(self, state, action) -> TT:
        return self._q1(state, action)

    def Q2(self, state, action) -> TT:
        return self._q2(state, action)
