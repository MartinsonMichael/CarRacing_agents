# import torch
# from torch import nn
#
# from common_agents_utils import make_it_batched_torch_tensor
#
#
# class StateAdaptiveActorCritic(nn.Module):
#     def __init__(self, state_shape, action_size, device, hidden_size=256):
#         super(StateAdaptiveActorCritic, self).__init__()
#         self.device = device
#         assert isinstance(state_shape, tuple)
#         assert len(state_shape) == 3 or len(state_shape) == 1
#
#         self.is_state_picture = len(state_shape) == 3
#         print(f"init StateAdaptiveActor, state shape : {state_shape}", end='')
#         if self.is_state_picture:
#             print(f", channels number : {state_shape[0]}")
#         else:
#             print()
#
#         if self.is_state_picture:
#             self._state_layers = nn.Sequential(
#                 nn.Conv2d(state_shape[0], 32, (8, 8), (4, 4)),
#                 nn.Conv2d(32, 64, (4, 4), (2, 2)),
#                 nn.Conv2d(64, 64, (3, 3), (1, 1)),
#             ).to(self.device)
#         else:
#             self._state_layers = nn.Sequential(
#                 nn.Linear(state_shape[0], hidden_size)
#             ).to(self.device)
#
#         state_layer_out_shape = self._state_layers(
#             torch.zeros((1, *state_shape)).to(self.device)
#         ).view(-1).shape[0]
#         if self.is_state_picture:
#             print(f"out of state layer : {state_layer_out_shape}")
#
#         self._h1 = nn.Linear(state_layer_out_shape, hidden_size).to(self.device)
#         self._head = nn.Linear(hidden_size, action_size).to(self.device)
#
#     def forward(self, state):
#         x = make_it_batched_torch_tensor(state, self.device)
#
#         x = torch.relu(self._state_layers(x))
#         x = x.flatten(start_dim=1)
#
#         x = torch.relu(self._h1(x))
#         x = torch.tanh(self._head(x))
#
#         return x
