from typing import Callable, Union, List
import numpy as np
import torch

from stable_baseline_replay_buffer.replay_buffer import ReplayBuffer


class TorchReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            size: int,
            phi: Callable,
            device: str,
    ):
        super().__init__(size)
        self.phi = phi
        self.device = device

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(self.phi(obs_t), copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(self.phi(obs_tp1), copy=False))
            dones.append(done)
        return (
            torch.from_numpy(np.array(obses_t, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(obses_tp1, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(dones, dtype=np.float32)).to(self.device),
        )

    def get_all(self):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for data in self.storage:
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(self.phi(obs_t), copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(self.phi(obs_tp1), copy=False))
            dones.append(done)
        return (
            torch.from_numpy(np.array(obses_t, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(obses_tp1, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(dones, dtype=np.float32)).to(self.device),
        )
