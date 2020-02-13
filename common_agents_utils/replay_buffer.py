from collections import namedtuple, deque
import random
import torch
import numpy as np


class Torch_Separated_Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(
            self, buffer_size, batch_size, seed, device, state_extractor, state_producer,
            sample_order=['state', 'action', 'reward', 'next_state', 'done']
    ):
        # do not change order!
        names_for_stored = [
            "state",
            "action",
            "log_prob",
            "reward",
            "next_state",
            "done"
        ]
        assert len(set(sample_order) - set(names_for_stored)) == 0, \
            "You have mistake in sample_order\n"\
            f"must be some of {set(names_for_stored)}\n"\
            f"but you pass : {set(sample_order)}"
        self._sample_order = sample_order
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state_picture",
            "state_vector",
            "action",
            "log_prob",
            "reward",
            "next_state_picture",
            "next_state_vector",
            "done"
        ])
        self.seed = random.seed(seed)
        self.device = device
        self._state_extractor = state_extractor
        self._state_producer = state_producer

    def add_experience(self, states, actions, rewards, next_states, dones, log_probs=None):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            if log_probs is None:
                log_probs = [None for _ in range(dones)]
            for state, action, reward, next_state, done, log_prob \
                    in zip(states, actions, rewards, next_states, dones, log_probs):
                self._add_single_experience(state, action, reward, next_state, done, log_prob)
        else:
            self._add_single_experience(states, actions, rewards, next_states, dones, log_probs)

    def _add_single_experience(self, state, action, reward, next_state, done, log_prob):
        state_picture, state_vector = self._state_extractor(state)
        next_state_picture, next_state_vector = self._state_extractor(next_state)

        if state_picture is not None:
            assert isinstance(state_picture, np.ndarray), \
                f"state_picture must be None or np.ndarray, and it has {type(state_picture)}"
            assert state_picture.dtype == np.uint8, \
                f"state_picture must have type np.uint8, and it has {state_picture.dtype}"

            assert isinstance(next_state_picture, np.ndarray), \
                f"state_picture must be None or np.ndarray, and it has {type(next_state_picture)}"
            assert next_state_picture.dtype == np.uint8, \
                f"state_picture must have type np.uint8, and it has {next_state_picture.dtype}"

        if state_vector is not None:
            assert isinstance(state_vector, np.ndarray), \
                f"state_vector must me None or np.ndarray, and it has {type(state_vector)}"
            assert state_vector.dtype == np.float32, \
                f"state_vector must have type np.float32, and it has {state_vector.dtype}"

            assert isinstance(next_state_vector, np.ndarray), \
                f"state_vector must me None or np.ndarray, and it has {type(next_state_vector)}"
            assert next_state_vector.dtype == np.float32, \
                f"state_vector must have type np.float32, and it has {next_state_vector.dtype}"

        self.memory.append(self.experience(
            state_picture,
            state_vector,
            action,
            log_prob,
            reward,
            next_state_picture,
            next_state_vector,
            done
        ))

    def _prepare_row_of_samples(self, experiences, attribute_name) -> torch.Tensor:
        if attribute_name == 'state':
            return torch.from_numpy(np.array([
                self._state_producer(e.state_picture, e.state_vector)
                for e in experiences
            ], dtype=np.float32)).to(self.device)
        if attribute_name == 'next_state':
            return torch.from_numpy(np.array([
                self._state_producer(e.next_state_picture, e.next_state_vector)
                for e in experiences
            ], dtype=np.float32)).to(self.device)
        if attribute_name == 'action':
            return torch.from_numpy(np.array([e.action for e in experiences], dtype=np.float32)).to(self.device)
        if attribute_name == 'reward':
            return torch.from_numpy(np.array([[e.reward] for e in experiences], dtype=np.float32)).to(self.device)
        if attribute_name == 'done':
            return torch.from_numpy(np.array([[e.done] for e in experiences], dtype=np.float32)).to(self.device)
        if attribute_name == 'log_prob':
            return torch.from_numpy(np.array([e.log_prob for e in experiences], dtype=np.float32)).to(self.device)

    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        batch = (
            self._prepare_row_of_samples(experiences, name)
            for name in self._sample_order
        )

        # print(f"exp state_picture.shape: {experiences[0].state_picture.shape}")
        # print(f"exp state_vector.shape: {experiences[0].state_vector.shape}")
        # print('whole shape : ', batch[0][0].shape)

        return batch

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

    def clean_all_buffer(self):
        self.memory.clear()

    def get_all(self):
        batch = (
            self._prepare_row_of_samples(self.memory, name)
            for name in self._sample_order
        )
        return batch
