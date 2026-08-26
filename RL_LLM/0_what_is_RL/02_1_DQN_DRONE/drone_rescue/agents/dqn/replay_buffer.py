from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, int, int], device: str) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, observation, action, reward, next_observation, done) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.as_tensor(self.observations[indices], device=self.device),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(self.rewards[indices], device=self.device),
            torch.as_tensor(self.next_observations[indices], device=self.device),
            torch.as_tensor(self.dones[indices], device=self.device),
        )

