from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .network import QNetwork
from .replay_buffer import ReplayBuffer
from .schedule import LinearSchedule


class DQNAgent:
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        actions: int,
        device: str = "cpu",
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        target_update_frequency: int = 500,
) -> None:
        self.device = torch.device(device)
        self.actions = actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.online_network = QNetwork(observation_shape[0], actions).to(self.device)
        self.target_network = QNetwork(observation_shape[0], actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = torch.optim.AdamW(self.online_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size, observation_shape, device)
        self.exploration = LinearSchedule(1.0, 0.05, 20_000)
        self.update_count = 0

    def select_action(self, observation: np.ndarray, greedy: bool = False) -> int:
        epsilon = 0.0 if greedy else self.exploration.value
        if np.random.random() < epsilon:
            return int(np.random.randint(self.actions))
        with torch.no_grad():
            tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.online_network(tensor)).item())

    def remember(self, observation, action, reward, next_observation, done) -> None:
        self.replay_buffer.add(observation, action, reward, next_observation, done)

    def train_step(self) -> float:
        if self.replay_buffer.size < self.batch_size:
            return 0.0
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(
            self.batch_size
        )
        current_q_values = self.online_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(dim=1).values
            target_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values
        loss = nn.functional.huber_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())
