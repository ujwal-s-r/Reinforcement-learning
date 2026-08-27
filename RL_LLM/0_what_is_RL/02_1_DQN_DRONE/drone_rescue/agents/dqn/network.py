from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, channels: int = 3, actions: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_shared = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
        )
        # Dueling streams: State Value V(s) and Action Advantages A(s, a)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        feat = self.features(observation)
        h = self.fc_shared(feat)
        value = self.value_stream(h)
        advantage = self.advantage_stream(h)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

