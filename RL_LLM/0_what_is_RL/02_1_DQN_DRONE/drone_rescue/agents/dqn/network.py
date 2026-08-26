from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, channels: int = 3, actions: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, actions),
        )

    def forward(self, observation):
        return self.network(observation)

