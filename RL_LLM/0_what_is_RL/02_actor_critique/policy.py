import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim=1, action_dim=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, state):
        logits = self.network(state)
        return logits

    def get_action(self, state):
        logits = self.forward(state)

        distribution = Categorical(logits=logits)

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim=1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)