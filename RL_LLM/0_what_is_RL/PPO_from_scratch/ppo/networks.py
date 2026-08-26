import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def get_distribution(self, state):
        logits = self.network(state)
        return Categorical(logits=logits)

    def get_action(self, state):
        distribution = self.get_distribution(state)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action, log_prob

    def get_log_prob(self, state, action):
        distribution = self.get_distribution(state)

        return distribution.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)