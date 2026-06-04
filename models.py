from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(input_dim: int, hidden_units: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, output_dim),
    )


class GaussianActor(nn.Module):
    """
    State-dependent diagonal Gaussian policy.

    The mean is bounded to the action range by tanh. The standard deviation is
    positive through softplus. Sampled actions are clamped before being passed to
    the environment/critic.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_units: int, action_low, action_high,
                 min_std: float = 1e-3, max_std: float = 2.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_units), nn.ReLU())
        self.mean_head = nn.Linear(hidden_units, act_dim)
        self.std_head = nn.Linear(hidden_units, act_dim)

        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_mid", (action_high_t + action_low_t) / 2.0)
        self.min_std = float(min_std)
        self.max_std = float(max_std)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = torch.tanh(self.mean_head(h)) * self.action_scale + self.action_mid
        std = F.softplus(self.std_head(h)) + self.min_std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        return mean, std

    def sample(self, obs: torch.Tensor, sample_size: int = 1, detach_action: bool = True):
        mean, std = self(obs)
        dist = Normal(mean, std)
        if sample_size == 1:
            action = dist.sample()
            if detach_action:
                action = action.detach()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = torch.max(torch.min(action, self.action_high), self.action_low)
            return action, log_prob, mean, std

        action = dist.sample((sample_size,))
        if detach_action:
            action = action.detach()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action, log_prob, mean, std


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_units: int, action_low, action_high) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_units, act_dim)
        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_mid", (action_high_t + action_low_t) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale + self.action_mid


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_units: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))
