import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_csv(row: Dict, path: str | Path) -> None:
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.act[idx], device=self.device),
            torch.as_tensor(self.rew[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )


class OUNoise:
    def __init__(self, act_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2) -> None:
        self.act_dim = act_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.act_dim, dtype=np.float32) * self.mu

    def reset(self) -> None:
        self.state = np.ones(self.act_dim, dtype=np.float32) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.act_dim).astype(np.float32)
        self.state = self.state + dx
        return self.state


@torch.no_grad()
def evaluate_policy(env, actor: torch.nn.Module, method: str, device: torch.device, num_episodes: int = 5) -> float:
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if method == "DPG":
                action = actor(obs_t).cpu().numpy()[0]
            else:
                mean, _ = actor(obs_t)
                action = mean.cpu().numpy()[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
    return float(np.mean(returns))
