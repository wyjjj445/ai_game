from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    state_size: int = 11
    action_size: int = 3
    hidden_size: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_every: int = 20
    seed: Optional[int] = 42


class DQNAgent:
    def __init__(self, cfg: DQNConfig, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = QNetwork(cfg.state_size, cfg.action_size, cfg.hidden_size).to(self.device)
        self.target_net = QNetwork(cfg.state_size, cfg.action_size, cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(cfg.buffer_size, seed=cfg.seed)
        self.epsilon = cfg.epsilon_start

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.cfg.action_size))
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        q_pred = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            q_target = rewards_t + self.cfg.gamma * q_next * (1.0 - dones_t)

        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.q_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", self.cfg.epsilon_end))
