"""
Minimal Deep Q-Network (DQN) agent for SafeExit-RL.

Implements a lightweight 2-layer MLP Q-network, target network,
experience replay buffer, epsilon-greedy action selection, and
stable TD-error optimization via PyTorch.
"""
from __future__ import annotations
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env import ACTIONS, ACTION_DELTAS


def encode_observation(obs) -> np.ndarray:
    """Convert an Observation into a fixed-size 14-dimensional float vector.

    Features (all relative to agent position):
      - 0..3:  Local wall traversability (forward, backward, left, right: 1.0 if free, 0.0 if wall)
      - 4..5:  Exit bearing unit vector (dr, dc)
      - 6:     Average believed hazard in sensing range
      - 7:     Believed hazard at current cell
      - 8..11: Believed hazard at adjacent cells (forward, backward, left, right)
      - 12:    Normalized total crowd count in sensing range
      - 13:    Normalized crowd count at current cell
    """
    pos = obs.position

    # 1. Traversability (4 features) — always provided by SafeExitEnv._build_observation
    #    local_walls = (can_fwd, can_bwd, can_left, can_right) as bool
    fwd_free, bwd_free, left_free, right_free = [float(w) for w in obs.local_walls]

    # 2. Exit bearing (2 features)
    b_dr, b_dc = obs.exit_bearing if obs.exit_bearing is not None else (0.0, 0.0)

    # 3. Hazard features (6 features)
    haz_vals = list(obs.local_hazard_estimate.values())
    avg_haz = float(np.mean(haz_vals)) if haz_vals else 0.0
    cur_haz = float(obs.local_hazard_estimate.get(pos, 0.0))

    adj_haz = [
        float(obs.local_hazard_estimate.get((pos[0] - 1, pos[1]), 0.0)),  # forward
        float(obs.local_hazard_estimate.get((pos[0] + 1, pos[1]), 0.0)),  # backward
        float(obs.local_hazard_estimate.get((pos[0], pos[1] - 1), 0.0)),  # left
        float(obs.local_hazard_estimate.get((pos[0], pos[1] + 1), 0.0)),  # right
    ]

    # 4. Crowd features (2 features)
    total_crowd = float(sum(obs.local_crowd_density.values())) / 10.0 if obs.local_crowd_density else 0.0
    cur_crowd = float(obs.local_crowd_density.get(pos, 0.0)) / 5.0

    vec = np.array([
        fwd_free, bwd_free, left_free, right_free,
        float(b_dr), float(b_dc),
        avg_haz, cur_haz,
        adj_haz[0], adj_haz[1], adj_haz[2], adj_haz[3],
        total_crowd, cur_crowd,
    ], dtype=np.float32)

    return vec


class QNetwork(nn.Module):
    """Small 2-layer MLP for action-value approximation."""

    def __init__(self, in_features: int = 14, num_actions: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer backed by a ring deque."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action_idx: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Minimal Deep Q-Network agent adhering to the SafeExit-RL agent contract."""

    def __init__(
        self,
        in_features: int = 14,
        num_actions: int = len(ACTIONS),
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 1500,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_interval: int = 100,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)
        self.train_steps = 0

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Networks
        self.q_net = QNetwork(in_features, num_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(in_features, num_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss: stable for large Q-value outliers
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def set_epsilon(self, episode: int):
        frac = min(1.0, episode / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs, eval_mode: bool = False) -> str:
        eps = 0.0 if eval_mode else self.epsilon
        if self.rng.random() < eps:
            return ACTIONS[self.rng.integers(len(ACTIONS))]

        state = encode_observation(obs)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state_t).squeeze(0).cpu().numpy()
        return ACTIONS[int(np.argmax(q_vals))]

    def update(self, obs, action: str, reward: float, next_obs, done: bool):
        state = encode_observation(obs)
        next_state = encode_observation(next_obs)
        action_idx = ACTIONS.index(action)

        self.replay_buffer.push(state, action_idx, reward, next_state, done)
        self.train_steps += 1

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q estimates
        current_q = self.q_net(states).gather(1, actions)

        # Bellman target using target network; terminal states zero out future value
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        """Save model weights and training metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.q_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
            self.train_steps = checkpoint.get("train_steps", 0)
            self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        else:
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
        self.q_net.eval()
        self.target_net.eval()
