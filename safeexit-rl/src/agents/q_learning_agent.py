"""
Tabular Q-learning agent over a discretized partial observation.

Observation discretization:
  - position: (row, col) — already discrete on the grid
  - avg_hazard_bin: average believed hazard in sensing radius,
    binned into {0: none, 1: low, 2: med, 3: high}
  - crowd_bin: local crowd count binned into {0, 1, 2, 3+}
  - exit_dir: exit_bearing quantized to 8 compass directions (+ center)
"""
from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from src.env import ACTIONS


def _bearing_to_dir(bearing: tuple) -> int:
    """Quantize a (dr, dc) bearing vector to one of 9 directions.
    0=center, 1-8 = N, NE, E, SE, S, SW, W, NW."""
    dr, dc = bearing
    if abs(dr) < 0.1 and abs(dc) < 0.1:
        return 0  # center / on exit
    angle = np.arctan2(dc, -dr)  # north = negative row
    # Map angle to 8 sectors
    sector = int(np.round(angle / (np.pi / 4))) % 8
    return sector + 1


def _hazard_bin(avg_hazard: float) -> int:
    """Bin average hazard into 4 levels."""
    if avg_hazard < 0.05:
        return 0  # none
    elif avg_hazard < 0.2:
        return 1  # low
    elif avg_hazard < 0.5:
        return 2  # med
    else:
        return 3  # high


def _crowd_bin(count: int) -> int:
    """Bin crowd count."""
    return min(count, 3)


def discretize_observation(obs) -> tuple:
    """Map an Observation object to a small discrete state key."""
    pos = obs.position

    # Average believed hazard from local_hazard_estimate
    hazard_vals = list(obs.local_hazard_estimate.values())
    avg_haz = np.mean(hazard_vals) if hazard_vals else 0.0
    haz_b = _hazard_bin(avg_haz)

    # Total crowd count in sensing range
    crowd_total = sum(obs.local_crowd_density.values()) if obs.local_crowd_density else 0
    crowd_b = _crowd_bin(crowd_total)

    # Exit direction
    exit_dir = _bearing_to_dir(obs.exit_bearing)

    return (pos[0], pos[1], haz_b, crowd_b, exit_dir)


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_episodes: int = 1500, seed: int = 0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def set_epsilon(self, episode: int):
        frac = min(1.0, episode / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs) -> str:
        state = discretize_observation(obs)
        if self.rng.random() < self.epsilon:
            return ACTIONS[self.rng.integers(len(ACTIONS))]
        q_values = self.q_table[state]
        return ACTIONS[int(np.argmax(q_values))]

    def update(self, obs, action: str, reward: float, next_obs, done: bool):
        state = discretize_observation(obs)
        next_state = discretize_observation(next_obs)
        a_idx = ACTIONS.index(action)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state][a_idx]
        self.q_table[state][a_idx] += self.alpha * td_error

    def save(self, path: str):
        """Serialize Q-table to file via pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert defaultdict to regular dict for safe serialization
        data = {k: v.tolist() for k, v in self.q_table.items()}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load Q-table from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
        for k, v in data.items():
            self.q_table[k] = np.array(v)
