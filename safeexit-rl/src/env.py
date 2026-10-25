"""
SafeExit-RL environment core.

Implements:
  - GridBuilding: loads a text layout, tracks walls/free cells/exits
  - HazardField: synthetic (non-CFD) fire hazard that expands over time
  - CrowdSim: scripted NPC evacuees contributing to local congestion
  - Observation construction: noisy, delayed, limited-radius sensing
    (the key point of novelty vs. a fully-observed MDP)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

ACTIONS = ["left", "right", "forward", "backward", "wait"]
ACTION_DELTAS = {
    "left": (0, -1),
    "right": (0, 1),
    "forward": (-1, 0),
    "backward": (1, 0),
    "wait": (0, 0),
}


@dataclass
class Observation:
    position: tuple
    local_hazard_estimate: dict  # {(r,c): noisy_value}
    local_crowd_density: dict    # {(r,c): count}
    staleness: dict              # {(r,c): steps_since_last_refresh}
    exit_bearing: tuple          # unit-ish direction vector toward believed-nearest exit
    local_walls: tuple = ()      # (can_fwd, can_bwd, can_left, can_right) boolean traversability


class HazardField:
    """Synthetic radial-expansion hazard model (proxy for FED).

    Hazard expands from a single origin cell using a sigmoid-in-radius growth:
        radius(t) = max_radius / (1 + exp(-growth_rate * (t - t_mid)))
        hazard(cell, t) = max(0, 1 - distance(cell, origin) / radius(t))

    This is O(1) per query — no PDE solver needed.
    """

    def __init__(self, origin: tuple, grid_shape: tuple, growth_rate: float = 0.05,
                 max_radius: float = 20.0):
        self.origin = origin
        self.grid_shape = grid_shape
        self.growth_rate = growth_rate
        self.max_radius = max_radius
        # t_mid controls when radius reaches half of max_radius
        self.t_mid = 60.0

    def _radius(self, t: int) -> float:
        """Effective hazard radius at time t using sigmoid growth."""
        return self.max_radius / (1.0 + np.exp(-self.growth_rate * (t - self.t_mid)))

    def true_value(self, cell: tuple, t: int) -> float:
        """Ground-truth hazard value at a cell at time t, in [0, 1]."""
        dist = np.sqrt((cell[0] - self.origin[0]) ** 2 +
                       (cell[1] - self.origin[1]) ** 2)
        r = self._radius(t)
        if r <= 0:
            return 0.0
        val = max(0.0, 1.0 - dist / r)
        return val

    def full_grid(self, t: int) -> np.ndarray:
        """Return the ground-truth hazard value for every cell at time t."""
        grid = np.zeros(self.grid_shape, dtype=float)
        for r in range(self.grid_shape[0]):
            for c in range(self.grid_shape[1]):
                grid[r, c] = self.true_value((r, c), t)
        return grid


class CrowdSim:
    """Scripted NPC evacuees doing biased random walk toward nearest exit.

    Each NPC moves one cell per tick, biased toward the nearest exit with
    some randomness to produce realistic congestion patterns.
    """

    def __init__(self, n_agents: int, building: "GridBuilding", seed: Optional[int] = None,
                 panic_prob: float = 0.0, herding_prob: float = 0.0):
        self.n_agents = n_agents
        self.building = building
        self.rng = np.random.default_rng(seed)
        self.panic_prob = panic_prob
        self.herding_prob = herding_prob
        self.positions: list[tuple] = []
        self.evacuated: list[bool] = []
        self._init_positions()

    def _init_positions(self):
        """Place NPCs on random free (non-exit, non-fire) cells."""
        free_cells = []
        for r in range(self.building.shape[0]):
            for c in range(self.building.shape[1]):
                cell = (r, c)
                if (self.building.is_free(cell) and
                        not self.building.is_exit(cell) and
                        cell != self.building.fire_origin):
                    free_cells.append(cell)
        if not free_cells:
            free_cells = [(1, 1)]  # fallback
        chosen = self.rng.choice(len(free_cells), size=self.n_agents, replace=True)
        self.positions = [free_cells[i] for i in chosen]
        self.evacuated = [False] * self.n_agents

    def _nearest_exit(self, cell: tuple) -> tuple:
        """Return the closest exit to the given cell (Euclidean)."""
        best = None
        best_dist = float('inf')
        for ex in self.building.exits:
            d = abs(cell[0] - ex[0]) + abs(cell[1] - ex[1])  # Manhattan
            if d < best_dist:
                best_dist = d
                best = ex
        return best

    def _nearby_crowd_target(self, idx: int, max_dist: int = 3) -> Optional[tuple]:
        """Find the nearest fellow evacuee to herd toward."""
        pos = self.positions[idx]
        best_target = None
        best_d = float('inf')
        for j, other_pos in enumerate(self.positions):
            if j == idx or self.evacuated[j]:
                continue
            d = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
            if 0 < d <= max_dist and d < best_d:
                best_d = d
                best_target = other_pos
        return best_target

    def step(self, hazard: Optional[HazardField] = None, t: int = 0):
        """Move each non-evacuated NPC one step toward nearest exit with
        some noise. Optionally includes herding (following nearby peers) and
        panic (increased randomness under high local hazard). NPCs that reach
        an exit are marked as evacuated."""
        for i in range(self.n_agents):
            if self.evacuated[i]:
                continue
            pos = self.positions[i]

            # 1. Target determination: check for herding tendency
            target = None
            if self.herding_prob > 0 and self.rng.random() < self.herding_prob:
                target = self._nearby_crowd_target(i)

            if target is None:
                target = self._nearest_exit(pos)

            if target is None:
                continue

            # 2. Check for hazard panic condition
            in_panic = False
            if self.panic_prob > 0 and hazard is not None:
                local_haz = hazard.true_value(pos, t)
                if local_haz > 0.2 and self.rng.random() < self.panic_prob:
                    in_panic = True

            # Compute biased direction toward target
            dr = np.sign(target[0] - pos[0])
            dc = np.sign(target[1] - pos[1])

            # Possible moves: biased direction + random cardinal directions
            candidates = []
            # Primary: move toward target
            if dr != 0:
                candidates.append((pos[0] + dr, pos[1]))
            if dc != 0:
                candidates.append((pos[0], pos[1] + dc))
            # Noise: add all cardinal directions with lower weight
            for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cand = (pos[0] + delta[0], pos[1] + delta[1])
                if cand not in candidates:
                    candidates.append(cand)

            # Filter to free cells
            valid = [c for c in candidates if self.building.is_free(c)]
            if not valid:
                continue  # stuck, wait

            if in_panic:
                # Under panic, directional bias collapses into uniform/erratic movement
                weights = np.ones(len(valid))
            else:
                # Weighted selection: first candidates (toward target) get higher weight
                n_biased = min(2, len(valid))
                weights = np.ones(len(valid))
                for j in range(n_biased):
                    if valid[j] in candidates[:2]:  # toward-target candidates
                        weights[j] = 4.0
            weights /= weights.sum()

            idx = self.rng.choice(len(valid), p=weights)
            self.positions[i] = valid[idx]

            # Check if NPC reached an exit
            if self.building.is_exit(self.positions[i]):
                self.evacuated[i] = True

    def density_grid(self) -> np.ndarray:
        """Return a grid with NPC counts per cell."""
        grid = np.zeros(self.building.shape, dtype=int)
        for i, (r, c) in enumerate(self.positions):
            if not self.evacuated[i]:
                grid[r, c] += 1
        return grid


class GridBuilding:
    """Loads a text layout and exposes walls/free cells/exits."""

    def __init__(self, layout_path: str):
        self.layout_path = layout_path
        self.grid_chars = self._load(layout_path)
        self.shape = (len(self.grid_chars), len(self.grid_chars[0]))
        self.exits = self._find('E')
        self.fire_origin = self._find('F')[0] if self._find('F') else None

    def _load(self, path):
        # Only strip comment lines (start with "# "), not wall rows made of '#'.
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if not l.startswith("# ")]
        return [list(l) for l in lines if l.strip() != ""]

    def _find(self, ch):
        coords = []
        for r, row in enumerate(self.grid_chars):
            for c, val in enumerate(row):
                if val == ch:
                    coords.append((r, c))
        return coords

    def is_free(self, cell: tuple) -> bool:
        r, c = cell
        if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
            return False
        return self.grid_chars[r][c] != '#'

    def is_exit(self, cell: tuple) -> bool:
        return cell in self.exits

    def free_cells(self) -> list:
        """Return list of all walkable cells (non-wall)."""
        cells = []
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                if self.is_free((r, c)):
                    cells.append((r, c))
        return cells


class SafeExitEnv:
    """Top-level environment tying together building + hazard + crowd +
    partial observability. Gym-like reset()/step() interface.
    """

    def __init__(self, layout_path: str, sensing_radius: int = 2,
                 obs_noise_std: float = 0.1, obs_delay: int = 2,
                 n_npcs: int = 10, max_steps: int = 200, seed: Optional[int] = None,
                 panic_prob: float = 0.0, herding_prob: float = 0.0):
        self.building = GridBuilding(layout_path)
        self.sensing_radius = sensing_radius
        self.obs_noise_std = obs_noise_std
        self.obs_delay = obs_delay
        self.n_npcs = n_npcs
        self.max_steps = max_steps
        self.panic_prob = panic_prob
        self.herding_prob = herding_prob
        self.rng = np.random.default_rng(seed)

        self.hazard: Optional[HazardField] = None
        self.crowd: Optional[CrowdSim] = None
        self.t = 0
        self.agent_pos = None
        self.visited = set()
        self.belief_state = {}       # agent's remembered hazard estimate per cell
        self.last_refresh = {}       # cell -> last timestep it was observed
        self.cumulative_true_exposure = 0.0
        self.cumulative_believed_exposure = 0.0

    def reset(self, start_pos: Optional[tuple] = None) -> Observation:
        """Place agent, init HazardField at building.fire_origin,
        init CrowdSim, reset counters, return initial Observation."""
        self.t = 0
        self.cumulative_true_exposure = 0.0
        self.cumulative_believed_exposure = 0.0
        self.belief_state = {}
        self.last_refresh = {}
        self.visited = set()

        # Place agent
        if start_pos is not None:
            self.agent_pos = start_pos
        else:
            # Random free cell that isn't fire or exit
            free = [c for c in self.building.free_cells()
                    if c != self.building.fire_origin and
                    not self.building.is_exit(c)]
            if free:
                idx = self.rng.integers(len(free))
                self.agent_pos = free[idx]
            else:
                self.agent_pos = (1, 1)

        self.visited.add(self.agent_pos)

        # Init hazard field
        origin = self.building.fire_origin if self.building.fire_origin else (0, 0)
        self.hazard = HazardField(
            origin=origin,
            grid_shape=self.building.shape,
        )

        # Init crowd simulation
        self.crowd = CrowdSim(
            n_agents=self.n_npcs,
            building=self.building,
            seed=int(self.rng.integers(2**31)),
            panic_prob=self.panic_prob,
            herding_prob=self.herding_prob,
        )

        return self._build_observation()

    def step(self, action: str):
        """Returns (observation, reward, done, info)."""
        from src.reward import compute_reward, RewardWeights

        self.t += 1
        weights = RewardWeights()

        # 1. Attempt to move agent
        old_pos = self.agent_pos
        dr, dc = ACTION_DELTAS[action]
        new_pos = (old_pos[0] + dr, old_pos[1] + dc)
        hit_wall = False
        moved = False

        if action == "wait":
            moved = False
        elif self.building.is_free(new_pos):
            self.agent_pos = new_pos
            moved = True
        else:
            hit_wall = True

        revisited = self.agent_pos in self.visited and moved
        self.visited.add(self.agent_pos)

        # 2. Advance hazard field (time advances automatically) and crowd sim
        self.crowd.step(hazard=self.hazard, t=self.t)

        # 3. Update cumulative true exposure using ground-truth hazard
        true_haz = self.hazard.true_value(self.agent_pos, self.t)
        self.cumulative_true_exposure += true_haz

        # 4. Build observation (noisy/delayed/radius-limited)
        obs = self._build_observation()

        # 5. Update cumulative believed exposure
        believed_haz = self.belief_state.get(self.agent_pos, 0.0)
        self.cumulative_believed_exposure += believed_haz

        # 6. Check done conditions
        reached_exit = self.building.is_exit(self.agent_pos)
        timed_out = self.t >= self.max_steps
        done = reached_exit or timed_out

        # 7. Compute crowd density at agent position
        density = self.crowd.density_grid()
        local_crowd = int(density[self.agent_pos[0], self.agent_pos[1]])

        # 8. Compute reward
        reward = compute_reward(
            moved=moved,
            revisited=revisited,
            hit_wall=hit_wall,
            believed_hazard_at_cell=believed_haz,
            local_crowd_count=local_crowd,
            reached_exit=reached_exit,
            cumulative_true_exposure=self.cumulative_true_exposure,
            weights=weights,
        )

        # Add a penalty for timing out without reaching exit
        if timed_out and not reached_exit:
            reward -= 10.0

        info = {
            "true_hazard": true_haz,
            "believed_hazard": believed_haz,
            "cumulative_true_exposure": self.cumulative_true_exposure,
            "cumulative_believed_exposure": self.cumulative_believed_exposure,
            "reached_exit": reached_exit,
            "timed_out": timed_out,
            "steps": self.t,
            "local_crowd": local_crowd,
        }

        return obs, reward, done, info

    def _build_observation(self) -> Observation:
        """Construct partial observation with noise, staleness, and radius limit.

        - Cells within sensing_radius: observe true hazard + Gaussian noise,
          but only refresh every obs_delay steps.
        - Cells outside sensing_radius: use last-known (stale) value, or a
          fixed "unknown" placeholder (0.0) if never observed.
        - Crowd density: count NPCs within sensing radius.
        - Exit bearing: direction to nearest known-safe exit based on belief.
        """
        pos = self.agent_pos
        local_hazard = {}
        local_crowd = {}
        staleness = {}
        density_grid = self.crowd.density_grid()

        for dr in range(-self.sensing_radius, self.sensing_radius + 1):
            for dc in range(-self.sensing_radius, self.sensing_radius + 1):
                cell = (pos[0] + dr, pos[1] + dc)
                r, c = cell
                if not (0 <= r < self.building.shape[0] and
                        0 <= c < self.building.shape[1]):
                    continue
                dist = abs(dr) + abs(dc)  # Manhattan distance for radius check
                if dist > self.sensing_radius:
                    continue

                # Crowd density: always fresh within sensing radius
                local_crowd[cell] = int(density_grid[r, c])

                # Hazard: only refresh if enough time passed since last observation
                last = self.last_refresh.get(cell, -self.obs_delay - 1)
                if self.t - last >= self.obs_delay:
                    # Refresh: observe true value + noise
                    true_val = self.hazard.true_value(cell, self.t)
                    noise = self.rng.normal(0, self.obs_noise_std)
                    observed = np.clip(true_val + noise, 0.0, 1.0)
                    self.belief_state[cell] = float(observed)
                    self.last_refresh[cell] = self.t

                staleness[cell] = self.t - self.last_refresh.get(cell, 0)

        # For cells in belief but not in sensing radius, keep stale values
        # (they stay in self.belief_state from previous observations)

        # Construct local_hazard_estimate from belief_state for cells within radius
        # plus any stale beliefs for cells outside radius
        all_hazard = dict(self.belief_state)  # full belief map

        # Exit bearing: direction to nearest exit that agent believes is safe
        exit_bearing = self._compute_exit_bearing()

        # Local wall / traversability for cardinal directions: forward, backward, left, right
        local_walls = (
            bool(self.building.is_free((pos[0] - 1, pos[1]))),
            bool(self.building.is_free((pos[0] + 1, pos[1]))),
            bool(self.building.is_free((pos[0], pos[1] - 1))),
            bool(self.building.is_free((pos[0], pos[1] + 1))),
        )

        return Observation(
            position=pos,
            local_hazard_estimate=all_hazard,
            local_crowd_density=local_crowd,
            staleness=staleness,
            exit_bearing=exit_bearing,
            local_walls=local_walls,
        )

    def _compute_exit_bearing(self) -> tuple:
        """Compute a direction vector toward the nearest believed-safe exit."""
        pos = self.agent_pos
        best_exit = None
        best_dist = float('inf')

        for ex in self.building.exits:
            # Check if exit is believed safe (low hazard or unknown)
            believed = self.belief_state.get(ex, 0.0)
            if believed < 0.5:  # exit is believed safe enough
                d = abs(pos[0] - ex[0]) + abs(pos[1] - ex[1])
                if d < best_dist:
                    best_dist = d
                    best_exit = ex

        if best_exit is None:
            # Fall back to any exit
            for ex in self.building.exits:
                d = abs(pos[0] - ex[0]) + abs(pos[1] - ex[1])
                if d < best_dist:
                    best_dist = d
                    best_exit = ex

        if best_exit is None or best_dist == 0:
            return (0, 0)

        # Normalize to a unit-ish direction
        dr = best_exit[0] - pos[0]
        dc = best_exit[1] - pos[1]
        mag = max(abs(dr), abs(dc), 1)
        return (dr / mag, dc / mag)
