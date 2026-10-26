"""
Static shortest-path baseline.

Computes a path ONCE at episode start using only the agent's initial,
limited/noisy observation -- never replans. This is the weakest baseline
and should perform worst under high noise/delay.
"""
from collections import deque
from src.env import ACTION_DELTAS


def plan_path(start: tuple, believed_hazard: dict, is_free_fn, exits: list,
              hazard_block_threshold: float = 0.3) -> list:
    """Return a list of cells from start to the nearest reachable exit,
    treating cells with believed_hazard >= hazard_block_threshold as blocked.
    Unknown cells (not in believed_hazard) are assumed free/safe.
    Uses BFS (unweighted grid, so BFS == Dijkstra here).
    """
    queue = deque()
    queue.append(start)
    parent = {start: None}

    while queue:
        cell = queue.popleft()

        # Check if we reached an exit
        if cell in exits:
            # Reconstruct path
            path = []
            cur = cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        # Explore neighbors (4-directional)
        for delta_name, (dr, dc) in ACTION_DELTAS.items():
            if delta_name == "wait":
                continue
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor in parent:
                continue
            if not is_free_fn(neighbor):
                continue
            # Block cells with high believed hazard
            haz = believed_hazard.get(neighbor, 0.0)  # unknown = safe
            if haz >= hazard_block_threshold:
                continue
            parent[neighbor] = cell
            queue.append(neighbor)

    # No path found — return empty (agent will wait)
    return []


def _path_to_action(current: tuple, target: tuple) -> str:
    """Convert a (current -> target) cell transition to an action string."""
    dr = target[0] - current[0]
    dc = target[1] - current[1]
    for action_name, (adr, adc) in ACTION_DELTAS.items():
        if adr == dr and adc == dc:
            return action_name
    return "wait"


class StaticShortestPathAgent:
    def __init__(self):
        self.path = None
        self.path_idx = 0

    def reset(self, start: tuple, initial_obs, is_free_fn, exits: list):
        believed = initial_obs.local_hazard_estimate
        self.path = plan_path(start, believed, is_free_fn, exits)
        self.path_idx = 0

    def select_action(self, obs) -> str:
        """Follow precomputed path. Fall back to 'wait' if path exhausted."""
        if not self.path or self.path_idx >= len(self.path) - 1:
            return "wait"

        current = self.path[self.path_idx]
        # If agent drifted from path (shouldn't happen with static, but be safe)
        if obs.position != current:
            # Try to find current position in remaining path
            for i in range(self.path_idx, len(self.path)):
                if self.path[i] == obs.position:
                    self.path_idx = i
                    break
            else:
                return "wait"  # lost, just wait

        self.path_idx += 1
        if self.path_idx >= len(self.path):
            return "wait"

        next_cell = self.path[self.path_idx]
        return _path_to_action(obs.position, next_cell)
