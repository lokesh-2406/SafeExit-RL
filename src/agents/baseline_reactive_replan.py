"""
Reactive replanning baseline.

Re-runs shortest-path planning every `replan_interval` steps using the
agent's CURRENT belief_state (same noise/delay model as the RL agent).
This parity in sensing is what makes the RL-vs-baseline comparison fair,
unlike SafeStep's DA baseline which gets ground-truth hazard.
"""
from src.agents.baseline_shortest_path import plan_path, _path_to_action


class ReactiveReplanAgent:
    def __init__(self, replan_interval: int = 5):
        self.replan_interval = replan_interval
        self.path = None
        self.path_idx = 0
        self.steps_since_replan = 0
        self.is_free_fn = None
        self.exits = None

    def reset(self, start: tuple, initial_obs, is_free_fn, exits: list):
        self.is_free_fn = is_free_fn
        self.exits = exits
        self.path = plan_path(start, initial_obs.local_hazard_estimate, is_free_fn, exits)
        self.path_idx = 0
        self.steps_since_replan = 0

    def select_action(self, obs, is_free_fn=None, exits: list = None) -> str:
        """If steps_since_replan >= replan_interval, replan from current
        position using current belief. Otherwise continue along path."""
        if is_free_fn is not None:
            self.is_free_fn = is_free_fn
        if exits is not None:
            self.exits = exits

        self.steps_since_replan += 1

        # Replan if interval reached or path is exhausted
        if (self.steps_since_replan >= self.replan_interval or
                not self.path or self.path_idx >= len(self.path) - 1):
            self.path = plan_path(
                obs.position,
                obs.local_hazard_estimate,
                self.is_free_fn,
                self.exits,
            )
            self.path_idx = 0
            self.steps_since_replan = 0

        # Follow the path
        if not self.path or self.path_idx >= len(self.path) - 1:
            return "wait"

        # Ensure we're at the expected position
        if obs.position != self.path[self.path_idx]:
            # Re-sync: find position in path
            for i in range(len(self.path)):
                if self.path[i] == obs.position:
                    self.path_idx = i
                    break
            else:
                # Lost, replan immediately
                self.path = plan_path(
                    obs.position,
                    obs.local_hazard_estimate,
                    self.is_free_fn,
                    self.exits,
                )
                self.path_idx = 0
                self.steps_since_replan = 0
                if not self.path or len(self.path) < 2:
                    return "wait"

        self.path_idx += 1
        if self.path_idx >= len(self.path):
            return "wait"

        next_cell = self.path[self.path_idx]
        return _path_to_action(obs.position, next_cell)
