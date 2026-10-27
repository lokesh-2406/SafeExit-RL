"""
Reward function for SafeExit-RL.

r_t = w1*R_progress + w2*R_hazard_belief + w3*R_congestion + w4*R_terminal

Design:
  - R_progress and R_hazard_belief/R_congestion use information the agent
    ACTUALLY has (its belief_state), not ground truth.
  - R_terminal uses TRUE cumulative exposure (ground truth) when the agent
    reaches an exit -- i.e. the world can punish risks the agent didn't see.

"""
from dataclasses import dataclass


@dataclass
class RewardWeights:
    w_progress: float = 1.0
    w_hazard_belief: float = 1.0
    w_congestion: float = 1.0
    w_terminal: float = 1.0

    step_cost: float = -0.1            # small per-step time penalty
    revisit_penalty: float = -0.3
    invalid_move_penalty: float = -1.0
    exit_base_reward: float = 20.0     # before scaling by true exposure


def r_progress(moved: bool, revisited: bool, hit_wall: bool, weights: RewardWeights) -> float:
    if hit_wall:
        return weights.invalid_move_penalty
    if revisited:
        return weights.revisit_penalty
    return weights.step_cost


def r_hazard_belief(believed_hazard_at_cell: float, weights: RewardWeights) -> float:
    """Penalize based on what the agent BELIEVES about its current cell,
    not ground truth. believed_hazard_at_cell in [0, 1]."""
    return -believed_hazard_at_cell


def r_congestion(local_crowd_count: int, weights: RewardWeights) -> float:
    return -0.2 * local_crowd_count


def r_terminal(reached_exit: bool, cumulative_true_exposure: float,
               weights: RewardWeights) -> float:
    """Big reward for reaching exit, scaled down by TRUE cumulative
    hazard exposure -- ground truth is used here deliberately."""
    if not reached_exit:
        return 0.0
    # TODO (Phase 3): tune the scaling function; a simple starting point:
    penalty_factor = max(0.0, 1.0 - cumulative_true_exposure)
    return weights.exit_base_reward * penalty_factor


def compute_reward(moved: bool, revisited: bool, hit_wall: bool,
                    believed_hazard_at_cell: float, local_crowd_count: int,
                    reached_exit: bool, cumulative_true_exposure: float,
                    weights: RewardWeights) -> float:
    r = 0.0
    r += weights.w_progress * r_progress(moved, revisited, hit_wall, weights)
    r += weights.w_hazard_belief * r_hazard_belief(believed_hazard_at_cell, weights)
    r += weights.w_congestion * r_congestion(local_crowd_count, weights)
    r += weights.w_terminal * r_terminal(reached_exit, cumulative_true_exposure, weights)
    return r
