"""
Sanity tests for GridBuilding, HazardField, CrowdSim, SafeExitEnv,
and the reward function.
"""
import pytest
import numpy as np
from src.env import GridBuilding, HazardField, CrowdSim, SafeExitEnv, ACTIONS


LAYOUT = "layouts/case1.txt"


# --- Phase 1: GridBuilding ---

def test_layout_loads():
    b = GridBuilding(LAYOUT)
    assert b.shape[0] > 0 and b.shape[1] > 0
    assert len(b.exits) >= 1


def test_exit_cells_are_free():
    b = GridBuilding(LAYOUT)
    for e in b.exits:
        assert b.is_free(e)


def test_wall_is_not_free():
    b = GridBuilding(LAYOUT)
    assert b.is_free((0, 0)) is False


def test_out_of_bounds_not_free():
    b = GridBuilding(LAYOUT)
    assert b.is_free((-1, 0)) is False
    assert b.is_free((b.shape[0], 0)) is False


# --- Phase 1: HazardField ---

def test_hazard_monotonic_with_time():
    """Hazard at origin should increase (or stay) with time."""
    h = HazardField(origin=(5, 5), grid_shape=(10, 10))
    prev = h.true_value((5, 5), 0)
    for t in range(1, 120):
        cur = h.true_value((5, 5), t)
        assert cur >= prev - 1e-12, f"Hazard decreased at t={t}"
        prev = cur


def test_hazard_decreases_with_distance():
    """At a fixed time, hazard should decrease as you move away from origin."""
    h = HazardField(origin=(5, 5), grid_shape=(20, 20))
    t = 80  # well after t_mid so there's meaningful spread
    val_origin = h.true_value((5, 5), t)
    val_near = h.true_value((5, 7), t)
    val_far = h.true_value((5, 15), t)
    assert val_origin >= val_near >= val_far


def test_hazard_full_grid_shape():
    h = HazardField(origin=(3, 3), grid_shape=(8, 12))
    g = h.full_grid(50)
    assert g.shape == (8, 12)
    assert g.min() >= 0.0 and g.max() <= 1.0


# --- Phase 1: CrowdSim ---

def test_crowd_reaches_exits():
    """After many steps, at least some NPCs should evacuate."""
    b = GridBuilding(LAYOUT)
    crowd = CrowdSim(n_agents=10, building=b, seed=42)
    for _ in range(300):
        crowd.step()
    assert sum(crowd.evacuated) > 0


def test_crowd_density_grid_shape():
    b = GridBuilding(LAYOUT)
    crowd = CrowdSim(n_agents=5, building=b, seed=0)
    dg = crowd.density_grid()
    assert dg.shape == b.shape


# --- Phase 2: SafeExitEnv & observation ---

def test_env_reset_returns_observation():
    env = SafeExitEnv(LAYOUT, seed=42)
    obs = env.reset()
    assert obs.position is not None
    assert isinstance(obs.local_hazard_estimate, dict)


def test_env_step_actions():
    """Every action should be accepted without crashing."""
    env = SafeExitEnv(LAYOUT, seed=42, max_steps=50)
    env.reset()
    for a in ACTIONS:
        obs, r, done, info = env.step(a)
        assert isinstance(r, float)
        if done:
            break


def test_random_rollout_completes():
    """Full random rollout should not crash."""
    rng = np.random.default_rng(7)
    env = SafeExitEnv(LAYOUT, seed=7, max_steps=100)
    env.reset()
    done = False
    steps = 0
    while not done:
        a = ACTIONS[rng.integers(len(ACTIONS))]
        _, _, done, info = env.step(a)
        steps += 1
    assert steps <= 100


def test_belief_state_staleness():
    """Cells outside sensing radius should not be refreshed every step."""
    env = SafeExitEnv(LAYOUT, sensing_radius=1, obs_delay=3, seed=0)
    env.reset(start_pos=(4, 1))
    obs1 = env.step("right")[0]
    # After one step, cells far away should not be in belief at all
    far_cell = (1, 25)
    assert far_cell not in env.last_refresh or env.last_refresh[far_cell] < env.t


def test_observation_noise():
    """Observed hazard within radius should differ from true value on average."""
    env = SafeExitEnv(LAYOUT, sensing_radius=2, obs_noise_std=0.3, obs_delay=1, seed=123)
    env.reset(start_pos=(4, 15))
    # Step many times near the fire origin to get meaningful hazard
    diffs = []
    for _ in range(50):
        obs, _, done, _ = env.step("wait")
        true_haz = env.hazard.true_value(env.agent_pos, env.t)
        believed = env.belief_state.get(env.agent_pos, 0.0)
        diffs.append(abs(true_haz - believed))
        if done:
            break
    # With noise_std=0.3, there should be SOME nonzero differences
    if len(diffs) > 5:
        assert np.mean(diffs) > 0.001


# --- Phase 3: Reward function ---

def test_reward_exit_positive():
    from src.reward import compute_reward, RewardWeights
    w = RewardWeights()
    r = compute_reward(moved=True, revisited=False, hit_wall=False,
                       believed_hazard_at_cell=0.0, local_crowd_count=0,
                       reached_exit=True, cumulative_true_exposure=0.0,
                       weights=w)
    assert r > 0, "Reaching exit with zero exposure should give positive reward"


def test_reward_wall_penalty():
    from src.reward import compute_reward, RewardWeights
    w = RewardWeights()
    r = compute_reward(moved=False, revisited=False, hit_wall=True,
                       believed_hazard_at_cell=0.0, local_crowd_count=0,
                       reached_exit=False, cumulative_true_exposure=0.0,
                       weights=w)
    assert r < 0, "Hitting a wall should give negative reward"


def test_reward_hazard_penalty():
    from src.reward import compute_reward, RewardWeights
    w = RewardWeights()
    r_safe = compute_reward(moved=True, revisited=False, hit_wall=False,
                            believed_hazard_at_cell=0.0, local_crowd_count=0,
                            reached_exit=False, cumulative_true_exposure=0.0,
                            weights=w)
    r_danger = compute_reward(moved=True, revisited=False, hit_wall=False,
                              believed_hazard_at_cell=0.8, local_crowd_count=0,
                              reached_exit=False, cumulative_true_exposure=0.0,
                              weights=w)
    assert r_danger < r_safe, "Higher believed hazard should give worse reward"
