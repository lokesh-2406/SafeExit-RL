"""
Comprehensive tests for SafeExit-RL extended features and three-case revision:
  1. Case 3 layout (layouts/case2.txt) geometry, connectivity, and environment execution
  2. Pre-evacuation delay, stationary occupant, hazard & crowd progression, exposure accounting
  3. Observation encoding (fixed-size 14D) and DQN agent
  4. DQN training update, terminal Bellman target, checkpoint save/load
  5. Q-learning state representations (coordinate vs relative) and unseen fallback
  6. CrowdSim panic and herding behaviors, regression safety, seed determinism
  7. Evaluation non-mutation guarantee and 4 primary comparison methods
  8. Case 3 zero-shot cross-layout transfer execution without retraining
"""
import os
import tempfile
import pytest
import numpy as np
import torch

from src.env import GridBuilding, SafeExitEnv, CrowdSim, ACTIONS
from src.agents.q_learning_agent import QLearningAgent, discretize_observation
from src.agents.dqn_agent import DQNAgent, encode_observation, QNetwork, ReplayBuffer
from src.agents.baseline_shortest_path import StaticShortestPathAgent
from src.agents.baseline_reactive_replan import ReactiveReplanAgent
from src.evaluate import evaluate_agent, aggregate


CASE1 = "layouts/case1.txt"
CASE3_LAYOUT = "layouts/case2.txt"  # Case 3 cross-layout evaluation environment


# ============================================================================
# 1. Case 3 Layout Geometry & Connectivity Tests
# ============================================================================

def test_case3_layout_loads_and_has_valid_dimensions():
    b = GridBuilding(CASE3_LAYOUT)
    assert b.shape[0] == 13
    assert b.shape[1] == 36
    assert len(b.exits) >= 2, "case2.txt (Case 3 layout) should have multiple exits"
    assert b.fire_origin is not None, "case2.txt (Case 3 layout) must define a fire origin"


def test_case3_layout_exits_and_walls():
    b = GridBuilding(CASE3_LAYOUT)
    for exit_cell in b.exits:
        assert b.is_free(exit_cell), f"Exit {exit_cell} must be a free cell"
    # Corner cells should be walls
    assert not b.is_free((0, 0))
    assert not b.is_free((0, b.shape[1] - 1))
    assert not b.is_free((b.shape[0] - 1, 0))
    assert not b.is_free((b.shape[0] - 1, b.shape[1] - 1))


def test_case3_layout_full_connectivity():
    """Verify all free cells in case2.txt can reach at least one exit."""
    from collections import deque
    b = GridBuilding(CASE3_LAYOUT)
    start = b.exits[0]
    queue = deque([start])
    visited = {start}

    while queue:
        cur = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (cur[0] + dr, cur[1] + dc)
            if b.is_free(neighbor) and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    all_free = set(b.free_cells())
    unreachable = all_free - visited
    assert len(unreachable) == 0, f"Unreachable cells found in case2.txt: {unreachable}"


def test_case3_env_reset_and_rollout():
    env = SafeExitEnv(CASE3_LAYOUT, seed=123, max_steps=50)
    obs = env.reset()
    assert obs.position is not None
    assert env.building.is_free(obs.position)

    for action in ACTIONS:
        next_obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        if done:
            break


# ============================================================================
# 2. Case 2 Delay, Stationary Occupant, & Exposure Accounting Tests
# ============================================================================

def test_zero_delay_backward_compatibility():
    """Default delay=0 must strictly preserve original environment reset/step behavior."""
    env_default = SafeExitEnv(CASE1, seed=42)
    env_zero = SafeExitEnv(CASE1, seed=42, pre_evacuation_delay=0)

    obs_def = env_default.reset(start_pos=(4, 5))
    obs_zero = env_zero.reset(start_pos=(4, 5))

    assert obs_def.position == obs_zero.position
    assert env_default.t == 0
    assert env_zero.t == 0
    assert env_zero.evacuation_steps == 0
    assert env_zero.pre_delay_exposure == 0.0
    assert env_zero.post_delay_exposure == 0.0

    # Step identical action
    obs_d, r_d, done_d, info_d = env_default.step("right")
    obs_z, r_z, done_z, info_z = env_zero.step("right")

    assert obs_d.position == obs_z.position
    assert r_d == pytest.approx(r_z)
    assert info_d["cumulative_true_exposure"] == pytest.approx(info_z["cumulative_true_exposure"])
    assert info_z["pre_delay_exposure"] == 0.0
    assert info_z["post_delay_exposure"] == pytest.approx(info_z["cumulative_true_exposure"])


def test_stationary_occupant_and_hazard_growth_during_delay():
    """During delay, occupant stays stationary while hazard expands and accumulates."""
    delay = 20
    start = (4, 21)  # cell adjacent to fire origin (4, 22)
    env = SafeExitEnv(CASE1, seed=10, pre_evacuation_delay=delay)
    obs = env.reset(start_pos=start)

    # Occupant position must remain start_pos
    assert env.agent_pos == start
    assert obs.position == start
    # Timeline advances by delay
    assert env.t == delay
    assert env.evacuation_steps == 0
    # Pre-delay exposure accumulates stationary exposure
    assert env.pre_delay_exposure > 0.0
    assert env.post_delay_exposure == 0.0
    assert env.cumulative_true_exposure == pytest.approx(env.pre_delay_exposure)
    # Hazard field expanded to reach adjacent cells over 20 steps
    h_adjacent = env.hazard.true_value(start, env.t)
    assert h_adjacent > env.hazard.true_value(start, 0)


def test_crowd_advances_during_delay():
    """Crowd must advance during pre-evacuation delay."""
    delay = 20
    env_immediate = SafeExitEnv(CASE1, n_npcs=10, seed=123, pre_evacuation_delay=0)
    env_immediate.reset()
    init_positions = list(env_immediate.crowd.positions)

    env_delayed = SafeExitEnv(CASE1, n_npcs=10, seed=123, pre_evacuation_delay=delay)
    env_delayed.reset()
    delayed_positions = list(env_delayed.crowd.positions)

    # Crowd positions must have changed after 20 steps of movement
    assert init_positions != delayed_positions
    # Any active NPC must remain on a free cell
    for pos, evac in zip(env_delayed.crowd.positions, env_delayed.crowd.evacuated):
        if not evac:
            assert env_delayed.building.is_free(pos)


def test_delay_does_not_consume_evacuation_steps():
    """Pre-evacuation delay must not consume the post-evacuation max_steps budget."""
    delay = 25
    budget = 10
    env = SafeExitEnv(CASE1, seed=42, max_steps=budget, pre_evacuation_delay=delay)
    env.reset(start_pos=(1, 1))

    steps_taken = 0
    done = False
    while not done:
        _, _, done, info = env.step("wait")
        steps_taken += 1

    assert steps_taken == budget, f"Expected {budget} steps, took {steps_taken}"
    assert info["evacuation_steps"] == budget
    assert info["simulation_time"] == delay + budget
    assert info["timed_out"] is True


def test_exposure_accounting_separation():
    """Verify total exposure = pre_delay_exposure + post_delay_exposure without double-counting."""
    delay = 15
    env = SafeExitEnv(CASE1, seed=7, pre_evacuation_delay=delay)
    env.reset(start_pos=(4, 21))  # cell adjacent to fire origin (4, 22)

    pre_exp = env.pre_delay_exposure
    assert pre_exp > 0.0

    for step_i in range(5):
        _, _, done, info = env.step("left")
        assert info["pre_delay_exposure"] == pytest.approx(pre_exp)
        assert info["total_exposure"] == pytest.approx(info["pre_delay_exposure"] + info["post_delay_exposure"])
        assert info["cumulative_true_exposure"] == pytest.approx(info["total_exposure"])
        if done:
            break


# ============================================================================
# 3. DQN Observation Encoding & Network Tests
# ============================================================================

def test_encode_observation_fixed_shape():
    env1 = SafeExitEnv(CASE1, seed=1)
    obs1 = env1.reset()
    vec1 = encode_observation(obs1)

    env2 = SafeExitEnv(CASE3_LAYOUT, seed=2)
    obs2 = env2.reset()
    vec2 = encode_observation(obs2)

    assert vec1.shape == (14,)
    assert vec2.shape == (14,)
    assert vec1.dtype == np.float32
    assert np.all(np.isfinite(vec1))
    assert np.all(np.isfinite(vec2))


def test_dqn_action_selection_and_eval_mode():
    agent = DQNAgent(seed=42)
    env = SafeExitEnv(CASE1, seed=42)
    obs = env.reset()

    # Training action (epsilon-greedy)
    act_train = agent.select_action(obs, eval_mode=False)
    assert act_train in ACTIONS

    # Evaluation action (greedy)
    act_eval = agent.select_action(obs, eval_mode=True)
    assert act_eval in ACTIONS


def test_dqn_training_step_and_terminal_handling():
    agent = DQNAgent(batch_size=4, seed=42)
    env = SafeExitEnv(CASE1, seed=42)

    obs = env.reset()
    for step_i in range(10):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs, done=bool(step_i % 2 == 0))
        obs = next_obs if not done else env.reset()

    assert len(agent.replay_buffer) >= 10
    assert agent.train_steps >= 10


def test_dqn_save_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dqn_test.pt")
        agent1 = DQNAgent(seed=99, batch_size=2)
        env = SafeExitEnv(CASE1, seed=99)
        obs = env.reset()

        for _ in range(5):
            action = agent1.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent1.update(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()

        agent1.save(ckpt_path)

        agent2 = DQNAgent(seed=123)
        agent2.load(ckpt_path)

        for p1, p2 in zip(agent1.q_net.parameters(), agent2.q_net.parameters()):
            assert torch.allclose(p1, p2), "Loaded model weights must match saved weights exactly"


# ============================================================================
# 4. Tabular Q-Learning State Discretization & Transfer Tests
# ============================================================================

def test_q_learning_discretize_modes():
    env = SafeExitEnv(CASE1, seed=10)
    obs = env.reset()

    # Coordinate mode includes absolute (r, c)
    state_coord = discretize_observation(obs, state_mode="coordinate")
    assert state_coord[0] == obs.position[0]
    assert state_coord[1] == obs.position[1]

    # Relative mode uses local walls instead of (r, c)
    state_rel = discretize_observation(obs, state_mode="relative")
    assert state_rel[0] == getattr(obs, "local_walls", ())
    assert len(state_rel) == 4


def test_q_learning_unseen_state_fallback():
    """Unseen states should not crash and should fall back safely to exit bearing."""
    agent = QLearningAgent(unseen_action_fallback="exit_bearing", seed=42)
    env = SafeExitEnv(CASE3_LAYOUT, seed=42)
    obs = env.reset()

    action = agent.select_action(obs, eval_mode=True)
    assert action in ACTIONS, "Fallback action must be a valid environment action"


# ============================================================================
# 5. Crowd Simulation: Panic & Herding Tests
# ============================================================================

def test_crowd_disabled_panic_herding_matches_default():
    """When panic and herding are 0.0, step behaves identically to default."""
    b = GridBuilding(CASE1)
    sim1 = CrowdSim(n_agents=5, building=b, seed=77, panic_prob=0.0, herding_prob=0.0)
    sim2 = CrowdSim(n_agents=5, building=b, seed=77)

    for _ in range(20):
        sim1.step()
        sim2.step()
        assert sim1.positions == sim2.positions
        assert sim1.evacuated == sim2.evacuated


def test_crowd_panic_keeps_agents_on_free_cells():
    """Even under high panic and local hazard, NPCs must stay on free cells."""
    env = SafeExitEnv(CASE3_LAYOUT, seed=42, panic_prob=0.8, herding_prob=0.0)
    env.reset()

    for _ in range(100):
        env.step("wait")
        for pos, evac in zip(env.crowd.positions, env.crowd.evacuated):
            if not evac:
                assert env.building.is_free(pos), f"NPC reached non-free cell {pos}"


def test_crowd_herding_keeps_agents_on_free_cells():
    """Even under high herding tendency, NPCs must stay on free cells."""
    env = SafeExitEnv(CASE3_LAYOUT, seed=42, panic_prob=0.0, herding_prob=0.8)
    env.reset()

    for _ in range(100):
        env.step("wait")
        for pos, evac in zip(env.crowd.positions, env.crowd.evacuated):
            if not evac:
                assert env.building.is_free(pos), f"NPC reached non-free cell {pos}"


# ============================================================================
# 6. Evaluation Non-Mutation & Four Primary Methods
# ============================================================================

def test_evaluation_does_not_mutate_agent_parameters():
    """Evaluating an agent should not change its weights or Q-table."""
    agent = DQNAgent(seed=42)
    initial_weights = [p.clone() for p in agent.q_net.parameters()]

    evaluate_agent(agent, "dqn", CASE1, n_episodes=4, seeds=[0])

    for p_before, p_after in zip(initial_weights, agent.q_net.parameters()):
        assert torch.allclose(p_before, p_after), "Evaluation must not alter network weights"


def test_four_primary_methods_evaluate_successfully():
    """Verify all four primary methods evaluate and produce valid comparison metrics."""
    dqn = DQNAgent(seed=0)
    rel_q = QLearningAgent(state_mode="relative", unseen_action_fallback="exit_bearing", seed=0)
    static_sp = StaticShortestPathAgent()
    reactive = ReactiveReplanAgent(replan_interval=5)

    methods = [
        (dqn, "dqn"),
        (rel_q, "rl"),
        (static_sp, "static"),
        (reactive, "reactive"),
    ]

    for agent, atype in methods:
        results = evaluate_agent(agent, atype, CASE1, n_episodes=2, seeds=[42], pre_evacuation_delay=5)
        agg = aggregate(results)
        assert "success_rate" in agg
        assert "avg_exposure" in agg
        assert "post_delay_exposure" in agg
        assert "pre_delay_exposure" in agg
        assert "avg_steps" in agg
        assert "avg_crowd_time" in agg
        assert agg["avg_exposure"] >= agg["post_delay_exposure"]


# ============================================================================
# 7. Case 3 Zero-Shot Generalization Test
# ============================================================================

def test_case3_zero_shot_execution():
    """Policies trained or initialized on Case 1 execute zero-shot on Case 3 layout."""
    agent_dqn = DQNAgent(seed=1)
    agent_rel = QLearningAgent(state_mode="relative", unseen_action_fallback="exit_bearing", seed=1)

    # Evaluate zero-shot on CASE3_LAYOUT
    dqn_res = evaluate_agent(agent_dqn, "dqn", CASE3_LAYOUT, n_episodes=2, seeds=[0])
    rel_res = evaluate_agent(agent_rel, "rl", CASE3_LAYOUT, n_episodes=2, seeds=[0])

    assert len(dqn_res) == 2
    assert len(rel_res) == 2
    for r in dqn_res + rel_res:
        assert isinstance(r["success"], bool)
        assert r["steps"] > 0
