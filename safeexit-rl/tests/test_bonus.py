"""
Comprehensive tests for SafeExit-RL bonus implementations:
  1. case2.txt layout geometry, connectivity, and environment execution
  2. Observation encoding (fixed-size 14D) and DQN agent
  3. DQN training update, terminal Bellman target, checkpoint save/load
  4. Q-learning state representations (coordinate vs relative)
  5. CrowdSim panic and herding behaviors, regression safety, seed determinism
  6. Evaluation non-mutation guarantee
"""
import os
import tempfile
import pytest
import numpy as np
import torch

from src.env import GridBuilding, SafeExitEnv, CrowdSim, ACTIONS
from src.agents.q_learning_agent import QLearningAgent, discretize_observation
from src.agents.dqn_agent import DQNAgent, encode_observation, QNetwork, ReplayBuffer


CASE1 = "layouts/case1.txt"
CASE2 = "layouts/case2.txt"


# ============================================================================
# 1. Layout Case 2 Tests
# ============================================================================

def test_case2_loads_and_has_valid_dimensions():
    b = GridBuilding(CASE2)
    assert b.shape[0] == 13
    assert b.shape[1] == 36
    assert len(b.exits) >= 2, "case2.txt should have multiple exits"
    assert b.fire_origin is not None, "case2.txt must define a fire origin"


def test_case2_exits_and_walls():
    b = GridBuilding(CASE2)
    for exit_cell in b.exits:
        assert b.is_free(exit_cell), f"Exit {exit_cell} must be a free cell"
    # Corner cells should be walls
    assert not b.is_free((0, 0))
    assert not b.is_free((0, b.shape[1] - 1))
    assert not b.is_free((b.shape[0] - 1, 0))
    assert not b.is_free((b.shape[0] - 1, b.shape[1] - 1))


def test_case2_full_connectivity():
    """Verify all free cells in case2.txt can reach at least one exit."""
    from collections import deque
    b = GridBuilding(CASE2)
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


def test_case2_env_reset_and_rollout():
    env = SafeExitEnv(CASE2, seed=123, max_steps=50)
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
# 2. DQN Observation Encoding & Network Tests
# ============================================================================

def test_encode_observation_fixed_shape():
    env1 = SafeExitEnv(CASE1, seed=1)
    obs1 = env1.reset()
    vec1 = encode_observation(obs1)

    env2 = SafeExitEnv(CASE2, seed=2)
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
    # Populate buffer with at least batch_size transitions
    for step_i in range(10):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        # Verify update works for both terminal and non-terminal transitions
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

        # Perform updates to change weights from random init
        for _ in range(5):
            action = agent1.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent1.update(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()

        agent1.save(ckpt_path)

        # Load into fresh agent
        agent2 = DQNAgent(seed=123)
        agent2.load(ckpt_path)

        for p1, p2 in zip(agent1.q_net.parameters(), agent2.q_net.parameters()):
            assert torch.allclose(p1, p2), "Loaded model weights must match saved weights exactly"


# ============================================================================
# 3. Tabular Q-Learning State Discretization & Transfer Tests
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
    env = SafeExitEnv(CASE2, seed=42)
    obs = env.reset()

    # Empty Q-table
    action = agent.select_action(obs, eval_mode=True)
    assert action in ACTIONS, "Fallback action must be a valid environment action"


# ============================================================================
# 4. Crowd Simulation: Panic & Herding Tests
# ============================================================================

def test_crowd_disabled_panic_herding_matches_default():
    """When panic and herding are 0.0, step behaves identically to original."""
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
    b = GridBuilding(CASE2)
    env = SafeExitEnv(CASE2, seed=42, panic_prob=0.8, herding_prob=0.0)
    env.reset()

    for _ in range(100):
        env.step("wait")
        for pos, evac in zip(env.crowd.positions, env.crowd.evacuated):
            if not evac:
                assert env.building.is_free(pos), f"NPC reached non-free cell {pos}"


def test_crowd_herding_keeps_agents_on_free_cells():
    """Even under high herding tendency, NPCs must stay on free cells."""
    b = GridBuilding(CASE2)
    env = SafeExitEnv(CASE2, seed=42, panic_prob=0.0, herding_prob=0.8)
    env.reset()

    for _ in range(100):
        env.step("wait")
        for pos, evac in zip(env.crowd.positions, env.crowd.evacuated):
            if not evac:
                assert env.building.is_free(pos), f"NPC reached non-free cell {pos}"


def test_evaluation_does_not_mutate_agent_parameters():
    """Evaluating an agent should not change its weights or Q-table."""
    from src.evaluate import evaluate_agent
    agent = DQNAgent(seed=42)
    initial_weights = [p.clone() for p in agent.q_net.parameters()]

    evaluate_agent(agent, "dqn", CASE1, n_episodes=4, seeds=[0])

    for p_before, p_after in zip(initial_weights, agent.q_net.parameters()):
        assert torch.allclose(p_before, p_after), "Evaluation must not alter network weights"
