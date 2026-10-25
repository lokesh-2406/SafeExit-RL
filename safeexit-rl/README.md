# SafeExit-RL

Adaptive fire-evacuation guidance using reinforcement learning under **partial, noisy, and delayed** hazard observations, with dynamic crowd congestion.

Inspired by, and benchmarked conceptually against, Fang et al. (2026), *"Development of a tenability-based path planning model for building fire evacuations using reinforcement learning"* (SafeStep), Journal of Building Engineering 118. This project intentionally differs from SafeStep in three key ways:
1. **Partial Observability**: The agent only senses hazard within a localized radius subject to Gaussian noise and measurement latency.
2. **Crowd Congestion**: Scripted evacuees create dynamic bottlenecks that factor into the state representation and reward function.
3. **Baseline Sensing Parity**: Both RL and classical graph-search baselines operate under identical noisy, delayed observations, isolating the value of RL policy learning.

---

## Repository Structure
```text
safeexit-rl/
├── layouts/
│   ├── case1.txt                  # Hand-authored building grid (9x30, 2 exits, fire origin)
│   └── case2.txt                  # Multi-wing layout (13x36, 4 exits, rooms, central fire)
├── src/
│   ├── agents/
│   │   ├── q_learning_agent.py        # Tabular Q-learning (coordinate & relative state modes)
│   │   ├── dqn_agent.py               # Minimal PyTorch Deep Q-Network (14-dim relative MLP)
│   │   ├── baseline_shortest_path.py  # Static BFS shortest-path baseline (no replanning)
│   │   └── baseline_reactive_replan.py# Periodic reactive replanning baseline (k=5 steps)
│   ├── env.py                     # SafeExitEnv, GridBuilding, HazardField, CrowdSim (panic/herding)
│   ├── reward.py                  # Congestion-, hazard-, and progress-aware reward
│   ├── train.py                   # RL training loop for Tabular Q and DQN
│   ├── evaluate.py                # Comparative evaluation pipeline across all agents
│   └── visualize.py               # ASCII debug render and matplotlib state snapshots
├── experiments/
│   ├── sweep_noise_delay.py       # Sensor noise (sigma) & latency (tau) sensitivity sweep
│   ├── eval_cross_layout.py       # Zero-shot cross-layout transfer evaluation
│   └── sweep_crowd_dynamics.py    # Panic and herding behavioral sweep
├── report/
│   └── report.tex                 # 2-page two-column Overleaf-compatible LaTeX report
├── results/                       # Checkpoints, plots, and CSV results
├── tests/
│   ├── test_env.py                # 17 core environment and reward tests
│   └── test_bonus.py              # 14 bonus tests (Case 2, DQN, transfer, panic/herding)
├── PLAN.md                        # Phased project specification
└── requirements.txt
```

---

## Reward Formulation & Tuned Hyperparameters
The reward formulation balances rapid egress, safety, and congestion avoidance:
$$r_t = w_1 R_{\text{progress}} + w_2 R_{\text{hazard}} + w_3 R_{\text{congestion}} + w_4 R_{\text{terminal}}$$

| Term | Value / Formula | Rationale |
| :--- | :--- | :--- |
| $w_1, w_2, w_3, w_4$ | `1.0, 1.0, 1.0, 1.0` | Balanced weighting across all four objectives. |
| Per-step cost | `-0.1` | Time pressure encouraging prompt evacuation. |
| Revisit penalty | `-0.3` | Discourages looping in dead ends and corridors. |
| Collision penalty | `-1.0` | Discourages invalid movements into walls. |
| Believed hazard penalty | `-\hat{h}(p_t)` | Agent is penalized based on its subjective belief of cell danger. |
| Crowd congestion penalty | `-0.2 * count` | Encourages routing away from congested choke points. |
| Terminal reward | `20.0 * max(0, 1 - true_exposure)` | Large completion reward, scaled down by ground-truth cumulative exposure. |

---

## Setup & Quickstart

### 1. Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m pytest -v
```
*(31 comprehensive unit tests covering grid parsing, hazard monotonicity, crowd movement, noise/staleness, reward signs, DQN training/replay, Case 2 connectivity, and crowd panic/herding).*

### 3. Train RL Agents
**Tabular Q-Learning (Case 1, coordinate state):**
```bash
python -m src.train --layout layouts/case1.txt --agent q_learning --episodes 3000 --seed 42 --out results/qtable.npy
```

**Tabular Q-Learning (Case 1, relative state):**
```bash
python -m src.train --layout layouts/case1.txt --agent q_learning --state-mode relative --episodes 1500 --seed 42 --out results/qtable_relative.npy
```

**Deep Q-Network (Case 1, 14-dim relative features):**
```bash
python -m src.train --layout layouts/case1.txt --agent dqn --episodes 800 --seed 42 --out results/dqn_model.pt
```

### 4. Evaluate In-Layout Benchmarks
```bash
python -m src.evaluate --checkpoint results/qtable.npy --dqn-checkpoint results/dqn_model.pt --layout layouts/case1.txt --n-eval-episodes 100
```
Outputs:
- Comparison plot: `results/comparison.png`
- Summary data: `results/eval_results.csv`

### 5. Evaluate Cross-Layout Generalization (Case 1 -> Case 2)
```bash
python -m experiments.eval_cross_layout --layout layouts/case2.txt --n-eval-episodes 100
```
Outputs:
- Cross-layout comparison plot: `results/cross_layout_comparison.png`
- Summary data: `results/cross_layout_results.csv`

### 6. Reproduce Noise & Latency Sensitivity Sweep
```bash
python -m experiments.sweep_noise_delay --checkpoint results/qtable.npy --dqn-checkpoint results/dqn_model.pt --layout layouts/case1.txt
```
Outputs:
- 4-panel sweep figure: `results/sweep_noise_delay.png`
- Detailed CSV: `results/sweep_results.csv`

### 7. Run Crowd Panic and Herding Sweep
```bash
python -m experiments.sweep_crowd_dynamics --layout layouts/case1.txt --n-eval-episodes 50
```
Outputs:
- Comparison figure: `results/crowd_dynamics_comparison.png`
- Summary data: `results/crowd_dynamics_results.csv`

---

## Experimental Results

### 1. In-Layout Benchmark (Case 1, $\sigma = 0.1, \tau = 2$)
| Method | Success % | True Exposure | Steps to Exit | High-Crowd Steps |
| :--- | :---: | :---: | :---: | :---: |
| **Tabular Q (Ours)** | 99.0% | 1.41 | 17.8 | 0.2 |
| **DQN (Ours)** | **100.0%** | **0.05** | 13.6 | 0.1 |
| **Static Shortest-Path** | 100.0% | 0.13 | 9.4 | 0.1 |
| **Reactive Replanning** | 100.0% | 0.12 | 10.1 | 0.1 |

*Takeaway:* DQN reduces cumulative hazard exposure by 96% compared to tabular Q-learning and cuts step count from 17.8 to 13.6, demonstrating superior value function smoothing over continuous hazard and bearing estimates.

### 2. Cross-Layout Generalization (Case 2, $13 \times 36$, 4 exits)
Policies trained exclusively on Case 1 deployed directly on Case 2 without retraining:

| Method | Success % | True Exposure | Steps to Exit | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Tabular Q (Coordinate)** | 12.0% | 53.52 | 177.6 | Overfits to absolute grid coordinates |
| **Tabular Q (Relative)** | 24.0% | 57.19 | 153.4 | Invariant to coordinates; limited by discrete binning |
| **DQN (Relative 14D)** | **37.0%** | **42.20** | **130.5** | Continuous relative features improve zero-shot transfer |
| **Static SP (Global)** | 100.0% | 0.03 | 10.7 | Fresh BFS path over full floor plan |
| **Reactive (Global)** | 99.0% | 1.83 | 12.9 | Periodic BFS replan over full floor plan |
| *Tabular Q (Retrained on Case 2)* | *90.0%* | *14.80* | *42.1* | *Calibrates layout reachability* |

*Takeaway:* Absolute coordinate representations cannot transfer across layouts (success drops to 12.0%). Relative feature representations double (24.0%) and triple (37.0% for DQN) zero-shot transfer performance, though global graph search remains superior for topological pathfinding in unseen buildings.

### 3. Noise Sensitivity Sweep ($\sigma \in [0.0, 0.3]$, delay $\tau = 2$)
| Noise Level ($\sigma$) | Tabular Q SR (Exp) | DQN SR (Exp) | Static SP SR (Exp) | Reactive Replan SR (Exp) |
| :---: | :---: | :---: | :---: | :---: |
| $\sigma = 0.0$ | 100.0% (0.09) | 100.0% (0.09) | 100.0% (0.14) | 100.0% (0.11) |
| $\sigma = 0.1$ | 97.9% (2.92) | 100.0% (0.01) | 100.0% (0.16) | 100.0% (0.13) |
| $\sigma = 0.2$ | 100.0% (0.06) | 100.0% (0.19) | 97.9% (1.91) | 97.9% (3.90) |
| $\sigma = 0.3$ | 95.8% (3.39) | **97.9% (2.59)** | **87.5% (3.74)** | **97.9% (4.21)** |

*Takeaway:* Under heavy sensor noise ($\sigma = 0.3$), classical deterministic shortest path degrades to 87.5% success with high cumulative exposure (3.74). Learned RL agents maintain $\ge 95.8\%$ success.

---

## Methodological Limitations & Interpretation

1. **Simulated Hazard vs. Physical FED**: Hazard values are derived from an analytical sigmoidal radial model, serving as a computationally efficient proxy for Fractional Effective Dose (FED), not a numerical CFD/FDS fluid simulation.
2. **Crowd Simulation**: NPCs execute scripted stochastic biased random walks with configurable panic (directional entropy) and herding (peer attraction), not full game-theoretic or multi-agent equilibrium behaviors.
3. **Cross-Layout Transfer**: Local perception-action agents transferred to completely new topologies cannot see through walls or anticipate non-convex corridor mazes without global graph representations or explicit spatial memory.
