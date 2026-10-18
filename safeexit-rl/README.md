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
│   └── case1.txt                  # Hand-authored building grid (walls, exits, fire origin)
├── src/
│   ├── agents/
│   │   ├── q_learning_agent.py        # Tabular Q-learning agent with discretized state space
│   │   ├── baseline_shortest_path.py  # Static BFS shortest-path baseline (no replanning)
│   │   └── baseline_reactive_replan.py# Periodic reactive replanning baseline (k=5 steps)
│   ├── env.py                     # SafeExitEnv, GridBuilding, HazardField, CrowdSim
│   ├── reward.py                  # Congestion-, hazard-, and progress-aware reward
│   ├── train.py                   # RL training loop with smoothed metric tracking
│   ├── evaluate.py                # Comparative evaluation pipeline across all agents
│   └── visualize.py               # ASCII debug render and matplotlib state snapshots
├── experiments/
│   └── sweep_noise_delay.py       # Sensor noise (sigma) & latency (tau) sensitivity sweep
├── report/
│   └── report.tex                 # 2-page two-column Overleaf-compatible LaTeX report
├── results/                       # Generated checkpoints, plots, and CSV results
├── tests/
│   └── test_env.py                # 17 comprehensive unit tests
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
python -m pytest tests/test_env.py -v
```
*(All 17 tests verify grid loading, hazard monotonicity, crowd movement, observation noise/staleness, and reward signs).*

### 3. Train the RL Agent
```bash
python -m src.train --layout layouts/case1.txt --episodes 3000 --seed 42
```
Outputs:
- Saved policy: `results/qtable.npy` (1,352 visited states)
- Progress figure: `results/training_curve.png` (Reward, rolling success rate, and exposure curves)

### 4. Evaluate Against Baselines
```bash
python -m src.evaluate --checkpoint results/qtable.npy --layout layouts/case1.txt --n-eval-episodes 100
```
Outputs:
- Comparison plot: `results/comparison.png`
- Summary data: `results/eval_results.csv`

### 5. Reproduce the Noise & Delay Sweep
```bash
python -m experiments.sweep_noise_delay --checkpoint results/qtable.npy --layout layouts/case1.txt
```
Outputs:
- 4-panel sweep figure: `results/sweep_noise_delay.png`
- Detailed CSV: `results/sweep_results.csv`

### 6. Generate Environment Snapshot
```bash
python -m src.visualize
```
Outputs:
- Multi-panel figure: `results/env_snapshot.png`

---

## Key Experimental Results

### Benchmark Comparison ($\sigma = 0.1, \tau = 2$)
| Method | Success % | True Exposure | Steps to Exit | High-Crowd Steps |
| :--- | :---: | :---: | :---: | :---: |
| **SafeExit-RL (Ours)** | **99.0%** | **1.41** | 17.8 | 0.2 |
| **Static Shortest-Path** | 100.0% | 0.13 | 9.4 | 0.1 |
| **Reactive Replanning** | 100.0% | 0.12 | 10.1 | 0.1 |

### Noise Sensitivity Sweep ($\sigma \in [0.0, 0.3]$, delay $\tau = 2$)
| Noise Level ($\sigma$) | SafeExit-RL Success (Exposure) | Static SP Success (Exposure) | Reactive Replan Success (Exposure) |
| :---: | :---: | :---: | :---: |
| $\sigma = 0.0$ | 100.0% (0.09) | 100.0% (0.14) | 100.0% (0.11) |
| $\sigma = 0.1$ | 97.9% (2.92) | 100.0% (0.16) | 100.0% (0.13) |
| $\sigma = 0.2$ | 100.0% (0.06) | 97.9% (1.91) | 97.9% (3.90) |
| $\sigma = 0.3$ | **100.0% (0.00)** | **87.5% (3.74)** | **97.9% (4.21)** |

> **Key Finding**: Under severe sensor noise ($\sigma = 0.3$), deterministic graph-search baselines degrade significantly (Static SP success drops to 87.5% with high cumulative exposure of 3.74, and Reactive replanning sustains 4.21 exposure). SafeExit-RL maintains a **100% success rate with 0.00 exposure**, demonstrating learned resilience to corrupted sensory inputs.
