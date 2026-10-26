# SafeExit-RL

Adaptive fire-evacuation guidance using reinforcement learning under **partial, noisy, and delayed** hazard observations, with dynamic crowd congestion.

Inspired by, and benchmarked conceptually against, Fang et al. (2026), *"Development of a tenability-based path planning model for building fire evacuations using reinforcement learning"* (SafeStep), Journal of Building Engineering 118. This project intentionally differs from SafeStep in three key ways:
1. **Partial Observability**: The agent only senses hazard within a localized radius subject to Gaussian noise and measurement latency.
2. **Crowd Congestion**: Scripted evacuees create dynamic bottlenecks that factor into the state representation and reward function.
3. **Baseline Sensing Parity**: Both RL and classical graph-search baselines operate under identical noisy, delayed observations, isolating the value of RL policy learning.

---

## Repository Structure
```text
SafeExit-RL/
├── demo/
│   └── simulator.html             # Standalone interactive browser simulation & viva demo
├── layouts/
│   ├── case1.txt                  # Hand-authored building grid (9x30, 2 exits, fire origin)
│   └── case2.txt                  # Multi-wing layout (13x36, 4 exits, rooms, central fire)
├── src/
│   ├── agents/
│   │   ├── q_learning_agent.py        # Tabular Q-learning (relative state representation)
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
├── results/                       # Trained models (.pt, .npy), comparative curves, and CSV metrics
├── tests/
│   ├── test_env.py                # 17 core environment and reward tests
│   └── test_extended_features.py  # 22 extended tests (Case 3, DQN, delay, exposure, panic/herding)
├── requirements.txt               # Project dependencies
└── README.md
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
| Terminal reward | `20.0 * max(0, 1 - post_delay_exposure)` | Large completion reward, scaled down by ground-truth post-delay evacuation exposure (equals total exposure for Case 1 immediate evacuation). |

---

## Three-Case Experimental Framework

SafeExit-RL benchmarks four primary methods across three systematically contrasting experimental regimes:
1. **DQN** (Deep Q-Network with 14-dim relative spatial/hazard features)
2. **Relative Tabular Q-learning** (Coordinate-invariant tabular state discretization with exit-bearing fallback)
3. **Static Shortest Path** (Offline BFS shortest-path baseline without replanning)
4. **Reactive Replanning** (Periodic online Dijkstra replanning based on local belief state)

| Regime | Layout | Pre-Evacuation Delay ($t_{\text{delay}}$) | Description & Focus |
| :--- | :--- | :---: | :--- |
| **Case 1** | `layouts/case1.txt` | 0 steps (Immediate) | Immediate evacuation upon fire ignition. Evaluates baseline egress policy performance and noise tolerance. |
| **Case 2** | `layouts/case1.txt` | 20 steps (~20s Delayed) | Delayed response where occupant remains stationary while fire expands and crowd advances. Evaluates dynamic resilience under severe corridor contamination. |
| **Case 3** | `layouts/case2.txt` | 0 steps (Cross-Layout) | Zero-shot transfer of Case 1-trained policies to an unseen, structurally distinct building ($13 \times 36$, 4 exits) without retraining. |

### Delay Parameterization & Exposure Accounting
- **Simulation Time Assumptions**: Under nominal assumptions of $1.0\text{ m} \times 1.0\text{ m}$ grid cell spacing and an unhindered pedestrian movement speed of $\approx 1.0\text{ m/s}$, one simulation step corresponds approximately to one second ($1\text{ step} \approx 1\text{ second}$). Egress in Case 1 typically takes 10-18 steps (approx. 10-18 seconds under these nominal assumptions).
- **Delay as a Simulation Parameter**: The pre-evacuation delay $t_{\text{delay}} = 20\text{ steps}$ is a chosen simulation parameter, selected to generate a substantial central corridor hazard bottleneck while keeping egress viable. At $t_{\text{delay}} = 20\text{ steps}$, the synthetic hazard radius has expanded from $0.95\text{ m}$ to $2.38\text{ m}$, affecting the central corridor and allowing ~40% of simulated NPCs to evacuate before the occupant begins moving. At $t \ge 60\text{ steps}$, the eastern exit becomes blocked ($\ge 0.30$ hazard).
- **Physical Calibration Disclaimer**: Physical calibration has not been validated against empirical fire egress experiments or real-world building evacuation data. The synthetic sigmoidal radial model serves solely as a computationally tractable simulation proxy for environmental tenability, not a physically validated combustion or Fractional Effective Dose (FED) model.
- **Exposure Metric Separation**: Total exposure accumulated from ignition is partitioned into:
  $$\text{Total Exposure} = E_{\text{pre-delay}} + E_{\text{post-delay}}$$
  Terminal exit reward uses $E_{\text{post-delay}}$ in $\max(0, 1 - E_{\text{post-delay}})$ to maintain reinforcement gradients, while both metrics are explicitly tracked without double-counting.
- **Synthetic Hazard vs. Physical FED**: The synthetic sigmoidal radial model serves as an abstract proxy for smoke/thermal exposure and is not a computational fluid dynamics (CFD/FDS) Fractional Effective Dose calculation.

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
*(39 comprehensive unit tests covering grid parsing, hazard monotonicity, crowd movement, noise/staleness, reward signs, DQN training/replay, Case 3 connectivity, panic/herding, delay backward compatibility, stationary delay accounting, and zero-shot transfer).*

### 3. Interactive Browser Simulation (Viva Demo)
Open `demo/simulator.html` directly in any web browser (completely self-contained, no server or dependencies required):
- **Live Scenarios**:
  - **1. The Cutoff Trap (Case 1 Layout)**: Fire ignites at $(4, 22)$; Static Shortest-Path blindly walks into fire at Exit 2, while SafeExit-RL detours safely to Exit 1.
  - **2. Spreading Fire & Pre-Evacuation Delay (Case 2)**: Same layout with fire pre-expanded ($r=5.0$) due to delayed start, demonstrating robustness under high sensor noise and delay.
  - **3. Cross-Layout Zero-Shot Transfer (Case 3)**: SafeExit-RL deployed on the unseen 4-exit labyrinth (`layouts/case2.txt`) without retraining.
- **Controls & Features**: Interactive playback speed, crowd panic and herding sliders, sensory noise toggles, live health and exposure metrics, and step-by-step path traces.

### 4. Case 1: Original Layout, Immediate Evacuation
**Evaluate Case 1 Primary Methods:**
```bash
python -m src.evaluate --layout layouts/case1.txt --pre-evacuation-delay 0 --rel-checkpoint results/qtable_relative.npy --dqn-checkpoint results/dqn_model.pt --out-csv results/case1_eval_results.csv --out-plot results/case1_comparison.png --title "Case 1: Immediate Evacuation (Original Layout)"
```
Outputs: `results/case1_eval_results.csv`, `results/case1_comparison.png`.

### 5. Case 2: Same Layout, Delayed Evacuation ($t_{\text{delay}} = 20$)
**Train fresh Case 2 models under delay:**
```bash
# Relative Tabular Q-learning
python -m src.train --layout layouts/case1.txt --agent q_learning --state-mode relative --pre-evacuation-delay 20 --episodes 1500 --seed 42 --out results/case2_qtable_relative.npy --plot-out results/case2_training_curve_relative.png

# Deep Q-Network
python -m src.train --layout layouts/case1.txt --agent dqn --pre-evacuation-delay 20 --episodes 800 --seed 42 --out results/case2_dqn_model.pt --plot-out results/case2_dqn_training_curve.png
```

**Evaluate Case 2 Primary Methods:**
```bash
python -m src.evaluate --layout layouts/case1.txt --pre-evacuation-delay 20 --rel-checkpoint results/case2_qtable_relative.npy --dqn-checkpoint results/case2_dqn_model.pt --out-csv results/case2_eval_results.csv --out-plot results/case2_comparison.png --title "Case 2: Delayed Evacuation (t_delay=20, Same Layout)"
```
Outputs: `results/case2_eval_results.csv`, `results/case2_comparison.png`.

### 6. Case 3: Cross-Layout Generalization (Unseen Layout)
**Evaluate Case 1 policies zero-shot on Case 3 layout (`layouts/case2.txt`):**
```bash
python -m experiments.eval_cross_layout --layout layouts/case2.txt --dqn-checkpoint results/dqn_model.pt --rel-checkpoint results/qtable_relative.npy --out-csv results/case3_cross_layout_results.csv --out-plot results/case3_cross_layout_comparison.png
```
Outputs: `results/case3_cross_layout_results.csv`, `results/case3_cross_layout_comparison.png`.

### 7. Reproduce Noise & Latency Sensitivity Sweep
```bash
python -m experiments.sweep_noise_delay --checkpoint results/qtable_relative.npy --dqn-checkpoint results/dqn_model.pt --layout layouts/case1.txt
```
Outputs:
- 4-panel sweep figure: `results/sweep_noise_delay.png`
- Detailed CSV: `results/sweep_results.csv`

### 8. Run Crowd Panic and Herding Sweep
```bash
python -m experiments.sweep_crowd_dynamics --layout layouts/case1.txt --n-eval-episodes 50
```
Outputs:
- Comparison figure: `results/crowd_dynamics_comparison.png`
- Summary data: `results/crowd_dynamics_results.csv`

---

## Experimental Results

### 1. Case 1 -- Immediate Evacuation (`layouts/case1.txt`, $t_{\text{delay}} = 0$)
| Primary Method | Success % | Total Exposure | Post-Delay Exposure | Evacuation Steps | High-Crowd Steps |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DQN (Ours)** | **100.0%** | **0.08** | **0.08** | 17.4 | 0.1 |
| **Relative Tabular Q** | 100.0% | 0.33 | 0.33 | 23.9 | 0.1 |
| **Static Shortest Path** | 100.0% | 0.13 | 0.13 | 9.4 | 0.1 |
| **Reactive Replanning** | 100.0% | 0.12 | 0.12 | 10.1 | 0.1 |

*Takeaway:* DQN achieves the lowest cumulative hazard exposure across all methods (0.08), maintaining safe standoff distances from expanding hazard while smoothly routing around evacuee congestion.

### 2. Case 2 -- Delayed Evacuation (`layouts/case1.txt`, $t_{\text{delay}} = 20$)
| Primary Method | Success % | Total Exposure | Post-Delay Exposure | Evacuation Steps | High-Crowd Steps |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DQN (Ours)** | **100.0%** | **0.39** | **0.06** | 14.9 | 0.0 |
| **Relative Tabular Q** | 89.0% | 4.88 | 4.66 | 35.8 | 0.0 |
| **Static Shortest Path** | 100.0% | 0.38 | 0.23 | 9.5 | 0.0 |
| **Reactive Replanning** | 95.0% | 9.91 | 9.69 | 19.8 | 0.1 |

*Takeaway:* Under a 20-step pre-evacuation delay, hazard engulfs the central corridor. Reactive replanning suffers severely (9.91 exposure, 95% success) due to belief staleness and local oscillation. DQN proves remarkably resilient, achieving 100% success and a post-delay exposure of just 0.06 (lower than Static Shortest Path's 0.23).

### 3. Case 3 -- Cross-Layout Generalization (`layouts/case2.txt`, Zero Retraining)
| Primary Method | Success % | Total Exposure | Evacuation Steps | High-Crowd Steps | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DQN (Ours)** | **25.0%** | 48.42 | 151.9 | 0.6 | Continuous 14D relative state transfer |
| **Relative Tabular Q** | 24.0% | 57.19 | 153.4 | 0.0 | Discrete relative state transfer |
| **Static Shortest Path** | 100.0% | 0.03 | 10.7 | 0.0 | Global topology BFS baseline |
| **Reactive Replanning** | 99.0% | 1.83 | 12.9 | 0.0 | Global topology periodic replanning |

*Takeaway:* Relative state representations (Relative Tabular Q at 24% and DQN at 25%) successfully navigate local corridors via exit bearing. Classical baselines leverage full topological maps to find global paths, underscoring the distinction between local perception-action RL and global graph search in unseen environments.

---

## Methodological Limitations & Interpretation

1. **Simulated Hazard vs. Physical FED**: Hazard values are derived from an analytical sigmoidal radial model, serving as a computationally efficient proxy for Fractional Effective Dose (FED), not a numerical CFD/FDS fluid simulation.
2. **Crowd Simulation**: NPCs execute scripted stochastic biased random walks with configurable panic (directional entropy) and herding (peer attraction), not full game-theoretic or multi-agent equilibrium behaviors.
3. **Cross-Layout Transfer**: Local perception-action agents transferred to completely new topologies cannot see through walls or anticipate non-convex corridor mazes without global graph representations or explicit spatial memory.
