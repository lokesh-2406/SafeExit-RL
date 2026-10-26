# SafeExit-RL — Execution Plan for Agentic Coding Assistant

> Hand this whole file to your coding agent (e.g. Antigravity) as the task spec.
> Work through phases IN ORDER. Each phase has a clear "done when" condition —
> do not move to the next phase until the current one's condition is met.
> Commit to git after each phase with a descriptive message (commit history
> matters for grading — do NOT squash everything into one commit at the end).

## Project summary

Build an RL agent that navigates a 2D grid-world building evacuation under
**partial, noisy, delayed observations** of fire hazard, plus crowd congestion
from other simulated evacuees. Compare it against two baselines that get the
SAME limited sensing (not ground truth), so the comparison isolates the value
of *learning* rather than the value of *better sensors*. This project is
inspired by, but intentionally diverges from, Fang et al. 2026 "SafeStep"
(Journal of Building Engineering) — do not reuse their exact reward constants,
exact state representation, or exact wording. Cite it as related work only.

Hard constraints from the course assignment:
- RL-only method. No heuristic-search-only or supervised-only submissions.
- GitHub repo must have real, incremental commit history.
- Final report is 2 pages (excluding references), two-column LaTeX.
- Deadline: Oct 26, 2026, 10:00 AM — build in buffer, don't cut it close.

---

## Phase 0 — Repo setup (Day 1, ~1 hr)
- [ ] `git init`, create `.gitignore` (python, venv, __pycache__, results/*.png except a few kept samples)
- [ ] Set up virtualenv, install `requirements.txt`
- [ ] Commit: "Initial project scaffold"

**Done when:** `pip install -r requirements.txt` works clean and repo has first commit.

---

## Phase 1 — Environment core (Day 1–2)
File: `src/env.py`

- [ ] Implement `GridBuilding` class: loads a layout from a simple text/array map
      (walls, free cells, exits). Start with ONE hand-authored layout (a
      corridor + a few rooms, similar complexity to SafeStep Case 1 — small
      enough to debug fast, do NOT copy their exact room labels/coordinates).
- [ ] Implement `HazardField`: synthetic fire spread. Do NOT try to use FDS.
      Model: hazard originates at one cell, expands outward over time as a
      soft radial gradient (e.g. hazard(cell, t) = f(distance_to_origin, t)
      with a sigmoid or linear growth in radius). Keep it cheap — this is a
      proxy for FED, not a physics simulation. Store TRUE hazard grid at every
      timestep (ground truth, used only for logging/reward-terminal/eval, not
      given directly to agent).
- [ ] Implement `CrowdSim`: N additional NPC evacuee agents doing simple
      biased-random-walk-to-nearest-exit. Track density per cell per step.
- [ ] Implement `step(action)` / `reset()` Gym-like interface.
- [ ] Unit test: hazard expands monotonically, crowd agents reach exits,
      env doesn't crash on edge movements (walls, corners).

**Done when:** `tests/test_env.py` passes and you can run a random-action
rollout and print a text-rendered grid per step.

---

## Phase 2 — Partial observability layer (Day 2–3)
File: `src/env.py` (extend) — this is the project's key novelty, give it care.

- [ ] Implement `Observation` construction from true state:
      - `position`: exact
      - `local_hazard_estimate`: true hazard value + Gaussian noise
        (configurable σ), only for cells within `sensing_radius` of agent
      - `staleness`: for each observed cell, only refresh its stored value
        every `delay` steps; outside sensing radius, use last-known value
        (or a fixed "unknown" placeholder if never observed)
      - `local_crowd_density`: count of NPCs within sensing radius
      - `exit_bearing`: relative direction to nearest KNOWN-safe exit based
        on agent's current belief state, not ground truth
- [ ] Make σ (noise) and delay configurable params on `GridBuilding.__init__`
      — you'll sweep these later in Phase 6.
- [ ] Add a `belief_state` dict the agent maintains across steps (its memory
      of hazard per cell, updated only when observed).

**Done when:** you can print an agent's belief grid next to the true hazard
grid and visibly see the gap (staleness/noise causing mismatch) in a demo
script.

---

## Phase 3 — Reward function (Day 3)
File: `src/reward.py`

Implement exactly this structure (keep weights as named constants you can
tune, document your chosen values in README — don't copy SafeStep's numbers):

```
r_t = w1*R_progress + w2*R_hazard_belief + w3*R_congestion + w4*R_terminal
```

- `R_progress`: small negative per-step cost (time pressure) + penalty for
  revisiting a cell / hitting a wall
- `R_hazard_belief`: penalty proportional to the agent's BELIEVED hazard at
  its current cell (uses belief_state, not ground truth — this is the
  intentional mismatch that creates interesting behavior)
- `R_congestion`: penalty proportional to local crowd density at current cell
- `R_terminal`: on reaching exit, big positive reward, SCALED DOWN by the
  agent's TRUE cumulative hazard exposure over the evacuation phase (ground truth used
  here deliberately — "the world punishes risks you didn't know about").
  *(Specification note: In Case 1 immediate evacuation, post-delay exposure equals total
  cumulative exposure. In Case 2 delayed evacuation, Phase 9 is authoritative: R_terminal
  scales with post-delay evacuation exposure to preserve policy learning gradients while
  total cumulative exposure is logged separately in metrics).*

- [ ] Log both `cumulative_true_exposure` and `cumulative_believed_exposure`
      per episode — you need both for the eval story in the report.

**Done when:** reward function is unit tested with a few hand-constructed
trajectories giving expected sign/magnitude of reward.

---

## Phase 4 — Agent (Day 3–4)
File: `src/agents/q_learning_agent.py`

- [ ] Start with tabular Q-learning over a DISCRETIZED observation
      (bin position, binned local hazard average, binned crowd count,
      exit bearing direction). This is much faster to get converging than a
      DQN and perfectly defensible given the discrete small grid.
- [ ] Epsilon-greedy exploration, standard Q-update.
- [ ] Stretch goal (ONLY if time remains after Phase 6): swap in a small DQN
      (`agents/dqn_agent.py`) with the same observation, compare.
- [ ] `train.py`: training loop over episodes, save Q-table / model
      checkpoint, log loss/reward curve to `results/`.

**Done when:** training loss/reward curve shows convergence (reward trending
up, plateauing) on the single layout from Phase 1, saved as a PNG plot.

---

## Phase 5 — Baselines (Day 5)
File: `src/agents/baseline_shortest_path.py`, `baseline_reactive_replan.py`

- [ ] **Static shortest-path**: BFS/Dijkstra computed once at episode start
      using only the agent's initial (noisy, limited) observation, never
      updated. No replanning.
- [ ] **Reactive replanning**: re-run Dijkstra every `k` steps using the
      agent's CURRENT belief_state (same noise/delay as the RL agent gets —
      this parity is what makes the comparison fair and is your strongest
      methodological point vs. SafeStep's DA comparison, which gives DA
      perfect information).
- [ ] Both baselines log the same metrics as the RL agent (see Phase 6).

**Done when:** both baselines run end-to-end on the same layout and produce
logged trajectories + metrics.

---

## Phase 6 — Evaluation & the noise/delay sweep (Day 6)
File: `src/evaluate.py`, `experiments/sweep_noise_delay.py`

Metrics per method (RL agent, static baseline, reactive baseline), averaged
over multiple random seeds / start positions:
- [ ] Success rate (reached an exit before episode timeout)
- [ ] True cumulative hazard exposure (ground truth)
- [ ] Time-to-exit (steps)
- [ ] Time spent in high-crowd-density cells
- [ ] Sweep: re-run all three methods across a grid of (sigma, delay) values,
      plot success rate / exposure vs. sigma and vs. delay -- THIS is your
      differentiated headline result, make sure it's in the report.

**Done when:** you have 3–4 clean comparison plots (bar charts for the main
comparison + line plots for the sweep) saved in `results/`.

---

## Phase 7 — Report (Day 7)
File: `report/report.tex` (two-column, Overleaf-compatible — template stub
already included in this repo).

Sections (2 pages, excluding references):
1. Team + GitHub link at top
2. State/Action/Reward formulation (be precise, use the math from Phase 2–3)
3. Environment & algorithm (grid, hazard model, crowd sim, Q-learning, the
   two baselines — explicitly note baselines share sensing limits with agent)
4. Results: main comparison table/bars + the noise/delay sweep plot
5. Short discussion: where RL wins, where it doesn't, limitations
6. Related work: cite SafeStep (Fang et al., J. Building Eng. 2026) honestly
   as the closest related work and state explicitly how this project differs
   (partial observability, congestion, synthetic hazard vs. FDS, baseline
   parity in sensing)

**Done when:** PDF compiles, is ≤2 pages excl. references, team names +
GitHub link are at top.

---

## Phase 8 — Polish & submit (Day 8, buffer day)
- [x] README.md: install steps, how to run training/eval/sweep, how to
      reproduce every figure in the report
- [x] Confirm commit history reflects real incremental work, not one dump
- [ ] If repo is private, grant access to GitHub handles `shubhambagwari`
      and `divyaanshmertia`
- [x] Final gut-check: does any code/report text read like a close paraphrase
      of the SafeStep paper's wording? Rewrite anything that does.
- [ ] Submit before 10:00 AM Oct 26 — do not wait until the deadline.

---

## Phase 9 — Final Three-Case Experiment Revision (Authoritative Specification)

### 1. Three-Case Experimental Framework
- [x] **Case 1: Original Layout, Immediate Evacuation (`layouts/case1.txt`, delay=0):**
  - Evaluates immediate evacuation response upon fire ignition at $(4, 22)$.
  - Primary comparison methods: (1) DQN, (2) Relative Tabular Q-learning, (3) Static Shortest Path, (4) Reactive Replanning.
  - Coordinate Tabular Q-learning excluded from primary comparisons (available as ablative baseline).
  - Outputs: `results/case1_eval_results.csv`, `results/case1_comparison.png`.
- [x] **Case 2: Same Layout, Delayed Evacuation (`layouts/case1.txt`, calibrated delay $t_{\text{delay}}=20$ steps):**
  - Primary occupant remains stationary during delay while hazard expands and crowd advances.
  - Delay does not consume the post-delay evacuation step budget ($T_{\text{max}}=200$).
  - Separate exposure accounting: Total Exposure = Pre-Delay Exposure + Post-Delay Exposure (no double counting).
  - Authoritative reward semantics: Terminal reward scaling uses post-delay evacuation exposure $E_{\text{post}}$ in $\max(0, 1 - E_{\text{post}})$ so passive pre-delay exposure does not nullify exit reward.
  - Fresh Case 2 models trained with delay: `results/case2_qtable_relative.npy` and `results/case2_dqn_model.pt`.
  - All four primary methods evaluated under identical conditions: `results/case2_eval_results.csv`, `results/case2_comparison.png`.
- [x] **Case 3: Cross-Layout Generalization (Unseen `layouts/case2.txt`, zero retraining):**
  - Tests zero-shot spatial transfer of Case 1-trained policies on a structurally distinct $13 \times 36$ layout with 4 exits and multi-wing corridors.
  - All four primary methods evaluated without weight updates: `results/case3_cross_layout_results.csv`, `results/case3_cross_layout_comparison.png`.

### 2. Delay Calibration & Physical Interpretation
- **Step-to-Time Semantics:** Grid cell size $\approx 1.0\text{ m}$, pedestrian speed $\approx 1.0\text{ m/s} \implies 1\text{ step} \approx 1\text{ s}$.
- **Hazard Dynamics:** Sigmoid radial growth $r(t) = 20.0 / (1 + \exp(-0.05(t - 60)))$.
  - At $t=0$, $r=0.95$ (only cell $(4, 22)$ active).
  - At $t=20$, $r=2.38$ (hazard blocks central corridor, 12 cells contaminated, ~40% crowd evacuated).
  - At $t=60$, $r=10.0$ (east exit blocked at hazard $\ge 0.30$).
  - Canonical delay $t_{\text{delay}}=20$ steps (~20s) represents realistic pre-movement delay, severely challenging local path planning while maintaining route feasibility.
- **Physical Distinction:** The synthetic radial hazard model is an abstract proxy for smoke/thermal tenability, distinct from computational fluid dynamics (CFD/FDS) and physical Fractional Effective Dose (FED).

### 3. Comprehensive Test Suite
- [x] Renamed `tests/test_bonus.py` $\to$ `tests/test_extended_features.py` (39 tests total across suite).
- [x] Added tests for zero-delay backward compatibility, stationary delay accumulation, crowd advancement during delay, step budget preservation, exposure separation, four primary methods evaluation, layout smoke execution, and trained Case 1 policy zero-shot transfer without weight mutation.

---

## Explicit non-goals (to stay in scope for 8 days)
- No FDS / CFD fire simulation — synthetic hazard field only, state this
  openly in the report as a simplification.
- No full DQN requirement — tabular Q-learning is sufficient; DQN is a
  stretch goal only.
- One building layout is enough; do not build a second complex layout unless
  Phases 0–7 finish early.
- No true multi-agent RL for the crowd (NPCs are scripted, not learning) —
  state this as a limitation/future work, same as SafeStep does for MARL.
