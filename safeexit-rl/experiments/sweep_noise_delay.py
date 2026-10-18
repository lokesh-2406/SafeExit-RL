"""
Headline experiment: sweep sensor noise (sigma) and observation delay
(tau), measuring how success rate and true exposure degrade for the
RL agent vs. both baselines.

This result doesn't exist in SafeStep (which assumes near-perfect sensing).

Usage:
    python -m experiments.sweep_noise_delay --checkpoint results/qtable.npy
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.env import SafeExitEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.baseline_shortest_path import StaticShortestPathAgent
from src.agents.baseline_reactive_replan import ReactiveReplanAgent
from src.evaluate import run_episode


SIGMA_VALUES = [0.0, 0.1, 0.2, 0.3]
DELAY_VALUES = [0, 2, 5, 10]
EPISODES_PER_COMBO = 50
SEEDS = [0, 1, 2]


def evaluate_combo(agent, agent_type, layout, sigma, delay, n_episodes, seeds):
    """Run agent for a specific (sigma, delay) combo, return avg metrics."""
    successes = 0
    exposures = []
    steps_list = []
    total_eps = 0

    for seed in seeds:
        env = SafeExitEnv(
            layout_path=layout,
            sensing_radius=2,
            obs_noise_std=sigma,
            obs_delay=delay,
            n_npcs=10,
            seed=seed,
        )
        eps_per_seed = max(1, n_episodes // len(seeds))
        for _ in range(eps_per_seed):
            m = run_episode(env, agent, agent_type)
            successes += int(m["success"])
            exposures.append(m["true_exposure"])
            steps_list.append(m["steps"])
            total_eps += 1

    return {
        "success_rate": successes / max(total_eps, 1) * 100,
        "avg_exposure": np.mean(exposures) if exposures else 0,
        "avg_steps": np.mean(steps_list) if steps_list else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/qtable.npy")
    parser.add_argument("--layout", default="layouts/case1.txt")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Agents
    rl_agent = QLearningAgent(epsilon_start=0.0, epsilon_end=0.0)
    rl_agent.load(args.checkpoint)
    sp_agent = StaticShortestPathAgent()
    rr_agent = ReactiveReplanAgent(replan_interval=5)

    agents = [
        ("RL (ours)", rl_agent, "rl"),
        ("Static SP", sp_agent, "static"),
        ("Reactive", rr_agent, "reactive"),
    ]
    colors = {"RL (ours)": "#2196F3", "Static SP": "#FF9800", "Reactive": "#4CAF50"}
    markers = {"RL (ours)": "o", "Static SP": "s", "Reactive": "^"}

    # --- Sweep sigma (fix delay=2) ---
    print("Sweeping sigma (delay=2)...")
    sigma_results = {name: {"sr": [], "exp": []} for name, _, _ in agents}
    for sigma in SIGMA_VALUES:
        for name, agent, atype in agents:
            m = evaluate_combo(agent, atype, args.layout, sigma, 2,
                               EPISODES_PER_COMBO, SEEDS)
            sigma_results[name]["sr"].append(m["success_rate"])
            sigma_results[name]["exp"].append(m["avg_exposure"])
            print(f"  sigma={sigma:.1f} {name}: SR={m['success_rate']:.1f}% "
                  f"Exp={m['avg_exposure']:.2f}")

    # --- Sweep delay (fix sigma=0.1) ---
    print("Sweeping delay (sigma=0.1)...")
    delay_results = {name: {"sr": [], "exp": []} for name, _, _ in agents}
    for delay in DELAY_VALUES:
        for name, agent, atype in agents:
            m = evaluate_combo(agent, atype, args.layout, 0.1, delay,
                               EPISODES_PER_COMBO, SEEDS)
            delay_results[name]["sr"].append(m["success_rate"])
            delay_results[name]["exp"].append(m["avg_exposure"])
            print(f"  delay={delay} {name}: SR={m['success_rate']:.1f}% "
                  f"Exp={m['avg_exposure']:.2f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: success rate vs sigma
    ax = axes[0, 0]
    for name, _, _ in agents:
        ax.plot(SIGMA_VALUES, sigma_results[name]["sr"],
                marker=markers[name], color=colors[name], label=name)
    ax.set_xlabel("Sensor Noise σ")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs. Noise (delay=2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: exposure vs sigma
    ax = axes[0, 1]
    for name, _, _ in agents:
        ax.plot(SIGMA_VALUES, sigma_results[name]["exp"],
                marker=markers[name], color=colors[name], label=name)
    ax.set_xlabel("Sensor Noise σ")
    ax.set_ylabel("True Cumulative Exposure")
    ax.set_title("Hazard Exposure vs. Noise (delay=2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: success rate vs delay
    ax = axes[1, 0]
    for name, _, _ in agents:
        ax.plot(DELAY_VALUES, delay_results[name]["sr"],
                marker=markers[name], color=colors[name], label=name)
    ax.set_xlabel("Observation Delay (steps)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs. Delay (σ=0.1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: exposure vs delay
    ax = axes[1, 1]
    for name, _, _ in agents:
        ax.plot(DELAY_VALUES, delay_results[name]["exp"],
                marker=markers[name], color=colors[name], label=name)
    ax.set_xlabel("Observation Delay (steps)")
    ax.set_ylabel("True Cumulative Exposure")
    ax.set_title("Hazard Exposure vs. Delay (σ=0.1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Noise/Delay Sensitivity Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/sweep_noise_delay.png", dpi=150)
    plt.close()
    print("Saved results/sweep_noise_delay.png")

    # Save CSV
    import csv
    with open("results/sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sweep", "Param", "Method", "SuccessRate", "AvgExposure"])
        for i, sigma in enumerate(SIGMA_VALUES):
            for name, _, _ in agents:
                writer.writerow(["sigma", sigma, name,
                                 f"{sigma_results[name]['sr'][i]:.1f}",
                                 f"{sigma_results[name]['exp'][i]:.2f}"])
        for i, delay in enumerate(DELAY_VALUES):
            for name, _, _ in agents:
                writer.writerow(["delay", delay, name,
                                 f"{delay_results[name]['sr'][i]:.1f}",
                                 f"{delay_results[name]['exp'][i]:.2f}"])
    print("Saved results/sweep_results.csv")


if __name__ == "__main__":
    main()
