"""
Evaluation: runs RL agent + both baselines over many seeds/start positions
and reports comparison metrics (success rate, true cumulative exposure,
time-to-exit, time in high-crowd-density cells).

Usage:
    python -m src.evaluate --checkpoint results/qtable.npy --layout layouts/case1.txt
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.env import SafeExitEnv, ACTIONS
from src.agents.q_learning_agent import QLearningAgent
from src.agents.baseline_shortest_path import StaticShortestPathAgent
from src.agents.baseline_reactive_replan import ReactiveReplanAgent


HIGH_CROWD_THRESHOLD = 2  # cells with >= this many NPCs count as "high density"


def run_episode(env, agent, agent_type="rl"):
    """Run one episode, return metrics dict."""
    obs = env.reset()

    if agent_type == "static":
        agent.reset(env.agent_pos, obs, env.building.is_free, env.building.exits)
    elif agent_type == "reactive":
        agent.reset(env.agent_pos, obs, env.building.is_free, env.building.exits)

    done = False
    high_crowd_steps = 0

    while not done:
        if agent_type == "rl":
            action = agent.select_action(obs)
        elif agent_type == "static":
            action = agent.select_action(obs)
        elif agent_type == "reactive":
            action = agent.select_action(obs)
        else:
            raise ValueError(agent_type)

        obs, reward, done, info = env.step(action)

        if info.get("local_crowd", 0) >= HIGH_CROWD_THRESHOLD:
            high_crowd_steps += 1

    return {
        "success": info["reached_exit"],
        "true_exposure": info["cumulative_true_exposure"],
        "believed_exposure": info["cumulative_believed_exposure"],
        "steps": info["steps"],
        "high_crowd_steps": high_crowd_steps,
    }


def evaluate_agent(agent, agent_type, layout_path, n_episodes, seeds,
                   obs_noise_std=0.1, obs_delay=2, sensing_radius=2, n_npcs=10):
    """Evaluate an agent over multiple episodes/seeds, return aggregated metrics."""
    results = []
    for seed in seeds:
        env = SafeExitEnv(
            layout_path=layout_path,
            sensing_radius=sensing_radius,
            obs_noise_std=obs_noise_std,
            obs_delay=obs_delay,
            n_npcs=n_npcs,
            seed=seed,
        )
        for _ in range(n_episodes // len(seeds)):
            metrics = run_episode(env, agent, agent_type)
            results.append(metrics)
    return results


def aggregate(results):
    """Compute mean metrics from a list of episode result dicts."""
    n = len(results)
    if n == 0:
        return {"success_rate": 0, "avg_exposure": 0, "avg_steps": 0, "avg_crowd_time": 0}
    return {
        "success_rate": sum(r["success"] for r in results) / n * 100,
        "avg_exposure": np.mean([r["true_exposure"] for r in results]),
        "avg_steps": np.mean([r["steps"] for r in results]),
        "avg_crowd_time": np.mean([r["high_crowd_steps"] for r in results]),
    }


def plot_comparison(all_metrics, save_path="results/comparison.png"):
    """Bar chart comparing methods across metrics."""
    methods = list(all_metrics.keys())
    metric_names = ["success_rate", "avg_exposure", "avg_steps", "avg_crowd_time"]
    display_names = ["Success %", "True Exposure", "Steps to Exit", "High-Crowd Steps"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(methods))

    for ax, mname, dname in zip(axes, metric_names, display_names):
        vals = [all_metrics[m][mname] for m in methods]
        bars = ax.bar(x, vals, color=["#2196F3", "#FF9800", "#4CAF50"])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(dname)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/qtable.npy")
    parser.add_argument("--layout", default="layouts/case1.txt")
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--obs-noise-std", type=float, default=0.1)
    parser.add_argument("--obs-delay", type=int, default=2)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    n_ep = args.n_eval_episodes

    # 1. RL Agent
    rl_agent = QLearningAgent(epsilon_start=0.0, epsilon_end=0.0)
    rl_agent.load(args.checkpoint)
    rl_results = evaluate_agent(rl_agent, "rl", args.layout, n_ep, seeds,
                                obs_noise_std=args.obs_noise_std,
                                obs_delay=args.obs_delay)

    # 2. Static shortest-path baseline
    sp_agent = StaticShortestPathAgent()
    sp_results = evaluate_agent(sp_agent, "static", args.layout, n_ep, seeds,
                                obs_noise_std=args.obs_noise_std,
                                obs_delay=args.obs_delay)

    # 3. Reactive replanning baseline
    rr_agent = ReactiveReplanAgent(replan_interval=5)
    rr_results = evaluate_agent(rr_agent, "reactive", args.layout, n_ep, seeds,
                                obs_noise_std=args.obs_noise_std,
                                obs_delay=args.obs_delay)

    all_metrics = {
        "RL (ours)": aggregate(rl_results),
        "Static SP": aggregate(sp_results),
        "Reactive": aggregate(rr_results),
    }

    # Print table
    print(f"\n{'Method':<15} {'Success%':>10} {'Exposure':>10} {'Steps':>10} {'CrowdTime':>10}")
    print("-" * 55)
    for method, m in all_metrics.items():
        print(f"{method:<15} {m['success_rate']:>10.1f} {m['avg_exposure']:>10.2f} "
              f"{m['avg_steps']:>10.1f} {m['avg_crowd_time']:>10.1f}")

    # Plot
    plot_comparison(all_metrics)

    # Save CSV
    import csv
    with open("results/eval_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "SuccessRate", "AvgExposure", "AvgSteps", "AvgCrowdTime"])
        for method, m in all_metrics.items():
            writer.writerow([method, f"{m['success_rate']:.1f}", f"{m['avg_exposure']:.2f}",
                             f"{m['avg_steps']:.1f}", f"{m['avg_crowd_time']:.1f}"])
    print("Saved results/eval_results.csv")


if __name__ == "__main__":
    main()
