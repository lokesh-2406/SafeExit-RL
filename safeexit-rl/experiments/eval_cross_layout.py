"""
Cross-Layout Transfer Experiment.

Evaluates policies trained on case1.txt when deployed directly to case2.txt
WITHOUT any retraining or weight updates, compared with classical baselines.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env import SafeExitEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.baseline_shortest_path import StaticShortestPathAgent
from src.agents.baseline_reactive_replan import ReactiveReplanAgent
from src.evaluate import evaluate_agent, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="layouts/case2.txt")
    parser.add_argument("--coord-checkpoint", default="results/qtable.npy")
    parser.add_argument("--rel-checkpoint", default="results/qtable_relative.npy")
    parser.add_argument("--dqn-checkpoint", default="results/dqn_model.pt")
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--obs-noise-std", type=float, default=0.1)
    parser.add_argument("--obs-delay", type=int, default=2)
    parser.add_argument("--out-csv", default="results/cross_layout_results.csv")
    parser.add_argument("--out-plot", default="results/cross_layout_comparison.png")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    seeds = list(range(args.n_seeds))
    n_ep = args.n_eval_episodes
    all_metrics = {}

    print(f"=== Cross-Layout Evaluation on {args.layout} (Trained on case1.txt) ===")

    # 1. Tabular Q (Coordinate-based)
    if os.path.exists(args.coord_checkpoint):
        agent_coord = QLearningAgent(state_mode="coordinate", unseen_action_fallback="arbitrary")
        agent_coord.load(args.coord_checkpoint)
        res_coord = evaluate_agent(agent_coord, "rl", args.layout, n_ep, seeds,
                                  obs_noise_std=args.obs_noise_std, obs_delay=args.obs_delay)
        all_metrics["Tabular Q (Coord)"] = aggregate(res_coord)

    # 2. Tabular Q (Relative-state)
    if os.path.exists(args.rel_checkpoint):
        agent_rel = QLearningAgent(state_mode="relative", unseen_action_fallback="exit_bearing")
        agent_rel.load(args.rel_checkpoint)
        res_rel = evaluate_agent(agent_rel, "rl", args.layout, n_ep, seeds,
                                obs_noise_std=args.obs_noise_std, obs_delay=args.obs_delay)
        all_metrics["Tabular Q (Rel)"] = aggregate(res_rel)

    # 3. DQN (14-dim Relative)
    if os.path.exists(args.dqn_checkpoint):
        agent_dqn = DQNAgent()
        agent_dqn.load(args.dqn_checkpoint)
        res_dqn = evaluate_agent(agent_dqn, "dqn", args.layout, n_ep, seeds,
                                obs_noise_std=args.obs_noise_std, obs_delay=args.obs_delay)
        all_metrics["DQN (Rel 14D)"] = aggregate(res_dqn)

    # 4. Static Shortest-Path Baseline
    sp_agent = StaticShortestPathAgent()
    res_sp = evaluate_agent(sp_agent, "static", args.layout, n_ep, seeds,
                            obs_noise_std=args.obs_noise_std, obs_delay=args.obs_delay)
    all_metrics["Static SP"] = aggregate(res_sp)

    # 5. Reactive Replanning Baseline
    rr_agent = ReactiveReplanAgent(replan_interval=5)
    res_rr = evaluate_agent(rr_agent, "reactive", args.layout, n_ep, seeds,
                            obs_noise_std=args.obs_noise_std, obs_delay=args.obs_delay)
    all_metrics["Reactive"] = aggregate(res_rr)

    # Print Table
    print(f"\n{'Method':<20} {'Success%':>10} {'Exposure':>10} {'Steps':>10} {'CrowdTime':>10}")
    print("-" * 65)
    for method, m in all_metrics.items():
        print(f"{method:<20} {m['success_rate']:>10.1f} {m['avg_exposure']:>10.2f} "
              f"{m['avg_steps']:>10.1f} {m['avg_crowd_time']:>10.1f}")

    # Plot Comparison
    methods = list(all_metrics.keys())
    metric_names = ["success_rate", "avg_exposure", "avg_steps", "avg_crowd_time"]
    display_names = ["Success %", "True Exposure", "Steps to Exit", "High-Crowd Steps"]
    palette = ["#F44336", "#2196F3", "#9C27B0", "#FF9800", "#4CAF50"]
    colors = [palette[i % len(palette)] for i in range(len(methods))]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    x = np.arange(len(methods))

    for ax, mname, dname in zip(axes, metric_names, display_names):
        vals = [all_metrics[m][mname] for m in methods]
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8, rotation=20)
        ax.set_title(dname)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    plt.suptitle("Cross-Layout Zero-Shot Transfer on Case 2 (Trained on Case 1)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    plt.close()
    print(f"Saved {args.out_plot}")

    # Save CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "SuccessRate", "AvgExposure", "AvgSteps", "AvgCrowdTime"])
        for method, m in all_metrics.items():
            writer.writerow([method, f"{m['success_rate']:.1f}", f"{m['avg_exposure']:.2f}",
                             f"{m['avg_steps']:.1f}", f"{m['avg_crowd_time']:.1f}"])
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
