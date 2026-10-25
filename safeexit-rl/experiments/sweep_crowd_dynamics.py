"""
Crowd Dynamics Sweep Experiment.

Measures the impact of crowd panic and herding behaviors on egress performance
across learned RL policies (Tabular Q, DQN) and classical baselines.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.baseline_shortest_path import StaticShortestPathAgent
from src.agents.baseline_reactive_replan import ReactiveReplanAgent
from src.evaluate import evaluate_agent, aggregate


CROWD_CONDITIONS = [
    ("Default (Independent)", 0.0, 0.0),
    ("Panic (p=0.2)", 0.2, 0.0),
    ("Panic (p=0.5)", 0.5, 0.0),
    ("Herding (p=0.3)", 0.0, 0.3),
    ("Combined (Panic 0.3 + Herd 0.3)", 0.3, 0.3),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="layouts/case1.txt")
    parser.add_argument("--checkpoint", default="results/qtable.npy")
    parser.add_argument("--dqn-checkpoint", default="results/dqn_model.pt")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--out-csv", default="results/crowd_dynamics_results.csv")
    parser.add_argument("--out-plot", default="results/crowd_dynamics_comparison.png")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    seeds = list(range(args.n_seeds))
    n_ep = args.n_eval_episodes

    agents = []
    if os.path.exists(args.checkpoint):
        rl_agent = QLearningAgent()
        rl_agent.load(args.checkpoint)
        agents.append(("Tabular Q", rl_agent, "rl"))

    if os.path.exists(args.dqn_checkpoint):
        dqn_agent = DQNAgent()
        dqn_agent.load(args.dqn_checkpoint)
        agents.append(("DQN", dqn_agent, "dqn"))

    agents.append(("Static SP", StaticShortestPathAgent(), "static"))
    agents.append(("Reactive", ReactiveReplanAgent(replan_interval=5), "reactive"))

    results_table = []
    print(f"=== Evaluating Crowd Dynamics on {args.layout} ===")

    for cond_name, p_panic, p_herd in CROWD_CONDITIONS:
        print(f"\nCondition: {cond_name} (panic={p_panic}, herding={p_herd})")
        for agent_name, agent, agent_type in agents:
            res = evaluate_agent(
                agent=agent,
                agent_type=agent_type,
                layout_path=args.layout,
                n_episodes=n_ep,
                seeds=seeds,
                panic_prob=p_panic,
                herding_prob=p_herd,
            )
            agg = aggregate(res)
            print(f"  {agent_name:<12}: SR={agg['success_rate']:>5.1f}% "
                  f"Exp={agg['avg_exposure']:>5.2f} Steps={agg['avg_steps']:>5.1f} "
                  f"CrowdTime={agg['avg_crowd_time']:>4.1f}")
            results_table.append({
                "Condition": cond_name,
                "PanicProb": p_panic,
                "HerdingProb": p_herd,
                "Method": agent_name,
                **agg,
            })

    # Save CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "PanicProb", "HerdingProb", "Method",
                         "SuccessRate", "AvgExposure", "AvgSteps", "AvgCrowdTime"])
        for row in results_table:
            writer.writerow([
                row["Condition"], row["PanicProb"], row["HerdingProb"], row["Method"],
                f"{row['success_rate']:.1f}", f"{row['avg_exposure']:.2f}",
                f"{row['avg_steps']:.1f}", f"{row['avg_crowd_time']:.1f}"
            ])
    print(f"\nSaved CSV to {args.out_csv}")

    # Plot Grouped Bar Chart of CrowdTime and Exposure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cond_labels = [c[0] for c in CROWD_CONDITIONS]
    agent_names = [a[0] for a in agents]
    x = np.arange(len(cond_labels))
    width = 0.8 / len(agent_names)

    # Panel 1: High Crowd Time
    ax = axes[0]
    for i, aname in enumerate(agent_names):
        vals = [r["avg_crowd_time"] for r in results_table if r["Method"] == aname]
        ax.bar(x + i * width - 0.4 + width/2, vals, width, label=aname)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8, rotation=20)
    ax.set_ylabel("Steps in High Crowd (>=2 NPCs)")
    ax.set_title("Crowd Congestion Under Panic & Herding")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Exposure
    ax = axes[1]
    for i, aname in enumerate(agent_names):
        vals = [r["avg_exposure"] for r in results_table if r["Method"] == aname]
        ax.bar(x + i * width - 0.4 + width/2, vals, width, label=aname)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8, rotation=20)
    ax.set_ylabel("Cumulative True Hazard Exposure")
    ax.set_title("Hazard Exposure Under Crowd Perturbations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Impact of Crowd Behavioral Extensions (Panic & Herding)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    plt.close()
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
