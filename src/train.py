"""
Training entry point.

Usage:
    python -m src.train --layout layouts/case1.txt --episodes 3000
"""
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.env import SafeExitEnv
from src.agents.q_learning_agent import QLearningAgent


def smooth(data, window=50):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="layouts/case1.txt")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--agent", choices=["q_learning", "dqn"], default="q_learning")
    parser.add_argument("--state-mode", choices=["relative"], default="relative", help="Tabular state representation (relative local features)")
    parser.add_argument("--sensing-radius", type=int, default=2)
    parser.add_argument("--obs-noise-std", type=float, default=0.1)
    parser.add_argument("--obs-delay", type=int, default=2)
    parser.add_argument("--n-npcs", type=int, default=10)
    parser.add_argument("--panic-prob", type=float, default=0.0)
    parser.add_argument("--herding-prob", type=float, default=0.0)
    parser.add_argument("--pre-evacuation-delay", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--plot-out", default=None)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    out_path = args.out or ("results/qtable_relative.npy" if args.agent == "q_learning" else "results/dqn_model.pt")
    plot_path = args.plot_out or ("results/training_curve_relative.png" if args.agent == "q_learning" else "results/dqn_training_curve.png")

    env = SafeExitEnv(
        layout_path=args.layout,
        sensing_radius=args.sensing_radius,
        obs_noise_std=args.obs_noise_std,
        obs_delay=args.obs_delay,
        n_npcs=args.n_npcs,
        seed=args.seed,
        panic_prob=args.panic_prob,
        herding_prob=args.herding_prob,
        pre_evacuation_delay=args.pre_evacuation_delay,
    )

    if args.agent == "dqn":
        from src.agents.dqn_agent import DQNAgent
        agent = DQNAgent(seed=args.seed, epsilon_decay_episodes=min(1500, max(1, args.episodes // 2)))
    else:
        agent = QLearningAgent(seed=args.seed, state_mode=args.state_mode)

    episode_rewards = []
    success_history = []
    exposure_history = []

    for ep in range(args.episodes):
        agent.set_epsilon(ep)
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        episode_rewards.append(total_reward)
        success_history.append(1 if info["reached_exit"] else 0)
        exposure_history.append(info["cumulative_true_exposure"])

        if (ep + 1) % 100 == 0:
            last100_sr = np.mean(success_history[-100:]) * 100
            print(f"episode {ep+1}/{args.episodes}  "
                  f"avg_reward(last100)={np.mean(episode_rewards[-100:]):.2f}  "
                  f"success_rate(last100)={last100_sr:.0f}%  "
                  f"epsilon={agent.epsilon:.3f}")

    # --- Plot training curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Reward curve
    ax = axes[0]
    ax.plot(episode_rewards, alpha=0.2, color='#2196F3', linewidth=0.5)
    if len(episode_rewards) > 50:
        ax.plot(range(49, len(episode_rewards)), smooth(episode_rewards),
                color='#2196F3', linewidth=2, label='Smoothed (50-ep)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success rate (rolling)
    ax = axes[1]
    rolling_sr = [np.mean(success_history[max(0,i-100):i+1])*100
                  for i in range(len(success_history))]
    ax.plot(rolling_sr, color='#4CAF50', linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (rolling 100)")
    ax.set_title("Success Rate")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Exposure
    ax = axes[2]
    ax.plot(exposure_history, alpha=0.2, color='#FF5722', linewidth=0.5)
    if len(exposure_history) > 50:
        ax.plot(range(49, len(exposure_history)), smooth(exposure_history),
                color='#FF5722', linewidth=2, label='Smoothed')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative True Exposure")
    ax.set_title("Hazard Exposure per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    agent_label = "DQN" if args.agent == "dqn" else f"Q-Learning ({args.state_mode})"
    plt.suptitle(f"SafeExit-RL Training Progress - {agent_label}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training plot to {plot_path}")

    agent.save(out_path)
    if hasattr(agent, "q_table"):
        print(f"Saved Q-table ({len(agent.q_table)} states) to {out_path}")
    else:
        print(f"Saved DQN model to {out_path}")
    print(f"Final success rate (last 100): "
          f"{np.mean(success_history[-100:])*100:.0f}%")


if __name__ == "__main__":
    main()
