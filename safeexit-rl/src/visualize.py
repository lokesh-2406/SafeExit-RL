"""
Rendering helpers for SafeExit-RL.

- render_text: ASCII grid for debugging (Phase 1-2)
- render_snapshot: matplotlib figure with hazard heatmap, crowd overlay,
  agent/exit markers (Phase 6, used in report figures)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def render_text(building, agent_pos, hazard_grid=None) -> str:
    """Quick ASCII render for debugging during Phase 1-2."""
    rows = []
    for r, row in enumerate(building.grid_chars):
        line = []
        for c, ch in enumerate(row):
            if (r, c) == agent_pos:
                line.append('A')
            else:
                line.append(ch)
        rows.append(''.join(line))
    return '\n'.join(rows)


def render_snapshot(building, agent_pos, true_hazard_grid, crowd_grid,
                    save_path: str, belief_grid=None, title=None, t=None):
    """Matplotlib figure: hazard heatmap + crowd overlay + agent marker.

    Args:
        building: GridBuilding instance
        agent_pos: (row, col) of the agent
        true_hazard_grid: 2D numpy array of true hazard values [0, 1]
        crowd_grid: 2D numpy array of NPC counts per cell
        save_path: path to save the figure
        belief_grid: optional 2D array of agent's believed hazard
        title: optional title string
        t: optional timestep for annotation
    """
    n_panels = 2 if belief_grid is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)

    # Custom fire colormap
    fire_cmap = LinearSegmentedColormap.from_list(
        "fire", ["#1a1a2e", "#e94560", "#ff6600", "#ffcc00", "#ffffff"])

    # Build wall mask
    wall_mask = np.zeros(building.shape, dtype=bool)
    for r in range(building.shape[0]):
        for c in range(building.shape[1]):
            if building.grid_chars[r][c] == '#':
                wall_mask[r, c] = True

    # --- Panel 1: True hazard ---
    ax = axes[0]
    haz_display = np.copy(true_hazard_grid)
    haz_display[wall_mask] = np.nan
    im = ax.imshow(haz_display, cmap=fire_cmap, vmin=0, vmax=1,
                   interpolation='nearest', aspect='equal')
    # Overlay walls in dark gray
    wall_overlay = np.full(building.shape + (4,), 0.0)
    wall_overlay[wall_mask] = [0.2, 0.2, 0.2, 1.0]
    ax.imshow(wall_overlay, aspect='equal')

    # Mark exits
    for ex in building.exits:
        ax.plot(ex[1], ex[0], 's', color='#00ff88', markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, label='Exit')
    # Mark agent
    ax.plot(agent_pos[1], agent_pos[0], '*', color='cyan', markersize=14,
            markeredgecolor='white', markeredgewidth=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hazard')
    ax.set_title("True Hazard Field")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 2: Crowd density ---
    ax = axes[1]
    crowd_display = crowd_grid.astype(float)
    crowd_display[wall_mask] = np.nan
    im2 = ax.imshow(crowd_display, cmap='YlOrRd', vmin=0,
                    vmax=max(crowd_grid.max(), 1),
                    interpolation='nearest', aspect='equal')
    wall_overlay2 = np.full(building.shape + (4,), 0.0)
    wall_overlay2[wall_mask] = [0.2, 0.2, 0.2, 1.0]
    ax.imshow(wall_overlay2, aspect='equal')
    for ex in building.exits:
        ax.plot(ex[1], ex[0], 's', color='#00ff88', markersize=10,
                markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(agent_pos[1], agent_pos[0], '*', color='cyan', markersize=14,
            markeredgecolor='white', markeredgewidth=1)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='NPC Count')
    ax.set_title("Crowd Density")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 3 (optional): Agent belief ---
    if belief_grid is not None:
        ax = axes[2]
        bel_display = np.copy(belief_grid)
        bel_display[wall_mask] = np.nan
        im3 = ax.imshow(bel_display, cmap=fire_cmap, vmin=0, vmax=1,
                        interpolation='nearest', aspect='equal')
        wall_overlay3 = np.full(building.shape + (4,), 0.0)
        wall_overlay3[wall_mask] = [0.2, 0.2, 0.2, 1.0]
        ax.imshow(wall_overlay3, aspect='equal')
        for ex in building.exits:
            ax.plot(ex[1], ex[0], 's', color='#00ff88', markersize=10,
                    markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(agent_pos[1], agent_pos[0], '*', color='cyan', markersize=14,
                markeredgecolor='white', markeredgewidth=1)
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Believed Hazard')
        ax.set_title("Agent Belief (noisy/delayed)")
        ax.set_xticks([])
        ax.set_yticks([])

    suptitle = title or "SafeExit-RL Environment Snapshot"
    if t is not None:
        suptitle += f" (t={t})"
    plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import os
    from src.env import SafeExitEnv
    os.makedirs("results", exist_ok=True)
    env = SafeExitEnv("layouts/case1.txt", seed=42)
    obs = env.reset(start_pos=(1, 1))

    # Advance environment a few steps so fire and crowd evolve
    for _ in range(40):
        env.step("wait")

    true_haz = env.hazard.full_grid(env.t)
    crowd_grid = env.crowd.density_grid()

    # Create belief grid
    belief_grid = np.zeros(env.building.shape, dtype=float)
    for (r, c), val in env.belief_state.items():
        belief_grid[r, c] = val

    save_path = "results/env_snapshot.png"
    render_snapshot(
        building=env.building,
        agent_pos=env.agent_pos,
        true_hazard_grid=true_haz,
        crowd_grid=crowd_grid,
        belief_grid=belief_grid,
        save_path=save_path,
        t=env.t,
    )
    print(f"Saved snapshot to {save_path}")


if __name__ == "__main__":
    main()
