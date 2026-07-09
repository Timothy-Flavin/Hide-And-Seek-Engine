"""Occupancy heatmaps from RL/record_positions.py output.

One figure per (level, algo): rows = agents (roles), columns = the three policy
regimes (no-BC / BC-const / BC-anneal) plus the human dataset. Each cell overlays
the agent's battery-alive occupancy density (log-scaled) on the dimmed level map,
so you can see *where* each role spends its time and how the human term / the human
demonstrator move the mass around.

    python -m RL.plot_position_heatmap
    python -m RL.plot_position_heatmap --levels test_level --algs ppo
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POS_DIR = os.path.join("offline_results", "positions")
LEVELS = ["test_level", "island_level", "neighborhood_level", "warehouse_level"]
ALGS = ["dqn", "ppo", "sac"]
# (column key, header). "human" is shared across algos for the same level.
COLUMNS = [("nobc", "no-BC"), ("bc", "BC-const"), ("anneal", "BC-anneal"), ("human", "human")]
CMAP = "magma"  # perceptually-uniform sequential (CVD-safe for density)


def _load(key, alg, level):
    """Return the npz for a column key, or None. 'human' ignores alg."""
    name = f"human_{level}.npz" if key == "human" else f"{key}_{alg}_{level}.npz"
    path = os.path.join(POS_DIR, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def _draw(ax, base, hist):
    """Dimmed map + log-density heatmap overlay (zero-visit cells transparent)."""
    ax.imshow(base // 3, interpolation="nearest")
    h = hist.astype(np.float64)
    if h.max() > 0:
        dens = np.log1p(h) / np.log1p(h.max())          # log-scaled 0..1
        rgba = plt.get_cmap(CMAP)(dens)
        rgba[..., 3] = np.where(h > 0, 0.85, 0.0)        # hide empty cells
        ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_level_alg(level, alg):
    cols = [(k, hdr, _load(k, alg, level)) for k, hdr in COLUMNS]
    if all(d is None for _k, _h, d in cols):
        return False
    # Agent count / names from the first available npz.
    ref = next(d for _k, _h, d in cols if d is not None)
    names = [str(x) for x in ref["agent_names"]]
    n_agents = len(names)

    fig, axes = plt.subplots(n_agents, len(cols),
                             figsize=(3.0 * len(cols), 3.0 * n_agents), squeeze=False)
    for c, (key, hdr, data) in enumerate(cols):
        for a in range(n_agents):
            ax = axes[a][c]
            if data is None or f"agent{a}_hist" not in data.files:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                base = data["base_map"]
                _draw(ax, base, data[f"agent{a}_hist"])
                alive = int(data["alive_frames_per_agent"][a])
                ax.set_xlabel(f"{alive:,} frames", fontsize=7)
            if a == 0:
                ax.set_title(hdr, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(names[a], fontsize=10)

    fig.suptitle(f"Agent occupancy heatmaps — {level} / {alg.upper()}  "
                 f"(log density, battery-alive frames)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(POS_DIR, f"heatmap_{level}_{alg}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] saved {out}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--algs", nargs="+", default=ALGS, choices=ALGS)
    args = ap.parse_args()
    if not os.path.isdir(POS_DIR):
        print(f"No {POS_DIR}/ -- run RL.record_positions first.")
        return
    any_done = False
    for level in args.levels:
        for alg in args.algs:
            any_done |= plot_level_alg(level, alg)
    if not any_done:
        print("No position npz files found; nothing to plot.")


if __name__ == "__main__":
    main()
