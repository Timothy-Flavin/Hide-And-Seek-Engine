"""Plot offline training performance WITH vs WITHOUT human behavior cloning.

The offline schedule (``time_files/*_offline_experiments.sh``) trains, for every
level and every algorithm, two per-agent-nets variants:

* the plain RL baseline   -> ``<alg>_decentralized_ego_radio_pa``
* the human-BC variant    -> ``<alg>_decentralized_ego_radio_pa_bc``   (ppo, sac)
                             ``dqn_decentralized_ego_radio_pa_cql``     (dqn)

Each is 5 seeds (run_1..run_5) of episodic returns recorded across 1M frames of
training and merged into ``offline_results/<level>/<alg>/``. This script draws
the BC-vs-no-BC comparison: mean +/- standard error over the 5 seeds, BC as a
solid line and no-BC dashed, one colour per algorithm.

Outputs (under ``offline_results/`` by default):
* ``bc_comparison_grid.png``            -- 4 levels (rows) x 3 algos (cols)
* ``bc_comparison_<level>.png``         -- one figure per level (3 algos overlaid)

    python -m RL.plot_bc_comparison
    RESULTS_ROOT=offline_results FRAMES=1000000 python -m RL.plot_bc_comparison
"""
import os
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Offline runs are 1M frames each (see the offline schedule). Only scales the
# x-axis; curves are interpolated across each run's duration so this is cosmetic.
TOTAL_FRAMES = int(os.environ.get("FRAMES", 1_000_000))
X_MAX = TOTAL_FRAMES / 1000.0
N_INTERP = 1000
N_RUNS = 5
EMA = 0.98  # smoothing factor, matching RL/plot_results.py

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "offline_results")

# Per-agent-nets variant that the offline schedule trains.
BASE_VARIANT = "decentralized_ego_radio_pa"

# (algorithm, colour). The BC-variant suffix differs: dqn's offline/BC objective
# is CQL, ppo/sac use a BC auxiliary loss. The annealed variant appends '_anneal'
# (bc_coef / cql_coef linearly decayed to 0 -- see --bc-anneal-frames).
ALGS = [("ppo", "#1f77b4"), ("dqn", "#d62728"), ("sac", "#2ca02c")]
BC_SUFFIX = {"ppo": "bc", "sac": "bc", "dqn": "cql"}

# (condition-key, human label, variant suffix appended to BASE_VARIANT, style).
LEVELS = ["test_level", "island_level", "neighborhood_level", "warehouse_level"]


def _bc_variant(alg):
    return f"{BASE_VARIANT}_{BC_SUFFIX[alg]}"


def _anneal_variant(alg):
    return f"{BASE_VARIANT}_{BC_SUFFIX[alg]}_anneal"


def load_runs(level, alg, variant):
    """Interpolated, EMA-smoothed per-seed curves for one alg+variant.

    Returns (target_x, list-of-curves). Missing seeds are skipped; an empty list
    means nothing to plot for this cell."""
    target_x = np.linspace(0, X_MAX, N_INTERP)
    runs = []
    for seed in range(1, N_RUNS + 1):
        fn = os.path.join(RESULTS_ROOT, level, alg,
                          f"{alg}_{variant}_episodic_returns_run_{seed}.npy")
        if not os.path.exists(fn):
            continue
        returns = np.load(fn).astype(np.float64)
        if returns.size == 0:
            continue
        for i in range(1, len(returns)):  # EMA smoothing
            returns[i] = EMA * returns[i - 1] + (1.0 - EMA) * returns[i]
        x = np.linspace(0, X_MAX, len(returns))
        runs.append(np.interp(target_x, x, returns))
    return target_x, runs


def mean_se(runs):
    """(mean, standard-error) across seeds for a list of equal-length curves."""
    data = np.asarray(runs)
    mean = data.mean(axis=0)
    se = data.std(axis=0) / np.sqrt(data.shape[0])
    return mean, se


def _draw_cell(ax, level, alg, color, *, label_algo=False):
    """Draw no-BC (dashed), constant-BC (solid) and annealed-BC (dotted) for one
    (level, alg) onto ``ax``. Any variant with no runs is skipped. Returns True if
    anything was plotted."""
    target_x, no_bc = load_runs(level, alg, BASE_VARIANT)
    _, bc = load_runs(level, alg, _bc_variant(alg))
    _, bc_anneal = load_runs(level, alg, _anneal_variant(alg))
    plotted = False
    prefix = f"{alg.upper()} " if label_algo else ""
    for runs, style, lw, alpha, tag in (
        (no_bc, "--", 1.8, 0.10, "no-BC"),
        (bc, "-", 2.2, 0.15, "BC const"),
        (bc_anneal, ":", 2.4, 0.15, "BC anneal"),
    ):
        if not runs:
            continue
        m, se = mean_se(runs)
        ax.plot(target_x, m, color=color, linestyle=style, linewidth=lw,
                label=f"{prefix}{tag} (n={len(runs)})")
        ax.fill_between(target_x, m - se, m + se, color=color, alpha=alpha)
        plotted = True
    return plotted


def plot_grid(out_dir):
    """4 levels (rows) x 3 algos (cols); each cell BC vs no-BC (mean +/- SE)."""
    nrows, ncols = len(LEVELS), len(ALGS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows),
                             squeeze=False, sharex=True)
    for r, level in enumerate(LEVELS):
        for c, (alg, color) in enumerate(ALGS):
            ax = axes[r][c]
            any_plotted = _draw_cell(ax, level, alg, color)
            if r == 0:
                ax.set_title(alg.upper(), fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{level}\nEpisodic return", fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("Frames (x1000)")
            ax.grid(True, alpha=0.3)
            if any_plotted:
                ax.legend(fontsize=7, loc="lower right")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
    fig.suptitle("Offline training: human-BC vs no-BC (per-agent nets, mean +/- SE over seeds)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(out_dir, "bc_comparison_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] saved {out}")


def plot_per_level(out_dir):
    """One figure per level, all three algorithms overlaid (BC vs no-BC)."""
    for level in LEVELS:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for alg, color in ALGS:
            plotted |= _draw_cell(ax, level, alg, color, label_algo=True)
        if not plotted:
            plt.close(fig)
            print(f"[{level}] no data, skipped")
            continue
        ax.set_xlabel("Frames (x1000)")
        ax.set_ylabel("Episodic return")
        ax.set_title(f"'{level}': human-BC (solid) vs no-BC (dashed), mean +/- SE")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
        out = os.path.join(out_dir, f"bc_comparison_{level}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{level}] saved {out}")


def main():
    out_dir = RESULTS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(os.path.join(RESULTS_ROOT, "*", "*", "*_episodic_returns_run_*.npy")):
        print(f"No episodic-returns .npy files under {RESULTS_ROOT}/. Nothing to plot.")
        return
    plot_grid(out_dir)
    plot_per_level(out_dir)


if __name__ == "__main__":
    main()
