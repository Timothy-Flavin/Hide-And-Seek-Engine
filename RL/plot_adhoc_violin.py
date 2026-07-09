"""Violin plots of ad-hoc teamplay performance (RL/eval_adhoc.py output).

Each ad-hoc matrix ``offline_results/adhoc_eval/<alg>_<condition>.npz`` holds, per
level, a 5x5x5 grid of team-score means -- one per seed-mixed team composition
(role a uses seed s_a). Flattened, that is the DISTRIBUTION of ad-hoc team scores
over all 125 compositions. This draws that distribution as a violin per condition
(no-BC / BC-const / BC-anneal), so you can see how the human term reshapes ad-hoc
coordination -- not just the mean, but the spread and tails.

Layout: 4 levels (rows) x 3 algos (cols); each cell has the three condition
violins (shared y within a level row). Median line + mean marker per violin, with
the median printed above it (identity is never colour-alone -> CVD-safe).

Colours: dataviz categorical slots 1-3, validated CVD-safe
(``validate_palette.js "#2a78d6,#1baf7a,#eda100"``).

    python -m RL.plot_adhoc_violin
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ADHOC_DIR = os.path.join("offline_results", "adhoc_eval")
LEVELS = ["test_level", "island_level", "neighborhood_level", "warehouse_level"]
ALGS = ["ppo", "dqn", "sac"]
# (condition key, label, colour) -- fixed order; colours are dataviz slots 1-3.
CONDITIONS = [
    ("nobc", "no-BC", "#2a78d6"),
    ("bc", "BC-const", "#1baf7a"),
    ("anneal", "BC-anneal", "#eda100"),
]


def team_scores(alg, condition, level):
    """The 125 (finite) team-score means for one (alg, condition, level), or None."""
    path = os.path.join(ADHOC_DIR, f"{alg}_{condition}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    key = f"{level}__mean"
    if key not in z.files:
        return None
    v = z[key].ravel()
    v = v[np.isfinite(v)]
    return v if v.size else None


def _draw_cell(ax, alg, level):
    """Three condition violins for one (alg, level). Returns True if any drawn."""
    data, positions, colors, drawn = [], [], [], False
    for i, (cond, _label, color) in enumerate(CONDITIONS, start=1):
        v = team_scores(alg, cond, level)
        if v is None:
            continue
        data.append(v)
        positions.append(i)
        colors.append((cond, color, v))
        drawn = True
    if not drawn:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        return False

    parts = ax.violinplot(data, positions=positions, widths=0.8,
                          showextrema=False, showmedians=False)
    for body, (_cond, color, _v) in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    # Median line + mean marker + median value label (secondary encoding).
    for pos, (_cond, color, v) in zip(positions, colors):
        med, mean = float(np.median(v)), float(np.mean(v))
        ax.hlines(med, pos - 0.34, pos + 0.34, color=color, lw=2.2, zorder=3)
        ax.plot(pos, mean, marker="o", ms=5, color="white",
                markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        ax.annotate(f"{med:.0f}", (pos, med), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="#0b0b0b",
                    fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([CONDITIONS[p - 1][1] for p in positions], fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    return True


def main():
    if not os.path.isdir(ADHOC_DIR):
        print(f"No {ADHOC_DIR}/ -- run RL.eval_adhoc first.")
        return
    nrows, ncols = len(LEVELS), len(ALGS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows),
                             squeeze=False, sharey="row")
    any_drawn = False
    for r, level in enumerate(LEVELS):
        for c, alg in enumerate(ALGS):
            ax = axes[r][c]
            any_drawn |= _draw_cell(ax, alg, level)
            if r == 0:
                ax.set_title(alg.upper(), fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{level}\nteam score", fontsize=9)
    if not any_drawn:
        print("No ad-hoc matrices found; nothing to plot.")
        plt.close(fig)
        return

    legend = [Patch(facecolor=col, alpha=0.55, edgecolor=col, label=lab)
              for _c, lab, col in CONDITIONS]
    fig.legend(handles=legend, loc="upper center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Ad-hoc teamplay: team-score distribution over 125 seed-mixed "
                 "compositions per (level, algo)", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join("offline_results", "adhoc_violin_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[violin] saved {out}")


if __name__ == "__main__":
    main()
