import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Total frames per run (5M). Only used to scale the x-axis; the curves are
# interpolated across each run's duration so the absolute value is cosmetic.
TOTAL_FRAMES = int(os.environ.get("FRAMES", 5_000_000))
X_MAX = TOTAL_FRAMES / 1000.0
N_INTERP = 1000
N_RUNS = 5  # runs/seeds to average over (run_1 .. run_N)

# (algorithm, color). Mode determines the line style.
ALGS = [("ppo", "blue"), ("dqn", "red"), ("sac", "green")]
# (variant suffix, human label, line style)
VARIANTS = [
    ("centralized", "Centralized", "-"),
    ("decentralized", "Decentralized", "--"),
    ("decentralized_ego", "Decentralized+Ego", ":"),
]

RESULTS_ROOT = "experiments/results"
LEVELS_ROOT = "levels"


def discover_levels():
    """Level names that have at least one result file under RESULTS_ROOT/<level>/.
    Falls back to the levels/ directory listing."""
    levels = []
    if os.path.isdir(LEVELS_ROOT):
        for name in sorted(os.listdir(LEVELS_ROOT)):
            if os.path.isdir(os.path.join(LEVELS_ROOT, name)):
                levels.append(name)
    # Include any result subdir not present in levels/ (defensive).
    if os.path.isdir(RESULTS_ROOT):
        for name in sorted(os.listdir(RESULTS_ROOT)):
            d = os.path.join(RESULTS_ROOT, name)
            if os.path.isdir(d) and name not in levels and glob.glob(os.path.join(d, "*_episodic_returns_run_*.npy")):
                levels.append(name)
    return levels


def load_variant_runs(level_name, alg, variant):
    """Return a list of interpolated, EMA-smoothed runs for one alg+variant."""
    target_x = np.linspace(0, X_MAX, N_INTERP)
    runs = []
    for run_num in range(1, N_RUNS + 1):
        fn = os.path.join(RESULTS_ROOT, level_name, f"{alg}_{variant}_episodic_returns_run_{run_num}.npy")
        if not os.path.exists(fn):
            continue
        returns = np.load(fn).astype(np.float64)
        if len(returns) == 0:
            continue
        for xi in range(1, len(returns)):  # EMA smoothing (0.98)
            returns[xi] = 0.98 * returns[xi - 1] + 0.02 * returns[xi]
        x = np.linspace(0, X_MAX, len(returns))
        runs.append(np.interp(target_x, x, returns))
    return target_x, runs


def plot_level(level_name):
    target_x = np.linspace(0, X_MAX, N_INTERP)
    plt.figure(figsize=(10, 6))
    plotted_any = False

    for alg, color in ALGS:
        for variant, vlabel, linestyle in VARIANTS:
            _, runs = load_variant_runs(level_name, alg, variant)
            if not runs:
                continue
            plotted_any = True
            data = np.array(runs)
            mean_returns = data.mean(axis=0)
            ste = data.std(axis=0) / np.sqrt(data.shape[0])
            label = f"{alg.upper()} {vlabel}"
            plt.plot(target_x, mean_returns, label=label, color=color, linestyle=linestyle, linewidth=2)
            plt.fill_between(target_x, mean_returns - ste, mean_returns + ste, color=color, alpha=0.15)

    if not plotted_any:
        plt.close()
        print(f"[{level_name}] no result files found, skipping.")
        return

    plt.xlabel("Frames (x1000)")
    plt.ylabel("Episodic Return")
    plt.title(f"Algorithm Comparison on '{level_name}' (Mean +/- Standard Error, {TOTAL_FRAMES/1e6:.0f}M frames)")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    out_dir = os.path.join(RESULTS_ROOT, level_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "combined_learning_curves.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{level_name}] saved combined plot to {out_file}")


def plot_results():
    levels = discover_levels()
    if not levels:
        print("No levels found.")
        return
    for level_name in levels:
        plot_level(level_name)


if __name__ == "__main__":
    plot_results()
