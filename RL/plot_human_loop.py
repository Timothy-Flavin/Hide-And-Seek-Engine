"""Plot human-in-the-loop vs policy-only TEAM performance for the record->train loop.

Per (level, alg, run):

* solid, one per agent -- TEAM return (summed over agents, full 250-frame episode)
  on episodes where the HUMAN drove that agent, from the recorder's
  ``experiments/results/<level>/human_returns.jsonl`` ``team_return_by_driver``
  field (incl. the step-0 bootstrap with random teammates);
* dotted baseline -- TEAM return with NO human (all agents on policy), from
  ``experiments/results/<level>/<alg>/eval_returns.jsonl`` ``team_return_mean``
  (incl. the step-0 random-policy baseline).

Both keyed on cumulative training frames (parsed from the demos' checkpoint tag /
the eval's ``cumulative_step``). Regenerated after each training step by the loop
scripts.

    python -m RL.plot_human_loop --level levels/test_level --alg ppo --run 1
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from RL.human_data import level_name_of

RESULTS_ROOT = "experiments/results"


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _ckpt_to_step(ckpt):
    """Cumulative training frames a human session was recorded against."""
    if not ckpt or ckpt == "none":
        return 0
    m = re.search(r"step(\d+)", ckpt)
    return int(m.group(1)) if m else None


def _mean_std_by_x(x_to_vals):
    xs = sorted(x_to_vals)
    means = [float(np.mean(x_to_vals[x])) for x in xs]
    stds = [float(np.std(x_to_vals[x])) for x in xs]
    return xs, means, stds


def plot_level(level, alg, run=1, out_path=None, runs=None):
    """Plot for a single run (``run``) or, when ``runs`` (a list of seeds) is
    given, aggregate the dotted no-human baseline across those seeds into one
    mean line with a std band. The solid human-driven curves are seed-invariant
    (the demos are not seeded), so they are drawn the same either way."""
    run_list = [int(r) for r in (runs if runs else [run])]
    lname = level_name_of(level)
    human_path = os.path.join(RESULTS_ROOT, lname, "human_returns.jsonl")
    eval_path = os.path.join(RESULTS_ROOT, lname, alg, "eval_returns.jsonl")

    # --- Solid: team return by driver (human in the loop) ---
    # {agent_name: {step: [per-episode team returns]}}
    human_by_driver = defaultdict(lambda: defaultdict(list))
    for e in _read_jsonl(human_path):
        if e.get("teammate_alg", alg) != alg:
            continue
        step = _ckpt_to_step(e.get("checkpoint"))
        if step is None:
            continue
        for name, vals in (e.get("team_return_by_driver") or {}).items():
            human_by_driver[name][step].extend(vals)

    # --- Dotted: all-policy team return baseline (no human) ---
    # Across the requested seeds: each seed contributes one team_return_mean per
    # cumulative step, so std over the list at a step is the across-seed spread.
    run_suffixes = tuple(f"_run_{r}" for r in run_list)
    baseline = defaultdict(list)  # step -> [team_return_mean per seed]
    for e in _read_jsonl(eval_path):
        prefix = str(e.get("prefix", ""))
        if not (prefix.startswith(f"{alg}_") and any(s in prefix for s in run_suffixes)):
            continue
        step = e.get("cumulative_step")
        if step is None:
            continue
        baseline[int(step)].append(float(e.get("team_return_mean", 0.0)))

    runs_label = ",".join(str(r) for r in run_list)
    if not human_by_driver and not baseline:
        print(f"[plot] no data for {lname}/{alg} run(s) {runs_label}; skipping.")
        return None

    names = list(human_by_driver)
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        xs, means, stds = _mean_std_by_x(human_by_driver[name])
        c = colors[name]
        ax.plot(xs, means, "-o", color=c, label=f"human drives {name}")
        if any(stds):
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.12)

    if baseline:
        xs, means, stds = _mean_std_by_x(baseline)
        n_seeds = max((len(baseline[x]) for x in baseline), default=1)
        blabel = ("no human (all-policy team)" if n_seeds <= 1
                  else f"no human (all-policy team, mean of {n_seeds} seeds)")
        ax.plot(xs, means, "--k", marker="s", markerfacecolor="none", label=blabel)
        # Across-seed std band (only meaningful with >1 seed at a step).
        if n_seeds > 1 and any(stds):
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(xs, lo, hi, color="k", alpha=0.10)

    ax.set_xlabel("cumulative training frames")
    ax.set_ylabel("team return (sum over agents, 250-frame episode)")
    run_title = f"run {run_list[0]}" if len(run_list) == 1 else f"runs {runs_label}"
    ax.set_title(f"{lname} / {alg} {run_title}: team return by human-driven agent (solid) "
                 f"vs all-policy (dotted)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if out_path is None:
        run_tag = f"run{run_list[0]}" if len(run_list) == 1 else f"runs{'-'.join(str(r) for r in run_list)}"
        out_path = os.path.join(RESULTS_ROOT, lname, f"human_vs_eval_{alg}_{run_tag}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", required=True, help="level dir or name (e.g. levels/test_level)")
    ap.add_argument("--alg", default="ppo", choices=["ppo", "dqn", "sac"])
    ap.add_argument("--run", type=int, default=1, help="single seed to plot")
    ap.add_argument("--runs", type=int, nargs="+", default=None,
                    help="aggregate the dotted no-human baseline across these seeds "
                         "(mean +/- std band), e.g. --runs 1 2 3")
    ap.add_argument("--out", default=None, help="output PNG path (default: under the level dir)")
    args = ap.parse_args()
    plot_level(args.level, args.alg, run=args.run, runs=args.runs, out_path=args.out)


if __name__ == "__main__":
    main()
