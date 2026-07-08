"""Schedule the ONLINE+OFFLINE (per-agent) sweep across machines.

Sibling of ``run_scheduler.py``. Same machines, pinning, benchmarks and makespan
assignment -- only the sweep differs: this schedules the apples-to-apples
comparison run by ``run_offline_online.sh`` with per-agent networks:

    per-agent nets, decentralized_ego_radio, 1M frames, 5 seeds, 4 levels,
    {ppo, dqn, sac} x {RL baseline, RL + offline term}

where the offline term is ``--human-bc`` (ppo/sac) or ``--human-cql`` (dqn). The
two arms save to separate variants (``..._pa`` vs ``..._pa_bc`` / ``..._pa_cql``),
so both run in one sweep without colliding.

Timing correction (see run_scheduler.py's model)
------------------------------------------------
``run_scheduler.py`` estimates each job as FRAMES / steady_sps, where steady_sps
is a 20 s window with the torch.compile warmup skipped, all burn-in zeroed, and
NO eval / checkpoint I/O / offline term. That is a throughput floor: the first
5M-frame sweep predicted ~24 h but took ~40 h (~1.67x) because the real wall
clock also pays, per job, for compile warmup, the step-0 + post-training eval,
periodic checkpoint writes, episodic logging, and -- over tens of hours -- GPU
clocks settling below their 20 s boost. Those are not in the benchmark, so we
fold them in with ``WALLCLOCK_FACTOR`` (multiplicative, calibrated to that 1.67x)
plus a small fixed ``PER_JOB_OVERHEAD_MIN`` for compile+eval. Re-tune both from
the observed makespan of this sweep. Because the per-agent + offline job is a
DIFFERENT code path than the online benchmark, benchmark this experiment's own
keys (RL/benchmark_offline.sh); until then we fall back to the online-only key
times ``FALLBACK_SLOWDOWN`` and warn.

Usage (from the repo root):
    python -m RL.run_scheduler_offline                     # generate schedules
    python -m RL.run_scheduler_offline --benchmark-plan timpc   # for benchmark_offline.sh
"""
import argparse
import os
import sys

# Reuse the topology, pinning, benchmark loading and assignment from the online
# scheduler -- single source of truth for machines/devices.
from RL.run_scheduler import (
    REPO_ROOT, TIME_DIR, MACHINES, DEVICE_NAMES, DEVICE_MACHINE, DEFAULT_SPS,
    device_prefix, device_flags, level_name, num_envs_for, ppo_minibatches_for,
    load_benchmarks, assign_ilp, assign_greedy,
    RUN_IF_MISSING_FN, TRAP_KILL_ALL, checkpoint_relpath,
)

# --------------------------------------------------------------------------- #
# Sweep definition -- keep in sync with run_offline_online.sh
# --------------------------------------------------------------------------- #
FRAMES = 1_000_000                       # total env frames per job (1M)
SEEDS = [1, 2, 3, 4, 5]
LEVELS = [
    "levels/test_level",
    "levels/neighborhood_level",
    "levels/island_level",
    "levels/warehouse_level",
]
ALGS = ["ppo", "dqn", "sac"]
# One config (decentralized_ego_radio) with per-agent networks; the two arms are
# the RL baseline and RL + the offline human-data term.
EGO_SIZE = 32
PER_AGENT = True
# Arms: "rl" (baseline), "offline" (constant human-BC/CQL term), "anneal" (the
# human term linearly decayed to 0 -- see --bc-anneal-frames). Default sweep is the
# original rl+offline; pass --arms anneal to schedule just the annealing arm.
ARMS_ALL = ["rl", "offline", "anneal"]
ARMS = ["rl", "offline"]

BC_COEF = 1.0                            # matches run_offline_online.sh default
BC_ANNEAL_FRAMES = int(os.environ.get("BC_ANNEAL_FRAMES", "250000"))  # anneal arm horizon

# --- Per-machine memory safety (prevent the OOM crashes) ---------------------- #
# Cap num_envs (GPU rollout VRAM scales with it) and shrink the off-policy replay
# buffer (pinned HOST RAM: ~num_frames * 2 * obs_bytes) on RAM/VRAM-tight boxes.
#   white-machine: RTX 2070 = 8 GB VRAM/GPU (tight) but 62 GB RAM -> cap envs only.
#   alienware    : 1080 Ti = 11 GB VRAM (fine) but 15 GB system RAM -> cap envs AND
#                  halve the SAC/DQN buffer so two concurrent GPUs' pinned buffers
#                  fit (2x ~2.3 GB at 100k crashed before).
# A machine absent here uses the defaults (num_envs_for, buffer_size=100000).
MEM_PROFILE = {
    "white-machine": {"num_envs_cap": 64},
    "alienware":     {"num_envs_cap": 64, "buffer_size": 50000},
}
REPLAY_ALGS = {"dqn", "sac"}             # have a --buffer-size replay buffer

# --- Wall-clock correction (see module docstring) ---
WALLCLOCK_FACTOR = float(os.environ.get("WALLCLOCK_FACTOR", "1.6"))
PER_JOB_OVERHEAD_MIN = float(os.environ.get("PER_JOB_OVERHEAD_MIN", "2.0"))
# When this experiment's own benchmark key is missing, fall back to the online
# shared-net key (decentralized_ego_radio) scaled by this, since per-agent nets
# (n_agents sequential encoder passes) + the offline term are slower per step.
FALLBACK_SLOWDOWN = float(os.environ.get("FALLBACK_SLOWDOWN", "1.4"))


def variant_for(per_agent: bool, alg: str, arm: str) -> str:
    """Saved/benchmark variant string, matching the runners' variant tagging:
    decentralized_ego_radio [+ _pa] [+ _bc/_cql for offline, + _..._anneal for anneal]."""
    v = "decentralized_ego_radio"
    if per_agent:
        v += "_pa"
    if arm in ("offline", "anneal"):
        v += "_cql" if alg == "dqn" else "_bc"
    if arm == "anneal":
        v += "_anneal"
    return v


def build_experiments(seeds=None, arms=None):
    seeds = seeds if seeds is not None else SEEDS
    arms = arms if arms is not None else ARMS
    return [
        {"level": lv, "alg": al, "seed": sd, "arm": arm, "per_agent": PER_AGENT}
        for sd in seeds
        for lv in LEVELS
        for al in ALGS
        for arm in arms
    ]


EXPERIMENTS = build_experiments()


# --------------------------------------------------------------------------- #
# Benchmarks -> per-job runtime (with the wall-clock correction)
# --------------------------------------------------------------------------- #
def steps_per_sec(bench, device, exp):
    variant = variant_for(exp["per_agent"], exp["alg"], exp["arm"])
    # The anneal arm runs the SAME per-step work as the offline arm (the BC/CQL loss
    # is still computed every step, just scaled by a decaying coef), so reuse the
    # offline benchmark timing rather than falling back.
    bench_variant = variant[:-len("_anneal")] if variant.endswith("_anneal") else variant
    key = f"{level_name(exp['level'])}_{exp['alg']}_{bench_variant}"
    entry = bench.get(device, {}).get(key)
    if entry and entry.get("steps_per_sec", 0) > 0:
        return entry["steps_per_sec"], False
    # Fallback: this experiment's key was never benchmarked -- approximate from the
    # online shared-net key (if present) slowed by FALLBACK_SLOWDOWN.
    base_key = f"{level_name(exp['level'])}_{exp['alg']}_decentralized_ego_radio"
    base = bench.get(device, {}).get(base_key)
    if base and base.get("steps_per_sec", 0) > 0:
        return base["steps_per_sec"] / FALLBACK_SLOWDOWN, True
    return DEFAULT_SPS, True


def runtime_minutes(bench, device, exp):
    """Corrected wall-clock estimate: benchmarked steady-state throughput inflated
    by WALLCLOCK_FACTOR (overheads the 20 s benchmark excludes) plus a small fixed
    per-job compile+eval cost."""
    sps, _ = steps_per_sec(bench, device, exp)
    compute_min = (FRAMES / sps) / 60.0
    return compute_min * WALLCLOCK_FACTOR + PER_JOB_OVERHEAD_MIN


def any_fallback(bench, experiments, devices):
    """True if any (device, exp) had to fall back to a non-exact benchmark key."""
    for device in devices:
        for exp in experiments:
            if steps_per_sec(bench, device, exp)[1]:
                return True
    return False


# --------------------------------------------------------------------------- #
# Job command -- mirrors run_offline_online.sh's per-alg flags
# --------------------------------------------------------------------------- #
def envs_for_device(exp, device):
    """num_envs for this job, capped by the device's machine memory profile."""
    envs = num_envs_for(exp["level"])
    cap = MEM_PROFILE.get(DEVICE_MACHINE[device], {}).get("num_envs_cap")
    return min(envs, cap) if cap else envs


def job_command(exp, device):
    prefix = device_prefix(device)
    profile = MEM_PROFILE.get(DEVICE_MACHINE[device], {})
    parts = [
        "python", "-m", f"RL.cleanrl_{exp['alg']}",
        "--run-number", str(exp["seed"]),
        "--total-timesteps", str(FRAMES),
        "--level", exp["level"],
        "--num-envs", str(envs_for_device(exp, device)),
        "--no-centralized", "--ego-view", "--ego-size", str(EGO_SIZE), "--use-radio",
    ]
    if exp["per_agent"]:
        parts.append("--per-agent-nets")
    if exp["alg"] == "ppo":
        parts += ["--num-minibatches", str(ppo_minibatches_for(exp["level"]))]
    # DQN's epsilon schedule spans the whole run in BOTH arms (it is an RL knob).
    if exp["alg"] == "dqn":
        parts += ["--exploration-timesteps", str(FRAMES)]
    # Shrink the off-policy replay buffer on RAM-tight machines (pinned host RAM).
    if exp["alg"] in REPLAY_ALGS and profile.get("buffer_size"):
        parts += ["--buffer-size", str(profile["buffer_size"])]
    if exp["arm"] in ("offline", "anneal"):
        if exp["alg"] == "dqn":
            parts += ["--human-cql"]
        else:
            parts += ["--human-bc", "--bc-coef", str(BC_COEF)]
    if exp["arm"] == "anneal":
        parts += ["--bc-anneal-frames", str(BC_ANNEAL_FRAMES)]
    flags = device_flags(device)
    if flags:
        parts += flags.split()
    return f"{prefix} {' '.join(parts)}".strip()


def job_desc(exp):
    return (f"{level_name(exp['level'])}/{exp['alg']}/"
            f"{variant_for(exp['per_agent'], exp['alg'], exp['arm'])} seed {exp['seed']}")


# --------------------------------------------------------------------------- #
# Schedule generation
# --------------------------------------------------------------------------- #
def write_machine_schedule(machine, assignment, bench, label="offline", regen_cmd="python -m RL.run_scheduler_offline"):
    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated {label.upper()} schedule for {machine} (devices concurrent, no MPS).",
        f"# Regenerate with: {regen_cmd}",
        "set -uo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
        'cd "${REPO_ROOT}" || exit 1',
        'export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"',
        'VENV="${VENV:-${REPO_ROOT}/.venv}"',
        'if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";',
        'elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then',
        '  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;',
        'fi',
        "",
    ] + TRAP_KILL_ALL + RUN_IF_MISSING_FN

    prof = MEM_PROFILE.get(machine)
    if prof:
        lines.append(f"# Memory-safe overrides for {machine}: {prof}")
        lines.append("")

    machine_est = 0.0
    active = 0
    for device in MACHINES[machine]:
        jobs = assignment.get(device, [])
        if not jobs:
            continue
        active += 1
        order = sorted(jobs, key=lambda e: -runtime_minutes(bench, device, e))
        load = sum(runtime_minutes(bench, device, e) for e in order)
        machine_est = max(machine_est, load)

        lines.append(f"# --- Device {device} [{device_prefix(device) or 'no pinning'}]: "
                     f"{len(order)} jobs, ~{load:.1f} min ---")
        lines.append("(")
        for n, exp in enumerate(order, 1):
            desc = f"[{machine}:{device}] {n}/{len(order)}: {job_desc(exp)}"
            variant = variant_for(exp["per_agent"], exp["alg"], exp["arm"])
            ckpt = checkpoint_relpath(exp["alg"], variant, exp["seed"], exp["level"])
            lines.append(f'  run_if_missing "{ckpt}" "{desc}" -- \\')
            lines.append(f'    {job_command(exp, device)}')
        lines.append(") &")
        lines.append("")

    lines.append(f'echo "Launched {active} device(s) on {machine}; waiting..."')
    lines.append("wait")
    lines.append(f'echo "All {label} jobs on {machine} complete."')
    lines.append("")

    os.makedirs(TIME_DIR, exist_ok=True)
    path = os.path.join(TIME_DIR, f"run_{machine}_{label}_experiments.sh")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)
    return path, machine_est


def generate_schedules(machines=None, seeds=None, arms=None, label=None):
    """Generate schedules. ``machines`` restricts BOTH which devices jobs may be
    assigned to AND which schedules are written (default: all machines). ``seeds``
    restricts which seeds are swept (default: all five). ``arms`` restricts which
    arms are scheduled (default rl+offline); ``label`` names the output scripts
    (``run_<machine>_<label>_experiments.sh``), defaulting to the sole arm's name
    when a single arm is selected, else 'offline'."""
    machines = machines or list(MACHINES)
    unknown = [m for m in machines if m not in MACHINES]
    if unknown:
        raise SystemExit(f"unknown machine(s) {unknown}; known: {', '.join(MACHINES)}")
    devices = [d for m in machines for d in MACHINES[m]]
    experiments = build_experiments(seeds, arms)
    if label is None:
        label = arms[0] if arms and len(arms) == 1 else "offline"
    regen = "python -m RL.run_scheduler_offline" + (f" --arms {' '.join(arms)}" if arms else "")

    bench = load_benchmarks()
    if any_fallback(bench, experiments, devices):
        print("[!] Some per-agent/offline benchmark keys are missing; those jobs are "
              f"estimated from the online key / {FALLBACK_SLOWDOWN}x. Run "
              "RL/benchmark_offline.sh on each machine for accurate timing.")
    runtime = [
        [runtime_minutes(bench, device, exp) for device in devices]
        for exp in experiments
    ]

    print(f"Assigning {len(experiments)} jobs (seeds={seeds or SEEDS}) across "
          f"{len(devices)} devices on {len(machines)} machine(s) [{', '.join(machines)}] "
          f"(wallclock_factor={WALLCLOCK_FACTOR}, +{PER_JOB_OVERHEAD_MIN:.0f} min/job).")
    assignment = assign_ilp(experiments, devices, runtime)
    if assignment is None:
        print("OR-Tools unavailable or no solution; using greedy assignment.")
        assignment = assign_greedy(experiments, devices, runtime)
    else:
        print("Optimal/feasible assignment found via OR-Tools (SCIP).")

    print("-" * 68)
    overall = 0.0
    for machine in machines:
        if not any(assignment[dev] for dev in MACHINES[machine]):
            continue
        path, est = write_machine_schedule(machine, assignment, bench, label=label, regen_cmd=regen)
        overall = max(overall, est)
        devs = ", ".join(f"{dev}:{len(assignment[dev])}" for dev in MACHINES[machine])
        print(f"{machine:<15} ~{est:7.1f} min | {devs}")
        print(f"{'':<15}   -> {os.path.relpath(path, REPO_ROOT)}")
    print("-" * 68)
    print(f"Estimated overall wall-clock makespan: {overall:.1f} min "
          f"({overall / 60.0:.1f} h)")


def cmd_benchmark_plan(machine):
    """Emit 'device|prefix|flags' lines for benchmark_offline.sh (one per device)."""
    if machine not in MACHINES:
        print(f"unknown machine '{machine}'; known: {', '.join(MACHINES)}", file=sys.stderr)
        sys.exit(2)
    for device in MACHINES[machine]:
        print(f"{device}|{device_prefix(device)}|{device_flags(device)}")


def cmd_mem_plan(machine):
    """Emit the machine's memory-safety overrides as shell-evalable assignments, so
    benchmark_offline.sh measures the SAME num_envs / buffer the real jobs use."""
    if machine not in MACHINES:
        print(f"unknown machine '{machine}'; known: {', '.join(MACHINES)}", file=sys.stderr)
        sys.exit(2)
    prof = MEM_PROFILE.get(machine, {})
    print(f"NUM_ENVS_CAP={prof.get('num_envs_cap', '')}")
    print(f"BUFFER_SIZE={prof.get('buffer_size', '')}")


def main():
    ap = argparse.ArgumentParser(description="Generate per-machine offline+online schedules.")
    ap.add_argument("--benchmark-plan", metavar="MACHINE",
                    help="print 'device|prefix|flags' lines for MACHINE and exit "
                         "(used by RL/benchmark_offline.sh)")
    ap.add_argument("--mem-plan", metavar="MACHINE",
                    help="print MACHINE's NUM_ENVS_CAP / BUFFER_SIZE overrides and exit "
                         "(used by RL/benchmark_offline.sh to match the real config)")
    ap.add_argument("--machines", nargs="+", metavar="MACHINE",
                    help="restrict assignment + schedule generation to these machines "
                         f"(default: all). Known: {', '.join(MACHINES)}")
    ap.add_argument("--seeds", nargs="+", type=int, metavar="N",
                    help="restrict the sweep to these seeds (default: 1 2 3 4 5)")
    ap.add_argument("--arms", nargs="+", metavar="ARM", choices=ARMS_ALL,
                    help=f"restrict to these arms (default: rl offline). Choices: {', '.join(ARMS_ALL)}. "
                         "e.g. --arms anneal schedules only the BC-annealing runs into "
                         "run_<machine>_anneal_experiments.sh")
    args = ap.parse_args()
    if args.benchmark_plan is not None:
        cmd_benchmark_plan(args.benchmark_plan)
        return
    if args.mem_plan is not None:
        cmd_mem_plan(args.mem_plan)
        return
    generate_schedules(machines=args.machines, seeds=args.seeds, arms=args.arms)


if __name__ == "__main__":
    main()
