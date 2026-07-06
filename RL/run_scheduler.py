"""Split the experiment sweep across machines using benchmarked throughput.

Reads the per-device steady-state benchmarks written by ``benchmark.sh`` /
``--benchmark`` (``time_files/<device>.json``), estimates how long each
``level x alg x config x seed`` job takes on each device, then assigns jobs to
devices to minimise the wall-clock makespan. One auto-generated bash schedule is
written per physical machine into ``time_files/run_<machine>_experiments.sh``.

Model
-----
* A **device** is one scheduling atom: a single physical GPU, or a CPU NUMA
  node. Each device runs exactly one job at a time (no CUDA MPS) and has its own
  benchmark file ``time_files/<device>.json``.
* A **physical machine** groups the devices that live in the same box and run
  *concurrently*. Two-GPU boxes have two GPU devices; lab-comp has a GPU device
  (on the GPU-affine NUMA node) plus a CPU device (on the other node, its own
  memory controller). The generated run script launches all of a machine's
  devices at once, each pinned with ``taskset``/``numactl`` so they don't
  contend, exactly as they were benchmarked.

Because the devices on a machine run together during both benchmarking and the
real sweep, the recorded steps/sec already reflect the true memory-bandwidth /
cache contention of the concurrent configuration.

Usage (from the repo root):
    python -m RL.run_scheduler                        # generate schedules
    python -m RL.run_scheduler --benchmark-plan lab-comp   # (used by benchmark.sh)
    ./time_files/run_<machine>_experiments.sh         # on each machine
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIME_DIR = os.path.join(REPO_ROOT, "time_files")

# --------------------------------------------------------------------------- #
# Sweep definition -- keep in sync with RL/run_experiments.sh and benchmark.sh
# --------------------------------------------------------------------------- #
FRAMES = 5_000_000                       # total env frames per job
SEEDS = [1, 2, 3, 4, 5]
LEVELS = [
    "levels/test_level",
    "levels/neighborhood_level",
    "levels/island_level",
    "levels/warehouse_level",
]
ALGS = ["ppo", "dqn", "sac"]
CONFIGS = ["decentralized_ego", "decentralized_ego_radio"]
EGO_SIZE = 32

# Per-map tuning (the big neighborhood map uses fewer envs / more minibatches).
NUM_ENVS_DEFAULT = 128
NUM_ENVS_OVERRIDES = {"neighborhood": 64}
PPO_MINIBATCHES_DEFAULT = 8
PPO_MINIBATCHES_OVERRIDES = {"neighborhood": 16}

# --------------------------------------------------------------------------- #
# Topology: physical machine -> {device: pinning}. No MPS; one job per device.
# All devices of a machine run concurrently (benchmark and real sweep alike).
#   cuda : CUDA_VISIBLE_DEVICES value; omit to inherit the environment's value
#          (right for single-GPU boxes and shared/allocated nodes). Set it only
#          to split the GPUs on a multi-GPU box.
#   omp  : OMP_NUM_THREADS (match the pinned logical-core count)
#   wrap : CPU/NUMA pinning wrapper (taskset / numactl ...)
#   cpu  : True -> CPU-only device (runner gets --no-cuda)
# NOTE: even a CPU device must keep a GPU visible -- the env/replay buffers
# allocate pinned host memory (pin_memory=True), which needs a live CUDA
# context. --no-cuda keeps all *compute* on the CPU; do not hide the GPU.
# Core lists mirror each box's original taskset/numactl layout.
# --------------------------------------------------------------------------- #
MACHINES = {
    # Single GPU, single NUMA node -> one device, inherit CUDA_VISIBLE_DEVICES.
    "timpc": {
        "timpc_gpu": {},
    },
    # lab-comp: GPU device on the GPU-affine NUMA node (node 1) + CPU device on
    # node 0 (its own memory controller). They run side by side. Both inherit the
    # visible GPU (the CPU device needs it for pinned memory; --no-cuda keeps its
    # compute on the CPU cores).
    # "lab-comp": {
    #     "lab-comp_gpu": {"omp": 8, "wrap": "numactl --preferred=1 taskset -c 16-31,48-63"},
    #     "lab-comp_cpu": {"omp": 16, "wrap": "numactl --preferred=0 taskset -c 0-15,32-47", "cpu": True},
    # },
    # Two GPUs: split the logical cores so each GPU is fed by its own half.
    "white-machine": {
        "white-machine_gpu0": {"cuda": "0", "omp": 8, "wrap": "taskset -c 0-3,8-11"},
        "white-machine_gpu1": {"cuda": "1", "omp": 8, "wrap": "taskset -c 4-7,12-15"},
    },
    "alienware": {
        "alienware_gpu0": {"cuda": "0", "omp": 4, "wrap": "taskset -c 0,1,4,5"},
        "alienware_gpu1": {"cuda": "1", "omp": 4, "wrap": "taskset -c 2,3,6,7"},
    },
}

# Flattened device views.
DEVICES = {dev: cfg for machine in MACHINES for dev, cfg in MACHINES[machine].items()}
DEVICE_NAMES = list(DEVICES.keys())
DEVICE_MACHINE = {dev: machine for machine in MACHINES for dev in MACHINES[machine]}

DEFAULT_SPS = 200.0  # fallback when a benchmark entry is missing


def device_prefix(device):
    """Env + CPU-pinning prefix for a device, e.g.
    ``OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11``.
    Contains no quotes, so it is safe both pasted into a generated script and
    expanded unquoted from a shell variable (benchmark.sh wraps it in `env`)."""
    cfg = DEVICES[device]
    parts = []
    if "omp" in cfg:
        parts.append(f"OMP_NUM_THREADS={cfg['omp']}")
    if "cuda" in cfg:
        parts.append(f"CUDA_VISIBLE_DEVICES={cfg['cuda']}")
    if cfg.get("wrap"):
        parts.append(cfg["wrap"])
    return " ".join(parts)


def device_flags(device):
    """Extra runner flags for a device (CPU devices force --no-cuda)."""
    return "--no-cuda" if DEVICES[device].get("cpu") else ""


# --------------------------------------------------------------------------- #
# Sweep helpers
# --------------------------------------------------------------------------- #
def level_name(level):
    return os.path.basename(os.path.normpath(level))


def _override(overrides, level, default):
    ln = level_name(level)
    for token, value in overrides.items():
        if token in ln:
            return value
    return default


def num_envs_for(level):
    return _override(NUM_ENVS_OVERRIDES, level, NUM_ENVS_DEFAULT)


def ppo_minibatches_for(level):
    return _override(PPO_MINIBATCHES_OVERRIDES, level, PPO_MINIBATCHES_DEFAULT)


def config_flags(config):
    return {
        "centralized": "--centralized",
        "decentralized": "--no-centralized",
        "decentralized_radio": "--no-centralized --use-radio",
        "decentralized_ego": f"--no-centralized --ego-view --ego-size {EGO_SIZE}",
        "decentralized_ego_radio": f"--no-centralized --ego-view --ego-size {EGO_SIZE} --use-radio",
    }[config]


EXPERIMENTS = [
    {"level": lv, "alg": al, "config": cf, "seed": sd}
    for sd in SEEDS
    for lv in LEVELS
    for al in ALGS
    for cf in CONFIGS
]


# --------------------------------------------------------------------------- #
# Benchmarks -> per-job runtime
# --------------------------------------------------------------------------- #
def load_benchmarks():
    data = {}
    for device in DEVICE_NAMES:
        path = os.path.join(TIME_DIR, device + ".json")
        try:
            with open(path) as f:
                data[device] = json.load(f)
        except (OSError, ValueError):
            print(f"[!] Missing/invalid benchmark for '{device}': {path} "
                  f"(jobs there fall back to {DEFAULT_SPS} sps)")
            data[device] = {}
    return data


def steps_per_sec(bench, device, exp):
    key = f"{level_name(exp['level'])}_{exp['alg']}_{exp['config']}"
    entry = bench.get(device, {}).get(key)
    if not entry:
        return DEFAULT_SPS
    sps = entry.get("steps_per_sec", DEFAULT_SPS)
    return sps if sps and sps > 0 else DEFAULT_SPS


def runtime_minutes(bench, device, exp):
    return (FRAMES / steps_per_sec(bench, device, exp)) / 60.0


# --------------------------------------------------------------------------- #
# Assignment: minimise makespan across devices (each device runs one job at a time)
# --------------------------------------------------------------------------- #
def assign_ilp(experiments, devices, runtime):
    """Exact assignment via OR-Tools; returns {device: [exp, ...]} or None."""
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        return None

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        return None

    n, m = len(experiments), len(devices)
    x = {(i, j): solver.IntVar(0, 1, f"x_{i}_{j}") for i in range(n) for j in range(m)}
    makespan = solver.NumVar(0, solver.infinity(), "makespan")

    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(m)) == 1)
    for j in range(m):
        solver.Add(sum(runtime[i][j] * x[i, j] for i in range(n)) <= makespan)

    solver.Minimize(makespan)
    solver.SetTimeLimit(120000)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

    assignment = {device: [] for device in devices}
    for i, exp in enumerate(experiments):
        for j, device in enumerate(devices):
            if x[i, j].solution_value() > 0.5:
                assignment[device].append(exp)
                break
    return assignment


def assign_greedy(experiments, devices, runtime):
    """Fallback: longest-job-first onto the least-loaded device (list scheduling)."""
    order = sorted(range(len(experiments)), key=lambda i: -min(runtime[i]))
    load = {device: 0.0 for device in devices}
    assignment = {device: [] for device in devices}
    for i in order:
        best_j = min(range(len(devices)), key=lambda j: load[devices[j]] + runtime[i][j])
        device = devices[best_j]
        assignment[device].append(experiments[i])
        load[device] += runtime[i][best_j]
    return assignment


# --------------------------------------------------------------------------- #
# Schedule generation
# --------------------------------------------------------------------------- #
def job_command(exp, device):
    prefix = device_prefix(device)
    parts = [
        "python", "-m", f"RL.cleanrl_{exp['alg']}",
        "--run-number", str(exp["seed"]),
        "--total-timesteps", str(FRAMES),
        "--level", exp["level"],
        "--num-envs", str(num_envs_for(exp["level"])),
    ]
    if exp["alg"] == "ppo":
        parts += ["--num-minibatches", str(ppo_minibatches_for(exp["level"]))]
    parts += config_flags(exp["config"]).split()
    flags = device_flags(device)
    if flags:
        parts += flags.split()
    cmd = " ".join(parts)
    return f"{prefix} {cmd}".strip()


def write_machine_schedule(machine, assignment, bench):
    """One script per physical machine: each device is a concurrent subshell
    that runs its assigned jobs sequentially (longest first)."""
    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated schedule for {machine} (devices run concurrently, no MPS).",
        "# Regenerate with: python -m RL.run_scheduler",
        "set -uo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
        'cd "${REPO_ROOT}" || exit 1',
        'export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"',
        '# Use the repo-local venv (its CUDA torch), not whatever python is on PATH.',
        '# Set VENV=... if your venv lives elsewhere.',
        'VENV="${VENV:-${REPO_ROOT}/.venv}"',
        'if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";',
        'elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then',
        '  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;',
        'fi',
        "",
    ]

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
            desc = f"{level_name(exp['level'])}/{exp['alg']}/{exp['config']} seed {exp['seed']}"
            lines.append(f'  echo "[{machine}:{device}] {n}/{len(order)}: {desc}"')
            lines.append(f'  {job_command(exp, device)} \\')
            lines.append(f'    || echo "[{machine}:{device}] FAILED: {desc}"')
        lines.append(") &")
        lines.append("")

    lines.append(f'echo "Launched {active} device(s) on {machine}; waiting..."')
    lines.append("wait")
    lines.append(f'echo "All jobs on {machine} complete."')
    lines.append("")

    os.makedirs(TIME_DIR, exist_ok=True)
    path = os.path.join(TIME_DIR, f"run_{machine}_experiments.sh")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)
    return path, machine_est


# --------------------------------------------------------------------------- #
def generate_schedules():
    bench = load_benchmarks()
    runtime = [
        [runtime_minutes(bench, device, exp) for device in DEVICE_NAMES]
        for exp in EXPERIMENTS
    ]

    print(f"Assigning {len(EXPERIMENTS)} jobs across {len(DEVICE_NAMES)} devices "
          f"on {len(MACHINES)} machines.")
    assignment = assign_ilp(EXPERIMENTS, DEVICE_NAMES, runtime)
    if assignment is None:
        print("OR-Tools unavailable or no solution; using greedy assignment.")
        assignment = assign_greedy(EXPERIMENTS, DEVICE_NAMES, runtime)
    else:
        print("Optimal/feasible assignment found via OR-Tools (SCIP).")

    print("-" * 68)
    overall = 0.0
    for machine in MACHINES:
        if not any(assignment[dev] for dev in MACHINES[machine]):
            continue
        path, est = write_machine_schedule(machine, assignment, bench)
        overall = max(overall, est)
        devs = ", ".join(f"{dev}:{len(assignment[dev])}" for dev in MACHINES[machine])
        print(f"{machine:<15} ~{est:7.1f} min | {devs}")
        print(f"{'':<15}   -> {os.path.relpath(path, REPO_ROOT)}")
    print("-" * 68)
    print(f"Estimated overall wall-clock makespan: {overall:.1f} min "
          f"({overall / 60.0:.1f} h)")


def cmd_benchmark_plan(machine):
    """Emit `device|prefix|flags` lines for benchmark.sh (one per device)."""
    if machine not in MACHINES:
        print(f"unknown machine '{machine}'; known: {', '.join(MACHINES)}", file=sys.stderr)
        sys.exit(2)
    for device in MACHINES[machine]:
        print(f"{device}|{device_prefix(device)}|{device_flags(device)}")


def main():
    ap = argparse.ArgumentParser(description="Generate per-machine experiment schedules.")
    ap.add_argument("--benchmark-plan", metavar="MACHINE",
                    help="print 'device|prefix|flags' lines for MACHINE's devices and exit "
                         "(used by benchmark.sh to run them concurrently under pinning)")
    args = ap.parse_args()

    if args.benchmark_plan is not None:
        cmd_benchmark_plan(args.benchmark_plan)
        return

    generate_schedules()


if __name__ == "__main__":
    main()
