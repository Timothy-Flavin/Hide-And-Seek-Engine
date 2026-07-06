"""Steady-state throughput benchmark shared by the cleanrl_* runners.

`--benchmark` on any runner skips the normal training/plotting and instead times
~20 s of steady-state env stepping, then records steps/sec into
``time_files/<machine>.json`` under the key ``<level>_<alg>_<variant>``.

The timer does NOT start at the very first action: torch.compile builds the
rollout and update graphs on the first couple of update-containing iterations,
which would dominate a 20 s window. ``BenchmarkClock`` skips those warmup
iterations and only then starts the clock, so the recorded number is genuine
steady-state throughput. Pair it with ``benchmark.sh``, which zeroes all burn-in
periods (DQN/SAC ``learning_starts``) so updates run inside the timed window.

``run_scheduler.py`` reads these files to estimate per-job runtime and split the
sweep across machines.
"""
import json
import os
import time


BENCHMARK_DURATION_S = 20.0
# Number of update-containing iterations to skip before timing so the one-time
# torch.compile of the forward + backward graphs is excluded from the window.
BENCHMARK_WARMUP_UPDATES = 2


class BenchmarkClock:
    """Times ``duration_s`` of steady-state stepping after a compile warmup.

    Call :meth:`tick` once per env iteration; it returns True when the window is
    complete. ``did_update`` marks iterations in which a gradient update ran --
    the warmup counts those (not idle burn-in iterations) so we start timing only
    once the update graph is compiled and we are truly in steady state.
    """

    def __init__(self, duration_s=BENCHMARK_DURATION_S, warmup_updates=BENCHMARK_WARMUP_UPDATES):
        self.duration_s = duration_s
        self.warmup_updates = warmup_updates
        self._updates_seen = 0
        self.started = False
        self.t0 = None
        self.steps = 0

    def tick(self, steps_this_iter, did_update):
        if not self.started:
            if did_update:
                self._updates_seen += 1
            if self._updates_seen >= self.warmup_updates:
                self.started = True
                self.t0 = time.time()
            return False
        self.steps += steps_this_iter
        return (time.time() - self.t0) >= self.duration_s

    def summary(self, num_envs):
        elapsed = (time.time() - self.t0) if self.t0 is not None else 0.0
        sps = self.steps / elapsed if elapsed > 0 else 0.0
        return {
            "steps_per_sec": sps,
            "steps": int(self.steps),
            "seconds": elapsed,
            "num_envs": int(num_envs),
        }


def benchmark_key(level, alg, variant):
    """Stable per-job key: ``<level_basename>_<alg>_<variant>``."""
    level_name = os.path.basename(os.path.normpath(level))
    return f"{level_name}_{alg}_{variant}"


def write_benchmark(machine, level, alg, variant, summary, out_dir="time_files"):
    """Merge one job's steady-state summary into ``time_files/<machine>.json``.

    benchmark.sh runs the jobs sequentially, so read-modify-write is race-free.
    """
    machine = machine or "unknown"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{machine}.json")

    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (ValueError, OSError):
            data = {}

    key = benchmark_key(level, alg, variant)
    data[key] = summary
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path, key
