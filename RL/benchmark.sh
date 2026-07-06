#!/bin/bash
# Steady-state throughput benchmark for ONE physical machine.
#
# Usage:  ./benchmark.sh --machine <machine> [--seed N]
#    or:  ./benchmark.sh <machine> [seed]
#
# Machines (see RL/run_scheduler.py): timpc, lab-comp, white-machine, alienware.
#
# All of a machine's devices run CONCURRENTLY, one job per device, each pinned
# with the same taskset/numactl/OMP/CUDA prefix it uses in the real sweep. For
# every (level x alg x config) combo the job is launched on all devices at once
# and we wait for them together, so each device is measured while the others are
# also training -- capturing real memory-bandwidth / cache contention. Each
# device times ~20s of steady state (after a torch.compile warmup) and records
# steps/sec into time_files/<device>.json, with all burn-in periods zeroed.
#
# Copy the resulting time_files/*.json to the scheduling machine and run:
#   python -m RL.run_scheduler
#
# Override the sweep from the environment, e.g.:
#   ALGS="dqn" LEVELS="levels/test_level" ./benchmark.sh --machine lab-comp
set -uo pipefail

MACHINE=""
SEED=1
while [ $# -gt 0 ]; do
    case "$1" in
        --machine) MACHINE="${2:?--machine needs a value}"; shift 2 ;;
        --seed)    SEED="${2:?--seed needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *)
            if [ -z "${MACHINE}" ]; then MACHINE="$1"
            elif [ "${SEED}" = "1" ]; then SEED="$1"
            else echo "unexpected argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done
[ -n "${MACHINE}" ] || { echo "usage: benchmark.sh --machine <machine> [--seed N]" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source .venv/bin/activate

# Pull this machine's devices + per-device pinning from the scheduler topology
# (single source of truth). One line per device: "device|prefix|flags".
PLAN="$(python -m RL.run_scheduler --benchmark-plan "${MACHINE}")" || exit 1

DEV_NAMES=(); DEV_PREFIX=(); DEV_FLAGS=()
while IFS='|' read -r name prefix flags; do
    [ -n "${name}" ] || continue
    DEV_NAMES+=("${name}"); DEV_PREFIX+=("${prefix}"); DEV_FLAGS+=("${flags}")
done <<< "${PLAN}"

ALGS=(${ALGS:-ppo dqn sac})
LEVELS=(${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level})
CONFIGS=(${CONFIGS:-centralized decentralized_ego decentralized_ego_radio})
EGO_SIZE="${EGO_SIZE:-32}"

# Per-map env counts / PPO minibatches -- mirror run_experiments.sh so the
# benchmarked throughput matches the real sweep on each map.
NUM_ENVS="${NUM_ENVS:-128}"
NUM_ENVS_NEIGHBORHOOD="${NUM_ENVS_NEIGHBORHOOD:-64}"
PPO_MINIBATCHES="${PPO_MINIBATCHES:-8}"
PPO_MINIBATCHES_NEIGHBORHOOD="${PPO_MINIBATCHES_NEIGHBORHOOD:-16}"

envs_for () {
    case "$1" in *neighborhood*) echo "${NUM_ENVS_NEIGHBORHOOD}" ;; *) echo "${NUM_ENVS}" ;; esac
}
ppo_minibatches_for () {
    case "$1" in *neighborhood*) echo "${PPO_MINIBATCHES_NEIGHBORHOOD}" ;; *) echo "${PPO_MINIBATCHES}" ;; esac
}

config_flags () {
    case "$1" in
        centralized)             echo "--centralized" ;;
        decentralized)           echo "--no-centralized" ;;
        decentralized_radio)     echo "--no-centralized --use-radio" ;;
        decentralized_ego)       echo "--no-centralized --ego-view --ego-size ${EGO_SIZE}" ;;
        decentralized_ego_radio) echo "--no-centralized --ego-view --ego-size ${EGO_SIZE} --use-radio" ;;
        *) echo "UNKNOWN_CONFIG" ;;
    esac
}

# Zero every burn-in period so the whole 20s window is steady state. PPO has none.
burn_in_flags () {
    case "$1" in
        dqn|sac) echo "--learning-starts 0" ;;
        *)       echo "" ;;
    esac
}

echo "Benchmarking machine '${MACHINE}' seed ${SEED}"
echo "  devices: ${DEV_NAMES[*]}  (run concurrently)"
echo "  levels : ${LEVELS[*]}"
echo "  algs   : ${ALGS[*]}"
echo "  configs: ${CONFIGS[*]}"

FAILURES=()
for level in "${LEVELS[@]}"; do
    lname="$(basename "${level}")"
    envs="$(envs_for "${lname}")"
    for alg in "${ALGS[@]}"; do
        tune="--num-envs ${envs}"
        if [ "${alg}" = "ppo" ]; then
            tune="${tune} --num-minibatches $(ppo_minibatches_for "${lname}")"
        fi
        for config in "${CONFIGS[@]}"; do
            flags="$(config_flags "${config}")"
            if [ "${flags}" = "UNKNOWN_CONFIG" ]; then
                echo "!!! Unknown config '${config}' -- skipping."
                continue
            fi
            echo "=== [${MACHINE} | ${lname} | ${alg} | ${config}] on ${#DEV_NAMES[@]} device(s) ==="

            # Launch this combo on every device at once, then wait -- keeps the
            # devices mutually loaded so contention is captured in each number.
            pids=()
            for k in "${!DEV_NAMES[@]}"; do
                dev="${DEV_NAMES[$k]}"; pin="${DEV_PREFIX[$k]}"; dflags="${DEV_FLAGS[$k]}"
                # `env` parses the leading OMP/CUDA assignments (a bare unquoted
                # "$pin python" would treat them as the command), then execs the
                # taskset/numactl wrapper + python.
                env ${pin} python -m "RL.cleanrl_${alg}" \
                        --benchmark --machine "${dev}" \
                        --run-number "${SEED}" \
                        --level "${level}" \
                        ${tune} \
                        $(burn_in_flags "${alg}") \
                        ${flags} ${dflags} &
                pids+=("$!:${dev}")
            done
            for pd in "${pids[@]}"; do
                if ! wait "${pd%%:*}"; then
                    echo "!!! FAILED: ${lname} | ${alg} | ${config} on ${pd#*:}"
                    FAILURES+=("${pd#*:}: ${lname}/${alg}/${config}")
                fi
            done
        done
    done
done

echo "############################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "Benchmark complete. Wrote:"
    for dev in "${DEV_NAMES[@]}"; do echo "  time_files/${dev}.json"; done
    echo "Copy those to the scheduling machine and run: python -m RL.run_scheduler"
else
    echo "Benchmark completed with ${#FAILURES[@]} failure(s):"
    printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
