#!/bin/bash
# Steady-state throughput benchmark for the OFFLINE+ONLINE (per-agent) sweep.
#
# Sibling of benchmark.sh: same machine/device pinning, but it benchmarks the
# ACTUAL job this experiment runs -- per-agent networks (--per-agent-nets) with
# each arm's objective (RL baseline, and RL + the offline term --human-bc for
# ppo/sac / --human-cql for dqn). Those change per-step cost versus the online
# shared-net benchmark, so run_scheduler_offline needs their own numbers.
#
# The offline arm loads the human demos, so the demos must already be on this
# machine (run ./sync_human_data.sh first). Results merge into the SAME
# time_files/<device>.json, keyed by the runner's variant (..._pa, ..._pa_bc,
# ..._pa_cql), which is exactly what run_scheduler_offline looks up.
#
# Usage:  ./RL/benchmark_offline.sh --machine <machine> [--seed N]
# Override the sweep from the environment, e.g.:
#   ALGS="dqn" LEVELS="levels/test_level" ./RL/benchmark_offline.sh timpc
set -uo pipefail

MACHINE=""; SEED=1
while [ $# -gt 0 ]; do
    case "$1" in
        --machine) MACHINE="${2:?}"; shift 2 ;;
        --seed)    SEED="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) if [ -z "${MACHINE}" ]; then MACHINE="$1"; else SEED="$1"; fi; shift ;;
    esac
done
[ -n "${MACHINE}" ] || { echo "usage: benchmark_offline.sh --machine <machine> [--seed N]" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

VENV="${VENV:-${REPO_ROOT}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
    echo "WARNING: no venv at ${VENV} and none active; using system python" >&2
fi

# Device list + pinning from the OFFLINE scheduler topology (same machines).
PLAN="$(python -m RL.run_scheduler_offline --benchmark-plan "${MACHINE}")" || exit 1
DEV_NAMES=(); DEV_PREFIX=(); DEV_FLAGS=()
while IFS='|' read -r name prefix flags; do
    [ -n "${name}" ] || continue
    DEV_NAMES+=("${name}"); DEV_PREFIX+=("${prefix}"); DEV_FLAGS+=("${flags}")
done <<< "${PLAN}"

# Per-machine memory-safety overrides (num_envs cap, replay buffer size) -- MUST
# match what run_scheduler_offline bakes into the real jobs, so the benchmarked
# throughput reflects the actual configuration. Sets NUM_ENVS_CAP / BUFFER_SIZE.
eval "$(python -m RL.run_scheduler_offline --mem-plan "${MACHINE}")"

ALGS=(${ALGS:-ppo dqn sac})
LEVELS=(${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level})
ARMS=(${ARMS:-rl offline})
EGO_SIZE="${EGO_SIZE:-32}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_ENVS_NEIGHBORHOOD="${NUM_ENVS_NEIGHBORHOOD:-64}"
PPO_MINIBATCHES="${PPO_MINIBATCHES:-8}"
PPO_MINIBATCHES_NEIGHBORHOOD="${PPO_MINIBATCHES_NEIGHBORHOOD:-16}"

# Apply the machine's num_envs cap (if any), exactly like the scheduler.
cap_envs () {  # $1 = uncapped num_envs
    if [ -n "${NUM_ENVS_CAP:-}" ] && [ "$1" -gt "${NUM_ENVS_CAP}" ]; then echo "${NUM_ENVS_CAP}"; else echo "$1"; fi
}
envs_for () {
    local e; case "$1" in *neighborhood*) e="${NUM_ENVS_NEIGHBORHOOD}" ;; *) e="${NUM_ENVS}" ;; esac
    cap_envs "${e}"
}
ppo_mb_for () { case "$1" in *neighborhood*) echo "${PPO_MINIBATCHES_NEIGHBORHOOD}" ;; *) echo "${PPO_MINIBATCHES}" ;; esac; }
# Replay-buffer override for sac/dqn on RAM-tight machines (matches the scheduler).
buffer_for () { case "$1" in dqn|sac) [ -n "${BUFFER_SIZE:-}" ] && echo "--buffer-size ${BUFFER_SIZE}" ;; esac; }

# Objective flags per (alg, arm). --exploration-timesteps (dqn epsilon horizon) is
# an RL knob present in both arms; the human-data term is added only in 'offline'.
# A large --exploration-timesteps keeps epsilon in steady state during the window.
objective_flags () {  # $1=alg  $2=arm
    local f=""
    [ "$1" = "dqn" ] && f="--exploration-timesteps 100000000"
    if [ "$2" = "offline" ]; then
        if [ "$1" = "dqn" ]; then f="${f} --human-cql"; else f="${f} --human-bc --bc-coef 1.0"; fi
    fi
    echo "${f}"
}

# Zero burn-in so updates (and the offline term) run inside the timed window.
burn_in_flags () { case "$1" in dqn|sac) echo "--learning-starts 0" ;; *) echo "" ;; esac; }

echo "Offline+online benchmark '${MACHINE}' seed ${SEED}"
echo "  devices: ${DEV_NAMES[*]}   arms: ${ARMS[*]}   levels: ${LEVELS[*]}   algs: ${ALGS[*]}"

FAILURES=()
for level in "${LEVELS[@]}"; do
    lname="$(basename "${level}")"; envs="$(envs_for "${lname}")"
    for alg in "${ALGS[@]}"; do
        tune="--num-envs ${envs}"
        [ "${alg}" = "ppo" ] && tune="${tune} --num-minibatches $(ppo_mb_for "${lname}")"
        for arm in "${ARMS[@]}"; do
            echo "=== [${MACHINE} | ${lname} | ${alg} | pa/${arm}] on ${#DEV_NAMES[@]} device(s) ==="
            pids=()
            for k in "${!DEV_NAMES[@]}"; do
                dev="${DEV_NAMES[$k]}"; pin="${DEV_PREFIX[$k]}"; dflags="${DEV_FLAGS[$k]}"
                env ${pin} python -m "RL.cleanrl_${alg}" \
                        --benchmark --machine "${dev}" \
                        --run-number "${SEED}" \
                        --level "${level}" \
                        ${tune} \
                        --no-centralized --ego-view --ego-size "${EGO_SIZE}" --use-radio \
                        --per-agent-nets \
                        $(buffer_for "${alg}") \
                        $(objective_flags "${alg}" "${arm}") \
                        $(burn_in_flags "${alg}") \
                        ${dflags} &
                pids+=("$!:${dev}")
            done
            for pd in "${pids[@]}"; do
                if ! wait "${pd%%:*}"; then
                    echo "!!! FAILED: ${lname} | ${alg} | pa/${arm} on ${pd#*:}"
                    FAILURES+=("${pd#*:}: ${lname}/${alg}/pa_${arm}")
                fi
            done
        done
    done
done

echo "############################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "Offline benchmark complete. Wrote per-device keys into time_files/<device>.json."
    echo "Then: python -m RL.run_scheduler_offline"
else
    echo "Completed with ${#FAILURES[@]} failure(s):"
    printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
