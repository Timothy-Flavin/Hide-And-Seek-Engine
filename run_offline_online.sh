#!/bin/bash
# Offline+online RL experiment: PPO / SAC / DQN, 1M frames each, evaluated as a
# no-human team every 200k frames (starting from random at frame 0), over 3 seeds
# and all levels. Produces the dotted "no human (all-policy team)" baseline that
# overlays the human-in-the-loop curves from run_human_loop.sh.
#
# This is the pure-training counterpart to run_human_loop.sh: it does NOT record
# any human demos and does NOT talk to another machine -- it runs end-to-end on
# ONE machine (the lab computer). It reuses the ALREADY-collected human demos as a
# fixed offline dataset via each runner's joint online+offline objective:
#     PPO / SAC : --human-bc   (cross-entropy behavior cloning)
#     DQN       : --human-cql  (offline Q-learning + conservative-Q penalty;
#                               online TD stays penalty-free)
#
# Each 1M-frame run is trained as 5 resumable 200k-frame chunks. Every runner
# evaluates the all-policy team at step 0 (random) and at the end of each chunk,
# appending to experiments/results/<level>/<alg>/eval_returns.jsonl -- so the
# dotted baseline gets points at 0/200k/400k/600k/800k/1M.
#
# PRECONDITIONS on this machine:
#   * The human demo segments must already be present under
#     experiments/results/<level>/<agent_type>/ (they are the offline dataset;
#     rsync them up from the recording machine first). The script checks and warns
#     loudly per level if they are missing -- without them the offline term is
#     silently skipped and you get plain online RL.
#   * Run against a CLEAN experiments/results for the trained outputs: the runs
#     reuse the <alg>_decentralized_ego_radio_run_<seed> prefixes, so a leftover
#     resume checkpoint / eval log from a prior experiment would be resumed and
#     appended to. (The demo buckets above are fine to keep -- they are inputs.)
#
# Override any setting from the environment, e.g.:
#   SEEDS="1 2 3" ALGS="dqn" LEVELS="levels/test_level" ./run_offline_online.sh
#   TOTAL=1000000 CHUNK=200000 NUM_ENVS=128 ./run_offline_online.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Activate the local venv if present (the RL trainers import the compiled engine).
# Override VENV to point elsewhere; set VENV="" to use whatever python is active.
VENV="${VENV:-${SCRIPT_DIR}/.venv}"
if [ -n "${VENV}" ] && [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
    echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2
fi

# ----- configuration (override from the environment) -----
SEEDS="${SEEDS:-1 2 3}"                   # run numbers; averaged into the dotted line
LEVELS="${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level}"
[ -n "${LEVEL:-}" ] && LEVELS="${LEVEL}"  # single-level override
ALGS="${ALGS:-ppo dqn sac}"
TOTAL="${TOTAL:-1000000}"                 # total env frames per run (1M)
CHUNK="${CHUNK:-1000000}"                  # frames per resumable chunk (== eval cadence)
EGO_SIZE="${EGO_SIZE:-32}"
USE_RADIO="${USE_RADIO:-1}"               # 1 -> decentralized_ego_radio variant
# --- Behavior-cloning term (ppo/sac only; dqn uses --human-cql, unaffected) ---
BC_COEF="${BC_COEF:-1.0}"                 # weight on the BC cross-entropy term (bumped)
BC_SEPARATE="${BC_SEPARATE:-0}"           # 1 -> BC trains a SEPARATE net; the online
                                          # policy trains pure-RL (identifiability probe)
NUM_ENVS="${NUM_ENVS:-128}"
NUM_ENVS_NEIGHBORHOOD="${NUM_ENVS_NEIGHBORHOOD:-64}"  # larger map -> fewer envs (VRAM)
# Extra flags appended verbatim to every runner invocation (e.g. EXTRA_FLAGS="--no-cuda"
# to force CPU for a debug run, or a different --learning-starts). Empty by default.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

RADIO_FLAG=""
[ "${USE_RADIO}" = "1" ] && RADIO_FLAG="--use-radio"

# Integer number of chunks; guard against a non-divisible CHUNK.
N_CHUNKS=$(( TOTAL / CHUNK ))
if [ $(( N_CHUNKS * CHUNK )) -ne "${TOTAL}" ]; then
    echo "!!! TOTAL (${TOTAL}) must be a multiple of CHUNK (${CHUNK})." >&2
    exit 2
fi

envs_for () {  # per-level num_envs (neighborhood map is the big one)
    case "$1" in
        *neighborhood*) echo "${NUM_ENVS_NEIGHBORHOOD}" ;;
        *)              echo "${NUM_ENVS}" ;;
    esac
}

# Offline objective flags for each algorithm. The BC term (ppo/sac) carries the
# bumped --bc-coef and, when BC_SEPARATE=1, --bc-separate (train the human demos on
# an independent net so the online policy learns unencumbered -- isolates whether a
# shared head was the blocker). DQN's CQL term takes neither.
offline_flags_for () {
    local sep=""
    [ "${BC_SEPARATE}" = "1" ] && sep="--bc-separate"
    case "$1" in
        ppo|sac) echo "--human-bc --bc-coef ${BC_COEF} ${sep}" ;;
        dqn)     echo "--human-cql --exploration-timesteps ${TOTAL}" ;;
        *)       echo "" ;;
    esac
}

FAILURES=()

# One full 1M-frame run for (alg, level, seed): N_CHUNKS resumable chunks.
train_run () {
    local alg="$1" level="$2" seed="$3"
    local level_name; level_name="$(basename "${level}")"
    local envs; envs="$(envs_for "${level_name}")"
    local offline; offline="$(offline_flags_for "${alg}")"

    local c
    for (( c=1; c<=N_CHUNKS; c++ )); do
        echo "=== [seed ${seed} | ${level_name} | ${alg}] chunk ${c}/${N_CHUNKS} (${CHUNK} frames, ${envs} envs) ==="
        if ! python -m "RL.cleanrl_${alg}" \
                --run-number "${seed}" \
                --level "${level}" \
                --no-centralized --ego-view --ego-size "${EGO_SIZE}" ${RADIO_FLAG} \
                ${offline} --resume \
                --total-timesteps "${CHUNK}" \
                --num-envs "${envs}" ${EXTRA_FLAGS}; then
            echo "!!! FAILED: seed ${seed} | ${level_name} | ${alg} | chunk ${c}"
            FAILURES+=("seed=${seed} level=${level_name} alg=${alg} chunk=${c}")
            return 1  # abort the rest of this run's chunks (resume state is per-run)
        fi
    done
}

# Regenerate the human-vs-eval figure aggregating the dotted line over all seeds.
plot_all () {
    for level in ${LEVELS}; do
        for alg in ${ALGS}; do
            python -m RL.plot_human_loop --level "${level}" --alg "${alg}" --runs ${SEEDS} \
                || echo "!!! plot failed for $(basename "${level}") / ${alg} (continuing)"
        done
    done
}

# Warn (do not fail) if a level has no human demos: the offline term needs them.
check_demos () {
    for level in ${LEVELS}; do
        python - "${level}" "${EGO_SIZE}" <<'PY'
import sys
from RL.human_data import count_human_frames, level_name_of
level, ego = sys.argv[1], int(sys.argv[2])
n = count_human_frames(level, ego_size=ego)
tag = level_name_of(level)
if n <= 0:
    print(f"!!! WARNING: no human demos (ego_size={ego}) for {tag}; the offline "
          f"term will be SKIPPED -> plain online RL for this level.")
else:
    print(f"    demos OK: {tag} has {n} human frames (ego_size={ego}).")
PY
    done
}

echo "Offline+online RL experiment:"
echo "  seeds=[${SEEDS}] levels=[${LEVELS}] algs=[${ALGS}]"
echo "  total=${TOTAL} chunk=${CHUNK} (${N_CHUNKS} chunks/run) ego=${EGO_SIZE} radio=${USE_RADIO}"
echo "  bc_coef=${BC_COEF} bc_separate=${BC_SEPARATE} (ppo/sac)"
echo "--- checking offline dataset (human demos) ---"
check_demos

# Seed is the OUTERMOST loop (like run_experiments.sh): after seed 1 you have
# touched every level/alg, so you can sanity-check before waiting on all seeds.
# Plots are regenerated after each seed pass, aggregating whatever seeds exist.
for seed in ${SEEDS}; do
    echo "############################################################"
    echo "##  SEED ${seed}: all levels / algs, ${TOTAL} frames each"
    echo "############################################################"
    for level in ${LEVELS}; do
        for alg in ${ALGS}; do
            train_run "${alg}" "${level}" "${seed}"
        done
    done
    echo "---- Seed ${seed} done. Regenerating aggregated plots... ----"
    plot_all
done

echo "############################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "All runs completed. Figures: experiments/results/<level>/human_vs_eval_<alg>_runs*.png"
    echo "  eval baselines: experiments/results/<level>/<alg>/eval_returns.jsonl"
    exit 0
else
    echo "Completed with ${#FAILURES[@]} failed run(s):"
    printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
