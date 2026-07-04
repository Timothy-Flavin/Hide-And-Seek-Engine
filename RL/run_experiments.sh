#!/bin/bash
# Full experiment sweep: 4 configs x 4 environments x 3 models x 5 seeds, 5M
# frames each, on the GPU.
#
# The 4 configurations are:
#   1. centralized
#   2. decentralized (no ego, no radio)
#   3. decentralized + ego-centric (32x32 window per agent, no radio)
#   4. decentralized + ego-centric + radio (trainable per-agent radio head)
#
# SEED IS THE OUTERMOST LOOP: every experiment is run once for seed 1, then once
# for seed 2, and so on. After the first seed pass you have touched every
# environment/config/model combination, so you can verify everything works
# before waiting on all 5 seeds. Plots are regenerated after each seed pass so
# results accumulate incrementally.
#
# Results are written to experiments/results/<level_name>/ and one combined plot
# per level is produced by plot_results.py after each seed pass.
#
# Runs continue on error; a summary of any failures is printed after each seed
# pass and at the very end (the script exits non-zero if anything failed).

# Run everything from the repo root so relative paths (levels/, experiments/)
# and the `RL` namespace package resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ----- configuration (override from the environment, e.g. SEEDS="1 2 3") -----
FRAMES="${FRAMES:-5000000}"          # 5M frames
SEEDS="${SEEDS:-1 2 3 4 5}"          # run numbers, averaged in the plot
EGO_SIZE="${EGO_SIZE:-32}"           # ego-centric window is 32x32
ALGS=(${ALGS:-ppo dqn sac})
LEVELS=(${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level})

# Config name -> extra CLI flags. Kept parallel so ordering is explicit.
CONFIG_NAMES=(centralized decentralized decentralized_ego decentralized_ego_radio)
config_flags () {
    case "$1" in
        centralized)              echo "--centralized" ;;
        decentralized)            echo "--no-centralized" ;;
        decentralized_ego)        echo "--no-centralized --ego-view --ego-size ${EGO_SIZE}" ;;
        decentralized_ego_radio)  echo "--no-centralized --ego-view --ego-size ${EGO_SIZE} --use-radio" ;;
        *) echo "UNKNOWN_CONFIG" ;;
    esac
}

FAILURES=()

run () {
    local alg="$1" level="$2" seed="$3" config="$4"
    local flags; flags="$(config_flags "${config}")"
    echo "=== [seed ${seed} | $(basename "${level}") | ${alg} | ${config}] ==="
    # cuda defaults to True in every runner -> trains on GPU when available.
    if ! python -m "RL.cleanrl_${alg}" \
            --run-number "${seed}" \
            --total-timesteps "${FRAMES}" \
            --level "${level}" \
            ${flags}; then
        echo "!!! FAILED: seed ${seed} | $(basename "${level}") | ${alg} | ${config}"
        FAILURES+=("seed=${seed} level=$(basename "${level}") alg=${alg} config=${config}")
    fi
}

for seed in ${SEEDS}; do
    echo "############################################################"
    echo "##  SEED ${seed}: running all environments / models / configs"
    echo "############################################################"
    seed_fail_start=${#FAILURES[@]}

    for level in "${LEVELS[@]}"; do
        for alg in "${ALGS[@]}"; do
            for config in "${CONFIG_NAMES[@]}"; do
                run "${alg}" "${level}" "${seed}" "${config}"
            done
        done
    done

    echo "---- Seed ${seed} done. Regenerating combined plots... ----"
    python -m RL.plot_results || echo "plot_results failed for seed ${seed}"

    seed_new_fail=$(( ${#FAILURES[@]} - seed_fail_start ))
    if [ "${seed_new_fail}" -eq 0 ]; then
        echo "---- Seed ${seed}: all ${#CONFIG_NAMES[@]}x${#LEVELS[@]}x${#ALGS[@]} runs succeeded. ----"
    else
        echo "---- Seed ${seed}: ${seed_new_fail} run(s) FAILED (see summary below). ----"
        printf '     %s\n' "${FAILURES[@]:${seed_fail_start}}"
    fi
done

echo "############################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "All experiments completed successfully."
    exit 0
else
    echo "Completed with ${#FAILURES[@]} failed run(s):"
    printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
