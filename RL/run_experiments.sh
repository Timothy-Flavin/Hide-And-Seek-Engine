#!/bin/bash
# Experiment sweep: <configs> x <environments> x <models> x <seeds>, 5M frames
# each, on the GPU.
#
# CONFIG SPACE (5 valid combinations, all supported by the runners):
#   centralized              : shared global map (state), per-agent heads
#   decentralized            : per-agent full-map FOV
#   decentralized_radio      : full-map FOV + trainable radio head
#   decentralized_ego        : per-agent 32x32 ego-centric crop
#   decentralized_ego_radio  : ego-centric crop + trainable radio head
#
# ACTIVE selection (this run): centralized, decentralized_ego,
# decentralized_ego_radio. The other two combos are wired up and can be enabled
# any time via the CONFIGS env var (see below) -- no code changes needed. Only
# one of the three active configs (centralized) uses the full-map observation,
# so the expensive large-map runs stay bounded without special-casing levels.
#
# SEED IS THE OUTERMOST LOOP: every experiment runs once for seed 1, then once
# for seed 2, and so on. After the first seed pass you have touched every
# environment/config/model combination, so you can verify everything works
# before waiting on all 5 seeds. Plots are regenerated after each seed pass so
# results accumulate incrementally.
#
# Results -> experiments/results/<level>/<alg>/ ; one combined plot per level.
# Runs continue on error; a failure summary is printed after each seed pass and
# at the end (the script exits non-zero if anything failed).

# Run everything from the repo root so relative paths (levels/, experiments/)
# and the `RL` namespace package resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ----- configuration (override from the environment, e.g. SEEDS="1 2 3") -----
FRAMES="${FRAMES:-5000000}"          # 5M frames (total env frames, seed-invariant)
SEEDS="${SEEDS:-1 2 3 4 5}"          # run numbers, averaged in the plot
EGO_SIZE="${EGO_SIZE:-32}"           # ego-centric window is 32x32
ALGS=(${ALGS:-ppo dqn sac})
LEVELS=(${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level})

# Per-map VRAM tuning. The 128x128 neighborhood map has a much larger full-map
# observation, so it uses fewer parallel envs and (PPO only) more minibatches to
# keep the peak activation memory within ~8 GB VRAM. All other maps use the
# defaults. total_timesteps is fixed, so fewer envs just means more iterations.
NUM_ENVS="${NUM_ENVS:-128}"                                  # default parallel envs
NUM_ENVS_NEIGHBORHOOD="${NUM_ENVS_NEIGHBORHOOD:-64}"         # neighborhood override
PPO_MINIBATCHES="${PPO_MINIBATCHES:-8}"                      # PPO default
PPO_MINIBATCHES_NEIGHBORHOOD="${PPO_MINIBATCHES_NEIGHBORHOOD:-16}"  # neighborhood override

envs_for () {  # num_envs for a level (by basename)
    case "$1" in
        *neighborhood*) echo "${NUM_ENVS_NEIGHBORHOOD}" ;;
        *)              echo "${NUM_ENVS}" ;;
    esac
}
ppo_minibatches_for () {  # PPO num_minibatches for a level
    case "$1" in
        *neighborhood*) echo "${PPO_MINIBATCHES_NEIGHBORHOOD}" ;;
        *)              echo "${PPO_MINIBATCHES}" ;;
    esac
}

# The 5 supported configs; `config_flags` maps each to its runner flags. To
# change which experiments run, edit CONFIGS (or set it in the environment),
# e.g. CONFIGS="centralized decentralized decentralized_radio decentralized_ego decentralized_ego_radio".
CONFIGS=(${CONFIGS:-centralized decentralized_ego decentralized_ego_radio})

config_flags () {
    case "$1" in
        centralized)              echo "--centralized" ;;
        decentralized)            echo "--no-centralized" ;;
        decentralized_radio)      echo "--no-centralized --use-radio" ;;
        decentralized_ego)        echo "--no-centralized --ego-view --ego-size ${EGO_SIZE}" ;;
        decentralized_ego_radio)  echo "--no-centralized --ego-view --ego-size ${EGO_SIZE} --use-radio" ;;
        *) echo "UNKNOWN_CONFIG" ;;
    esac
}

FAILURES=()

run () {
    local alg="$1" level="$2" seed="$3" config="$4"
    local level_name; level_name="$(basename "${level}")"

    local flags; flags="$(config_flags "${config}")"
    if [ "${flags}" = "UNKNOWN_CONFIG" ]; then
        echo "!!! Unknown config '${config}' -- skipping."
        FAILURES+=("seed=${seed} level=${level_name} alg=${alg} config=${config} (unknown config)")
        return
    fi

    # Per-map env count (all algos) and PPO-only minibatch count.
    local envs; envs="$(envs_for "${level_name}")"
    local tune="--num-envs ${envs}"
    if [ "${alg}" = "ppo" ]; then
        tune="${tune} --num-minibatches $(ppo_minibatches_for "${level_name}")"
    fi

    echo "=== [seed ${seed} | ${level_name} | ${alg} | ${config}] (${tune}) ==="
    # cuda defaults to True in every runner -> trains on GPU when available.
    if ! python -m "RL.cleanrl_${alg}" \
            --run-number "${seed}" \
            --total-timesteps "${FRAMES}" \
            --level "${level}" \
            ${tune} \
            ${flags}; then
        echo "!!! FAILED: seed ${seed} | ${level_name} | ${alg} | ${config}"
        FAILURES+=("seed=${seed} level=${level_name} alg=${alg} config=${config}")
    fi
}

echo "Configs this run: ${CONFIGS[*]}"

for seed in ${SEEDS}; do
    echo "############################################################"
    echo "##  SEED ${seed}: running all environments / models / configs"
    echo "############################################################"
    seed_fail_start=${#FAILURES[@]}

    for level in "${LEVELS[@]}"; do
        for alg in "${ALGS[@]}"; do
            for config in "${CONFIGS[@]}"; do
                run "${alg}" "${level}" "${seed}" "${config}"
            done
        done
    done

    echo "---- Seed ${seed} done. Regenerating combined plots... ----"
    python -m RL.plot_results || echo "plot_results failed for seed ${seed}"

    seed_new_fail=$(( ${#FAILURES[@]} - seed_fail_start ))
    if [ "${seed_new_fail}" -eq 0 ]; then
        echo "---- Seed ${seed}: all scheduled runs succeeded. ----"
    else
        echo "---- Seed ${seed}: ${seed_new_fail} run(s) FAILED (see summary below). ----"
        printf '     %s\n' "${FAILURES[@]:${seed_fail_start}}"
    fi
done

echo "############################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "All scheduled experiments completed successfully."
    exit 0
else
    echo "Completed with ${#FAILURES[@]} failed run(s):"
    printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
