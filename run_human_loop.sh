#!/bin/bash
# Iterative human-in-the-loop pre-training for the decentralized ego agents:
#
#   record human demos (ALL levels) -> online+offline (BC) RL chunk (each level)
#   -> record -> RL chunk -> ...
#
# Each iteration trains ONE resumable RL chunk per level (online rollouts + an
# offline behavior-cloning term from the recorded human demos), then records a
# fresh per-agent-type quota of demos against the NEW checkpoint. The recorder
# tags demos by the checkpoint they were gathered against and only tops up the
# agents that still lack data for the current checkpoint, so re-running is
# idempotent. Every level gets FRAMES_PER_AGENT frames for every agent type
# before its models are trained.
#
# Teammates in the recorder are driven by the latest checkpoint and (with radio)
# choose both movement and radio, so their shared info shows up in the human's
# ego view.
#
# Run from anywhere; everything executes from the repo root so relative paths
# (levels/, experiments/) and the `RL` namespace package resolve.
#
# Override any setting from the environment, e.g.:
#   LEVELS="levels/test_level" ALG=ppo ITERS=6 CHUNK=1000000 ./run_human_loop.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ----- configuration (override from the environment) -----
# Space-separated list of level dirs; every one is covered before training.
LEVELS="${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level}"
# LEVEL is a single-level override (records/trains just that one).
[ -n "${LEVEL:-}" ] && LEVELS="${LEVEL}"
ALG="${ALG:-ppo}"                        # ppo | dqn | sac
RUN="${RUN:-1}"                          # run/seed number (recorder teammate-run must match)
EGO_SIZE="${EGO_SIZE:-32}"               # ego window side (recorder and trainer must match)
FRAMES_PER_AGENT="${FRAMES_PER_AGENT:-2500}"  # demos per agent type, per level, per checkpoint
ITERS="${ITERS:-4}"                      # number of train->record iterations
CHUNK="${CHUNK:-500000}"                 # online env frames per RL chunk (per level)
# Epsilon anneals along cumulative frames, so span the whole loop (DQN only).
EXPLORE_TOTAL="${EXPLORE_TOTAL:-$((CHUNK * ITERS))}"
USE_RADIO="${USE_RADIO:-1}"              # 1 -> train+use the radio head (teammates transmit)
NUM_ENVS="${NUM_ENVS:-128}"
STEP_DELAY_MS="${STEP_DELAY_MS:-40}"     # recorder pacing (ms/step while a key is held)

RADIO_FLAG=""
[ "${USE_RADIO}" = "1" ] && RADIO_FLAG="--use-radio"
# --exploration-timesteps is a DQN-only knob (PPO/SAC have no epsilon schedule).
EXPLORE_FLAG=""
[ "${ALG}" = "dqn" ] && EXPLORE_FLAG="--exploration-timesteps ${EXPLORE_TOTAL}"

# Record FRAMES_PER_AGENT frames for every agent type of every level.
record_all () {
    for lvl in ${LEVELS}; do
        echo "=== [record] ${lvl}: ${FRAMES_PER_AGENT} frames/agent-type; teammates=${ALG} run ${RUN} ==="
        python human_dataset.py \
            --level "${lvl}" --ego-size "${EGO_SIZE}" \
            --frames-per-agent "${FRAMES_PER_AGENT}" \
            --teammate-alg "${ALG}" --teammate-run "${RUN}" \
            --step-delay-ms "${STEP_DELAY_MS}"
    done
}

# Train one resumable chunk per level.
train_all () {
    for lvl in ${LEVELS}; do
        echo "=== [train] ${ALG} decentralized_ego${RADIO_FLAG:+_radio} run ${RUN} on ${lvl}: ${CHUNK} frames (resume + human-bc) ==="
        python -m "RL.cleanrl_${ALG}" \
            --level "${lvl}" \
            --no-centralized --ego-view --ego-size "${EGO_SIZE}" ${RADIO_FLAG} \
            --run-number "${RUN}" \
            --human-bc --resume \
            --total-timesteps "${CHUNK}" ${EXPLORE_FLAG} \
            --num-envs "${NUM_ENVS}"
    done
}

# Regenerate the 6-curve human-vs-eval performance graph for every level.
plot_all () {
    for lvl in ${LEVELS}; do
        python -m RL.plot_human_loop --level "${lvl}" --alg "${ALG}" --run "${RUN}" \
            || echo "!!! plot failed for ${lvl} (continuing)"
    done
}

# ----- phase dispatch -----
# Split the loop into single passes so an orchestrator (e.g. run_human_loop_remote.sh)
# can run recording on one machine and training on another:
#   PHASE=record : one recording pass over all levels, then exit.
#   PHASE=train  : train one resumable chunk per level, then exit.
#   PHASE=loop   : (default) the full local record<->train loop below.
PHASE="${PHASE:-loop}"
case "${PHASE}" in
    record) record_all; exit $? ;;
    train)  train_all;  exit $? ;;
    loop)   ;;
    *) echo "unknown PHASE='${PHASE}' (want record|train|loop)" >&2; exit 2 ;;
esac

echo "Human-in-the-loop pre-training:"
echo "  levels=[${LEVELS}] alg=${ALG} run=${RUN} ego=${EGO_SIZE} radio=${USE_RADIO}"
echo "  iters=${ITERS} chunk=${CHUNK} frames/agent=${FRAMES_PER_AGENT} explore_total=${EXPLORE_TOTAL}"

# Bootstrap: first demos with no checkpoint yet (teammates random move + heuristic
# radio). These seed the BC term for iteration 1's training, for every level.
record_all

for (( i=1; i<=ITERS; i++ )); do
    echo "############################################################"
    echo "##  iteration ${i}/${ITERS}"
    echo "############################################################"
    train_all
    plot_all   # refresh the human-vs-eval graph after this training step
    # Record fresh demos against the new checkpoints for the next iteration; the
    # final iteration ends on trained models (no trailing record).
    if [ "${i}" -lt "${ITERS}" ]; then
        record_all
    fi
done

echo "############################################################"
echo "Done: ${ITERS} RL chunks per level (${ALG}, run ${RUN}) over [${LEVELS}]."
for lvl in ${LEVELS}; do
    ln="$(basename "${lvl}")"
    echo "  ${ln}: returns experiments/results/${ln}/human_returns.jsonl ; checkpoints experiments/results/${ln}/${ALG}/checkpoints/"
done
echo "  RL curves: python -m RL.plot_results"
