#!/bin/bash
# Iterative human-in-the-loop pre-training for the decentralized ego agents:
#
#   record human demos -> online+offline (BC) RL chunk -> record -> RL chunk -> ...
#
# Each iteration trains ONE resumable RL chunk (online rollouts + an offline
# behavior-cloning term from the recorded human demos), then records a fresh
# per-agent-type quota of demos against the NEW checkpoint. The recorder tags
# demos by the checkpoint they were gathered against and only tops up the agents
# that still lack data for the current checkpoint, so re-running is idempotent.
#
# Teammates in the recorder are driven by the latest checkpoint and (with radio)
# choose both movement and radio, so their shared info shows up in the human's
# ego view.
#
# Run from anywhere; everything executes from the repo root so relative paths
# (levels/, experiments/) and the `RL` namespace package resolve.
#
# Override any setting from the environment, e.g.:
#   LEVEL=levels/neighborhood_level ALG=ppo ITERS=6 CHUNK=1000000 ./run_human_loop.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ----- configuration (override from the environment) -----
LEVEL="${LEVEL:-levels/test_level}"      # level directory
ALG="${ALG:-dqn}"                        # dqn | ppo | sac
RUN="${RUN:-1}"                          # run/seed number (recorder teammate-run must match)
EGO_SIZE="${EGO_SIZE:-32}"               # ego window side (recorder and trainer must match)
FRAMES_PER_AGENT="${FRAMES_PER_AGENT:-2500}"  # demos per agent type, per checkpoint
ITERS="${ITERS:-4}"                      # number of train->record iterations
CHUNK="${CHUNK:-500000}"                 # online env frames per RL chunk
# Epsilon anneals along cumulative frames, so span the whole loop (DQN only).
EXPLORE_TOTAL="${EXPLORE_TOTAL:-$((CHUNK * ITERS))}"
USE_RADIO="${USE_RADIO:-1}"              # 1 -> train+use the radio head (teammates transmit)
NUM_ENVS="${NUM_ENVS:-128}"
STEP_DELAY_MS="${STEP_DELAY_MS:-132}"    # recorder pacing (ms/step)

RADIO_FLAG=""
[ "${USE_RADIO}" = "1" ] && RADIO_FLAG="--use-radio"
# --exploration-timesteps is a DQN-only knob (PPO/SAC have no epsilon schedule).
EXPLORE_FLAG=""
[ "${ALG}" = "dqn" ] && EXPLORE_FLAG="--exploration-timesteps ${EXPLORE_TOTAL}"

record () {
    echo "=== [record] ${LEVEL}: ${FRAMES_PER_AGENT} frames/agent-type; teammates=${ALG} run ${RUN} ==="
    python human_dataset.py \
        --level "${LEVEL}" --ego-size "${EGO_SIZE}" \
        --frames-per-agent "${FRAMES_PER_AGENT}" \
        --teammate-alg "${ALG}" --teammate-run "${RUN}" \
        --step-delay-ms "${STEP_DELAY_MS}"
}

train () {
    echo "=== [train] ${ALG} decentralized_ego${RADIO_FLAG:+_radio} run ${RUN}: ${CHUNK} frames (resume + human-bc) ==="
    python -m "RL.cleanrl_${ALG}" \
        --level "${LEVEL}" \
        --no-centralized --ego-view --ego-size "${EGO_SIZE}" ${RADIO_FLAG} \
        --run-number "${RUN}" \
        --human-bc --resume \
        --total-timesteps "${CHUNK}" ${EXPLORE_FLAG} \
        --num-envs "${NUM_ENVS}"
}

echo "Human-in-the-loop pre-training:"
echo "  level=${LEVEL} alg=${ALG} run=${RUN} ego=${EGO_SIZE} radio=${USE_RADIO}"
echo "  iters=${ITERS} chunk=${CHUNK} frames/agent=${FRAMES_PER_AGENT} explore_total=${EXPLORE_TOTAL}"

# Bootstrap: first demos with no checkpoint yet (teammates random move + heuristic
# radio). These seed the BC term for iteration 1's training.
record

for (( i=1; i<=ITERS; i++ )); do
    echo "############################################################"
    echo "##  iteration ${i}/${ITERS}"
    echo "############################################################"
    train
    # Record fresh demos against the new checkpoint for the next iteration; the
    # final iteration ends on a trained model (no trailing record).
    if [ "${i}" -lt "${ITERS}" ]; then
        record
    fi
done

LEVEL_NAME="$(basename "${LEVEL}")"
echo "############################################################"
echo "Done: ${ITERS} RL chunks on ${LEVEL_NAME} (${ALG}, run ${RUN})."
echo "  team-return history : experiments/results/${LEVEL_NAME}/human_returns.jsonl"
echo "  checkpoints         : experiments/results/${LEVEL_NAME}/${ALG}/checkpoints/"
echo "  RL curves           : python -m RL.plot_results"
