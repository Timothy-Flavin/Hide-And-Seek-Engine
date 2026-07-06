#!/bin/bash
# Human-in-the-loop pre-training, split across two machines:
#
#   * RECORD runs HERE (this machine has the display; the human plays).
#   * TRAIN runs on the LAB computer (headless, GPU, ~300 miles away) over SSH.
#
# Each iteration: record demos locally against the current checkpoint -> rsync the
# demos up to the lab -> train one resumable chunk per level on the lab -> rsync the
# fresh checkpoints back down (so the NEXT local recording drives its teammates from,
# and tags its demos against, the just-trained checkpoint). The interleave matches
# run_human_loop.sh exactly; only the train phase is offloaded.
#
# Requirements on the lab:
#   * the repo already checked out at REMOTE_DIR (git pull it -- this script does NOT
#     sync code, only the experiments/results data);
#   * its venv built/installed (the RL trainers import the compiled engine);
#   * run_human_loop.sh present (this drives training via `PHASE=train`).
#
# SSH auth is handled by your ~/.ssh/config profile (REMOTE_HOST, default 'lab-comp').
#
# Override any setting from the environment, e.g.:
#   LEVELS="levels/test_level" ALG=ppo ITERS=6 CHUNK=1000000 ./run_human_loop_remote.sh
#   REMOTE_HOST=lab-comp REMOTE_DIR=/home/tflavin/Hide-And-Seek-Engine ./run_human_loop_remote.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ----- remote target -----
REMOTE_HOST="${REMOTE_HOST:-lab-comp}"                       # ~/.ssh/config profile
REMOTE_DIR="${REMOTE_DIR:-/home/tflavin/Hide-And-Seek-Engine}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_DIR}/.venv}"           # activated before training

# ----- loop / model config (same names + defaults as run_human_loop.sh) -----
# Exported so the local `PHASE=record run_human_loop.sh` and the remote
# `PHASE=train run_human_loop.sh` both inherit them.
export LEVELS="${LEVELS:-levels/test_level levels/neighborhood_level levels/island_level levels/warehouse_level}"
[ -n "${LEVEL:-}" ] && export LEVELS="${LEVEL}"
export ALG="${ALG:-dqn}"
export RUN="${RUN:-1}"
export EGO_SIZE="${EGO_SIZE:-32}"
export FRAMES_PER_AGENT="${FRAMES_PER_AGENT:-2500}"
export ITERS="${ITERS:-4}"
export CHUNK="${CHUNK:-200000}"
export EXPLORE_TOTAL="${EXPLORE_TOTAL:-$((CHUNK * ITERS))}"  # DQN epsilon spans the whole loop
export USE_RADIO="${USE_RADIO:-1}"
export NUM_ENVS="${NUM_ENVS:-128}"
export STEP_DELAY_MS="${STEP_DELAY_MS:-132}"

LOCAL_RESULTS="${SCRIPT_DIR}/experiments/results"
REMOTE_RESULTS="${REMOTE_DIR}/experiments/results"
RSYNC=(rsync -a --partial --human-readable --info=progress2)

die () { echo "!!! $*" >&2; exit 1; }

# Push locally-recorded demos (segments + human_returns.jsonl) up to the lab.
# Excludes checkpoints/ so a stale local copy can never overwrite the lab's newer
# weights; no --delete, so append-only demos on either side are never removed.
sync_demos_up () {
    echo "--- [sync] demos  ${LOCAL_RESULTS}/  ->  ${REMOTE_HOST}:${REMOTE_RESULTS}/"
    mkdir -p "${LOCAL_RESULTS}"
    ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_RESULTS}'" \
        || die "cannot reach ${REMOTE_HOST} or create ${REMOTE_RESULTS}"
    "${RSYNC[@]}" --exclude='checkpoints/' \
        "${LOCAL_RESULTS}/" "${REMOTE_HOST}:${REMOTE_RESULTS}/" \
        || die "demo up-sync failed"
}

# Pull ONLY the trained checkpoints back down (weights + resume state), one level
# at a time so we copy exactly <level>/<alg>/checkpoints/ and nothing else -- no
# fragile rsync path filters, no touching the local demos.
sync_checkpoints_down () {
    echo "--- [sync] checkpoints  ${REMOTE_HOST}:${REMOTE_RESULTS}/*/${ALG}/checkpoints/  ->  ${LOCAL_RESULTS}/"
    for lvl in ${LEVELS}; do
        ln="$(basename "${lvl}")"
        local_ckpt="${LOCAL_RESULTS}/${ln}/${ALG}/checkpoints/"
        mkdir -p "${local_ckpt}"
        "${RSYNC[@]}" \
            "${REMOTE_HOST}:${REMOTE_RESULTS}/${ln}/${ALG}/checkpoints/" \
            "${local_ckpt}" \
            || die "checkpoint down-sync failed for ${ln}"
    done
}

# One local recording pass over all levels (reuses run_human_loop.sh's record_all).
record_local () {
    echo "=== [record @ local] all levels, ${FRAMES_PER_AGENT} frames/agent-type ==="
    PHASE=record bash "${SCRIPT_DIR}/run_human_loop.sh" || die "local recording failed"
}

# One training pass over all levels ON THE LAB (reuses run_human_loop.sh's train_all).
# The here-doc is expanded HERE, so config values are baked into the script text sent
# over stdin -- no ssh argv re-splitting, so LEVELS with spaces survives intact.
# Escaped \$ / \${...} are evaluated remotely.
train_remote () {
    echo "=== [train @ ${REMOTE_HOST}] all levels, ${CHUNK} frames/level (resume + human-bc) ==="
    ssh "${REMOTE_HOST}" bash -s <<REMOTE || die "remote training failed"
set -uo pipefail
cd "${REMOTE_DIR}" || { echo "no such dir: ${REMOTE_DIR}" >&2; exit 1; }
export PYTHONPATH="${REMOTE_DIR}:\${PYTHONPATH:-}"
if [ -f "${REMOTE_VENV}/bin/activate" ]; then
    source "${REMOTE_VENV}/bin/activate"
else
    echo "WARNING: no venv at ${REMOTE_VENV}; using \$(command -v python)" >&2
fi
[ -f run_human_loop.sh ] || { echo "run_human_loop.sh missing on ${REMOTE_HOST} (git pull the repo)" >&2; exit 1; }
export LEVELS="${LEVELS}" ALG="${ALG}" RUN="${RUN}" EGO_SIZE="${EGO_SIZE}" \
       CHUNK="${CHUNK}" EXPLORE_TOTAL="${EXPLORE_TOTAL}" \
       USE_RADIO="${USE_RADIO}" NUM_ENVS="${NUM_ENVS}"
PHASE=train bash run_human_loop.sh
REMOTE
}

echo "Remote human-in-the-loop pre-training:"
echo "  local record  -> ${REMOTE_HOST}:${REMOTE_DIR} train -> checkpoints back"
echo "  levels=[${LEVELS}] alg=${ALG} run=${RUN} ego=${EGO_SIZE} radio=${USE_RADIO}"
echo "  iters=${ITERS} chunk=${CHUNK} frames/agent=${FRAMES_PER_AGENT} explore_total=${EXPLORE_TOTAL}"

# Fail fast if the lab is unreachable or the repo/script isn't there yet.
ssh "${REMOTE_HOST}" "test -f '${REMOTE_DIR}/run_human_loop.sh'" \
    || die "can't reach ${REMOTE_HOST} or ${REMOTE_DIR}/run_human_loop.sh missing (git pull the repo on the lab)"

# Bootstrap: first demos with no checkpoint yet (random-move + heuristic-radio
# teammates). These seed the BC term for iteration 1.
record_local

for (( i=1; i<=ITERS; i++ )); do
    echo "############################################################"
    echo "##  iteration ${i}/${ITERS}"
    echo "############################################################"
    sync_demos_up          # ship the freshly recorded demos to the lab
    train_remote           # lab trains one chunk/level from them (+ resume state)
    sync_checkpoints_down  # bring the new checkpoints home for the next record
    # Final iteration ends on trained models (no trailing record), matching
    # run_human_loop.sh.
    if [ "${i}" -lt "${ITERS}" ]; then
        record_local       # record against the checkpoints we just pulled
    fi
done

echo "############################################################"
echo "Done: ${ITERS} remote RL chunks per level (${ALG}, run ${RUN}) over [${LEVELS}]."
echo "Checkpoints pulled to: ${LOCAL_RESULTS}/<level>/${ALG}/checkpoints/"
for lvl in ${LEVELS}; do
    ln="$(basename "${lvl}")"
    echo "  ${ln}: demos experiments/results/${ln}/ ; checkpoints experiments/results/${ln}/${ALG}/checkpoints/"
done
echo "  RL curves (from the lab's run logs): python -m RL.plot_results"
