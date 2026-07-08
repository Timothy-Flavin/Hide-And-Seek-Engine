#!/usr/bin/env bash
# Push the merged offline_results/ (all machines' runs, gathered here by
# pull_offline_results.sh) PLUS the existing ad-hoc matrices
# (offline_results/adhoc_eval/) UP to a runner, so the remaining ad-hoc
# evaluation can finish there. Default target: timpc.
#
# Why: RL/eval_adhoc.py builds seed-mixed teams from ALL 5 seeds per role, but
# those seeds were trained across the 4 machines. Only this laptop has the merged
# set (in offline_results/). timpc needs that merged tree to run the eval, and the
# adhoc_eval/ matrices so --resume skips the ones already computed here.
#
# Workflow:
#   ./pull_offline_results.sh              # 1. merge all machines' runs (incl. anneal) here
#   ./push_offline_to_runner.sh            # 2. send merged tree (+ adhoc_eval) to timpc
#   SLIM=1 ./push_offline_to_runner.sh     #    ...or send only pct100 ckpts + adhoc_eval
#   ./push_offline_to_runner.sh lab-comp   #    ...or push to a different runner
#   # 3. on the runner, from the repo root:
#   #   TORCHDYNAMO_DISABLE=1 python -m RL.eval_adhoc --alg all \
#   #       --conditions nobc bc anneal --resume --device cuda
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/offline_results/"
if [ ! -d "${SRC}" ]; then
  echo "!!! no ${SRC} -- run ./pull_offline_results.sh first." >&2
  exit 1
fi

# SSH alias + remote repo path -- kept identical to pull_offline_results.sh so the
# remote layout matches (each Host entry lives in ~/.ssh/config).
MACHINE="${1:-timpc}"
DEFAULT_REMOTE_REPO="/Hide-And-Seek-Engine"
declare -A REMOTE_REPO=(
  [timpc]="~/Desktop/${DEFAULT_REMOTE_REPO}"
  [lab-comp]="~/${DEFAULT_REMOTE_REPO}"
  [white-machine]="~/Desktop/${DEFAULT_REMOTE_REPO}"
  [alienware]="/mnt/hdd/Timmy${DEFAULT_REMOTE_REPO}"
)
repo="${REMOTE_REPO[$MACHINE]:-~${DEFAULT_REMOTE_REPO}}"
DEST="${MACHINE}:${repo}/offline_results/"

# SLIM=1: send only what the ad-hoc eval reads -- the final (pct100) checkpoints and
# the adhoc_eval/ matrices -- instead of every fractional checkpoint + npy/png.
FILTERS=()
if [ "${SLIM:-0}" = "1" ]; then
  FILTERS=(--include='*/'
           --include='*_pct100.pt'
           --include='adhoc_eval/**'
           --exclude='*')
  echo "=== [push:SLIM] pct100 checkpoints + adhoc_eval only ==="
else
  echo "=== [push:FULL] entire offline_results/ ==="
fi

echo "=== ${SRC} -> ${DEST} ==="
# -a archive, -z compress, -h human, --partial resume, --info=progress2 overall %.
# No --delete: only ever add on the remote, never remove.
rsync -azh --partial --info=progress2 "${FILTERS[@]}" "${SRC}" "${DEST}"

echo
echo "Done -> ${DEST}"
echo "On ${MACHINE}, from the repo root, finish the ad-hoc eval:"
echo "  TORCHDYNAMO_DISABLE=1 python -m RL.eval_adhoc --alg all \\"
echo "      --conditions nobc bc anneal --resume --device cuda"
