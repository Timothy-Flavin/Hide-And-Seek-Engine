#!/usr/bin/env bash
# Pull the ad-hoc teamplay matrices (offline_results/adhoc_eval/) back from a
# runner after the remaining evaluations were finished there (see
# push_offline_to_runner.sh + RL/eval_adhoc.py). Default source: timpc.
#
# The runner accumulates its results in the SAME offline_results/adhoc_eval/ dir
# it was seeded with, so after it runs the leftover conditions that folder holds
# the complete set (sac/dqn/ppo x nobc/bc/anneal). This merges them back here.
#
# Usage:
#   ./pull_adhoc_results.sh                 # pull from timpc
#   ./pull_adhoc_results.sh lab-comp        # pull from a different runner
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/offline_results/adhoc_eval/"
mkdir -p "${DEST}"

# SSH alias + remote repo path -- identical to pull/push_offline_results.sh.
MACHINE="${1:-timpc}"
DEFAULT_REMOTE_REPO="/Hide-And-Seek-Engine"
declare -A REMOTE_REPO=(
  [timpc]="~/Desktop/${DEFAULT_REMOTE_REPO}"
  [lab-comp]="~/${DEFAULT_REMOTE_REPO}"
  [white-machine]="~/Desktop/${DEFAULT_REMOTE_REPO}"
  [alienware]="/mnt/hdd/Timmy${DEFAULT_REMOTE_REPO}"
)
repo="${REMOTE_REPO[$MACHINE]:-~${DEFAULT_REMOTE_REPO}}"
SRC="${MACHINE}:${repo}/offline_results/adhoc_eval/"

echo "=== [pull-adhoc] ${SRC} -> ${DEST} ==="
# -a archive, -z compress, -h human, --partial resume, --info=progress2 overall %.
# No --delete: only add/update, never remove local matrices. The runner's copies
# are the authoritative complete set, so newer files overwrite the local partials.
rsync -azh --partial --info=progress2 "${SRC}" "${DEST}" \
  || { echo "!!! FAILED: ${MACHINE}"; exit 1; }

echo
echo "Done -> ${DEST}"
echo "Now inspect / compare the matrices, e.g.:"
echo "  ls ${DEST}"
