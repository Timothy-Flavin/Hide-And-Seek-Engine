#!/usr/bin/env bash
# Pull offline_experiments results from all runner machines into one merged
# tree on this laptop: ./offline_results/
#
# Machines are referenced by SSH config aliases (~/.ssh/config Host entries):
#   timpc, lab-comp, white-machine, alienware
#
# Each machine writes its runs to  <repo>/experiments/results/  and the level/
# algo/seed checkpoints don't collide across machines, so everything merges
# into a single offline_results/ tree.
#
# Usage:
#   ./pull_offline_results.sh              # pull from all machines
#   ./pull_offline_results.sh timpc alienware   # pull from a subset
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/offline_results"

# SSH aliases to pull from (must match Host entries in ~/.ssh/config).
MACHINES=(timpc lab-comp white-machine alienware)

# Path to the repo on each remote. Same for all by default; override per-host
# below if a machine keeps the repo somewhere else.
DEFAULT_REMOTE_REPO="/Hide-And-Seek-Engine"
declare -A REMOTE_REPO=(
  [timpc]="~/Desktop/${DEFAULT_REMOTE_REPO}"
  [lab-comp]="~/${DEFAULT_REMOTE_REPO}"
  [white-machine]="~/Desktop/${DEFAULT_REMOTE_REPO}"
  [alienware]="/mnt/hdd/Timmy${DEFAULT_REMOTE_REPO}"
)

# Allow selecting a subset on the command line.
if [ "$#" -gt 0 ]; then
  MACHINES=("$@")
fi

mkdir -p "${DEST}"

rc=0
for m in "${MACHINES[@]}"; do
  repo="${REMOTE_REPO[$m]:-$DEFAULT_REMOTE_REPO}"
  src="${m}:${repo}/experiments/results/"
  echo "=== [pull] ${m} : ${src} -> ${DEST}/ ==="
  # -a archive, -z compress, -h human, --partial resume, --info=progress2 overall %.
  # Trailing slash on src means "contents of results/", merged into DEST.
  # No --delete: we only ever add, never remove local files.
  rsync -azh --partial --info=progress2 "${src}" "${DEST}/" \
    || { echo "!!! FAILED: ${m}"; rc=1; }
done

echo
if [ "${rc}" -eq 0 ]; then
  echo "All machines pulled into ${DEST}/"
else
  echo "Done with errors (see !!! FAILED lines above)."
fi
exit "${rc}"
