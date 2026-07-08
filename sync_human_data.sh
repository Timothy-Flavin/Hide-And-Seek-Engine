#!/bin/bash
# Push the recorded HUMAN DEMO data to every training machine.
#
# The offline+online experiment (run_offline_online.sh) and the per-agent sweep
# read the human demos as their fixed offline dataset, so every machine that runs
# a job needs an up-to-date copy under experiments/results/<level>/<agent_type>/.
# This ships exactly that -- the append-only demo segments, their meta/backups,
# and human_returns.jsonl -- to each host listed below.
#
# What is NOT sent: the RL output dirs (ppo/ dqn/ sac/), which hold each machine's
# OWN trained checkpoints and eval logs. Overwriting those with this box's copy
# would clobber a remote run, so they are excluded. No --delete either, so demos
# already on a machine are never removed (recording is append-only).
#
# SSH auth comes from your ~/.ssh/config (the keys below are ssh profile names).
#
# Usage:
#   ./sync_human_data.sh                 # sync to all machines
#   ./sync_human_data.sh lab-comp timpc  # sync to a subset
#   DRY_RUN=1 ./sync_human_data.sh       # show what would transfer, change nothing
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# host (ssh profile)  ->  repo root on that host
declare -A ROOTS=(
    [lab-comp]="/home/tflavin/Hide-And-Seek-Engine"
    [white-machine]="/home/tflavin/Desktop/Hide-And-Seek-Engine"
    [alienware]="/mnt/hdd/Timmy/Hide-And-Seek-Engine"
    [timpc]="/home/tim/Desktop/Hide-And-Seek-Engine"
)

LOCAL_RESULTS="${SCRIPT_DIR}/experiments/results"
[ -d "${LOCAL_RESULTS}" ] || { echo "!!! no ${LOCAL_RESULTS} to sync" >&2; exit 1; }

# Which hosts: all by default, or the ones named on the command line.
if [ "$#" -gt 0 ]; then
    HOSTS=("$@")
else
    HOSTS=("${!ROOTS[@]}")
fi

RSYNC=(rsync -a --human-readable --info=progress2
       # Human demos only -- never touch a machine's own trained RL outputs.
       --exclude='ppo/' --exclude='dqn/' --exclude='sac/'
       # Skip editor/OS cruft.
       --exclude='__pycache__/' --exclude='.DS_Store')
[ -n "${DRY_RUN:-}" ] && RSYNC+=(--dry-run) && echo "(DRY_RUN: no files will be written)"

fail=0
for host in "${HOSTS[@]}"; do
    root="${ROOTS[$host]:-}"
    if [ -z "${root}" ]; then
        echo "!!! unknown host '${host}' (known: ${!ROOTS[*]})" >&2
        fail=1; continue
    fi
    remote_results="${root}/experiments/results"
    echo "=== [sync] human demos  ->  ${host}:${remote_results}/ ==="
    if ! ssh "${host}" "mkdir -p '${remote_results}'"; then
        echo "!!! cannot reach ${host} or create ${remote_results}" >&2
        fail=1; continue
    fi
    # Trailing slash on the source: copy the CONTENTS of results/ into the remote
    # results/, so <level>/<agent_type>/ lands at the same path on the far side.
    if ! "${RSYNC[@]}" "${LOCAL_RESULTS}/" "${host}:${remote_results}/"; then
        echo "!!! rsync to ${host} failed" >&2
        fail=1; continue
    fi
    echo "    done: ${host}"
done

if [ "${fail}" -ne 0 ]; then
    echo "############################################################"
    echo "Completed with errors (see above)."
    exit 1
fi
echo "############################################################"
echo "Human demo data synced to: ${HOSTS[*]}"
