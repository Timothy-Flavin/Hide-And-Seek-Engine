#!/bin/bash
# Clear the per-agent OFFLINE+ONLINE experiment's outputs from experiments/results/
# so a fresh offline loop re-runs every job cleanly.
#
# Why: the generated schedules guard each job with run_if_missing, which SKIPS a
# job whose pct100 checkpoint already exists. A stale per-agent checkpoint from an
# aborted attempt (e.g. a run killed mid-way) would be wrongly skipped, and a
# half-written checkpoint could be resumed. Wipe them first for a clean run.
#
# Scope: ONLY the per-agent variants this experiment produces --
#   <alg>_decentralized_ego_radio_pa[_bc|_cql]_run_<seed>_*
# The base RL / centralized / plain-radio sweep results are left untouched (their
# variant names have no `_pa`), so this never deletes the consolidated results.
#
# Run this ON EACH MACHINE that will run the offline loop (checkpoints are local),
# BEFORE launching time_files/run_<machine>_offline_experiments.sh.
#
# Usage:
#   ./clear_offline_checkpoints.sh             # delete (prints every path)
#   DRY_RUN=1 ./clear_offline_checkpoints.sh   # preview only, delete nothing
#   PATTERN='*_pa_bc*' ./clear_offline_checkpoints.sh   # narrow to one arm
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

RESULTS="${RESULTS:-experiments/results}"
# Matches _pa, _pa_bc, _pa_cql; the leading "_pa" after the radio variant means it
# can never match the plain decentralized_ego_radio (non-per-agent) files.
PATTERN="${PATTERN:-*decentralized_ego_radio_pa*}"
DRY="${DRY_RUN:-}"

if [ ! -d "${RESULTS}" ]; then
    echo "No ${RESULTS}/ here; nothing to clear."
    exit 0
fi
[ -n "${DRY}" ] && echo "(DRY_RUN: previewing only, nothing will be deleted)"
echo "Clearing per-agent offline artifacts under ${RESULTS}/ matching '${PATTERN}'"

# 1) Checkpoints (.pt) + per-run episodic-return artifacts (.npy/.png).
n_files=0
while IFS= read -r f; do
    [ -n "${f}" ] || continue
    echo "  rm ${f}"
    [ -z "${DRY}" ] && rm -f "${f}"
    n_files=$((n_files + 1))
done < <(find "${RESULTS}" -type f -name "${PATTERN}")
echo "  -> ${n_files} file(s) matched"

# 2) Prune this experiment's per-agent rows from each eval_returns.jsonl (that file
#    is shared/appended by every variant, so only drop entries whose prefix has _pa).
n_pruned=0
while IFS= read -r jsonl; do
    [ -f "${jsonl}" ] || continue
    removed=$(DRY="${DRY}" python - "${jsonl}" <<'PY'
import json, os, sys
path = sys.argv[1]
dry = os.environ.get("DRY", "")
keep, removed = [], 0
for line in open(path):
    s = line.rstrip("\n")
    if not s.strip():
        continue
    try:
        pref = json.loads(s).get("prefix", "")
    except ValueError:
        keep.append(s); continue
    if "_pa" in pref:          # per-agent entry -> drop
        removed += 1
    else:
        keep.append(s)
if not dry and removed:
    with open(path, "w") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
print(removed)
PY
)
    if [ "${removed:-0}" -gt 0 ]; then
        echo "  ${jsonl}: pruned ${removed} _pa eval row(s)"
        n_pruned=$((n_pruned + removed))
    fi
done < <(find "${RESULTS}" -type f -name "eval_returns.jsonl")
echo "  -> ${n_pruned} eval row(s) pruned"

if [ -n "${DRY}" ]; then
    echo "(DRY_RUN complete: nothing deleted. Re-run without DRY_RUN=1 to apply.)"
else
    echo "Cleared. The offline loop will now re-run every per-agent job from scratch."
fi
