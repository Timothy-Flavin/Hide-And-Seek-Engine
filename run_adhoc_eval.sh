#!/usr/bin/env bash
# Ad-hoc teamplay evaluation for the BC-impact study. Runs on timpc (GPU),
# everything in eval / inference mode with no gradients, over the offline
# per-agent-nets checkpoints in ./offline_results/.
#
# For every algorithm (sac, dqn, ppo) and every condition (bc, nobc) it
# evaluates all 5**3 = 125 seed-mixed team compositions per level x 4 levels
# = 500 compositions, 32 episodes each over 32 parallel envs, and writes a
# 5x5x5 team-score matrix per level to offline_results/adhoc_eval/.
#
#   ./run_adhoc_eval.sh                 # all algos, both conditions
#   ./run_adhoc_eval.sh sac dqn         # subset of algos, both conditions
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

VENV="${VENV:-${SCRIPT_DIR}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;
fi

ALGS=("$@")
[ "${#ALGS[@]}" -eq 0 ] && ALGS=(sac dqn ppo)

DEVICE="${DEVICE:-cuda}"

for alg in "${ALGS[@]}"; do
  echo "=== [adhoc] ${alg} : bc + nobc ==="
  python -m RL.eval_adhoc --alg "${alg}" --conditions bc nobc \
    --device "${DEVICE}" --num-envs 32 --episodes 32 \
    || echo "!!! FAILED: ${alg}"
done

echo "All ad-hoc evaluations complete -> offline_results/adhoc_eval/"
