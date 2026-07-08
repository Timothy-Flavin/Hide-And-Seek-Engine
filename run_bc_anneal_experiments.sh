#!/usr/bin/env bash
# BC-annealing arm of the offline sweep: same per-agent-nets offline runs as
# time_files/*_offline_experiments.sh, but with the human-imitation weight
# linearly annealed 1.0 -> 0 over the first ANNEAL frames (default 250k), so BC
# acts as a warmup that online RL then surpasses instead of a permanent anchor.
#
# Saves under a DISTINCT variant name (ppo/sac: '..._pa_bc_anneal', dqn:
# '..._pa_cql_anneal') so these never overwrite the pure-RL ('..._pa') or the
# constant-coefficient ('..._pa_bc' / '..._pa_cql') runs -- all three conditions
# coexist for comparison (see RL/plot_bc_comparison.py).
#
#   ./run_bc_anneal_experiments.sh                 # all algos, all levels, seeds 1-5
#   ANNEAL=250000 NUM_ENVS=128 ./run_bc_anneal_experiments.sh ppo sac
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV="${VENV:-${SCRIPT_DIR}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;
fi

trap 'trap - INT TERM; echo; echo "[interrupted] killing jobs..."; kill 0' INT TERM

ANNEAL="${ANNEAL:-250000}"
NUM_ENVS="${NUM_ENVS:-128}"
FRAMES="${FRAMES:-1000000}"
SEEDS="${SEEDS:-1 2 3 4 5}"
LEVELS="${LEVELS:-test_level island_level neighborhood_level warehouse_level}"
ALGS=("$@"); [ "${#ALGS[@]}" -eq 0 ] && ALGS=(ppo sac dqn)

# obj suffix used in the saved checkpoint name, per algorithm.
obj_suffix () { case "$1" in dqn) echo "cql_anneal";; *) echo "bc_anneal";; esac; }

run_if_missing () {
  local ckpt="$1" desc="$2"; shift 2
  if [ -f "${ckpt}" ]; then echo "[skip] ${desc} (pct100 exists)"; return 0; fi
  echo "=== [run] ${desc} ==="
  "$@" || echo "!!! FAILED: ${desc}"
}

COMMON=(--total-timesteps "${FRAMES}" --num-envs "${NUM_ENVS}" --no-centralized \
        --ego-view --ego-size 32 --use-radio --per-agent-nets \
        --bc-anneal-frames "${ANNEAL}")

for alg in "${ALGS[@]}"; do
  suf="$(obj_suffix "${alg}")"
  for level in ${LEVELS}; do
    for s in ${SEEDS}; do
      ckpt="experiments/results/${level}/${alg}/checkpoints/${alg}_decentralized_ego_radio_pa_${suf}_run_${s}_pct100.pt"
      desc="${alg}/${level}/${suf} seed ${s} (anneal=${ANNEAL})"
      case "${alg}" in
        ppo) run_if_missing "${ckpt}" "${desc}" python -m RL.cleanrl_ppo --run-number "${s}" --level "levels/${level}" "${COMMON[@]}" --human-bc --bc-coef 1.0 ;;
        sac) run_if_missing "${ckpt}" "${desc}" python -m RL.cleanrl_sac --run-number "${s}" --level "levels/${level}" "${COMMON[@]}" --human-bc --bc-coef 1.0 ;;
        dqn) run_if_missing "${ckpt}" "${desc}" python -m RL.cleanrl_dqn --run-number "${s}" --level "levels/${level}" "${COMMON[@]}" --exploration-timesteps "${FRAMES}" --human-cql ;;
      esac
    done
  done
done

echo "All BC-anneal runs complete. Plot with: python -m RL.plot_bc_comparison"
