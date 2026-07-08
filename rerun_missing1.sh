#!/usr/bin/env bash
# Rerun of MISSING 5M-frame runs (centralized + a few decentralized_ego SAC).
# Auto-generated; half 1 of 28 total missing jobs (~595 min est). Run on lab-comp.
# Idempotent: each job is skipped if its pct100 checkpoint already exists.
set -uo pipefail
trap 'trap - INT TERM; echo; echo "[interrupted] killing all jobs on this machine..."; kill 0' INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV="${VENV:-${SCRIPT_DIR}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;
fi

run () {  # $1=level $2=alg $3=variant $4=seed  ; remaining args = the command
  local ckpt="experiments/results/$1/$2/checkpoints/$2_$3_run_$4_pct100.pt"
  if [ -f "${ckpt}" ]; then echo "[skip] $1/$2/$3 seed $4 (pct100 exists)"; return 0; fi
  echo "=== [run] $1/$2/$3 seed $4 ==="
  shift 4
  "$@" || echo "!!! FAILED: (see above)"
}

run island_level dqn centralized 4 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --centralized
run island_level sac decentralized_ego 4 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
run island_level sac decentralized_ego 5 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
run neighborhood_level dqn centralized 4 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --centralized
run neighborhood_level ppo centralized 4 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --centralized
run neighborhood_level sac centralized 4 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --centralized
run test_level dqn centralized 4 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --centralized
run test_level ppo centralized 4 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --centralized
run test_level sac centralized 4 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --centralized
run test_level sac centralized 5 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --centralized
run warehouse_level dqn centralized 4 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --centralized
run warehouse_level ppo centralized 4 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --centralized
run warehouse_level sac centralized 3 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --centralized
run warehouse_level sac centralized 5 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --centralized
echo "rerun_missing1 complete."
