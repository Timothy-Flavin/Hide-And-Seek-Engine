#!/usr/bin/env bash
# Auto-generated schedule for white-machine (devices run concurrently, no MPS).
# Regenerate with: python -m RL.run_scheduler
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Use the repo-local venv (its CUDA torch), not whatever python is on PATH.
# Set VENV=... if your venv lives elsewhere.
VENV="${VENV:-${REPO_ROOT}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;
fi

# run_if_missing <ckpt-relpath> <desc> -- <command...>
run_if_missing () {
  local ckpt="$1" desc="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  if [ -f "${ckpt}" ]; then echo "[skip] ${desc} (pct100 exists)"; return 0; fi
  echo "=== [run] ${desc} ==="
  "$@" || echo "!!! FAILED: ${desc}"
}

# --- Device white-machine_gpu0 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11]: 17 jobs, ~858.3 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 1/17: warehouse_level/sac/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 2/17: warehouse_level/sac/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 3/17: warehouse_level/sac/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 4/17: test_level/dqn/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 5/17: warehouse_level/ppo/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 6/17: warehouse_level/ppo/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 7/17: warehouse_level/ppo/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 8/17: warehouse_level/ppo/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_5_pct100.pt" "[white-machine:white-machine_gpu0] 9/17: warehouse_level/ppo/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 10/17: warehouse_level/ppo/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 11/17: warehouse_level/ppo/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 12/17: warehouse_level/ppo/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 13/17: warehouse_level/ppo/decentralized_ego seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_run_5_pct100.pt" "[white-machine:white-machine_gpu0] 14/17: warehouse_level/ppo/decentralized_ego seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 15/17: test_level/ppo/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 16/17: test_level/ppo/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 17/17: test_level/ppo/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
) &

# --- Device white-machine_gpu1 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15]: 7 jobs, ~857.0 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 1/7: warehouse_level/sac/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_run_5_pct100.pt" "[white-machine:white-machine_gpu1] 2/7: warehouse_level/sac/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_3_pct100.pt" "[white-machine:white-machine_gpu1] 3/7: island_level/dqn/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_5_pct100.pt" "[white-machine:white-machine_gpu1] 4/7: island_level/dqn/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 5/7: test_level/dqn/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_3_pct100.pt" "[white-machine:white-machine_gpu1] 6/7: test_level/dqn/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_5_pct100.pt" "[white-machine:white-machine_gpu1] 7/7: test_level/dqn/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
) &

echo "Launched 2 device(s) on white-machine; waiting..."
wait
echo "All jobs on white-machine complete."
