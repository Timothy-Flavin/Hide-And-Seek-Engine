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

# --- Device white-machine_gpu0 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11]: 23 jobs, ~1491.2 min ---
(
  echo "[white-machine:white-machine_gpu0] 1/23: warehouse_level/sac/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio seed 1"
  echo "[white-machine:white-machine_gpu0] 2/23: warehouse_level/sac/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio seed 2"
  echo "[white-machine:white-machine_gpu0] 3/23: warehouse_level/sac/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio seed 3"
  echo "[white-machine:white-machine_gpu0] 4/23: warehouse_level/sac/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio seed 4"
  echo "[white-machine:white-machine_gpu0] 5/23: warehouse_level/sac/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio seed 5"
  echo "[white-machine:white-machine_gpu0] 6/23: neighborhood_level/dqn/decentralized_ego seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego seed 1"
  echo "[white-machine:white-machine_gpu0] 7/23: neighborhood_level/dqn/decentralized_ego seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego seed 2"
  echo "[white-machine:white-machine_gpu0] 8/23: neighborhood_level/dqn/decentralized_ego seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego seed 3"
  echo "[white-machine:white-machine_gpu0] 9/23: neighborhood_level/dqn/decentralized_ego seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego seed 4"
  echo "[white-machine:white-machine_gpu0] 10/23: neighborhood_level/dqn/decentralized_ego seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego seed 5"
  echo "[white-machine:white-machine_gpu0] 11/23: warehouse_level/ppo/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio seed 1"
  echo "[white-machine:white-machine_gpu0] 12/23: warehouse_level/ppo/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio seed 2"
  echo "[white-machine:white-machine_gpu0] 13/23: warehouse_level/ppo/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio seed 3"
  echo "[white-machine:white-machine_gpu0] 14/23: warehouse_level/ppo/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio seed 4"
  echo "[white-machine:white-machine_gpu0] 15/23: warehouse_level/ppo/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio seed 5"
  echo "[white-machine:white-machine_gpu0] 16/23: warehouse_level/ppo/decentralized_ego seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego seed 1"
  echo "[white-machine:white-machine_gpu0] 17/23: warehouse_level/ppo/decentralized_ego seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego seed 2"
  echo "[white-machine:white-machine_gpu0] 18/23: warehouse_level/ppo/decentralized_ego seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego seed 3"
  echo "[white-machine:white-machine_gpu0] 19/23: warehouse_level/ppo/decentralized_ego seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego seed 4"
  echo "[white-machine:white-machine_gpu0] 20/23: warehouse_level/ppo/decentralized_ego seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego seed 5"
  echo "[white-machine:white-machine_gpu0] 21/23: test_level/ppo/decentralized_ego seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: test_level/ppo/decentralized_ego seed 1"
  echo "[white-machine:white-machine_gpu0] 22/23: test_level/ppo/decentralized_ego seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: test_level/ppo/decentralized_ego seed 4"
  echo "[white-machine:white-machine_gpu0] 23/23: test_level/ppo/decentralized_ego seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu0] FAILED: test_level/ppo/decentralized_ego seed 5"
) &

# --- Device white-machine_gpu1 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15]: 17 jobs, ~1494.8 min ---
(
  echo "[white-machine:white-machine_gpu1] 1/17: island_level/dqn/decentralized_ego seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego seed 2"
  echo "[white-machine:white-machine_gpu1] 2/17: island_level/dqn/decentralized_ego seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego seed 4"
  echo "[white-machine:white-machine_gpu1] 3/17: island_level/dqn/decentralized_ego seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego seed 5"
  echo "[white-machine:white-machine_gpu1] 4/17: island_level/dqn/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego_radio seed 1"
  echo "[white-machine:white-machine_gpu1] 5/17: island_level/dqn/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego_radio seed 2"
  echo "[white-machine:white-machine_gpu1] 6/17: island_level/dqn/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego_radio seed 3"
  echo "[white-machine:white-machine_gpu1] 7/17: island_level/dqn/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego_radio seed 4"
  echo "[white-machine:white-machine_gpu1] 8/17: island_level/dqn/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego_radio seed 5"
  echo "[white-machine:white-machine_gpu1] 9/17: warehouse_level/dqn/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: warehouse_level/dqn/decentralized_ego_radio seed 3"
  echo "[white-machine:white-machine_gpu1] 10/17: test_level/dqn/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio seed 1"
  echo "[white-machine:white-machine_gpu1] 11/17: test_level/dqn/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio seed 2"
  echo "[white-machine:white-machine_gpu1] 12/17: test_level/dqn/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio seed 3"
  echo "[white-machine:white-machine_gpu1] 13/17: test_level/dqn/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio seed 4"
  echo "[white-machine:white-machine_gpu1] 14/17: test_level/dqn/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio seed 5"
  echo "[white-machine:white-machine_gpu1] 15/17: test_level/dqn/decentralized_ego seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego seed 1"
  echo "[white-machine:white-machine_gpu1] 16/17: test_level/dqn/decentralized_ego seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego seed 4"
  echo "[white-machine:white-machine_gpu1] 17/17: test_level/dqn/decentralized_ego seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego seed 5"
) &

echo "Launched 2 device(s) on white-machine; waiting..."
wait
echo "All jobs on white-machine complete."
