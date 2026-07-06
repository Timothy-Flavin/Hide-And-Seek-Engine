#!/usr/bin/env bash
# Auto-generated schedule for alienware (devices run concurrently, no MPS).
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

# --- Device alienware_gpu0 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5]: 15 jobs, ~1481.1 min ---
(
  echo "[alienware:alienware_gpu0] 1/15: island_level/sac/decentralized_ego seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/sac/decentralized_ego seed 1"
  echo "[alienware:alienware_gpu0] 2/15: island_level/sac/decentralized_ego seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/sac/decentralized_ego seed 3"
  echo "[alienware:alienware_gpu0] 3/15: warehouse_level/sac/decentralized_ego seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: warehouse_level/sac/decentralized_ego seed 4"
  echo "[alienware:alienware_gpu0] 4/15: neighborhood_level/ppo/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio seed 1"
  echo "[alienware:alienware_gpu0] 5/15: neighborhood_level/ppo/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio seed 2"
  echo "[alienware:alienware_gpu0] 6/15: neighborhood_level/ppo/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio seed 3"
  echo "[alienware:alienware_gpu0] 7/15: neighborhood_level/ppo/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio seed 4"
  echo "[alienware:alienware_gpu0] 8/15: neighborhood_level/ppo/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio seed 5"
  echo "[alienware:alienware_gpu0] 9/15: neighborhood_level/ppo/decentralized_ego seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego seed 1"
  echo "[alienware:alienware_gpu0] 10/15: neighborhood_level/ppo/decentralized_ego seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego seed 2"
  echo "[alienware:alienware_gpu0] 11/15: neighborhood_level/ppo/decentralized_ego seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego seed 4"
  echo "[alienware:alienware_gpu0] 12/15: neighborhood_level/ppo/decentralized_ego seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego seed 5"
  echo "[alienware:alienware_gpu0] 13/15: island_level/ppo/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/ppo/decentralized_ego_radio seed 3"
  echo "[alienware:alienware_gpu0] 14/15: island_level/ppo/decentralized_ego seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/ppo/decentralized_ego seed 3"
  echo "[alienware:alienware_gpu0] 15/15: island_level/ppo/decentralized_ego seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/ppo/decentralized_ego seed 5"
) &

# --- Device alienware_gpu1 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7]: 18 jobs, ~1494.8 min ---
(
  echo "[alienware:alienware_gpu1] 1/18: island_level/sac/decentralized_ego seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/sac/decentralized_ego seed 2"
  echo "[alienware:alienware_gpu1] 2/18: island_level/sac/decentralized_ego seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/sac/decentralized_ego seed 4"
  echo "[alienware:alienware_gpu1] 3/18: island_level/sac/decentralized_ego seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/sac/decentralized_ego seed 5"
  echo "[alienware:alienware_gpu1] 4/18: neighborhood_level/ppo/decentralized_ego seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: neighborhood_level/ppo/decentralized_ego seed 3"
  echo "[alienware:alienware_gpu1] 5/18: island_level/ppo/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio seed 1"
  echo "[alienware:alienware_gpu1] 6/18: island_level/ppo/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio seed 2"
  echo "[alienware:alienware_gpu1] 7/18: island_level/ppo/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio seed 4"
  echo "[alienware:alienware_gpu1] 8/18: island_level/ppo/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio seed 5"
  echo "[alienware:alienware_gpu1] 9/18: island_level/ppo/decentralized_ego seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego seed 1"
  echo "[alienware:alienware_gpu1] 10/18: island_level/ppo/decentralized_ego seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego seed 2"
  echo "[alienware:alienware_gpu1] 11/18: island_level/ppo/decentralized_ego seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego seed 4"
  echo "[alienware:alienware_gpu1] 12/18: test_level/ppo/decentralized_ego_radio seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio seed 1"
  echo "[alienware:alienware_gpu1] 13/18: test_level/ppo/decentralized_ego_radio seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio seed 2"
  echo "[alienware:alienware_gpu1] 14/18: test_level/ppo/decentralized_ego_radio seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio seed 3"
  echo "[alienware:alienware_gpu1] 15/18: test_level/ppo/decentralized_ego_radio seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio seed 4"
  echo "[alienware:alienware_gpu1] 16/18: test_level/ppo/decentralized_ego_radio seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio seed 5"
  echo "[alienware:alienware_gpu1] 17/18: test_level/ppo/decentralized_ego seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego seed 2"
  echo "[alienware:alienware_gpu1] 18/18: test_level/ppo/decentralized_ego seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego seed 3"
) &

echo "Launched 2 device(s) on alienware; waiting..."
wait
echo "All jobs on alienware complete."
