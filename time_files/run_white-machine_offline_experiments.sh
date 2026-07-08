#!/usr/bin/env bash
# Auto-generated OFFLINE+ONLINE schedule for white-machine (devices concurrent, no MPS).
# Regenerate with: python -m RL.run_scheduler_offline
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
VENV="${VENV:-${REPO_ROOT}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate";
elif [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "WARNING: no venv at ${VENV} and none active; using $(command -v python)" >&2;
fi

# --- Device white-machine_gpu0 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11]: 17 jobs, ~423.1 min ---
(
  echo "[white-machine:white-machine_gpu0] 1/17: island_level/dqn/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu0] FAILED: island_level/dqn/decentralized_ego_radio_pa seed 1"
  echo "[white-machine:white-machine_gpu0] 2/17: island_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu0] FAILED: island_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  echo "[white-machine:white-machine_gpu0] 3/17: test_level/dqn/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu0] FAILED: test_level/dqn/decentralized_ego_radio_pa seed 2"
  echo "[white-machine:white-machine_gpu0] 4/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  echo "[white-machine:white-machine_gpu0] 5/17: neighborhood_level/dqn/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa seed 2"
  echo "[white-machine:white-machine_gpu0] 6/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  echo "[white-machine:white-machine_gpu0] 7/17: neighborhood_level/dqn/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa seed 3"
  echo "[white-machine:white-machine_gpu0] 8/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  echo "[white-machine:white-machine_gpu0] 9/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  echo "[white-machine:white-machine_gpu0] 10/17: neighborhood_level/dqn/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa seed 5"
  echo "[white-machine:white-machine_gpu0] 11/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu0] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  echo "[white-machine:white-machine_gpu0] 12/17: warehouse_level/ppo/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa seed 2"
  echo "[white-machine:white-machine_gpu0] 13/17: warehouse_level/ppo/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa seed 3"
  echo "[white-machine:white-machine_gpu0] 14/17: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  echo "[white-machine:white-machine_gpu0] 15/17: warehouse_level/ppo/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa seed 4"
  echo "[white-machine:white-machine_gpu0] 16/17: warehouse_level/ppo/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa seed 5"
  echo "[white-machine:white-machine_gpu0] 17/17: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[white-machine:white-machine_gpu0] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 5"
) &

# --- Device white-machine_gpu1 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15]: 12 jobs, ~424.7 min ---
(
  echo "[white-machine:white-machine_gpu1] 1/12: island_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu1] FAILED: island_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  echo "[white-machine:white-machine_gpu1] 2/12: test_level/dqn/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa seed 1"
  echo "[white-machine:white-machine_gpu1] 3/12: test_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  echo "[white-machine:white-machine_gpu1] 4/12: test_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  echo "[white-machine:white-machine_gpu1] 5/12: test_level/dqn/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa seed 3"
  echo "[white-machine:white-machine_gpu1] 6/12: test_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  echo "[white-machine:white-machine_gpu1] 7/12: test_level/dqn/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa seed 4"
  echo "[white-machine:white-machine_gpu1] 8/12: test_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  echo "[white-machine:white-machine_gpu1] 9/12: test_level/dqn/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa seed 5"
  echo "[white-machine:white-machine_gpu1] 10/12: test_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[white-machine:white-machine_gpu1] FAILED: test_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  echo "[white-machine:white-machine_gpu1] 11/12: warehouse_level/ppo/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[white-machine:white-machine_gpu1] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa seed 1"
  echo "[white-machine:white-machine_gpu1] 12/12: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[white-machine:white-machine_gpu1] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 2"
) &

echo "Launched 2 device(s) on white-machine; waiting..."
wait
echo "All offline+online jobs on white-machine complete."
