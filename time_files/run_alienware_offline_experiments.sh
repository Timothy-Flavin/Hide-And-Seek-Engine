#!/usr/bin/env bash
# Auto-generated OFFLINE+ONLINE schedule for alienware (devices concurrent, no MPS).
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

# --- Device alienware_gpu0 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5]: 16 jobs, ~417.2 min ---
(
  echo "[alienware:alienware_gpu0] 1/16: warehouse_level/sac/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[alienware:alienware_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio_pa seed 4"
  echo "[alienware:alienware_gpu0] 2/16: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 5"
  echo "[alienware:alienware_gpu0] 3/16: neighborhood_level/ppo/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa seed 1"
  echo "[alienware:alienware_gpu0] 4/16: neighborhood_level/ppo/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa seed 2"
  echo "[alienware:alienware_gpu0] 5/16: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  echo "[alienware:alienware_gpu0] 6/16: neighborhood_level/ppo/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa seed 4"
  echo "[alienware:alienware_gpu0] 7/16: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  echo "[alienware:alienware_gpu0] 8/16: island_level/ppo/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/ppo/decentralized_ego_radio_pa seed 1"
  echo "[alienware:alienware_gpu0] 9/16: island_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: island_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  echo "[alienware:alienware_gpu0] 10/16: test_level/ppo/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa seed 1"
  echo "[alienware:alienware_gpu0] 11/16: test_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  echo "[alienware:alienware_gpu0] 12/16: test_level/ppo/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa seed 2"
  echo "[alienware:alienware_gpu0] 13/16: test_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  echo "[alienware:alienware_gpu0] 14/16: test_level/ppo/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa seed 3"
  echo "[alienware:alienware_gpu0] 15/16: test_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  echo "[alienware:alienware_gpu0] 16/16: test_level/ppo/decentralized_ego_radio_pa_bc seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu0] FAILED: test_level/ppo/decentralized_ego_radio_pa_bc seed 5"
) &

# --- Device alienware_gpu1 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7]: 18 jobs, ~423.8 min ---
(
  echo "[alienware:alienware_gpu1] 1/18: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 1"
  echo "[alienware:alienware_gpu1] 2/18: warehouse_level/sac/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[alienware:alienware_gpu1] FAILED: warehouse_level/sac/decentralized_ego_radio_pa seed 2"
  echo "[alienware:alienware_gpu1] 3/18: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  echo "[alienware:alienware_gpu1] 4/18: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  echo "[alienware:alienware_gpu1] 5/18: neighborhood_level/ppo/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 \
    || echo "[alienware:alienware_gpu1] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa seed 3"
  echo "[alienware:alienware_gpu1] 6/18: neighborhood_level/ppo/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 \
    || echo "[alienware:alienware_gpu1] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa seed 5"
  echo "[alienware:alienware_gpu1] 7/18: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 5"
  echo "[alienware:alienware_gpu1] 8/18: island_level/ppo/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa seed 2"
  echo "[alienware:alienware_gpu1] 9/18: island_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa_bc seed 2"
  echo "[alienware:alienware_gpu1] 10/18: island_level/ppo/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa seed 3"
  echo "[alienware:alienware_gpu1] 11/18: island_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa_bc seed 3"
  echo "[alienware:alienware_gpu1] 12/18: island_level/ppo/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa seed 4"
  echo "[alienware:alienware_gpu1] 13/18: island_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  echo "[alienware:alienware_gpu1] 14/18: island_level/ppo/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa seed 5"
  echo "[alienware:alienware_gpu1] 15/18: island_level/ppo/decentralized_ego_radio_pa_bc seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: island_level/ppo/decentralized_ego_radio_pa_bc seed 5"
  echo "[alienware:alienware_gpu1] 16/18: test_level/ppo/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio_pa seed 4"
  echo "[alienware:alienware_gpu1] 17/18: test_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  echo "[alienware:alienware_gpu1] 18/18: test_level/ppo/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 \
    || echo "[alienware:alienware_gpu1] FAILED: test_level/ppo/decentralized_ego_radio_pa seed 5"
) &

echo "Launched 2 device(s) on alienware; waiting..."
wait
echo "All offline+online jobs on alienware complete."
