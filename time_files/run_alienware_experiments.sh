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

# run_if_missing <ckpt-relpath> <desc> -- <command...>
# The command may start with NAME=VALUE env assignments + a taskset/numactl
# wrapper (device pinning); run it through `env` so those leading assignments
# are applied instead of being treated as a command name.
run_if_missing () {
  local ckpt="$1" desc="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  if [ -f "${ckpt}" ]; then echo "[skip] ${desc} (pct100 exists)"; return 0; fi
  echo "=== [run] ${desc} ==="
  env "$@" || echo "!!! FAILED: ${desc}"
}

# --- Device alienware_gpu0 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5]: 3 jobs, ~853.3 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_run_2_pct100.pt" "[alienware:alienware_gpu0] 1/3: warehouse_level/sac/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_run_3_pct100.pt" "[alienware:alienware_gpu0] 2/3: warehouse_level/sac/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_2_pct100.pt" "[alienware:alienware_gpu0] 3/3: neighborhood_level/ppo/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio
) &

# --- Device alienware_gpu1 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7]: 27 jobs, ~850.6 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_run_1_pct100.pt" "[alienware:alienware_gpu1] 1/27: warehouse_level/sac/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_1_pct100.pt" "[alienware:alienware_gpu1] 2/27: neighborhood_level/ppo/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_3_pct100.pt" "[alienware:alienware_gpu1] 3/27: neighborhood_level/ppo/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_4_pct100.pt" "[alienware:alienware_gpu1] 4/27: neighborhood_level/ppo/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_5_pct100.pt" "[alienware:alienware_gpu1] 5/27: neighborhood_level/ppo/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_run_1_pct100.pt" "[alienware:alienware_gpu1] 6/27: neighborhood_level/ppo/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_run_2_pct100.pt" "[alienware:alienware_gpu1] 7/27: neighborhood_level/ppo/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_run_3_pct100.pt" "[alienware:alienware_gpu1] 8/27: neighborhood_level/ppo/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_run_4_pct100.pt" "[alienware:alienware_gpu1] 9/27: neighborhood_level/ppo/decentralized_ego seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_run_5_pct100.pt" "[alienware:alienware_gpu1] 10/27: neighborhood_level/ppo/decentralized_ego seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --num-minibatches 16 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_1_pct100.pt" "[alienware:alienware_gpu1] 11/27: island_level/ppo/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_2_pct100.pt" "[alienware:alienware_gpu1] 12/27: island_level/ppo/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_3_pct100.pt" "[alienware:alienware_gpu1] 13/27: island_level/ppo/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_4_pct100.pt" "[alienware:alienware_gpu1] 14/27: island_level/ppo/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_5_pct100.pt" "[alienware:alienware_gpu1] 15/27: island_level/ppo/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_run_1_pct100.pt" "[alienware:alienware_gpu1] 16/27: island_level/ppo/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_run_2_pct100.pt" "[alienware:alienware_gpu1] 17/27: island_level/ppo/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_run_3_pct100.pt" "[alienware:alienware_gpu1] 18/27: island_level/ppo/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_run_4_pct100.pt" "[alienware:alienware_gpu1] 19/27: island_level/ppo/decentralized_ego seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_run_5_pct100.pt" "[alienware:alienware_gpu1] 20/27: island_level/ppo/decentralized_ego seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_1_pct100.pt" "[alienware:alienware_gpu1] 21/27: test_level/ppo/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_2_pct100.pt" "[alienware:alienware_gpu1] 22/27: test_level/ppo/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_3_pct100.pt" "[alienware:alienware_gpu1] 23/27: test_level/ppo/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_4_pct100.pt" "[alienware:alienware_gpu1] 24/27: test_level/ppo/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_run_5_pct100.pt" "[alienware:alienware_gpu1] 25/27: test_level/ppo/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_run_4_pct100.pt" "[alienware:alienware_gpu1] 26/27: test_level/ppo/decentralized_ego seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_run_5_pct100.pt" "[alienware:alienware_gpu1] 27/27: test_level/ppo/decentralized_ego seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --num-minibatches 8 --no-centralized --ego-view --ego-size 32
) &

echo "Launched 2 device(s) on alienware; waiting..."
wait
echo "All jobs on alienware complete."
