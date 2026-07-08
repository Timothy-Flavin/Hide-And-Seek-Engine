#!/usr/bin/env bash
# Auto-generated schedule for lab-comp (devices run concurrently, no MPS).
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

# --- Device lab-comp_gpu [OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63]: 23 jobs, ~857.0 min ---
(
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 1/23: island_level/sac/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 2/23: island_level/sac/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 3/23: island_level/sac/decentralized_ego seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 4/23: island_level/sac/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 5/23: island_level/sac/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 6/23: island_level/sac/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 7/23: island_level/sac/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 8/23: island_level/sac/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 9/23: test_level/sac/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 10/23: test_level/sac/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 11/23: test_level/sac/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 12/23: test_level/sac/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 13/23: test_level/sac/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 14/23: neighborhood_level/sac/decentralized_ego_radio seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 15/23: neighborhood_level/sac/decentralized_ego_radio seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 16/23: neighborhood_level/sac/decentralized_ego_radio seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 17/23: neighborhood_level/sac/decentralized_ego_radio seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 18/23: neighborhood_level/sac/decentralized_ego_radio seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 19/23: neighborhood_level/sac/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 20/23: neighborhood_level/sac/decentralized_ego seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 21/23: neighborhood_level/sac/decentralized_ego seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 22/23: neighborhood_level/sac/decentralized_ego seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 23/23: neighborhood_level/sac/decentralized_ego seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
) &

# --- Device lab-comp_cpu [OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47]: 2 jobs, ~833.3 min ---
(
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_run_1_pct100.pt" "[lab-comp:lab-comp_cpu] 1/2: island_level/sac/decentralized_ego seed 1" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --no-cuda
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_run_4_pct100.pt" "[lab-comp:lab-comp_cpu] 2/2: island_level/sac/decentralized_ego seed 4" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --no-cuda
) &

echo "Launched 2 device(s) on lab-comp; waiting..."
wait
echo "All jobs on lab-comp complete."
