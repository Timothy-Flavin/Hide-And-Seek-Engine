#!/usr/bin/env bash
# Auto-generated OFFLINE+ONLINE schedule for lab-comp (devices concurrent, no MPS).
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

trap 'trap - INT TERM; echo; echo "[interrupted] killing all jobs on this machine..."; kill 0' INT TERM

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

# --- Device lab-comp_gpu [OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63]: 22 jobs, ~319.8 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 1/22: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 2/22: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 3/22: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 4/22: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 5/22: island_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 6/22: island_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 7/22: island_level/sac/decentralized_ego_radio_pa seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 8/22: island_level/sac/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 9/22: island_level/sac/decentralized_ego_radio_pa seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 10/22: warehouse_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 11/22: warehouse_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 12/22: warehouse_level/sac/decentralized_ego_radio_pa seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 13/22: warehouse_level/sac/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 14/22: warehouse_level/sac/decentralized_ego_radio_pa seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 15/22: test_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 16/22: test_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 17/22: test_level/sac/decentralized_ego_radio_pa seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 18/22: test_level/sac/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 19/22: test_level/sac/decentralized_ego_radio_pa seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 20/22: neighborhood_level/sac/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 21/22: neighborhood_level/sac/decentralized_ego_radio_pa seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 22/22: test_level/dqn/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
) &

# --- Device lab-comp_cpu [OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47]: 5 jobs, ~315.8 min ---
(
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_1_pct100.pt" "[lab-comp:lab-comp_cpu] 1/5: warehouse_level/dqn/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[lab-comp:lab-comp_cpu] 2/5: warehouse_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_3_pct100.pt" "[lab-comp:lab-comp_cpu] 3/5: warehouse_level/dqn/decentralized_ego_radio_pa seed 3" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_4_pct100.pt" "[lab-comp:lab-comp_cpu] 4/5: warehouse_level/dqn/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_5_pct100.pt" "[lab-comp:lab-comp_cpu] 5/5: warehouse_level/dqn/decentralized_ego_radio_pa seed 5" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda
) &

echo "Launched 2 device(s) on lab-comp; waiting..."
wait
echo "All offline+online jobs on lab-comp complete."
