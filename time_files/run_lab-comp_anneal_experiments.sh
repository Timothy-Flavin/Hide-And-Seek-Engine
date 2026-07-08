#!/usr/bin/env bash
# Auto-generated ANNEAL schedule for lab-comp (devices concurrent, no MPS).
# Regenerate with: python -m RL.run_scheduler_offline --arms anneal
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

# --- Device lab-comp_gpu [OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63]: 11 jobs, ~173.5 min ---
(
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 1/11: island_level/sac/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 2/11: island_level/sac/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 3/11: island_level/sac/decentralized_ego_radio_pa_bc_anneal seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 4/11: island_level/sac/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 5/11: island_level/sac/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[lab-comp:lab-comp_gpu] 6/11: warehouse_level/sac/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[lab-comp:lab-comp_gpu] 7/11: warehouse_level/sac/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_3_pct100.pt" "[lab-comp:lab-comp_gpu] 8/11: warehouse_level/sac/decentralized_ego_radio_pa_bc_anneal seed 3" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[lab-comp:lab-comp_gpu] 9/11: warehouse_level/sac/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 10/11: warehouse_level/sac/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[lab-comp:lab-comp_gpu] 11/11: island_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
) &

# --- Device lab-comp_cpu [OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47]: 2 jobs, ~165.5 min ---
(
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_1_pct100.pt" "[lab-comp:lab-comp_cpu] 1/2: island_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 1" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000 --no-cuda
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_3_pct100.pt" "[lab-comp:lab-comp_cpu] 2/2: island_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 3" -- \
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000 --no-cuda
) &

echo "Launched 2 device(s) on lab-comp; waiting..."
wait
echo "All anneal jobs on lab-comp complete."
