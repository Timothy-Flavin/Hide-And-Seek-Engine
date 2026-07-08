#!/usr/bin/env bash
# Auto-generated ANNEAL schedule for white-machine (devices concurrent, no MPS).
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

# Memory-safe overrides for white-machine: {'num_envs_cap': 64}

# --- Device white-machine_gpu0 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11]: 15 jobs, ~173.7 min ---
(
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 1/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 2/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 3/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 4/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[white-machine:white-machine_gpu0] 5/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 6/15: island_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 7/15: warehouse_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 8/15: warehouse_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 9/15: warehouse_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 10/15: warehouse_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[white-machine:white-machine_gpu0] 11/15: warehouse_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 12/15: test_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_3_pct100.pt" "[white-machine:white-machine_gpu0] 13/15: test_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[white-machine:white-machine_gpu0] 14/15: test_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[white-machine:white-machine_gpu0] 15/15: test_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
) &

# --- Device white-machine_gpu1 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15]: 7 jobs, ~172.7 min ---
(
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 1/7: test_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_3_pct100.pt" "[white-machine:white-machine_gpu1] 2/7: test_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 3/7: neighborhood_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_3_pct100.pt" "[white-machine:white-machine_gpu1] 4/7: neighborhood_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 3" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 5/7: island_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[white-machine:white-machine_gpu1] 6/7: island_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 7/7: test_level/ppo/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
) &

echo "Launched 2 device(s) on white-machine; waiting..."
wait
echo "All anneal jobs on white-machine complete."
