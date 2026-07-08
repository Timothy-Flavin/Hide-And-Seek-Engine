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

# --- Device white-machine_gpu0 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11]: 17 jobs, ~676.0 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 1/17: warehouse_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 2/17: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 3/17: warehouse_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 4/17: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 5/17: island_level/dqn/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 6/17: island_level/dqn/decentralized_ego_radio_pa_cql seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 7/17: island_level/dqn/decentralized_ego_radio_pa_cql seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 8/17: test_level/dqn/decentralized_ego_radio_pa_cql seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 9/17: neighborhood_level/dqn/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 10/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 11/17: neighborhood_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 12/17: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 13/17: warehouse_level/ppo/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[white-machine:white-machine_gpu0] 14/17: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 15/17: warehouse_level/ppo/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8
  run_if_missing "experiments/results/warehouse_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 16/17: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu0] 17/17: test_level/ppo/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8
) &

# --- Device white-machine_gpu1 [OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15]: 11 jobs, ~662.8 min ---
(
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 1/11: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 2/11: neighborhood_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 3/11: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 4/11: island_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 5/11: warehouse_level/dqn/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 6/11: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 7/11: warehouse_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 8/11: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 9/11: test_level/dqn/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_1_pct100.pt" "[white-machine:white-machine_gpu1] 10/11: test_level/dqn/decentralized_ego_radio_pa_cql seed 1" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[white-machine:white-machine_gpu1] 11/11: test_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000
) &

echo "Launched 2 device(s) on white-machine; waiting..."
wait
echo "All offline+online jobs on white-machine complete."
