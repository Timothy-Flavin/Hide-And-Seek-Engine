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

# Memory-safe overrides for alienware: {'num_envs_cap': 64, 'buffer_size': 50000}

# --- Device alienware_gpu0 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5]: 12 jobs, ~319.3 min ---
(
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_4_pct100.pt" "[alienware:alienware_gpu0] 1/12: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_1_pct100.pt" "[alienware:alienware_gpu0] 2/12: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000 --human-cql
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_2_pct100.pt" "[alienware:alienware_gpu0] 3/12: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000 --human-cql
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_3_pct100.pt" "[alienware:alienware_gpu0] 4/12: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000 --human-cql
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_4_pct100.pt" "[alienware:alienware_gpu0] 5/12: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000 --human-cql
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_run_5_pct100.pt" "[alienware:alienware_gpu0] 6/12: neighborhood_level/dqn/decentralized_ego_radio_pa_cql seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000 --human-cql
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu0] 7/12: test_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu0] 8/12: neighborhood_level/dqn/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu0] 9/12: neighborhood_level/dqn/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_3_pct100.pt" "[alienware:alienware_gpu0] 10/12: neighborhood_level/dqn/decentralized_ego_radio_pa seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_4_pct100.pt" "[alienware:alienware_gpu0] 11/12: neighborhood_level/dqn/decentralized_ego_radio_pa seed 4" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_run_5_pct100.pt" "[alienware:alienware_gpu0] 12/12: neighborhood_level/dqn/decentralized_ego_radio_pa seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --buffer-size 50000
) &

# --- Device alienware_gpu1 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7]: 7 jobs, ~319.0 min ---
(
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[alienware:alienware_gpu1] 1/7: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[alienware:alienware_gpu1] 2/7: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_3_pct100.pt" "[alienware:alienware_gpu1] 3/7: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_5_pct100.pt" "[alienware:alienware_gpu1] 4/7: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 5" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu1] 5/7: neighborhood_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu1] 6/7: neighborhood_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_3_pct100.pt" "[alienware:alienware_gpu1] 7/7: neighborhood_level/sac/decentralized_ego_radio_pa seed 3" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
) &

echo "Launched 2 device(s) on alienware; waiting..."
wait
echo "All offline+online jobs on alienware complete."
