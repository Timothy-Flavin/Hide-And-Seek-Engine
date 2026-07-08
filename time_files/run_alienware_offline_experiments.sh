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

# run_if_missing <ckpt-relpath> <desc> -- <command...>
run_if_missing () {
  local ckpt="$1" desc="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  if [ -f "${ckpt}" ]; then echo "[skip] ${desc} (pct100 exists)"; return 0; fi
  echo "=== [run] ${desc} ==="
  "$@" || echo "!!! FAILED: ${desc}"
}

# Memory-safe overrides for alienware: {'num_envs_cap': 64, 'buffer_size': 50000}

# --- Device alienware_gpu0 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5]: 15 jobs, ~656.8 min ---
(
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu0] 1/15: neighborhood_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[alienware:alienware_gpu0] 2/15: island_level/sac/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu0] 3/15: test_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu0] 4/15: island_level/sac/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu0] 5/15: neighborhood_level/ppo/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[alienware:alienware_gpu0] 6/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu0] 7/15: neighborhood_level/ppo/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16
  run_if_missing "experiments/results/neighborhood_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[alienware:alienware_gpu0] 8/15: neighborhood_level/ppo/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 16 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu0] 9/15: island_level/ppo/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[alienware:alienware_gpu0] 10/15: island_level/ppo/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_2_pct100.pt" "[alienware:alienware_gpu0] 11/15: island_level/ppo/decentralized_ego_radio_pa seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8
  run_if_missing "experiments/results/island_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[alienware:alienware_gpu0] 12/15: island_level/ppo/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu0] 13/15: test_level/ppo/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[alienware:alienware_gpu0] 14/15: test_level/ppo/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/test_level/ppo/checkpoints/ppo_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[alienware:alienware_gpu0] 15/15: test_level/ppo/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python -m RL.cleanrl_ppo --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0
) &

# --- Device alienware_gpu1 [OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7]: 5 jobs, ~676.7 min ---
(
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu1] 1/5: test_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_1_pct100.pt" "[alienware:alienware_gpu1] 2/5: test_level/sac/decentralized_ego_radio_pa_bc seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_run_1_pct100.pt" "[alienware:alienware_gpu1] 3/5: island_level/sac/decentralized_ego_radio_pa seed 1" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[alienware:alienware_gpu1] 4/5: test_level/sac/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
  run_if_missing "experiments/results/island_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_run_2_pct100.pt" "[alienware:alienware_gpu1] 5/5: island_level/sac/decentralized_ego_radio_pa_bc seed 2" -- \
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --buffer-size 50000 --human-bc --bc-coef 1.0
) &

echo "Launched 2 device(s) on alienware; waiting..."
wait
echo "All offline+online jobs on alienware complete."
