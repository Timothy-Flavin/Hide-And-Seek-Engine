#!/usr/bin/env bash
# Auto-generated schedule for timpc (devices run concurrently, no MPS).
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

# --- Device timpc_gpu [no pinning]: 41 jobs, ~858.1 min ---
(
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_run_4_pct100.pt" "[timpc:timpc_gpu] 1/41: warehouse_level/sac/decentralized_ego seed 4" -- \
    python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/sac/checkpoints/sac_decentralized_ego_run_5_pct100.pt" "[timpc:timpc_gpu] 2/41: warehouse_level/sac/decentralized_ego seed 5" -- \
    python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_run_1_pct100.pt" "[timpc:timpc_gpu] 3/41: test_level/sac/decentralized_ego seed 1" -- \
    python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_run_2_pct100.pt" "[timpc:timpc_gpu] 4/41: test_level/sac/decentralized_ego seed 2" -- \
    python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_run_3_pct100.pt" "[timpc:timpc_gpu] 5/41: test_level/sac/decentralized_ego seed 3" -- \
    python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_run_4_pct100.pt" "[timpc:timpc_gpu] 6/41: test_level/sac/decentralized_ego seed 4" -- \
    python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_run_5_pct100.pt" "[timpc:timpc_gpu] 7/41: test_level/sac/decentralized_ego seed 5" -- \
    python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_1_pct100.pt" "[timpc:timpc_gpu] 8/41: island_level/dqn/decentralized_ego_radio seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_2_pct100.pt" "[timpc:timpc_gpu] 9/41: island_level/dqn/decentralized_ego_radio seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_4_pct100.pt" "[timpc:timpc_gpu] 10/41: island_level/dqn/decentralized_ego_radio seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_run_1_pct100.pt" "[timpc:timpc_gpu] 11/41: island_level/dqn/decentralized_ego seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_run_2_pct100.pt" "[timpc:timpc_gpu] 12/41: island_level/dqn/decentralized_ego seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_run_3_pct100.pt" "[timpc:timpc_gpu] 13/41: island_level/dqn/decentralized_ego seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_run_4_pct100.pt" "[timpc:timpc_gpu] 14/41: island_level/dqn/decentralized_ego seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_run_5_pct100.pt" "[timpc:timpc_gpu] 15/41: island_level/dqn/decentralized_ego seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_1_pct100.pt" "[timpc:timpc_gpu] 16/41: warehouse_level/dqn/decentralized_ego_radio seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_2_pct100.pt" "[timpc:timpc_gpu] 17/41: warehouse_level/dqn/decentralized_ego_radio seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_3_pct100.pt" "[timpc:timpc_gpu] 18/41: warehouse_level/dqn/decentralized_ego_radio seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_4_pct100.pt" "[timpc:timpc_gpu] 19/41: warehouse_level/dqn/decentralized_ego_radio seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_5_pct100.pt" "[timpc:timpc_gpu] 20/41: warehouse_level/dqn/decentralized_ego_radio seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_run_1_pct100.pt" "[timpc:timpc_gpu] 21/41: warehouse_level/dqn/decentralized_ego seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_run_2_pct100.pt" "[timpc:timpc_gpu] 22/41: warehouse_level/dqn/decentralized_ego seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_run_3_pct100.pt" "[timpc:timpc_gpu] 23/41: warehouse_level/dqn/decentralized_ego seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_run_4_pct100.pt" "[timpc:timpc_gpu] 24/41: warehouse_level/dqn/decentralized_ego seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_run_5_pct100.pt" "[timpc:timpc_gpu] 25/41: warehouse_level/dqn/decentralized_ego seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_2_pct100.pt" "[timpc:timpc_gpu] 26/41: test_level/dqn/decentralized_ego_radio seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_run_1_pct100.pt" "[timpc:timpc_gpu] 27/41: test_level/dqn/decentralized_ego seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_run_2_pct100.pt" "[timpc:timpc_gpu] 28/41: test_level/dqn/decentralized_ego seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_run_3_pct100.pt" "[timpc:timpc_gpu] 29/41: test_level/dqn/decentralized_ego seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_run_4_pct100.pt" "[timpc:timpc_gpu] 30/41: test_level/dqn/decentralized_ego seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_run_5_pct100.pt" "[timpc:timpc_gpu] 31/41: test_level/dqn/decentralized_ego seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_run_1_pct100.pt" "[timpc:timpc_gpu] 32/41: neighborhood_level/dqn/decentralized_ego seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_run_2_pct100.pt" "[timpc:timpc_gpu] 33/41: neighborhood_level/dqn/decentralized_ego seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_run_3_pct100.pt" "[timpc:timpc_gpu] 34/41: neighborhood_level/dqn/decentralized_ego seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_run_4_pct100.pt" "[timpc:timpc_gpu] 35/41: neighborhood_level/dqn/decentralized_ego seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_run_5_pct100.pt" "[timpc:timpc_gpu] 36/41: neighborhood_level/dqn/decentralized_ego seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_1_pct100.pt" "[timpc:timpc_gpu] 37/41: neighborhood_level/dqn/decentralized_ego_radio seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_2_pct100.pt" "[timpc:timpc_gpu] 38/41: neighborhood_level/dqn/decentralized_ego_radio seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_3_pct100.pt" "[timpc:timpc_gpu] 39/41: neighborhood_level/dqn/decentralized_ego_radio seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_4_pct100.pt" "[timpc:timpc_gpu] 40/41: neighborhood_level/dqn/decentralized_ego_radio seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
  run_if_missing "experiments/results/neighborhood_level/dqn/checkpoints/dqn_decentralized_ego_radio_run_5_pct100.pt" "[timpc:timpc_gpu] 41/41: neighborhood_level/dqn/decentralized_ego_radio seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio
) &

echo "Launched 1 device(s) on timpc; waiting..."
wait
echo "All jobs on timpc complete."
