#!/usr/bin/env bash
# Auto-generated OFFLINE+ONLINE schedule for timpc (devices concurrent, no MPS).
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

# --- Device timpc_gpu [no pinning]: 29 jobs, ~424.4 min ---
(
  echo "[timpc:timpc_gpu] 1/29: island_level/sac/decentralized_ego_radio_pa seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[timpc:timpc_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa seed 2"
  echo "[timpc:timpc_gpu] 2/29: test_level/sac/decentralized_ego_radio_pa_bc seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa_bc seed 1"
  echo "[timpc:timpc_gpu] 3/29: test_level/sac/decentralized_ego_radio_pa_bc seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa_bc seed 4"
  echo "[timpc:timpc_gpu] 4/29: test_level/sac/decentralized_ego_radio_pa_bc seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa_bc seed 5"
  echo "[timpc:timpc_gpu] 5/29: neighborhood_level/sac/decentralized_ego_radio_pa seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa seed 1"
  echo "[timpc:timpc_gpu] 6/29: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 1"
  echo "[timpc:timpc_gpu] 7/29: neighborhood_level/sac/decentralized_ego_radio_pa seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa seed 2"
  echo "[timpc:timpc_gpu] 8/29: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 2"
  echo "[timpc:timpc_gpu] 9/29: neighborhood_level/sac/decentralized_ego_radio_pa seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa seed 3"
  echo "[timpc:timpc_gpu] 10/29: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 3"
  echo "[timpc:timpc_gpu] 11/29: neighborhood_level/sac/decentralized_ego_radio_pa seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa seed 4"
  echo "[timpc:timpc_gpu] 12/29: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 4"
  echo "[timpc:timpc_gpu] 13/29: neighborhood_level/sac/decentralized_ego_radio_pa seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa seed 5"
  echo "[timpc:timpc_gpu] 14/29: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio_pa_bc seed 5"
  echo "[timpc:timpc_gpu] 15/29: island_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[timpc:timpc_gpu] FAILED: island_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  echo "[timpc:timpc_gpu] 16/29: island_level/dqn/decentralized_ego_radio_pa seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: island_level/dqn/decentralized_ego_radio_pa seed 4"
  echo "[timpc:timpc_gpu] 17/29: island_level/dqn/decentralized_ego_radio_pa seed 5"
  python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: island_level/dqn/decentralized_ego_radio_pa seed 5"
  echo "[timpc:timpc_gpu] 18/29: warehouse_level/dqn/decentralized_ego_radio_pa seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa seed 1"
  echo "[timpc:timpc_gpu] 19/29: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 1"
  echo "[timpc:timpc_gpu] 20/29: warehouse_level/dqn/decentralized_ego_radio_pa seed 2"
  python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa seed 2"
  echo "[timpc:timpc_gpu] 21/29: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 2"
  echo "[timpc:timpc_gpu] 22/29: warehouse_level/dqn/decentralized_ego_radio_pa seed 3"
  python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa seed 3"
  echo "[timpc:timpc_gpu] 23/29: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  echo "[timpc:timpc_gpu] 24/29: warehouse_level/dqn/decentralized_ego_radio_pa seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa seed 4"
  echo "[timpc:timpc_gpu] 25/29: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 4"
  echo "[timpc:timpc_gpu] 26/29: warehouse_level/dqn/decentralized_ego_radio_pa seed 5"
  python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa seed 5"
  echo "[timpc:timpc_gpu] 27/29: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  echo "[timpc:timpc_gpu] 28/29: neighborhood_level/dqn/decentralized_ego_radio_pa seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa seed 1"
  echo "[timpc:timpc_gpu] 29/29: neighborhood_level/dqn/decentralized_ego_radio_pa seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio_pa seed 4"
) &

echo "Launched 1 device(s) on timpc; waiting..."
wait
echo "All offline+online jobs on timpc complete."
