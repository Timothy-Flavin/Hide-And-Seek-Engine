#!/usr/bin/env bash
# Auto-generated ANNEAL schedule for timpc (devices concurrent, no MPS).
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

# --- Device timpc_gpu [no pinning]: 15 jobs, ~175.8 min ---
(
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_1_pct100.pt" "[timpc:timpc_gpu] 1/15: test_level/sac/decentralized_ego_radio_pa_bc_anneal seed 1" -- \
    python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[timpc:timpc_gpu] 2/15: test_level/sac/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[timpc:timpc_gpu] 3/15: test_level/sac/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_5_pct100.pt" "[timpc:timpc_gpu] 4/15: test_level/sac/decentralized_ego_radio_pa_bc_anneal seed 5" -- \
    python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_2_pct100.pt" "[timpc:timpc_gpu] 5/15: neighborhood_level/sac/decentralized_ego_radio_pa_bc_anneal seed 2" -- \
    python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/neighborhood_level/sac/checkpoints/sac_decentralized_ego_radio_pa_bc_anneal_run_4_pct100.pt" "[timpc:timpc_gpu] 6/15: neighborhood_level/sac/decentralized_ego_radio_pa_bc_anneal seed 4" -- \
    python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_2_pct100.pt" "[timpc:timpc_gpu] 7/15: island_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_4_pct100.pt" "[timpc:timpc_gpu] 8/15: island_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/island_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_5_pct100.pt" "[timpc:timpc_gpu] 9/15: island_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_1_pct100.pt" "[timpc:timpc_gpu] 10/15: warehouse_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 1" -- \
    python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_2_pct100.pt" "[timpc:timpc_gpu] 11/15: warehouse_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 2" -- \
    python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_3_pct100.pt" "[timpc:timpc_gpu] 12/15: warehouse_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 3" -- \
    python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_4_pct100.pt" "[timpc:timpc_gpu] 13/15: warehouse_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/warehouse_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_5_pct100.pt" "[timpc:timpc_gpu] 14/15: warehouse_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 5" -- \
    python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
  run_if_missing "experiments/results/test_level/dqn/checkpoints/dqn_decentralized_ego_radio_pa_cql_anneal_run_4_pct100.pt" "[timpc:timpc_gpu] 15/15: test_level/dqn/decentralized_ego_radio_pa_cql_anneal seed 4" -- \
    python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --bc-anneal-frames 250000
) &

echo "Launched 1 device(s) on timpc; waiting..."
wait
echo "All anneal jobs on timpc complete."
