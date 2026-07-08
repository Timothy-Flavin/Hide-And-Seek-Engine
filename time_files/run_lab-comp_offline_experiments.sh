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

# --- Device lab-comp_gpu [OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63]: 23 jobs, ~423.2 min ---
(
  echo "[lab-comp:lab-comp_gpu] 1/23: warehouse_level/sac/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/sac/decentralized_ego_radio_pa seed 1"
  echo "[lab-comp:lab-comp_gpu] 2/23: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 2"
  echo "[lab-comp:lab-comp_gpu] 3/23: warehouse_level/sac/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/sac/decentralized_ego_radio_pa seed 3"
  echo "[lab-comp:lab-comp_gpu] 4/23: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 3"
  echo "[lab-comp:lab-comp_gpu] 5/23: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 4"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/sac/decentralized_ego_radio_pa_bc seed 4"
  echo "[lab-comp:lab-comp_gpu] 6/23: warehouse_level/sac/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/sac/decentralized_ego_radio_pa seed 5"
  echo "[lab-comp:lab-comp_gpu] 7/23: island_level/sac/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa seed 1"
  echo "[lab-comp:lab-comp_gpu] 8/23: island_level/sac/decentralized_ego_radio_pa_bc seed 1"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa_bc seed 1"
  echo "[lab-comp:lab-comp_gpu] 9/23: island_level/sac/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa_bc seed 2"
  echo "[lab-comp:lab-comp_gpu] 10/23: island_level/sac/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa seed 3"
  echo "[lab-comp:lab-comp_gpu] 11/23: island_level/sac/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa_bc seed 3"
  echo "[lab-comp:lab-comp_gpu] 12/23: island_level/sac/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa seed 4"
  echo "[lab-comp:lab-comp_gpu] 13/23: island_level/sac/decentralized_ego_radio_pa_bc seed 4"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa_bc seed 4"
  echo "[lab-comp:lab-comp_gpu] 14/23: island_level/sac/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa seed 5"
  echo "[lab-comp:lab-comp_gpu] 15/23: island_level/sac/decentralized_ego_radio_pa_bc seed 5"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: island_level/sac/decentralized_ego_radio_pa_bc seed 5"
  echo "[lab-comp:lab-comp_gpu] 16/23: test_level/sac/decentralized_ego_radio_pa seed 1"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 1 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa seed 1"
  echo "[lab-comp:lab-comp_gpu] 17/23: test_level/sac/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa seed 2"
  echo "[lab-comp:lab-comp_gpu] 18/23: test_level/sac/decentralized_ego_radio_pa_bc seed 2"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 2 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa_bc seed 2"
  echo "[lab-comp:lab-comp_gpu] 19/23: test_level/sac/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa seed 3"
  echo "[lab-comp:lab-comp_gpu] 20/23: test_level/sac/decentralized_ego_radio_pa_bc seed 3"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 3 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa_bc seed 3"
  echo "[lab-comp:lab-comp_gpu] 21/23: test_level/sac/decentralized_ego_radio_pa seed 4"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 4 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa seed 4"
  echo "[lab-comp:lab-comp_gpu] 22/23: test_level/sac/decentralized_ego_radio_pa seed 5"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_sac --run-number 5 --total-timesteps 1000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets \
    || echo "[lab-comp:lab-comp_gpu] FAILED: test_level/sac/decentralized_ego_radio_pa seed 5"
  echo "[lab-comp:lab-comp_gpu] 23/23: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 4"
  OMP_NUM_THREADS=16 numactl --preferred=1 taskset -c 16-31,48-63 python -m RL.cleanrl_ppo --run-number 4 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 \
    || echo "[lab-comp:lab-comp_gpu] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 4"
) &

# --- Device lab-comp_cpu [OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47]: 5 jobs, ~396.9 min ---
(
  echo "[lab-comp:lab-comp_cpu] 1/5: island_level/dqn/decentralized_ego_radio_pa seed 2"
  OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda \
    || echo "[lab-comp:lab-comp_cpu] FAILED: island_level/dqn/decentralized_ego_radio_pa seed 2"
  echo "[lab-comp:lab-comp_cpu] 2/5: island_level/dqn/decentralized_ego_radio_pa seed 3"
  OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --no-cuda \
    || echo "[lab-comp:lab-comp_cpu] FAILED: island_level/dqn/decentralized_ego_radio_pa seed 3"
  echo "[lab-comp:lab-comp_cpu] 3/5: island_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --no-cuda \
    || echo "[lab-comp:lab-comp_cpu] FAILED: island_level/dqn/decentralized_ego_radio_pa_cql seed 3"
  echo "[lab-comp:lab-comp_cpu] 4/5: island_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 1000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --exploration-timesteps 1000000 --human-cql --no-cuda \
    || echo "[lab-comp:lab-comp_cpu] FAILED: island_level/dqn/decentralized_ego_radio_pa_cql seed 5"
  echo "[lab-comp:lab-comp_cpu] 5/5: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 1"
  OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= numactl --preferred=0 taskset -c 0-15,32-47 python -m RL.cleanrl_ppo --run-number 1 --total-timesteps 1000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio --per-agent-nets --num-minibatches 8 --human-bc --bc-coef 1.0 --no-cuda \
    || echo "[lab-comp:lab-comp_cpu] FAILED: warehouse_level/ppo/decentralized_ego_radio_pa_bc seed 1"
) &

echo "Launched 2 device(s) on lab-comp; waiting..."
wait
echo "All offline+online jobs on lab-comp complete."
