(
  echo "[timpc:timpc_gpu] 1/47: warehouse_level/sac/decentralized_ego seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/sac/decentralized_ego seed 1"
  echo "[timpc:timpc_gpu] 2/47: warehouse_level/sac/decentralized_ego seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/sac/decentralized_ego seed 2"
  echo "[timpc:timpc_gpu] 3/47: warehouse_level/sac/decentralized_ego seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/sac/decentralized_ego seed 3"
  echo "[timpc:timpc_gpu] 4/47: warehouse_level/sac/decentralized_ego seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/sac/decentralized_ego seed 5"
  echo "[timpc:timpc_gpu] 5/47: island_level/sac/decentralized_ego_radio seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: island_level/sac/decentralized_ego_radio seed 1"
  echo "[timpc:timpc_gpu] 6/47: island_level/sac/decentralized_ego_radio seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: island_level/sac/decentralized_ego_radio seed 2"
  echo "[timpc:timpc_gpu] 7/47: island_level/sac/decentralized_ego_radio seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: island_level/sac/decentralized_ego_radio seed 3"
  echo "[timpc:timpc_gpu] 8/47: island_level/sac/decentralized_ego_radio seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: island_level/sac/decentralized_ego_radio seed 4"
  echo "[timpc:timpc_gpu] 9/47: island_level/sac/decentralized_ego_radio seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: island_level/sac/decentralized_ego_radio seed 5"
  echo "[timpc:timpc_gpu] 10/47: test_level/sac/decentralized_ego seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego seed 1"
  echo "[timpc:timpc_gpu] 11/47: test_level/sac/decentralized_ego seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego seed 2"
  echo "[timpc:timpc_gpu] 12/47: test_level/sac/decentralized_ego seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego seed 3"
  echo "[timpc:timpc_gpu] 13/47: test_level/sac/decentralized_ego seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego seed 4"
  echo "[timpc:timpc_gpu] 14/47: test_level/sac/decentralized_ego seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego seed 5"
  echo "[timpc:timpc_gpu] 15/47: test_level/sac/decentralized_ego_radio seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio seed 1"
  echo "[timpc:timpc_gpu] 16/47: test_level/sac/decentralized_ego_radio seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio seed 2"
  echo "[timpc:timpc_gpu] 17/47: test_level/sac/decentralized_ego_radio seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio seed 3"
  echo "[timpc:timpc_gpu] 18/47: test_level/sac/decentralized_ego_radio seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio seed 4"
  echo "[timpc:timpc_gpu] 19/47: test_level/sac/decentralized_ego_radio seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: test_level/sac/decentralized_ego_radio seed 5"
  echo "[timpc:timpc_gpu] 20/47: neighborhood_level/sac/decentralized_ego_radio seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio seed 1"
  echo "[timpc:timpc_gpu] 21/47: neighborhood_level/sac/decentralized_ego_radio seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio seed 2"
  echo "[timpc:timpc_gpu] 22/47: neighborhood_level/sac/decentralized_ego_radio seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio seed 3"
  echo "[timpc:timpc_gpu] 23/47: neighborhood_level/sac/decentralized_ego_radio seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio seed 4"
  echo "[timpc:timpc_gpu] 24/47: neighborhood_level/sac/decentralized_ego_radio seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego_radio seed 5"
  echo "[timpc:timpc_gpu] 25/47: neighborhood_level/sac/decentralized_ego seed 1"
  python -m RL.cleanrl_sac --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego seed 1"
  echo "[timpc:timpc_gpu] 26/47: neighborhood_level/sac/decentralized_ego seed 2"
  python -m RL.cleanrl_sac --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego seed 2"
  echo "[timpc:timpc_gpu] 27/47: neighborhood_level/sac/decentralized_ego seed 3"
  python -m RL.cleanrl_sac --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego seed 3"
  echo "[timpc:timpc_gpu] 28/47: neighborhood_level/sac/decentralized_ego seed 4"
  python -m RL.cleanrl_sac --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego seed 4"
  echo "[timpc:timpc_gpu] 29/47: neighborhood_level/sac/decentralized_ego seed 5"
  python -m RL.cleanrl_sac --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/sac/decentralized_ego seed 5"
  echo "[timpc:timpc_gpu] 30/47: island_level/dqn/decentralized_ego seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: island_level/dqn/decentralized_ego seed 1"
  echo "[timpc:timpc_gpu] 31/47: island_level/dqn/decentralized_ego seed 3"
  python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/island_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: island_level/dqn/decentralized_ego seed 3"
  echo "[timpc:timpc_gpu] 32/47: warehouse_level/dqn/decentralized_ego_radio seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio seed 1"
  echo "[timpc:timpc_gpu] 33/47: warehouse_level/dqn/decentralized_ego_radio seed 2"
  python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio seed 2"
  echo "[timpc:timpc_gpu] 34/47: warehouse_level/dqn/decentralized_ego_radio seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio seed 4"
  echo "[timpc:timpc_gpu] 35/47: warehouse_level/dqn/decentralized_ego_radio seed 5"
  python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego_radio seed 5"
  echo "[timpc:timpc_gpu] 36/47: warehouse_level/dqn/decentralized_ego seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego seed 1"
  echo "[timpc:timpc_gpu] 37/47: warehouse_level/dqn/decentralized_ego seed 2"
  python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego seed 2"
  echo "[timpc:timpc_gpu] 38/47: warehouse_level/dqn/decentralized_ego seed 3"
  python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego seed 3"
  echo "[timpc:timpc_gpu] 39/47: warehouse_level/dqn/decentralized_ego seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego seed 4"
  echo "[timpc:timpc_gpu] 40/47: warehouse_level/dqn/decentralized_ego seed 5"
  python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/warehouse_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: warehouse_level/dqn/decentralized_ego seed 5"
  echo "[timpc:timpc_gpu] 41/47: test_level/dqn/decentralized_ego seed 2"
  python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/dqn/decentralized_ego seed 2"
  echo "[timpc:timpc_gpu] 42/47: test_level/dqn/decentralized_ego seed 3"
  python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/test_level --num-envs 128 --no-centralized --ego-view --ego-size 32 \
    || echo "[timpc:timpc_gpu] FAILED: test_level/dqn/decentralized_ego seed 3"
  echo "[timpc:timpc_gpu] 43/47: neighborhood_level/dqn/decentralized_ego_radio seed 1"
  python -m RL.cleanrl_dqn --run-number 1 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio seed 1"
  echo "[timpc:timpc_gpu] 44/47: neighborhood_level/dqn/decentralized_ego_radio seed 2"
  python -m RL.cleanrl_dqn --run-number 2 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio seed 2"
  echo "[timpc:timpc_gpu] 45/47: neighborhood_level/dqn/decentralized_ego_radio seed 3"
  python -m RL.cleanrl_dqn --run-number 3 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio seed 3"
  echo "[timpc:timpc_gpu] 46/47: neighborhood_level/dqn/decentralized_ego_radio seed 4"
  python -m RL.cleanrl_dqn --run-number 4 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio seed 4"
  echo "[timpc:timpc_gpu] 47/47: neighborhood_level/dqn/decentralized_ego_radio seed 5"
  python -m RL.cleanrl_dqn --run-number 5 --total-timesteps 5000000 --level levels/neighborhood_level --num-envs 64 --no-centralized --ego-view --ego-size 32 --use-radio \
    || echo "[timpc:timpc_gpu] FAILED: neighborhood_level/dqn/decentralized_ego_radio seed 5"
)