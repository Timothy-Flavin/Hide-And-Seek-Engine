#!/bin/bash
set -e

mkdir -p results
#"ppo" "dqn" "sac"
for alg in "dqn"; do
    for seed in {2..5}; do
        echo "Running $alg with seed $seed (run number $seed)..."
        python cleanrl_${alg}.py \
            --seed $seed \
            --run-number $seed \
            --total-timesteps 10000000
    done
done

echo "All experiments completed!"
