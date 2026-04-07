#!/bin/bash
set -e

mkdir -p results

for alg in "ppo" "dqn" "sac"; do
    for seed in {1..5}; do
        echo "Running $alg with seed $seed (run number $seed)..."
        python cleanrl_${alg}.py \
            --seed $seed \
            --run-number $seed \
            --total-timesteps 10000000
    done
done

echo "All experiments completed!"
