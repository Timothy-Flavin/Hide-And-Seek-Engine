#!/bin/bash
set -e

mkdir -p results
#"ppo" "dqn" "sac"
for alg in "ppo" "dqn" "sac"; do
    for seed in {1..1}; do
        echo "Running $alg with seed $seed (run number $seed)..."
        python cleanrl_${alg}.py \
            --run-number $seed \
            --total-timesteps 10000000 \
            --centralized
    done
done

echo "All experiments completed!"
