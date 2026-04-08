#!/bin/bash
TOTAL_STEPS=100000

echo "GPU Topology:"
nvidia-smi topo -m

# GPU 0 is on NUMA node 1 based on previous output (CPUs 16-31,48-63)
GPU_CPUS="16-31,48-63"
# NUMA node 0 (Misaligned) is CPUs 0-15,32-47
FAR_CPUS="0-15,32-47"

echo "Running Aligned (CPUs $GPU_CPUS)..."
taskset -c $GPU_CPUS python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=1

echo "Running Misaligned (CPUs $FAR_CPUS)..."
taskset -c $FAR_CPUS python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=1
