#!/bin/bash
TOTAL_STEPS=100000

GPU_CPUS="16-31,48-63"
FAR_CPUS="0-15,32-47"

echo "Running BEST_CASE: Aligned CPUs + Test C (torch_threads=1, OMP_WAIT_POLICY=passive)..."
export OMP_WAIT_POLICY=passive
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "gpu_util_BestCase.log" &
NVIDIA_PID=$!
taskset -c $GPU_CPUS python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=1
kill $NVIDIA_PID
avg_gpu=$(awk '{ total += $1; count++ } END { print total/count }' gpu_util_BestCase.log)
echo "Avg GPU Utilization for BestCase: ${avg_gpu}%"

echo "----------------------------------------"

echo "Running WORST_CASE: Misaligned CPUs + Control (torch_threads=0, default OMP_WAIT_POLICY)..."
unset OMP_WAIT_POLICY
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "gpu_util_WorstCase.log" &
NVIDIA_PID=$!
taskset -c $FAR_CPUS python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=0
kill $NVIDIA_PID
avg_gpu=$(awk '{ total += $1; count++ } END { print total/count }' gpu_util_WorstCase.log)
echo "Avg GPU Utilization for WorstCase: ${avg_gpu}%"
echo "----------------------------------------"
