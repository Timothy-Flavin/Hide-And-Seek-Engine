#!/bin/bash
TOTAL_STEPS=1000000

GPU_CPUS="16-31,48-63"
FAR_CPUS="0-15,32-47"

echo "Running BEST_CASE: Aligned CPUs + Test C (torch_threads=1, OMP_WAIT_POLICY=passive)..."
export OMP_WAIT_POLICY=default
for i in {4..6}; do
    echo "  Run $i..."
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "gpu_util_BestCase_run_${i}.log" &
    NVIDIA_PID=$!
    taskset -c $GPU_CPUS python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=1 > "output_BestCase_run_${i}.log" 2>&1
    kill $NVIDIA_PID 2>/dev/null
    avg_gpu=$(tail -n +6 "gpu_util_BestCase_run_${i}.log" | awk '{ total += $1; count++ } END { print (count > 0 ? total/count : 0) }')
    sps=$(tail -n 10 "output_BestCase_run_${i}.log" | grep "SPS=" | tail -n 1 | awk -F'SPS=' '{print $2}')
    echo "  End Run $i: SPS=$sps, GPU=${avg_gpu}%"
done

echo "----------------------------------------"

echo "Running WORST_CASE: Misaligned CPUs + Control (torch_threads=0, default OMP_WAIT_POLICY)..."
unset OMP_WAIT_POLICY
for i in {4..6}; do
    echo "  Run $i..."
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "gpu_util_WorstCase_run_${i}.log" &
    NVIDIA_PID=$!
    taskset -c $FAR_CPUS python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=0 > "output_WorstCase_run_${i}.log" 2>&1
    kill $NVIDIA_PID 2>/dev/null
    avg_gpu=$(tail -n +6 "gpu_util_WorstCase_run_${i}.log" | awk '{ total += $1; count++ } END { print (count > 0 ? total/count : 0) }')
    sps=$(tail -n 10 "output_WorstCase_run_${i}.log" | grep "SPS=" | tail -n 1 | awk -F'SPS=' '{print $2}')
    echo "  End Run $i: SPS=$sps, GPU=${avg_gpu}%"
done
