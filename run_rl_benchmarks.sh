#!/bin/bash
TOTAL_STEPS=100000

run_benchmark() {
    local name="$1"
    local torch_threads="$2"
    local omp_policy="$3"

    echo "Running $name (torch_threads=$torch_threads, OMP_WAIT_POLICY=$omp_policy)..."
    if [ "$omp_policy" != "default" ]; then
        export OMP_WAIT_POLICY="$omp_policy"
    else
        unset OMP_WAIT_POLICY
    fi
    
    # Monitor GPU usage
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "gpu_util_${name}.log" &
    NVIDIA_PID=$!
    
    python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=$torch_threads
    
    kill $NVIDIA_PID
    avg_gpu=$(awk '{ total += $1; count++ } END { print total/count }' gpu_util_${name}.log)
    echo "Avg GPU Utilization for $name: ${avg_gpu}%"
    echo "----------------------------------------"
}

run_benchmark "Control" "0" "default"
run_benchmark "Test_A" "1" "default"
run_benchmark "Test_B" "0" "passive"
run_benchmark "Test_C" "1" "passive"
