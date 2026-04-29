#!/bin/bash
TOTAL_STEPS=1000000

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
    
    for i in {4..6}; do
        echo "  Run $i..."
        # Monitor GPU usage
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "../rl_benchmark/gpu_util_${name}_run_${i}.log" &
        NVIDIA_PID=$!
        
        python ../../cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=$torch_threads > "../rl_benchmark/output_${name}_run_${i}.log" 2>&1
        
        kill $NVIDIA_PID 2>/dev/null
        avg_gpu=$(tail -n +6 "../rl_benchmark/gpu_util_${name}_run_${i}.log" | awk '{ total += $1; count++ } END { print (count > 0 ? total/count : 0) }')
        sps=$(tail -n 10 "../rl_benchmark/output_${name}_run_${i}.log" | grep "SPS=" | tail -n 1 | awk -F'SPS=' '{print $2}')
        echo "  End Run $i: SPS=$sps, GPU=${avg_gpu}%"
    done
    echo "----------------------------------------"
}

run_benchmark "Control" "0" "default"
run_benchmark "Test_A" "1" "default"
run_benchmark "Test_B" "0" "passive"
run_benchmark "Test_C" "1" "passive"
