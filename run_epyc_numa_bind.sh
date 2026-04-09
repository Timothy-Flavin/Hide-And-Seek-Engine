#!/bin/bash
TOTAL_STEPS=1000000

echo "GPU Topology:"
nvidia-smi topo -m

GPU_CPUS="16-31,48-63"
FAR_CPUS="0-15,32-47"

run_numa() {
    local name="$1"
    local cpus="$2"
    
    echo "Running $name (CPUs $cpus)..."
    for i in {4..6}; do
        echo "  Run $i..."
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "gpu_util_${name}_run_${i}.log" &
        NVIDIA_PID=$!
        
        taskset -c $cpus python cleanrl_ppo.py --total-timesteps=$TOTAL_STEPS --torch-threads=1 > "output_${name}_run_${i}.log" 2>&1
        
        kill $NVIDIA_PID 2>/dev/null
        avg_gpu=$(tail -n +6 "gpu_util_${name}_run_${i}.log" | awk '{ total += $1; count++ } END { print (count > 0 ? total/count : 0) }')
        sps=$(tail -n 10 "output_${name}_run_${i}.log" | grep "SPS=" | tail -n 1 | awk -F'SPS=' '{print $2}')
        echo "  End Run $i: SPS=$sps, GPU=${avg_gpu}%"
    done
    echo "----------------------------------------"
}

run_numa "Aligned" "$GPU_CPUS"
run_numa "Misaligned" "$FAR_CPUS"
