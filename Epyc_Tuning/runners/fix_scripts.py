import re

files_to_fix = [
    ('run_rl_benchmarks.sh', '../rl_benchmark/'),
    ('run_epyc_numa_bind.sh', '../numa_bind/'),
    ('run_best_vs_worst.sh', '../best_vs_worst/')
]

for filename, folder in files_to_fix:
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        content = content.replace('"gpu_util_', f'"{folder}gpu_util_')
        content = content.replace('"output_', f'"{folder}output_')
        content = content.replace('python cleanrl_ppo.py', 'python ../../cleanrl_ppo.py')
        
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Fixed {filename}")
    except Exception as e:
        pass


metrics = 'consolidate_metrics.py'
import os
try:
    with open(metrics, 'r') as f:
        content = f.read()
            
    # Modify paths
    content = content.replace('f"output_{name}_run_{i}.log"', 'f"../rl_benchmark/output_{name}_run_{i}.log" if name in ["Control", "Test_A", "Test_B", "Test_C"] else f"../numa_bind/output_{name}_run_{i}.log" if name in ["Aligned", "Misaligned"] else f"../best_vs_worst/output_{name}_run_{i}.log"')
    content = content.replace('f"gpu_util_{name}_run_{i}.log"', 'f"../rl_benchmark/gpu_util_{name}_run_{i}.log" if name in ["Control", "Test_A", "Test_B", "Test_C"] else f"../numa_bind/gpu_util_{name}_run_{i}.log" if name in ["Aligned", "Misaligned"] else f"../best_vs_worst/gpu_util_{name}_run_{i}.log"')

    with open(metrics, 'w') as f:
        f.write(content)
    print("Fixed consolidate_metrics.py")
except Exception as e:
    pass

