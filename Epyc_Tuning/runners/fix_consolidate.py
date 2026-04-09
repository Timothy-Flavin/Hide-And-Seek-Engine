with open("consolidate_metrics.py", "r") as f:
    content = f.read()

content = content.replace('f"output_{c}_run_{i}.log"', 'f"../rl_benchmark/output_{c}_run_{i}.log" if c in ["Control", "Test_A", "Test_B", "Test_C"] else f"../numa_bind/output_{c}_run_{i}.log" if c in ["Aligned", "Misaligned"] else f"../best_vs_worst/output_{c}_run_{i}.log"')
content = content.replace('f"gpu_util_{c}_run_{i}.log"', 'f"../rl_benchmark/gpu_util_{c}_run_{i}.log" if c in ["Control", "Test_A", "Test_B", "Test_C"] else f"../numa_bind/gpu_util_{c}_run_{i}.log" if c in ["Aligned", "Misaligned"] else f"../best_vs_worst/gpu_util_{c}_run_{i}.log"')

with open("consolidate_metrics.py", "w") as f:
    f.write(content)

