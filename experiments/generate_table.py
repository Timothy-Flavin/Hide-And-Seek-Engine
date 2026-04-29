import numpy as np
import glob
import os

def format_table(name, path_prefix):
    try:
        f = f"{path_prefix}/speedtest_results_decentralized_state_raw.npy"
        data = np.load(f)
        # max over runs (axis 2) and envs (axis 0) for each agent count 
        peak_per_agent = np.max(data, axis=(0, 2))
        return f"{name} & {peak_per_agent[0]:,.0f} & {peak_per_agent[4]:,.0f} & {peak_per_agent[9]:,.0f} \\\\"
    except Exception as e:
        return f"{name} & N/A & N/A & N/A \\\\"

print("\\begin{table}[htbp]")
print("\\centering")
print("\\begin{tabular}{lrrr}")
print("\\toprule")
print("\\textbf{Hardware} & \\textbf{1 Agent (SPS)} & \\textbf{5 Agents (SPS)} & \\textbf{10 Agents (SPS)} \\\\")
print("\\midrule")
print(format_table("AMD Ryzen 9950X", "experiments/Epyc_Tuning/env_configs_9950X"))
print(format_table("AMD EPYC 7282", "experiments/Epyc_Tuning/env_configs_epyc"))
print(format_table("Intel 4-Core Laptop", "experiments/Epyc_Tuning/env_configs_laptop_new"))
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Peak Throughput (SPS) scaling by Agent Count across different hardware configurations (Decentralized Observable State).}")
print("\\label{tab:hardware_comparison}")
print("\\end{table}")
