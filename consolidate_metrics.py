import numpy as np
import glob

configs = ["Control", "Test_A", "Test_B", "Test_C", "Aligned", "Misaligned"]

print("Config | Mean SPS | Std SPS | Mean GPU | Std GPU")
print("-" * 60)
for c in configs:
    sps_vals = []
    gpu_vals = []
    for i in range(1, 7):
        # sps
        try:
            with open(f"output_{c}_run_{i}.log", "r") as f:
                lines = f.readlines()
                sps_line = [l for l in lines[-10:] if "SPS=" in l][-1]
                sps_vals.append(float(sps_line.split("SPS=")[1].strip()))
        except:
            pass
        
        # gpu
        try:
            with open(f"gpu_util_{c}_run_{i}.log", "r") as f:
                lines = f.readlines()[5:] # drops 5 lines
                gpus = [float(x.strip()) for x in lines if x.strip()]
                if gpus:
                    gpu_vals.append(np.mean(gpus))
        except:
            pass
            
    if sps_vals:
        sps_mean = np.mean(sps_vals)
        sps_std = np.std(sps_vals)
        gpu_mean = np.mean(gpu_vals) if gpu_vals else 0.0
        gpu_std = np.std(gpu_vals) if gpu_vals else 0.0
        print(f"{c} | {sps_mean:.1f} | {sps_std:.1f} | {gpu_mean:.1f}% | {gpu_std:.1f}%")
