import os
import json
import yaml
from ortools.linear_solver import pywraplp

def solve_scheduling_problem():
    # ---------------------------------------------------
    # 1. Hardware Topology & Capacity Setup
    # ---------------------------------------------------
    # 'capacity': Concurrent jobs the device can handle.
    # 'efficiency': Multiplier for steps_per_sec (e.g., 0.85 = 15% slower per job).
    # 'use_mps': Strictly defines if the bash script should wrap execution in the MPS daemon.
    device_config = {
        "timpc_gpu": {"capacity": 2, "efficiency": 1.0, "use_mps": True},
        #"lab-comp_gpu": {"capacity": 4, "efficiency": 0.85, "use_mps": True},
        #"lab-comp_cpu": {"capacity": 1, "efficiency": 1.0, "use_mps": False},
        #"white-machine_gpu": {"capacity": 2, "efficiency": 1.0, "use_mps": False},
        #"alienware_gpu": {"capacity": 2, "efficiency": 1.0, "use_mps": False},
        #"mac_cpu": {"capacity": 1, "efficiency": 1.0, "use_mps": False},
        #"laptop_cpu": {"capacity": 1, "efficiency": 1.0, "use_mps": False},
    }
    
    devices = list(device_config.keys())
    # Explicitly map the new logical names to your existing folder structure
    benchmark_folders = {
        "timpc_gpu": "timpc",
        "lab-comp_gpu": "lab-comp_gpu",
        "lab-comp_cpu": "lab-comp_cpu",
        "white-machine_gpu": "white-machine_gpu0", # Just read gpu0's file for the unified estimate
        "alienware_gpu": "alienware_gpu_0",       # Just read gpu0's file for the unified estimate
        "mac_cpu": "mac",
        "laptop_cpu": "laptop"
    }
    
    env_activations = {
        "timpc_gpu": "source .venv/bin/activate",
        "lab-comp_gpu": "source .venv/bin/activate",
        "lab-comp_cpu": "source .venv/bin/activate",
        "white-machine_gpu": "source .venv/bin/activate",
        "alienware_gpu": "source .venv/bin/activate",
        "mac_cpu": "source ../.venv/bin/activate",
        "laptop_cpu": "source .venv/bin/activate",
    }
    
    # Preambles map directly to the concurrency queues. 
    # Queue 0 takes index 0, Queue 1 takes index 1, strictly enforcing hardware boundaries.
    command_pre_appends = {
        "timpc_gpu": [
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 ",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 "
        ],
        "lab-comp_gpu": [
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 ",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 ",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 ",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 "
        ],
        "lab-comp_cpu": [
            "CUDA_VISIBLE_DEVICES=\"\" numactl --preferred=0 taskset -c 0-15,32-47 "
        ],
        "white-machine_gpu": [
            "OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 ",
            "OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 "
        ],
        "alienware_gpu": [
            "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 ",
            "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 "
        ],
        "mac_cpu": [
            ""
        ],
        "laptop_cpu": [
            ""
        ]
    }

    envs = ["nchain"] # "minigrid", "cartpole", "mujoco", 
    models = ["dqn", "sac", "ppo"]
    ablations = [0, 1, 2, 3, 4, 5, 6]
    runs = [6,7,8,9,10,11,12,13,14,15]

    experiments = [
        {"env": env, "model": model, "ablation": abl, "run": run}
        for env in envs for model in models for abl in ablations for run in runs
    ]

    num_experiments = len(experiments)
    num_devices = len(devices)

    try:
        with open(os.path.join(os.path.dirname(__file__), "../env_config.yaml"), "r") as f:
            ENV_CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: env_config.yaml not found. Using fallback step counts.")
        ENV_CONFIG = {}

    def get_runtime_minutes(device, experiment):
        env = experiment["env"]
        model = experiment["model"]
        abl = experiment["ablation"]

        # Use the explicit mapping dictionary instead of string splitting
        folder_name = benchmark_folders[device]
        json_path = os.path.join("time_files", folder_name, f"{env}_{model}_best.json")

        default_sps = 30.0
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                abl_key = f"ablation_{abl}"
                steps_per_sec = data.get(abl_key, {}).get("steps_per_sec", default_sps)
        except Exception:
            # Print a loud warning if the benchmark is missing so it doesn't fail silently
            print(f"[!] Warning: Missing benchmark {json_path}. Defaulting to {default_sps} SPS.")
            steps_per_sec = default_sps
            
        if steps_per_sec <= 0:
            steps_per_sec = 1.0

        efficiency = device_config[device]["efficiency"]
        adjusted_sps = steps_per_sec * efficiency

        env_dict = ENV_CONFIG.get(env, {})
        max_steps_val = env_dict.get("max_steps", 300000)
        if isinstance(max_steps_val, dict):
            total_steps = int(max_steps_val.get(model, 300000))
        else:
            total_steps = int(max_steps_val)
            
        return (total_steps / adjusted_sps) / 60.0
    
    
    runtime_matrix = [
        [get_runtime_minutes(device, exp) for device in devices] for exp in experiments
    ]

    # ---------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("SCIP solver not available.")
        return

    solver.EnableOutput()
    solver.SetTimeLimit(600000) 

    # ---------------------------------------------------
    # 3. Variables & Constraints
    # ---------------------------------------------------
    x = {}
    for i in range(num_experiments):
        for j in range(num_devices):
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    makespan = solver.NumVar(0, solver.infinity(), "makespan")

    for i in range(num_experiments):
        solver.Add(sum(x[i, j] for j in range(num_devices)) == 1)

    for j in range(num_devices):
        device_name = devices[j]
        capacity = device_config[device_name]["capacity"]
        
        total_sequential_time = sum(
            runtime_matrix[i][j] * x[i, j] for i in range(num_experiments)
        )
        
        solver.Add(total_sequential_time <= makespan * capacity)

    # ---------------------------------------------------
    # 4. Objective & Execution
    # ---------------------------------------------------
    solver.Minimize(makespan)

    print(f"Assigning {num_experiments} experiments across {num_devices} device profiles.")
    print("Solving...\n" + "-" * 50)
    
    status = solver.Solve()
    
    print("-" * 50)

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print("Status: Optimal/Feasible Schedule Found!")
        print(f"Estimated true wall-clock time to finish all: {makespan.solution_value():.2f} minutes\n")

        os.makedirs("time_files", exist_ok=True)

        for j, device in enumerate(devices):
            capacity = device_config[device]["capacity"]
            use_mps = device_config[device]["use_mps"]
            activation_cmd = env_activations.get(device, "source .venv/bin/activate")
            preambles = command_pre_appends.get(device, [""])
            
            assigned_jobs = []
            total_device_sequential_time = 0
            for i, exp in enumerate(experiments):
                if x[i, j].solution_value() > 0.5:
                    time_taken = runtime_matrix[i][j]
                    total_device_sequential_time += time_taken
                    assigned_jobs.append((exp, time_taken))
            
            if not assigned_jobs:
                continue
                
            queues = [[] for _ in range(capacity)]
            for idx, job in enumerate(assigned_jobs):
                queues[idx % capacity].append(job)

            sh_lines = [
                "#!/usr/bin/env bash\n",
                f"# Auto-generated multi-process schedule for {device}\n",
                "# Capacity: " + str(capacity) + "\n",
                "set -euo pipefail\n\n",
                f"{activation_cmd}\n\n",
            ]
            
            # Only trigger MPS logic if explicitly allowed in the config map
            if use_mps:
                sh_lines.extend([
                    "export CUDA_VISIBLE_DEVICES=0\n",
                    "sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS || true\n",
                    "nvidia-cuda-mps-control -d || true\n",
                    "sleep 2\n\n"
                ])

            for q_idx, queue_jobs in enumerate(queues):
                if not queue_jobs: 
                    continue
                
                preamble = preambles[q_idx] if q_idx < len(preambles) else preambles[-1]
                
                sh_lines.append(f"# --- Concurrency Queue {q_idx} ---\n")
                sh_lines.append("(\n")
                
                for exp, time_taken in queue_jobs:
                    sh_lines.append(f"  echo \"[{device} - Q{q_idx}] Running {exp['model']} on {exp['env']} | Abl {exp['ablation']} | Run {exp['run']}\"\n")
                    
                    extra_args = ""
                    if exp['env'] == "nchain":
                        extra_args = " --num_envs 1"
                        if exp['model'] == "dqn":
                            extra_args += " --rnd_burn_in 20 --update_every 1 --dqn_batch_size 32"
                        elif exp['model'] == "ppo":
                            extra_args += " --num_steps 128 --ppo_lr 1e-3"
                            
                    cmd = f"  {preamble}python runner.py --algo {exp['model']} --env_name {exp['env']} --ablation {exp['ablation']} --run {exp['run']} --device_name {device}{extra_args}\n"
                    sh_lines.append(cmd)
                
                sh_lines.append(") &\n\n")

            sh_lines.append("echo \"All queues launched. Waiting for completion...\"\n")
            sh_lines.append("wait\n\n")
            
            if use_mps:
                sh_lines.extend([
                    "echo quit | nvidia-cuda-mps-control || true\n",
                    "sudo nvidia-smi -i 0 -c DEFAULT || true\n"
                ])
                
            sh_lines.append("echo \"All tasks completed on this machine.\"\n")

            sh_filename = os.path.join("time_files", f"run_{device}_experiments.sh")
            with open(sh_filename, "w", newline="\n") as sh_file:
                sh_file.writelines(sh_lines)

            print(f"--- Device: {device} ---")
            print(f"  Jobs Assigned: {len(assigned_jobs)}")
            print(f"  Raw Sequential Time: {total_device_sequential_time:.2f} mins")
            print(f"  Estimated Wall-Clock Time: {(total_device_sequential_time / capacity):.2f} mins")
            print(f"  Generated parallel shell script saved to {sh_filename}\n")

    else:
        print("The solver could not find a feasible solution.")

if __name__ == "__main__":
    solve_scheduling_problem()