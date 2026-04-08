import os
import re

# Map old patterns to new patterns
rename_patterns = [
    # SAC central
    (r"^sac_central_episodic_returns_run_(\d+)\.npy$", r"sac_centralized_episodic_returns_run_\1.npy"),
    # PPO
    (r"^ppo_episodic_returns_run_(\d+)\.npy$", r"ppo_centralized_episodic_returns_run_\1.npy"),
    # DQN
    (r"^dqn_episodic_returns_run_(\d+)\.npy$", r"dqn_centralized_episodic_returns_run_\1.npy"),
    # SAC decentralized (future-proof, if any)
    (r"^sac_individual_episodic_returns_run_(\d+)\.npy$", r"sac_decentralized_episodic_returns_run_\1.npy"),
    # PPO decentralized
    (r"^ppo_individual_episodic_returns_run_(\d+)\.npy$", r"ppo_decentralized_episodic_returns_run_\1.npy"),
    # DQN decentralized
    (r"^dqn_individual_episodic_returns_run_(\d+)\.npy$", r"dqn_decentralized_episodic_returns_run_\1.npy"),
]

def rename_results_files(results_dir="results"):
    for fname in os.listdir(results_dir):
        for pat, repl in rename_patterns:
            if re.match(pat, fname):
                new_name = re.sub(pat, repl, fname)
                src = os.path.join(results_dir, fname)
                dst = os.path.join(results_dir, new_name)
                print(f"Renaming {src} -> {dst}")
                os.rename(src, dst)
                break

if __name__ == "__main__":
    rename_results_files()
