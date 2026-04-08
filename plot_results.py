import os
import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    variants = [
        ("ppo_centralized", "PPO Centralized", "blue", "-"),
        ("ppo_decentralized", "PPO Decentralized", "blue", "--"),
        ("dqn_centralized", "DQN Centralized", "red", "-"),
        ("dqn_decentralized", "DQN Decentralized", "red", "--"),
        ("sac_centralized", "SAC Centralized", "green", "-"),
        ("sac_decentralized", "SAC Decentralized", "green", "--"),
    ]

    target_x = np.linspace(0, 10000, 1000)
    plt.figure(figsize=(10, 6))

    for prefix, label, color, linestyle in variants:
        interpolated_runs = []
        for run_num in range(1, 6):
            filename = f"results/{prefix}_episodic_returns_run_{run_num}.npy"
            if os.path.exists(filename):
                returns = np.load(filename)
                if len(returns) == 0:
                    continue
                # EMA smoothing
                for xi in range(1, len(returns)):
                    returns[xi] = 0.95 * returns[xi-1] + 0.05 * returns[xi]
                x = np.linspace(0, 10000, len(returns))
                interp_returns = np.interp(target_x, x, returns)
                interpolated_runs.append(interp_returns)
        if interpolated_runs:
            data = np.array(interpolated_runs)
            mean_returns = np.mean(data, axis=0)
            std_returns = np.std(data, axis=0)
            ste = std_returns / np.sqrt(data.shape[0])
            plt.plot(target_x, mean_returns, label=label, color=color, linestyle=linestyle, linewidth=2)
            plt.fill_between(target_x, mean_returns - ste, mean_returns + ste, color=color, alpha=0.2)

    plt.xlabel("Steps (x1000)")
    plt.ylabel("Episodic Return")
    plt.title("Algorithm Comparison (Mean ± Standard Error)")
    plt.grid(True)
    plt.legend()
    os.makedirs("results", exist_ok=True)
    out_file = "results/combined_learning_curves.png"
    plt.savefig(out_file)
    print(f"Saved combined plot to {out_file}")

if __name__ == "__main__":
    plot_results()
