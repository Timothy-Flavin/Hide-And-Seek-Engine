import os
import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    algorithms = ['ppo', 'dqn', 'sac_central']
    colors = {'ppo': 'blue', 'dqn': 'red', 'sac': 'green', 'sac_central': 'green'}
    
    # The x-axis represents 10 million steps, scaled to 10,000 units (e.g., thousands of steps).
    # We create a common x-axis of 1000 evenly spaced points to interpolate against
    # so we can average arrays of different lengths.
    target_x = np.linspace(0, 10000, 1000)
    
    plt.figure(figsize=(10, 6))
    
    for alg in algorithms:
        interpolated_runs = []
        for run_num in range(1, 6):
            filename = f"results/{alg}_episodic_returns_run_{run_num}.npy"
            if os.path.exists(filename):
                returns = np.load(filename)
                if len(returns) == 0:
                    continue
                
                for xi in range(1,len(returns)):
                    returns[xi] = 0.95*returns[xi-1]+0.05*returns[xi]

                # Create the corresponding x-axis for this specific run
                x = np.linspace(0, 10000, len(returns))
                
                # Interpolate the episodic returns onto the target x-axis
                interp_returns = np.interp(target_x, x, returns)
                interpolated_runs.append(interp_returns)
        
        if interpolated_runs:
            # Convert to numpy array for vectorized mean/ste
            data = np.array(interpolated_runs)
            mean_returns = np.mean(data, axis=0)
            
            # Standard error of the mean
            std_returns = np.std(data, axis=0)
            ste = std_returns / np.sqrt(data.shape[0])
            
            # Plot solid line for mean, fill_between for standard error
            plt.plot(target_x, mean_returns, label=alg.upper(), color=colors[alg], linewidth=2)
            plt.fill_between(target_x, mean_returns - ste, mean_returns + ste, color=colors[alg], alpha=0.3)
            
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
