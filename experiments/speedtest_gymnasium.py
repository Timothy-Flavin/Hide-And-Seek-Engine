import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import json
import tempfile
import time
import gymnasium as gym

import numpy as np

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv


class SingleSARGridEnv(gym.Env):
    def __init__(self, map_png, tiles_json, agents_json, survivors_json, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.env = SARBatchedGridEnv(
            num_envs=1,
            map_png=map_png,
            tiles_json=tiles_json,
            agents_json=agents_json,
            survivors_json=survivors_json,
            mode="centralized",
            requires_state=False,
        )

        # Define spaces so AsyncVectorEnv can allocate buffers
        # Assuming observation is a dict or array, using dummy spaces for action since we bypass them
        self.action_space = gym.spaces.Dict(
            {
                "move": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(num_agents, 2), dtype=np.float32
                ),
                "radio": gym.spaces.MultiDiscrete([num_agents] * num_agents),
            }
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )  # Dummy

    def reset(self, seed=None, options=None):
        # We might not need to reset during timing, just like speedtest.py
        # But for gym.Env compliance:
        if hasattr(self.env, "reset"):
            self.env.reset()
        return np.zeros(1), {}

    def step(self, action):
        move_acts = action["move"]
        radio_acts = action["radio"]
        # The underlying C++ env wants shape (1, agents, dims)
        move_acts = move_acts.reshape(1, self.num_agents, 2)
        radio_acts = radio_acts.reshape(1, self.num_agents)

        obs, rewards, terminated, truncated, infos = self.env.step(
            move_acts, radio_acts
        )
        return np.zeros(1), 0.0, False, False, {}


def _random_local_actions(num_envs: int, n_agents: int):
    """Generates split movement and radio actions matching the new C++ API."""
    move_actions = np.random.uniform(-1.0, 1.0, size=(num_envs, n_agents, 2)).astype(
        np.float32
    )
    radio_actions = np.random.randint(0, n_agents, size=(num_envs, n_agents)).astype(
        np.int32
    )
    return [
        {"move": move_actions[i], "radio": radio_actions[i]} for i in range(num_envs)
    ]


def generate_agents_json(num_agents: int) -> str:
    agent_template = {
        "flying": False,
        "aqueous": True,
        "altitude_min": 0.0,
        "altitude_max": 1.0,
        "base_speed": 2.0,
        "base_view": 5.0,
        "battery": 150,
        "deployment_delay": 0.0,
        "rgb": [255, 224, 189],
        "terrain_speed": {"grass": 1.0, "water": 0.2, "mountain": 0.5},
        "start": [0.5, 0.5],
    }
    agents = {}
    for i in range(num_agents):
        agents[f"agent_{i}"] = agent_template.copy()

    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(agents, f)
    return path


def run_speedtest_gymnasium(
    num_envs: int,
    num_agents: int,
    steps: int,
    assets: dict[str, str],
) -> float:
    print(f"Testing {num_agents} agents with {num_envs} Gymnasium parallel envs...")

    # Generate temporary agents JSON
    agents_json_path = generate_agents_json(num_agents)

    def make_env():
        return SingleSARGridEnv(
            map_png=assets["map_png"],
            tiles_json=assets["tiles_json"],
            agents_json=agents_json_path,
            survivors_json=assets["survivors_json"],
            num_agents=num_agents,
        )

    # Using AsyncVectorEnv to parallelize in Python
    try:
        vec_env = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
        vec_env.reset()
    except Exception as e:
        print(f"  Skipping due to memory/process limit: {e}")
        if os.path.exists(agents_json_path):
            os.remove(agents_json_path)
        return 0.0

    t0 = time.perf_counter()
    total_step_calls = 0

    t_random_act = 0.0
    t_step_cpp = 0.0

    # Number of steps per worker is steps // num_envs
    n_iters = steps // num_envs

    # Pre-generate some actions to not count array creation purely, though AsyncVectorEnv manages Python overhead
    for _ in range(n_iters):
        t_start_act = time.perf_counter()
        actions = _random_local_actions(num_envs, num_agents)

        # AsyncVectorEnv takes dict of batched shapes if space is Dict, or tuple.
        # But if action_space is Dict, it expects dict of arrays like:
        # {'move': array(N, num_agents, 2), 'radio': array(N, num_agents)}
        action_dict_batched = {
            "move": np.array([a["move"] for a in actions]),
            "radio": np.array([a["radio"] for a in actions]),
        }
        t_random_act += time.perf_counter() - t_start_act

        t_start_step = time.perf_counter()
        vec_env.step(action_dict_batched)
        t_step_cpp += time.perf_counter() - t_start_step

        total_step_calls += num_envs

    dt = time.perf_counter() - t0
    fps = total_step_calls / max(dt, 1e-8)

    # Cleanup
    vec_env.close()
    os.remove(agents_json_path)

    print(f"  Py Wall Clock:")
    print(f"    Random Acts: {t_random_act:.3f} s")
    print(f"    Env Step (Gymnasium): {t_step_cpp:.3f} s")
    print(f"    Total:       {dt:.3f} s")
    print(f"  FPS: {fps:,.1f}")
    return fps


def main():
    import matplotlib.pyplot as plt

    steps = 50_000
    env_counts = [1, 4, 16, 32]  # Cut down from 128/256 to avoid hitting 16GB RAM limit
    agent_counts = list(range(1, 11))
    num_runs = 3

    assets = {
        "map_png": "levels/test_level/level.png",
        "tiles_json": "levels/test_level/tiles.json",
        "survivors_json": "levels/test_level/survivors.json",
    }

    results_matrix = np.zeros((len(env_counts), len(agent_counts), num_runs))

    for env_idx, n_envs in enumerate(env_counts):
        if n_envs == 1:
            steps = 10000
        else:
            steps = 10000
        for agent_idx, n_agents in enumerate(agent_counts):
            print(f"\n--- Testing {n_envs} envs, {n_agents} agents ---")
            for run_idx in range(num_runs):
                fps = run_speedtest_gymnasium(n_envs, n_agents, steps, assets)
                results_matrix[env_idx, agent_idx, run_idx] = fps

    np.save("experiments/speedtest_results_gymnasium_raw.npy", results_matrix)

    # Plotting
    plt.figure(figsize=(10, 6))
    for env_idx, n_envs in enumerate(env_counts):
        run_data = results_matrix[env_idx]
        means = np.mean(run_data, axis=1)
        mins = np.min(run_data, axis=1)
        maxs = np.max(run_data, axis=1)

        yerr = [means - mins, maxs - means]

        plt.errorbar(
            agent_counts,
            means,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=f"{n_envs} envs",
        )

    plt.xlabel("Number of Agents")
    plt.ylabel("FPS (Total steps / second)")
    plt.title(
        "Framerate Scaling vs Number of Agents and Parallel Envs (Gymnasium Async)"
    )
    plt.legend()
    plt.grid(True)

    plot_path = "experiments/speedtest_results_gymnasium.png"
    plt.savefig(plot_path)
    print(
        f"\nSpeedtest complete. Plot saved to {plot_path} and raw arrays to experiments/speedtest_results_gymnasium_raw.npy"
    )


if __name__ == "__main__":
    main()
