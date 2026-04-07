import os
import time
import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv


def generate_agents_json(num_agents: int) -> str:
    agent_template = {
        "flying": False,
        "aqueous": True,
        "altitude_min": 0.0,
        "altitude_max": 1.0,
        "base_speed": 2.0,
        "base_view": 5.0,
        "battery": 150.0,
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


def run_speedtest_old(num_envs, num_agents, steps, assets):
    print(f"Testing {num_agents} agents with {num_envs} parallel envs (old API)...")
    agents_json_path = generate_agents_json(num_agents)
    env = SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=assets["map_png"],
        tiles_json=assets["tiles_json"],
        agents_json=agents_json_path,
        survivors_json=assets["survivors_json"],
        map_size=32,
        seed=42,
    )
    obs, info = env.reset()
    t0 = time.perf_counter()
    total_step_calls = 0
    for _ in range(steps // num_envs):
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_step_calls += num_envs
        # Handle resets if needed (old API may not require manual reset)
    dt = time.perf_counter() - t0
    fps = total_step_calls / max(dt, 1e-8)
    os.remove(agents_json_path)
    print(f"  FPS: {fps:,.1f}")
    return fps


def main():
    steps = 10_000
    env_counts = [1, 4, 16, 64, 128, 256]
    agent_counts = list(range(1, 11))
    assets = {
        "map_png": "test_level/level.png",
        "tiles_json": "test_level/tiles.json",
        "survivors_json": "test_level/survivors.json",
    }
    results = {n: [] for n in env_counts}
    for n_envs in env_counts:
        for n_agents in agent_counts:
            fps = run_speedtest_old(n_envs, n_agents, steps, assets)
            results[n_envs].append(fps)
    plt.figure(figsize=(10, 6))
    for n_envs, fps_list in results.items():
        plt.plot(agent_counts, fps_list, marker="o", label=f"{n_envs} envs")
    plt.xlabel("Number of Agents")
    plt.ylabel("FPS (Total steps / second)")
    plt.title("Framerate Scaling vs Number of Agents and Parallel Envs (Old API)")
    plt.legend()
    plt.grid(True)
    plot_path = "speedtest_old_results.png"
    plt.savefig(plot_path)
    print(f"\nSpeedtest complete. Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
