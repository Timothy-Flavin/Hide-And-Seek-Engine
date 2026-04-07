import argparse
import json
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv


def _random_local_actions(
    num_envs: int, n_agents: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generates split movement and radio actions matching the new C++ API."""
    move_actions = np.random.uniform(-1.0, 1.0, size=(num_envs, n_agents, 2)).astype(
        np.float32
    )
    radio_actions = np.random.randint(0, n_agents, size=(num_envs, n_agents)).astype(
        np.int32
    )
    return move_actions, radio_actions


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


def run_speedtest(
    num_envs: int,
    num_agents: int,
    steps: int,
    assets: dict[str, str],
) -> float:
    print(f"Testing {num_agents} agents with {num_envs} parallel envs...")

    # Generate temporary agents JSON
    agents_json_path = generate_agents_json(num_agents)

    env = SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=assets["map_png"],
        tiles_json=assets["tiles_json"],
        agents_json=agents_json_path,
        survivors_json=assets["survivors_json"],
        mode="decentralized",
        requires_state=False,
    )
    env.reset()

    t0 = time.perf_counter()
    total_step_calls = 0

    for _ in range(steps // num_envs):
        move_acts, radio_acts = _random_local_actions(num_envs, num_agents)
        obs, _, terminated, _, _ = env.step(move_acts, radio_acts)

        total_step_calls += num_envs

        term_np = (
            terminated.cpu().numpy()
            if isinstance(terminated, torch.Tensor)
            else terminated
        )
        for e in np.where(term_np)[0]:
            env.reset_env(int(e))

    dt = time.perf_counter() - t0
    fps = total_step_calls / max(dt, 1e-8)

    # Cleanup
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
            fps = run_speedtest(n_envs, n_agents, steps, assets)
            results[n_envs].append(fps)

    # Plotting
    plt.figure(figsize=(10, 6))
    for n_envs, fps_list in results.items():
        plt.plot(agent_counts, fps_list, marker="o", label=f"{n_envs} envs")

    plt.xlabel("Number of Agents")
    plt.ylabel("FPS (Total steps / second)")
    plt.title("Framerate Scaling vs Number of Agents and Parallel Envs")
    plt.legend()
    plt.grid(True)

    plot_path = "speedtest_results.png"
    plt.savefig(plot_path)
    print(f"\nSpeedtest complete. Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
