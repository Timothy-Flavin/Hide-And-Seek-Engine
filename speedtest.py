import argparse
import json
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv

rng = np.random.default_rng()


def _random_local_actions(
    num_envs: int, n_agents: int
) -> tuple[np.ndarray, np.ndarray]:
    # Generating directly into the correct dtype saves a copy/cast cycle
    move_actions = rng.random((num_envs, n_agents, 2), dtype=np.float32)
    # Scale and shift: [0, 1) -> [-1, 1)
    move_actions = move_actions * 2.0 - 1.0
    radio_actions = rng.integers(0, n_agents, size=(num_envs, n_agents), dtype=np.int32)
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
    mode: str,
    requires_state: bool,
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
        mode=mode,
        requires_state=requires_state,
    )
    # env.reset()

    t0 = time.perf_counter()
    total_step_calls = 0

    t_random_act = 0.0
    t_step_cpp = 0.0
    t_reset = 0.0

    for _ in range(steps // num_envs):
        t_start_act = time.perf_counter()
        move_acts, radio_acts = _random_local_actions(num_envs, num_agents)
        t_random_act += time.perf_counter() - t_start_act

        t_start_step = time.perf_counter()
        obs, _, terminated, _, _ = env.step(move_acts, radio_acts)
        t_step_cpp += time.perf_counter() - t_start_step

        total_step_calls += num_envs

        # t_start_reset = time.perf_counter()
        # term_np = (
        #     terminated.cpu().numpy()
        #     if isinstance(terminated, torch.Tensor)
        #     else terminated
        # )
        # # for e in np.where(term_np)[0]:
        # #    env.reset_env(int(e))
        # t_reset += time.perf_counter() - t_start_reset

    dt = time.perf_counter() - t0
    fps = total_step_calls / max(dt, 1e-8)

    # Cleanup
    os.remove(agents_json_path)

    print(f"  Py Wall Clock:")
    print(f"    Random Acts: {t_random_act:.3f} s")
    print(f"    Env Step:    {t_step_cpp:.3f} s")
    # print(f"    Env Reset:   {t_reset:.3f} s")
    print(f"    Total:       {dt:.3f} s")
    print(f"  FPS: {fps:,.1f}")
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="centralized",
        choices=["centralized", "decentralized", "no_obs"],
    )
    parser.add_argument(
        "--requires_state", type=str, default="False", choices=["True", "False"]
    )
    args = parser.parse_args()

    mode = args.mode
    requires_state = args.requires_state == "True"

    steps = 10_000_000
    env_counts = [1, 16, 128, 256, 1024]
    agent_counts = list(range(1, 11))
    num_runs = 3

    assets = {
        "map_png": "test_level/level.png",
        "tiles_json": "test_level/tiles.json",
        "survivors_json": "test_level/survivors.json",
    }

    results_matrix = np.zeros((len(env_counts), len(agent_counts), num_runs))

    os.makedirs("laptop_speedtest", exist_ok=True)

    for env_idx, n_envs in enumerate(env_counts):
        if n_envs == 1:
            steps = 50000
        else:
            steps = 500000
        for agent_idx, n_agents in enumerate(agent_counts):
            print(
                f"\n--- Testing {n_envs} envs, {n_agents} agents ({mode}, state={requires_state}) ---"
            )
            for run_idx in range(num_runs):
                fps = run_speedtest(
                    n_envs, n_agents, steps, assets, mode, requires_state
                )
                results_matrix[env_idx, agent_idx, run_idx] = fps

    state_str = "state" if requires_state else "nostate"
    base_name = f"speedtest_results_{mode}_{state_str}"

    np.save(os.path.join("laptop_speedtest", f"{base_name}_raw.npy"), results_matrix)

    # Plotting
    plt.figure(figsize=(10, 6))
    for env_idx, n_envs in enumerate(env_counts):
        run_data = results_matrix[env_idx]
        means = np.mean(run_data, axis=1)
        mins = np.min(run_data, axis=1)
        maxs = np.max(run_data, axis=1)

        # yerr takes [lower_errors, upper_errors] relative to the mean values
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
    plt.title(f"Framerate Scaling ({mode}, state={requires_state})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join("laptop_speedtest", f"{base_name}.png")
    plt.savefig(plot_path)
    print(
        f"\nSpeedtest complete. Plot saved to {plot_path} and raw arrays to {base_name}_raw.npy"
    )


if __name__ == "__main__":
    main()
