import argparse
import time

import numpy as np
import torch

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from hide_and_seek_engine.sar_loader import load_sar_config


def _random_local_actions(
    num_envs: int, n_agents: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generates split movement and radio actions matching the new C++ API."""
    move_actions = np.random.uniform(-1.0, 1.0, size=(num_envs, n_agents, 2)).astype(
        np.float32
    )
    # Assuming radio actions are discrete choices up to n_agents
    radio_actions = np.random.randint(0, n_agents, size=(num_envs, n_agents)).astype(
        np.int32
    )
    return move_actions, radio_actions


def unit_schema_checks(assets: dict[str, str]):
    """Validates that the new load_sar_config outputs correctly shaped structures."""
    cfg = load_sar_config(
        tiles_json=assets["tiles_json"],
        agents_json=assets["agents_json"],
        survivors_json=assets["survivors_json"],
        map_png=assets["map_png"],
    )

    assert len(cfg.supports_walking) == cfg.n_tiles, "Mismatch in walking support array"
    assert len(cfg.terrain_rgb) == cfg.n_tiles, "Mismatch in terrain RGB array"
    assert (
        len(cfg.type_map) == cfg.width * cfg.height
    ), "Type map does not match map area"
    assert (
        len(cfg.initial_agent_pos) == cfg.n_agents * 2
    ), "Agent positions array size incorrect"
    assert (
        len(cfg.saveable_map) == cfg.n_pois * cfg.n_agents
    ), "Saveable rules array size incorrect"
    print("[unit] schema checks: PASS")


def run_local_headless(
    num_envs: int,
    steps: int,
    assets: dict[str, str],
    mode: str,
    requires_state: bool,
    seed: int = 42,
) -> float:
    print("Making env")
    env = SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=assets["map_png"],
        tiles_json=assets["tiles_json"],
        agents_json=assets["agents_json"],
        survivors_json=assets["survivors_json"],
        mode=mode,
        requires_state=requires_state,
    )
    print("Resetting env")
    obs, _ = env.reset()

    t0 = time.perf_counter()
    total_step_calls = 0

    print("Stepping env")
    for _ in range(steps // num_envs):
        move_acts, radio_acts = _random_local_actions(num_envs, env.config.n_agents)
        obs, _, terminated, _, _ = env.step(move_acts, radio_acts)

        total_step_calls += num_envs

        # Handle env resets
        term_np = (
            terminated.cpu().numpy()
            if isinstance(terminated, torch.Tensor)
            else terminated
        )
        for e in np.where(term_np)[0]:
            env.reset_env(int(e))

    dt = time.perf_counter() - t0

    fps = total_step_calls / max(dt, 1e-8)

    # Print out shapes to verify the configuration
    spatial_shape = obs["spatial"].shape if obs["spatial"] is not None else "None"
    print(
        f"[local] mode={mode:<13} state={str(requires_state):<5} | "
        f"envs={num_envs:>3} steps={steps:>6} fps={fps:,.1f} | spatial_shape={spatial_shape}"
    )
    return fps


def run_renderer_episode(assets: dict[str, str], max_steps: int = 300, seed: int = 42):
    """Smoke test for the new consolidated JIT Pygame renderer."""
    env = SARBatchedGridEnv(
        num_envs=1,
        map_png=assets["map_png"],
        tiles_json=assets["tiles_json"],
        agents_json=assets["agents_json"],
        survivors_json=assets["survivors_json"],
        mode="decentralized",
        requires_state=True,
    )
    env.reset()

    for step in range(max_steps):
        move_acts, radio_acts = _random_local_actions(1, env.config.n_agents)
        _, _, terminated, _, _ = env.step(move_acts, radio_acts)

        if step < 30:
            # Render True Global State
            env.render(pov=-1, env_idx=0)
        else:
            # Render individual Agent Beliefs
            pov_agent = (step - 30) % env.config.n_agents
            env.render(pov=pov_agent, env_idx=0)

        term_val = (
            terminated[0].item()
            if isinstance(terminated, torch.Tensor)
            else terminated[0]
        )
        if bool(term_val):
            break

    print("[render] true state & per-agent POV renderer: PASS")


def benchmark_suite(assets: dict[str, str], steps: int, env_counts: list[int]):
    print("\n=== Local Mode Benchmarks ===")

    # Define the requested profiling configurations
    configs = [
        {"mode": "no_obs", "requires_state": True},  # State only (True)
        {"mode": "no_obs", "requires_state": False},  # Headless void (fastest baseline)
        {"mode": "decentralized", "requires_state": False},  # Standard Decentralized RL
        {"mode": "centralized", "requires_state": False},  # Standard Centralized RL
    ]

    for cfg in configs:
        print(
            f"\n--- Testing Config: {cfg['mode']} (Requires State: {cfg['requires_state']}) ---"
        )
        # if cfg["mode"] != "decentralized":
        #     continue
        # if cfg["requires_state"]:
        #     continue
        for n in env_counts:
            run_local_headless(
                num_envs=n,
                steps=steps,
                assets=assets,
                mode=cfg["mode"],
                requires_state=cfg["requires_state"],
            )


def main():
    parser = argparse.ArgumentParser(
        description="SAR environment unit/integration/benchmark suite"
    )
    parser.add_argument(
        "--steps", type=int, default=10_000, help="Headless benchmark total steps"
    )
    parser.add_argument(
        "--envs",
        type=int,
        nargs="*",
        default=[1],  # 2, 4, 8, 32, 64],
        help="Parallel env counts to test",
    )
    parser.add_argument(
        "--skip-render", action="store_true", help="Skip renderer smoke test"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-png", default="test_level/level.png")
    parser.add_argument("--tiles-json", default="test_level/tiles.json")
    parser.add_argument("--agents-json", default="test_level/agents.json")
    parser.add_argument("--survivors-json", default="test_level/survivors.json")
    args = parser.parse_args()

    assets = {
        "map_png": args.map_png,
        "tiles_json": args.tiles_json,
        "agents_json": args.agents_json,
        "survivors_json": args.survivors_json,
    }

    unit_schema_checks(assets)
    benchmark_suite(assets=assets, steps=args.steps, env_counts=args.envs)

    if not args.skip_render:
        print("\n=== Running Renderer Smoke Test ===")
        run_renderer_episode(assets=assets, seed=args.seed)


if __name__ == "__main__":
    main()
