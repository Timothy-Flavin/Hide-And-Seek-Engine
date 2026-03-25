import argparse
import time

import numpy as np

from hide_and_seek_engine.env_wrapper import (
    SARBatchedGridEnv,
    SARGlobalModeEnv,
    load_sar_config,
)


def _random_local_actions(num_envs: int, n_agents: int) -> np.ndarray:
    actions = np.zeros((num_envs, n_agents, 3), dtype=np.float32)
    actions[:, :, :2] = np.random.uniform(-1.0, 1.0, size=(num_envs, n_agents, 2)).astype(np.float32)
    actions[:, :, 2] = np.random.randint(0, 4, size=(num_envs, n_agents)).astype(np.float32)
    return actions


def _random_global_actions(num_envs: int, n_agents: int) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=(num_envs, n_agents * 2)).astype(np.float32)


def unit_schema_checks(tiles_json: str, agents_json: str, survivors_json: str, map_size: int = 32):
    cfg = load_sar_config(
        tiles_json=tiles_json,
        agents_json=agents_json,
        survivors_json=survivors_json,
        num_envs=1,
        map_size=map_size,
    )

    assert len(cfg.terrain_names) == cfg.terrain_rgb.shape[0]
    assert cfg.tile_supports_walking.shape[0] == len(cfg.terrain_names)
    assert cfg.tile_supports_aquatic.shape[0] == len(cfg.terrain_names)
    assert cfg.tile_supports_flying.shape[0] == len(cfg.terrain_names)
    assert cfg.tile_blocking.shape[0] == len(cfg.terrain_names)
    assert cfg.agent_specs.shape[0] == 4
    assert cfg.agent_specs.shape[1] == 10 + len(cfg.terrain_names)
    print("[unit] schema checks: PASS")


def run_local_headless(num_envs: int, steps: int, assets: dict[str, str], seed: int = 42) -> float:
    env = SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=assets["map_png"],
        tiles_json=assets["tiles_json"],
        agents_json=assets["agents_json"],
        survivors_json=assets["survivors_json"],
        map_size=32,
        seed=seed,
    )
    env.reset()

    t0 = time.perf_counter()
    total_step_calls = 0
    for _ in range(steps):
        actions = _random_local_actions(num_envs, env.n_agents)
        _, _, terminated, _, _ = env.step(actions)
        total_step_calls += num_envs
        term_np = terminated.cpu().numpy()
        for e in np.where(term_np)[0]:
            env.reset_env(int(e))
    dt = time.perf_counter() - t0
    env.close()

    fps = total_step_calls / max(dt, 1e-8)
    print(f"[local] envs={num_envs:>3} steps={steps:>6} step_calls={total_step_calls:>8} fps={fps:,.1f}")
    return fps


def run_global_headless(num_envs: int, steps: int, seed: int = 42) -> float:
    env = SARGlobalModeEnv(num_envs=num_envs, seed=seed)
    env.reset()

    t0 = time.perf_counter()
    total_step_calls = 0
    for _ in range(steps):
        actions = _random_global_actions(num_envs, env.n_agents)
        _, terminated = env.step(actions)
        total_step_calls += num_envs
        term_np = np.asarray(terminated, dtype=bool)
        if np.any(term_np):
            env.core.reset()
    dt = time.perf_counter() - t0

    fps = total_step_calls / max(dt, 1e-8)
    print(f"[global] envs={num_envs:>3} steps={steps:>6} step_calls={total_step_calls:>8} fps={fps:,.1f}")
    return fps


def run_renderer_episode(assets: dict[str, str], max_steps: int = 300, seed: int = 42):
    env = SARBatchedGridEnv(
        num_envs=1,
        map_png=assets["map_png"],
        tiles_json=assets["tiles_json"],
        agents_json=assets["agents_json"],
        survivors_json=assets["survivors_json"],
        map_size=32,
        seed=seed,
    )
    env.reset()

    for step in range(max_steps):
        actions = _random_local_actions(1, env.n_agents)
        _, _, terminated, _, _ = env.step(actions)

        if step < 30:
            env.render(env_idx=0)
        else:
            pov_agent = (step - 30) % env.n_agents
            env.render_pov(agent_idx=pov_agent, env_idx=0)

        if bool(terminated[0].item()):
            break

    env.close()
    print("[render] global + per-agent POV renderer: PASS")


def benchmark_suite(assets: dict[str, str], steps: int, env_counts: list[int]):
    print("\n=== Local Mode Benchmarks ===")
    for n in env_counts:
        run_local_headless(num_envs=n, steps=steps, assets=assets)

    print("\n=== Global Mode Benchmarks ===")
    for n in env_counts:
        run_global_headless(num_envs=n, steps=steps)


def main():
    parser = argparse.ArgumentParser(description="SAR environment unit/integration/benchmark suite")
    parser.add_argument("--steps", type=int, default=10_000, help="Headless benchmark steps")
    parser.add_argument("--envs", type=int, nargs="*", default=[1, 2, 4, 8], help="Parallel env counts to test")
    parser.add_argument("--skip-render", action="store_true", help="Skip renderer smoke test")
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

    unit_schema_checks(args.tiles_json, args.agents_json, args.survivors_json)
    benchmark_suite(assets=assets, steps=args.steps, env_counts=args.envs)

    if not args.skip_render:
        run_renderer_episode(assets=assets, seed=args.seed)


if __name__ == "__main__":
    main()
