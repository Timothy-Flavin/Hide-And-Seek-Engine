"""Record where agents go, for occupancy heatmaps.

For each regime (nobc / bc / anneal) x level x algorithm (dqn / ppo / sac) this
loads run seed 1 (all three roles from that one checkpoint), rolls ~100k frames
(auto-resetting episodes), and accumulates a per-agent occupancy histogram on the
level's grid -- counting an agent's (y, x) cell each frame ONLY while it still has
battery (internal[2] > 0). The internal obs vector is [y, x, battery, ...] in raw
grid cells, so position = internal[:2], gated on internal[2].

It also builds the same per-role occupancy for the HUMAN dataset, reading the
controlled agent's recorded positions from every segment (split on the recorded
``controlled_agent`` field, so each role's human map is comparable to that role's
policy map).

Output (default ``offline_results/positions/``), one npz per config + per human level:
    <regime>_<alg>_<level>.npz   /   human_<level>.npz
each with agent{a}_hist (H,W int64), grid_hw, base_map (H,W,3 uint8),
agent_names, frames_per_agent, alive_frames_per_agent.

    python -m RL.record_positions                       # all 36 configs + human
    python -m RL.record_positions --regimes anneal --algs ppo --levels test_level
    python -m RL.record_positions --human-only
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.eval_checkpoint import _make_act_fn, epsilon_move
from RL.eval_adhoc import (
    build_pa_net, load_policy_state, assemble_team, checkpoint_path, variant_for,
)

ALL_REGIMES = ["nobc", "bc", "anneal"]
ALL_LEVELS = ["test_level", "island_level", "neighborhood_level", "warehouse_level"]
ALL_ALGS = ["dqn", "ppo", "sac"]
OUT_DIR = os.path.join("offline_results", "positions")

# Internal-vector layout (raw grid cells): [y, x, battery, view_range, deploy, stuck, ...]
IY, IX, IBATT = 0, 1, 2


def make_env(level, num_envs, device):
    d = os.path.join("levels", level)
    return SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=os.path.join(d, "level.png"), tiles_json=os.path.join(d, "tiles.json"),
        agents_json=os.path.join(d, "agents.json"), survivors_json=os.path.join(d, "survivors.json"),
        mode="decentralized", requires_state=False, device=device,
        ego_view=True, ego_size=32,
    )


def base_map_and_names(env, level):
    env._init_renderer()  # builds _base_map_rgb (H, W, 3)
    base = np.asarray(env._base_map_rgb, dtype=np.uint8)
    with open(os.path.join("levels", level, "agents.json")) as f:
        names = list(json.load(f).keys())
    return base, names


def _bin_positions(hist, ys, xs, batt, H, W):
    """Add the battery-alive (y, x) cells of one agent-slice into ``hist``.

    Positions are continuous cell-CENTER coordinates (an agent in cell k sits at
    k+0.5), so the containing cell is floor(pos), NOT round(pos). np.round uses
    round-half-to-even, which maps the N.5 centers onto alternating cells and
    produces a spurious checkerboard (every other cell left empty)."""
    alive = batt > 0
    if not alive.any():
        return 0
    yi = np.clip(np.floor(ys[alive]).astype(np.int64), 0, H - 1)
    xi = np.clip(np.floor(xs[alive]).astype(np.int64), 0, W - 1)
    np.add.at(hist, (yi, xi), 1)
    return int(alive.sum())


def record_policy(regime, alg, level, *, frames, num_envs, device, results_root,
                  seed=1, epsilon=0.0):
    env = make_env(level, num_envs, device)
    n_agents = env.config.n_agents
    base, names = base_map_and_names(env, level)
    # Histogram grid is the TRUE map size (base_map), NOT env.map_spatial_shape --
    # in ego-view mode the latter is the 32x32 ego crop, so binning to it would
    # clamp every out-of-range position onto the border. Agent positions
    # (internal[:2]) are in true-map cells.
    H, W = base.shape[0], base.shape[1]

    roles = (seed,) * n_agents
    state = assemble_team(
        {seed: load_policy_state(checkpoint_path(level, alg, regime, seed, results_root), alg)},
        roles)
    net = build_pa_net(alg, env.map_spatial_shape, (n_agents, env.agent_internal_dim),
                       n_agents, n_agents).to(device)
    net.load_state_dict(state)
    net.eval()
    act = epsilon_move(_make_act_fn(alg, net, True, num_envs, n_agents),
                       epsilon, num_envs, n_agents)

    hist = [np.zeros((H, W), dtype=np.int64) for _ in range(n_agents)]
    alive = [0] * n_agents
    steps = int(np.ceil(frames / num_envs))

    with torch.inference_mode():
        env.reset()
        obs = env._get_obs_dict()
        spatial, internal = obs["spatial"], obs["internal"]
        for t in range(steps):
            io = internal.cpu().numpy()                      # (E, A, D)
            for a in range(n_agents):
                alive[a] += _bin_positions(hist[a], io[:, a, IY], io[:, a, IX],
                                           io[:, a, IBATT], H, W)
            move, radio = act(spatial, internal)
            next_obs, _, term, trunc, _ = env.step(move, radio)
            done = (term | trunc).cpu().numpy()
            if done.any():
                for e in range(num_envs):
                    if done[e]:
                        env.reset_env(e)
                obs = env._get_obs_dict()
                spatial, internal = obs["spatial"], obs["internal"]
            else:
                spatial, internal = next_obs["spatial"], next_obs["internal"]

    total = steps * num_envs
    out = {f"agent{a}_hist": hist[a] for a in range(n_agents)}
    out.update(grid_hw=np.array([H, W]), base_map=base, agent_names=np.array(names),
               frames_per_agent=np.array([total] * n_agents),
               alive_frames_per_agent=np.array(alive))
    path = os.path.join(OUT_DIR, f"{regime}_{alg}_{level}.npz")
    np.savez_compressed(path, **out)
    print(f"[policy] {regime}/{alg}/{level}: {total} frames/agent, "
          f"alive {alive} -> {path}")


def record_human(level, *, results_root_demos="experiments/results", device="cpu"):
    env = make_env(level, 1, device)
    n_agents = env.config.n_agents
    base, names = base_map_and_names(env, level)
    H, W = base.shape[0], base.shape[1]  # true map size (see record_policy)

    hist = [np.zeros((H, W), dtype=np.int64) for _ in range(n_agents)]
    alive = [0] * n_agents
    frames_seen = [0] * n_agents
    base_dir = os.path.join(results_root_demos, level)
    if os.path.isdir(base_dir):
        for bucket in sorted(os.listdir(base_dir)):
            for seg in sorted(glob.glob(os.path.join(base_dir, bucket, "segment_*"))):
                oi_p = os.path.join(seg, "obs_internal.npy")
                ca_p = os.path.join(seg, "controlled_agent.npy")
                if not (os.path.exists(oi_p) and os.path.exists(ca_p)):
                    continue
                oi = np.load(oi_p)                       # (N, D) controlled agent's ego internal
                ca = np.load(ca_p).astype(int).ravel()   # (N,) which role the human drove
                for a in range(n_agents):
                    m = ca == a
                    if not m.any():
                        continue
                    frames_seen[a] += int(m.sum())
                    alive[a] += _bin_positions(hist[a], oi[m, IY], oi[m, IX], oi[m, IBATT], H, W)

    out = {f"agent{a}_hist": hist[a] for a in range(n_agents)}
    out.update(grid_hw=np.array([H, W]), base_map=base, agent_names=np.array(names),
               frames_per_agent=np.array(frames_seen), alive_frames_per_agent=np.array(alive))
    path = os.path.join(OUT_DIR, f"human_{level}.npz")
    np.savez_compressed(path, **out)
    print(f"[human]  {level}: frames/role {frames_seen}, alive {alive} -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regimes", nargs="+", default=ALL_REGIMES, choices=ALL_REGIMES)
    ap.add_argument("--levels", nargs="+", default=ALL_LEVELS, choices=ALL_LEVELS)
    ap.add_argument("--algs", nargs="+", default=ALL_ALGS, choices=ALL_ALGS)
    ap.add_argument("--frames", type=int, default=100_000, help="total frames/config")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dqn-epsilon", type=float, default=0.05,
                    help="epsilon-greedy move exploration for DQN only (its eval is a "
                         "deterministic argmax that gets stuck); 0 disables")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-root", default="offline_results")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--human-only", action="store_true", help="only (re)build human maps")
    ap.add_argument("--no-human", action="store_true", help="skip the human maps")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    if not args.human_only:
        for regime in args.regimes:
            for alg in args.algs:
                for level in args.levels:
                    path = os.path.join(OUT_DIR, f"{regime}_{alg}_{level}.npz")
                    if args.skip_existing and os.path.exists(path):
                        print(f"[skip] {path}")
                        continue
                    record_policy(regime, alg, level, frames=args.frames,
                                  num_envs=args.num_envs, device=device,
                                  results_root=args.results_root, seed=args.seed,
                                  epsilon=args.dqn_epsilon if alg == "dqn" else 0.0)

    if not args.no_human:
        for level in args.levels:
            path = os.path.join(OUT_DIR, f"human_{level}.npz")
            if args.skip_existing and os.path.exists(path):
                print(f"[skip] {path}")
                continue
            record_human(level, device=str(device))


if __name__ == "__main__":
    main()
