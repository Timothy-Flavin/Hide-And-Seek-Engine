"""Qualitative viewer: load one run's three per-agent policies, play a few
episodes, and record the global (true-state) render to a short video so you can
eyeball what the agents actually do. The mean team return is baked into the
filename.

A per-agent-nets checkpoint already holds all three role policies, so a single
run is "three models". You can also build an ad-hoc team by mixing seeds per role
with ``--seeds s0 s1 s2`` (role a uses seed s_a), matching RL/eval_adhoc.py.

Output (default ``videos/``):
    <alg>_<variant>_<seedtag>_<level>_<N>ep_team<meanR>.gif

    # one run's team (all three roles from seed 1), 3 episodes:
    python -m RL.view_episodes --alg ppo --condition anneal --level test_level --seed 1

    # ad-hoc team: role0<-seed1, role1<-seed3, role2<-seed5:
    python -m RL.view_episodes --alg sac --condition bc --level island_level --seeds 1 3 5

Headless by default (SDL dummy driver); pass --show to also open a window.
"""
import argparse
import os
import sys

# Headless rendering unless the user asked for a window -- must be set before the
# env module (which imports pygame) is imported.
if "--show" not in sys.argv and not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
import pygame
from PIL import Image, ImageDraw

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.eval_checkpoint import _make_act_fn
from RL.eval_adhoc import (
    build_pa_net, load_policy_state, assemble_team, checkpoint_path, variant_for,
)

VIDEO_DIR = "videos"


def capture_frame(env):
    """Grab the current pygame screen as an (H, W, 3) uint8 array."""
    arr = pygame.surfarray.array3d(env._pygame_screen)  # (W, H, 3)
    return np.ascontiguousarray(arr.transpose(1, 0, 2))


def overlay(frame, text):
    """Stamp a status line (yellow, with a dark shadow for legibility)."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.text((5, 5), text, fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 230, 0))
    return np.asarray(img)


def load_team(alg, condition, level, roles, results_root):
    """Assemble the per-agent policy state for a team. ``roles`` is a length-3
    tuple of seed numbers (role a uses seed roles[a])."""
    states = {}
    for s in set(roles):
        path = checkpoint_path(level, alg, condition, s, results_root)
        st = load_policy_state(path, alg)
        if st is None:
            raise SystemExit(f"missing/invalid checkpoint for seed {s}: {path}")
        states[s] = st
    return assemble_team(states, tuple(roles))


def write_video(frames, path, fps):
    """Write frames to ``path``. GIF (Pillow) by default; .mp4 needs ffmpeg."""
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".mp4"):
        try:
            imageio.mimwrite(path, frames, fps=fps, codec="libx264",
                             macro_block_size=None)
            return path
        except Exception as e:
            path = path[:-4] + ".gif"
            print(f"[video] mp4 failed ({e}); falling back to GIF -> {path}")
    imageio.mimsave(path, frames, format="GIF", duration=1.0 / fps, loop=0)
    return path


def run(args):
    device = torch.device(args.device)
    roles = tuple(args.seeds) if args.seeds else (args.seed, args.seed, args.seed)
    variant = variant_for(args.alg, args.condition)
    level_dir = os.path.join("levels", args.level)

    env = SARBatchedGridEnv(
        num_envs=1,
        map_png=os.path.join(level_dir, "level.png"),
        tiles_json=os.path.join(level_dir, "tiles.json"),
        agents_json=os.path.join(level_dir, "agents.json"),
        survivors_json=os.path.join(level_dir, "survivors.json"),
        mode="decentralized", requires_state=True, device=device,  # state -> global render
        ego_view=True, ego_size=32,
    )
    n_agents = env.config.n_agents
    net = build_pa_net(args.alg, env.map_spatial_shape,
                       (n_agents, env.agent_internal_dim), n_agents, n_agents).to(device)
    net.load_state_dict(load_team(args.alg, args.condition, args.level, roles, args.results_root))
    net.eval()
    act = _make_act_fn(args.alg, net, True, 1, n_agents)

    max_steps = args.max_steps or int(getattr(env, "max_frames", 250))
    frames, ep_returns = [], []

    with torch.inference_mode():
        env.reset()
        obs = env._get_obs_dict()
        spatial, internal = obs["spatial"], obs["internal"]
        for ep in range(args.episodes):
            ep_ret = 0.0
            for step in range(max_steps):
                move, radio = act(spatial, internal)
                next_obs, rewards_raw, term, trunc, _ = env.step(move, radio)
                ep_ret += float(rewards_raw[0].sum().item())
                env.render(pov=-1, env_idx=0)
                frames.append(overlay(capture_frame(env),
                              f"{variant} seeds{roles}  "
                              f"ep {ep + 1}/{args.episodes}  step {step + 1}  teamR={ep_ret:.1f}"))
                done = bool(term[0].item() or trunc[0].item()) or step + 1 >= max_steps
                if done:
                    break
                spatial, internal = next_obs["spatial"], next_obs["internal"]
            ep_returns.append(ep_ret)
            print(f"  episode {ep + 1}: team return {ep_ret:.2f} ({step + 1} steps)")
            env.reset_env(0)
            obs = env._get_obs_dict()
            spatial, internal = obs["spatial"], obs["internal"]

    mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
    seedtag = f"seed{args.seed}" if not args.seeds else "seeds" + "-".join(map(str, roles))
    ext = "mp4" if args.format == "mp4" else "gif"
    default_name = (f"{variant}_{seedtag}_{args.level}_"
                    f"{args.episodes}ep_team{mean_ret:.0f}.{ext}")
    out_path = args.out or os.path.join(VIDEO_DIR, default_name)
    out_path = write_video(frames, out_path, args.fps)
    print(f"[video] {len(frames)} frames, mean team return {mean_ret:.2f} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alg", required=True, choices=["ppo", "dqn", "sac"])
    ap.add_argument("--condition", default="nobc", choices=["nobc", "bc", "anneal"])
    ap.add_argument("--level", default="test_level")
    ap.add_argument("--seed", type=int, default=1,
                    help="run number; all three roles use this run's checkpoint")
    ap.add_argument("--seeds", type=int, nargs=3, metavar=("S0", "S1", "S2"),
                    help="ad-hoc team: role a uses seed S_a (overrides --seed)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None, help="cap per episode (default env max_frames)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--format", default="gif", choices=["gif", "mp4"],
                    help="mp4 needs ffmpeg/imageio-ffmpeg; falls back to gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-root", default="offline_results")
    ap.add_argument("--out", default=None, help="explicit output path (else auto-named with reward)")
    ap.add_argument("--show", action="store_true", help="also open a live window")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
