"""Ad-hoc teamplay evaluation: does human-BC change how well independently-trained
agents cooperate when mixed into teams they never trained with?

Every offline run uses *per-agent nets* (``--per-agent-nets``): one independent
(encoder + head) module per agent role, stored together in a single checkpoint.
That lets us build "ad-hoc" teams by mixing roles across seeds -- role 0's net
from seed i, role 1's from seed j, role 2's from seed k -- none of which trained
alongside the others. With 5 seeds and 3 roles that is ``5**3 = 125`` team
compositions per level, ``x4 levels = 500`` per condition.

For each condition (``bc`` vs ``nobc``) and each algorithm we evaluate all 500
compositions over 32 episodes (32 parallel envs, everything in eval / inference
mode, no gradients), and record the mean and stdev of the *team* score. The
result is a ``5x5x5`` matrix per level (axis a = the seed used for role a),
saved to ``offline_results/adhoc_eval/<alg>_<condition>.npz``. Diffing the ``bc``
and ``nobc`` matrices is the ad-hoc BC effect.

Roughly 500 comps x 32 eps x ~156 avg frames ~= 2.5M frames per (alg, condition).

    # full run on timpc (all algos, both conditions):
    python -m RL.eval_adhoc --alg all --conditions bc nobc --device cuda

    # CPU smoke test (tiny):
    TORCHDYNAMO_DISABLE=1 python -m RL.eval_adhoc --alg sac --conditions nobc \
        --levels test_level --seeds 2 --num-envs 2 --episodes 2 --device cpu
"""
import argparse
import itertools
import json
import os
import re
import time

import numpy as np
import torch

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.eval_utils import evaluate_policy
from RL.eval_checkpoint import _make_act_fn, POLICY_KEY
from RL.checkpoint_utils import strip_compile_prefix

ALL_LEVELS = ["test_level", "island_level", "neighborhood_level", "warehouse_level"]
ALL_ALGS = ["sac", "dqn", "ppo"]

# Base per-agent-nets variant, and the BC-variant suffix per algorithm (dqn's
# offline/BC objective is CQL; ppo/sac use a BC auxiliary loss).
BASE_VARIANT = "decentralized_ego_radio_pa"
BC_SUFFIX = {"ppo": "bc", "sac": "bc", "dqn": "cql"}

# ModuleLists inside each net that are indexed by agent role; every other
# parameter (if any) is shared and taken from the first team member.
PER_AGENT_MODULES = ("encoders", "actor_heads", "adv_heads", "v_heads", "critics")
_PER_AGENT_RE = re.compile(r"^([a-zA-Z_]+)\.(\d+)\.")

OUT_DIR = os.path.join("offline_results", "adhoc_eval")


def build_pa_net(alg, spatial_shape, internal_dim, n_agents, n_radio_actions):
    """Construct the *per-agent-nets* policy for one algorithm (decentralized,
    radio on) -- matching how the offline runs were trained (``--per-agent-nets``).
    ``eval_checkpoint._build_net`` only builds the shared-net variant, so the
    per-agent ModuleLists are built explicitly here."""
    if alg == "sac":
        from RL.cleanrl_sac import Actor
        return Actor(spatial_shape, internal_dim, n_agents, centralized=False,
                     use_radio=True, n_radio_actions=n_radio_actions, per_agent=True)
    if alg == "dqn":
        from RL.cleanrl_dqn import QNetwork
        return QNetwork(spatial_shape, internal_dim, n_agents, centralized=False,
                        use_radio=True, n_radio_actions=n_radio_actions, per_agent=True)
    if alg == "ppo":
        from RL.cleanrl_ppo import Agent
        return Agent(spatial_shape, internal_dim, n_agents, centralized=False,
                     use_radio=True, n_radio_actions=n_radio_actions, per_agent=True)
    raise ValueError(f"unknown alg {alg}")


def variant_for(alg, condition):
    """Checkpoint variant string for (alg, condition)."""
    if condition == "nobc":
        return f"{alg}_{BASE_VARIANT}"
    return f"{alg}_{BASE_VARIANT}_{BC_SUFFIX[alg]}"


def checkpoint_path(level, alg, condition, seed, results_root):
    variant = variant_for(alg, condition)
    return os.path.join(results_root, level, alg, "checkpoints",
                        f"{variant}_run_{seed}_pct100.pt")


def load_policy_state(path, alg):
    """Return the (compile-prefix-stripped) policy state_dict from a checkpoint,
    or None if the file is missing / malformed."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key = POLICY_KEY[alg]
    if not isinstance(ckpt, dict) or key not in ckpt:
        return None
    return strip_compile_prefix(ckpt[key])


def assemble_team(seed_states, roles):
    """Franken state_dict: for role a use ``seed_states[roles[a]]`` for that
    role's per-agent modules; shared params come from ``roles[0]``.

    ``seed_states`` maps seed-index -> policy state_dict (identical key sets).
    ``roles`` is a length-n_agents tuple of seed indices, one per role."""
    template = seed_states[roles[0]]
    out = {}
    for k in template:
        m = _PER_AGENT_RE.match(k)
        if m and m.group(1) in PER_AGENT_MODULES:
            a = int(m.group(2))
            src = roles[a] if a < len(roles) else roles[0]
        else:
            src = roles[0]
        out[k] = seed_states[src][k]
    return out


def make_env(level, num_envs, device, results_root_levels="levels"):
    level_dir = os.path.join(results_root_levels, level)
    return SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=os.path.join(level_dir, "level.png"),
        tiles_json=os.path.join(level_dir, "tiles.json"),
        agents_json=os.path.join(level_dir, "agents.json"),
        survivors_json=os.path.join(level_dir, "survivors.json"),
        mode="decentralized", requires_state=False, device=device,
        ego_view=True, ego_size=32,
    )


def eval_level(alg, condition, level, *, seeds, num_envs, episodes, device,
               results_root, max_ep_steps, limit=None):
    """Evaluate all seed^n_agents compositions for one (alg, condition, level).

    Returns (result_dict, frames_simulated) or (None, 0) if checkpoints missing."""
    # Load the per-seed policy state dicts up front (cheap, reused across comps).
    seed_states = {}
    for s in range(1, seeds + 1):
        st = load_policy_state(checkpoint_path(level, alg, condition, s, results_root), alg)
        if st is None:
            print(f"  [warn] missing checkpoint: {level}/{alg}/{variant_for(alg, condition)} seed {s}")
        seed_states[s - 1] = st  # 0-indexed seed slot

    env = make_env(level, num_envs, device)
    n_agents = env.config.n_agents
    spatial_shape = env.map_spatial_shape
    internal_dim = (n_agents, env.agent_internal_dim)
    with open(os.path.join("levels", level, "agents.json")) as f:
        agent_names = list(json.load(f).keys())

    # One eager net for the level; we swap weights per composition (no recompile).
    net = build_pa_net(alg, spatial_shape, internal_dim, n_agents, n_agents).to(device)
    net.eval()
    act = _make_act_fn(alg, net, True, num_envs, n_agents)

    shape = (seeds,) * n_agents
    mean = np.full(shape, np.nan, dtype=np.float64)
    std = np.full(shape, np.nan, dtype=np.float64)
    nep = np.zeros(shape, dtype=np.int64)
    frames = 0

    comps = list(itertools.product(range(seeds), repeat=n_agents))
    if limit is not None:
        comps = comps[:limit]
    t0 = time.time()
    for c, roles in enumerate(comps):
        if any(seed_states[r] is None for r in roles):
            continue  # a required seed's checkpoint is missing -> leave NaN
        net.load_state_dict(assemble_team(seed_states, roles))
        stats = evaluate_policy(env, act, num_envs, n_agents,
                                n_episodes=episodes, max_ep_steps=max_ep_steps)
        mean[roles] = stats["team_return_mean"]
        std[roles] = stats["team_return_std"]
        nep[roles] = stats["n_episodes"]
        frames += int(round(stats["n_episodes"] * stats["ep_len_mean"]))
        if (c + 1) % 25 == 0 or c + 1 == len(comps):
            rate = (c + 1) / max(time.time() - t0, 1e-6)
            print(f"  {level}: {c + 1}/{len(comps)} comps "
                  f"({rate:.1f}/s) last team={stats['team_return_mean']:.2f}")
    result = {
        "mean": mean, "std": std, "nepisodes": nep,
        "agent_names": agent_names, "n_agents": n_agents,
    }
    return result, frames


def run(alg, condition, *, levels, seeds, num_envs, episodes, device,
        results_root, max_ep_steps, limit, skip_existing):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, f"{alg}_{condition}.npz")
    out_json = os.path.join(OUT_DIR, f"{alg}_{condition}.json")
    if skip_existing and os.path.exists(out_npz):
        print(f"[skip] {out_npz} exists")
        return
    print(f"=== {alg} / {condition} (variant {variant_for(alg, condition)}) ===")

    arrays = {}
    meta = {
        "alg": alg, "condition": condition, "variant": variant_for(alg, condition),
        "seeds": seeds, "num_envs": num_envs, "episodes": episodes,
        "max_ep_steps": max_ep_steps, "device": str(device),
        "axis_meaning": "matrix[s0, s1, s2] = team score with role a using seed s_a+1",
        "levels": {},
    }
    total_frames = 0
    for level in levels:
        res, frames = eval_level(
            alg, condition, level, seeds=seeds, num_envs=num_envs,
            episodes=episodes, device=device, results_root=results_root,
            max_ep_steps=max_ep_steps, limit=limit)
        if res is None:
            continue
        arrays[f"{level}__mean"] = res["mean"]
        arrays[f"{level}__std"] = res["std"]
        arrays[f"{level}__nepisodes"] = res["nepisodes"]
        finite = np.isfinite(res["mean"])
        meta["levels"][level] = {
            "agent_names": res["agent_names"],
            "n_compositions": int(finite.sum()),
            "team_mean_over_comps": float(np.nanmean(res["mean"])) if finite.any() else None,
            "team_std_over_comps": float(np.nanstd(res["mean"])) if finite.any() else None,
            "frames": frames,
        }
        total_frames += frames
        # Save incrementally so a crash mid-run keeps completed levels.
        np.savez(out_npz, **arrays)
        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {level}: mean-over-comps "
              f"{meta['levels'][level]['team_mean_over_comps']}, saved {out_npz}")
    meta["total_frames"] = total_frames
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {alg}/{condition}: {total_frames:,} frames -> {out_npz}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alg", default="all",
                    help="sac|dqn|ppo|all (default all)")
    ap.add_argument("--conditions", nargs="+", default=["bc", "nobc"],
                    choices=["bc", "nobc"])
    ap.add_argument("--levels", nargs="+", default=ALL_LEVELS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--max-ep-steps", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-root", default="offline_results",
                    help="root holding <level>/<alg>/checkpoints (default offline_results)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap compositions per level (smoke tests)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip an (alg,condition) whose .npz already exists")
    args = ap.parse_args()

    algs = ALL_ALGS if args.alg == "all" else [args.alg]
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.inference_mode():
        for alg in algs:
            for condition in args.conditions:
                run(alg, condition, levels=args.levels, seeds=args.seeds,
                    num_envs=args.num_envs, episodes=args.episodes, device=device,
                    results_root=args.results_root, max_ep_steps=args.max_ep_steps,
                    limit=args.limit, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
