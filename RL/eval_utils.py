"""Post-training evaluation for the record -> train -> record loop.

After a training chunk (and once at step 0 for the random policy), roll the
policy headless for a fixed number of episodes -- each clipped to the SAME max
length as the human demos / training episodes, so human-in-the-loop and
policy-only numbers are apples-to-apples -- and record per-agent + team return.
Results append to ``experiments/results/<level>/<alg>/eval_returns.jsonl`` (the
remote loop syncs the whole ``<level>/<alg>/`` dir back), and feed the 6-curve
human-vs-eval plot (see ``RL/plot_human_loop.py``).

The evaluator is policy-agnostic: the caller passes an ``act_fn`` mapping the
batched obs to ``(move_actions, radio_actions)`` numpy arrays, mirroring exactly
how that trainer selects actions during rollouts.
"""
import json
import os
import time

import numpy as np

RESULTS_ROOT = "experiments/results"


def evaluate_policy(env, act_fn, num_envs, n_agents, *, n_episodes=10, max_ep_steps=250):
    """Roll the batched ``env`` under ``act_fn`` until ``n_episodes`` episodes
    complete, each capped at ``max_ep_steps`` steps (matching the human/training
    episode length), collecting per-agent episodic return.

    ``act_fn(spatial, internal) -> (move_actions [E,A,2] float32,
    radio_actions [E,A] int32)``. Returns a stats dict with per-agent and team
    mean/std over the first ``n_episodes`` completed episodes.
    """
    env.reset()
    obs = env._get_obs_dict()
    spatial, internal = obs["spatial"], obs["internal"]

    ep_ret = np.zeros((num_envs, n_agents), dtype=np.float64)  # per-env, per-agent
    ep_len = np.zeros(num_envs, dtype=np.int64)
    returns: list[np.ndarray] = []   # each a (n_agents,) completed-episode return
    lengths: list[int] = []

    # Generous safety budget: with num_envs parallel episodes, n_episodes complete
    # in ~ceil(n_episodes/num_envs) full-length episodes; add slack for early ends.
    safety = (n_episodes + num_envs) * max_ep_steps
    steps = 0
    while len(returns) < n_episodes and steps < safety:
        move_actions, radio_actions = act_fn(spatial, internal)
        next_obs, rewards_raw, terminations, truncations, _ = env.step(move_actions, radio_actions)
        ep_ret += rewards_raw.cpu().numpy()
        ep_len += 1

        term = terminations.cpu().numpy()
        trunc = truncations.cpu().numpy()
        forced = ep_len >= max_ep_steps
        done = term | trunc | forced

        if done.any():
            for e in range(num_envs):
                if done[e]:
                    returns.append(ep_ret[e].copy())
                    lengths.append(int(ep_len[e]))
                    ep_ret[e] = 0.0
                    ep_len[e] = 0
                    env.reset_env(e)
            obs = env._get_obs_dict()
            spatial, internal = obs["spatial"], obs["internal"]
        else:
            spatial, internal = next_obs["spatial"], next_obs["internal"]
        steps += 1

    if returns:
        arr = np.stack(returns[:n_episodes], axis=0)          # (E, n_agents)
        len_arr = np.asarray(lengths[:n_episodes], dtype=np.float64)
    else:  # no episode completed within budget: fall back to partial returns
        arr = ep_ret.copy()
        len_arr = ep_len.astype(np.float64)
    team = arr.sum(axis=1)
    return {
        "n_episodes": int(arr.shape[0]),
        "max_ep_steps": int(max_ep_steps),
        "ep_len_mean": float(len_arr.mean()),
        "per_agent_return_mean": [float(x) for x in arr.mean(axis=0)],
        "per_agent_return_std": [float(x) for x in arr.std(axis=0)],
        "team_return_mean": float(team.mean()),
        "team_return_std": float(team.std()),
    }


def run_and_log_eval(env, act_fn, num_envs, n_agents, *, level, alg, prefix,
                     agent_names, cumulative_step, n_episodes=10, max_ep_steps=250):
    """Evaluate, tag with agent names + cumulative step, append to the level's
    eval log, and print a summary -- restoring RNG (torch/numpy/python) and the
    env afterward so training stays bit-identical (RNG-wise) to a no-eval run."""
    import random

    import torch

    rng_torch = torch.get_rng_state()
    rng_np = np.random.get_state()
    rng_py = random.getstate()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        stats = evaluate_policy(env, act_fn, num_envs, n_agents,
                                n_episodes=n_episodes, max_ep_steps=max_ep_steps)
    finally:
        torch.set_rng_state(rng_torch)
        np.random.set_state(rng_np)
        random.setstate(rng_py)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        env.reset()

    stats["cumulative_step"] = int(cumulative_step)
    stats["agent_names"] = list(agent_names)
    path = append_eval_stats(level, alg, prefix, stats)
    print(f"[{alg}] eval @ {cumulative_step}: team {stats['team_return_mean']:.3f} "
          f"per-agent {['%.2f' % x for x in stats['per_agent_return_mean']]} "
          f"over {stats['n_episodes']} eps -> {path}")
    return stats


def append_eval_stats(level, alg, prefix, stats, results_root=RESULTS_ROOT):
    """Append one eval summary to ``<results>/<level>/<alg>/eval_returns.jsonl``."""
    from RL.human_data import level_name_of

    out_dir = os.path.join(results_root, level_name_of(level), alg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_returns.jsonl")
    entry = {"timestamp": time.time(), "prefix": prefix, **stats}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return path
