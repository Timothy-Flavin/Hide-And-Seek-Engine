"""Backfill the policy-only eval baseline (the graph's dotted 'all-policy' line).

The trainers write ``eval_returns.jsonl`` at the end of training, but if a training
round ran on an older build without the eval feature, that baseline is missing and
``RL.plot_human_loop`` has no dotted line to draw. This evaluates, headless and
locally, the policy checkpoints you already have -- so no re-training is needed:

* the CURRENT synced-back checkpoint at its ``cumulative_step`` (from the resume
  file), and
* a step-0 random-policy baseline (a freshly initialised net), matching the
  trainers' own step-0 eval.

Each is 10 episodes clipped to the env's 250-frame length (apples-to-apples with
the human demos), appended to ``eval_returns.jsonl`` in the trainers' format.
Entries whose ``cumulative_step`` is already logged are skipped, so re-running is
safe. The intermediate historical rounds can't be recovered here (the resume file
only holds the latest weights) -- those fill in as future rounds run on the lab.

    python -m RL.eval_checkpoint --level levels/test_level --alg ppo --run 1
    # all loop levels:
    for L in levels/*_level; do python -m RL.eval_checkpoint --level "$L" --alg ppo --run 1; done
"""
import argparse
import json
import os

import numpy as np
import torch

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.eval_utils import evaluate_policy, append_eval_stats
from RL.checkpoint_utils import (
    resume_checkpoint_path,
    results_dir_for,
    load_resume,
    strip_compile_prefix,
    variant_name,
)
from RL.human_data import level_name_of

# Discrete move set shared with the trainers (index -> [dy, dx]).
ACTION_MAP = np.array(
    [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=np.float32
)
POLICY_KEY = {"dqn": "q_network", "ppo": "agent", "sac": "actor"}


def _build_net(alg, spatial_shape, internal_dim, n_agents, use_radio, n_radio_actions):
    """Construct the (uncompiled) policy net for one algorithm."""
    if alg == "ppo":
        from RL.cleanrl_ppo import Agent
        return Agent(spatial_shape, internal_dim, n_agents, centralized=False,
                     use_radio=use_radio, n_radio_actions=n_radio_actions)
    if alg == "dqn":
        from RL.cleanrl_dqn import QNetwork
        return QNetwork(spatial_shape, internal_dim, n_agents, centralized=False,
                        use_radio=use_radio, n_radio_actions=n_radio_actions)
    if alg == "sac":
        from RL.cleanrl_sac import Actor
        return Actor(spatial_shape, internal_dim, n_agents, centralized=False,
                     use_radio=use_radio, n_radio_actions=n_radio_actions)
    raise ValueError(f"unknown alg {alg}")


def _make_act_fn(alg, net, use_radio, num_envs, n_agents):
    """Batched act_fn mirroring how each trainer selects rollout actions."""
    def act(spatial, internal):
        with torch.no_grad():
            if alg == "ppo":
                out = net.get_action_and_value(spatial, internal)
                move, radio = out[0], (out[4] if use_radio else None)
            elif alg == "dqn":
                if use_radio:
                    move, radio = net.get_action(spatial, internal)  # bound to get_action_radio
                else:
                    move, radio = net.get_action(spatial, internal), None
            else:  # sac
                if use_radio:
                    out = net.get_action_radio(spatial, internal)
                    move, radio = out[0], out[3]
                else:
                    move, radio = net.get_action(spatial, internal)[0], None
        move_np = ACTION_MAP[move.cpu().numpy()]
        radio_np = (radio.cpu().numpy().astype(np.int32) if radio is not None
                    else np.zeros((num_envs, n_agents), dtype=np.int32))
        return move_np, radio_np
    return act


def epsilon_move(act, epsilon, num_envs, n_agents):
    """Wrap an act_fn with epsilon-greedy on the MOVE action only. A deterministic
    argmax policy (DQN's eval) can get stuck -- looping or wedged against a wall --
    which distorts occupancy maps and ad-hoc team scores; a small epsilon injects
    random moves so behavior reflects real coverage. Radio actions are untouched.
    ``epsilon <= 0`` returns ``act`` unchanged."""
    if epsilon <= 0:
        return act
    n_actions = len(ACTION_MAP)

    def wrapped(spatial, internal):
        move_np, radio_np = act(spatial, internal)
        explore = np.random.random((num_envs, n_agents)) < epsilon
        if explore.any():
            ridx = np.random.randint(0, n_actions, size=(num_envs, n_agents))
            move_np[explore] = ACTION_MAP[ridx[explore]]
        return move_np, radio_np
    return wrapped


def _load_state(level, alg, run, use_radio):
    """Return (state_dict, cumulative_step, prefix) for the current checkpoint, or
    (None, None, prefix). Prefers the rolling resume checkpoint (which carries the
    cumulative step), then the 100% weight snapshot."""
    variant = variant_name(centralized=False, ego_view=True, use_radio=use_radio)
    prefix = f"{alg}_{variant}_run_{run}"
    results_dir = results_dir_for(level, alg)
    key = POLICY_KEY[alg]

    resume = load_resume(resume_checkpoint_path(results_dir, prefix), map_location="cpu")
    if resume is not None and key in resume.get("models", {}):
        return resume["models"][key], int(resume.get("cumulative_step", 0)), prefix

    pct100 = os.path.join(results_dir, "checkpoints", f"{prefix}_pct100.pt")
    if os.path.exists(pct100):
        state = torch.load(pct100, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and key in state:
            return state[key], None, prefix
    return None, None, prefix


def _existing_steps(level, alg, prefix):
    """cumulative_steps already logged for this prefix (to avoid duplicates)."""
    path = os.path.join(results_dir_for(level, alg), "eval_returns.jsonl")
    steps = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("prefix") == prefix and e.get("cumulative_step") is not None:
                    steps.add(int(e["cumulative_step"]))
    return steps


def _eval_net(env, net, alg, use_radio, num_envs, n_agents, agent_names, eval_episodes):
    net.eval()
    act = _make_act_fn(alg, net, use_radio, num_envs, n_agents)
    stats = evaluate_policy(env, act, num_envs, n_agents, n_episodes=eval_episodes,
                            max_ep_steps=int(getattr(env, "max_frames", 250)))
    stats["agent_names"] = list(agent_names)
    return stats


def backfill(level, alg, run, *, num_envs=16, eval_episodes=10, device="cpu"):
    env = SARBatchedGridEnv(
        num_envs=num_envs,
        map_png=os.path.join(level, "level.png"),
        tiles_json=os.path.join(level, "tiles.json"),
        agents_json=os.path.join(level, "agents.json"),
        survivors_json=os.path.join(level, "survivors.json"),
        mode="decentralized", requires_state=False, device=device,
        ego_view=True, ego_size=32,
    )
    n_agents = env.config.n_agents
    spatial_shape = env.map_spatial_shape
    internal_dim = (n_agents, env.agent_internal_dim)
    n_radio_actions = n_agents
    with open(os.path.join(level, "agents.json")) as f:
        agent_names = list(json.load(f).keys())

    # Resolve which variant this run trained (radio preferred, matching the loop).
    state = cumulative = None
    use_radio = True
    for ur in (True, False):
        state, cumulative, prefix = _load_state(level, alg, run, ur)
        if state is not None:
            use_radio = ur
            break
    prefix = f"{alg}_{variant_name(centralized=False, ego_view=True, use_radio=use_radio)}_run_{run}"
    done_steps = _existing_steps(level, alg, prefix)

    def _maybe_log(net, step, tag):
        if step in done_steps:
            print(f"[backfill] {level_name_of(level)}/{alg}: step {step} already logged; skip {tag}.")
            return
        stats = _eval_net(env, net, alg, use_radio, num_envs, n_agents, agent_names, eval_episodes)
        stats["cumulative_step"] = int(step)
        path = append_eval_stats(level, alg, prefix, stats)
        done_steps.add(int(step))
        print(f"[backfill] {tag} @ {step}: team {stats['team_return_mean']:.3f} "
              f"per-agent {['%.2f' % x for x in stats['per_agent_return_mean']]} "
              f"over {stats['n_episodes']} eps -> {path}")

    # Step-0 random baseline (fresh net), matching the trainers' step-0 eval.
    rnd = _build_net(alg, spatial_shape, internal_dim, n_agents, use_radio, n_radio_actions).to(device)
    _maybe_log(rnd, 0, "random baseline")

    # Current trained checkpoint.
    if state is None:
        print(f"[backfill] {level_name_of(level)}/{alg}: no trained checkpoint for run {run}; "
              f"only the step-0 baseline was written.")
        return
    if cumulative is None:
        print(f"[backfill] {level_name_of(level)}/{alg}: checkpoint has no cumulative_step "
              f"(no resume file); skipping the trained point.")
        return
    net = _build_net(alg, spatial_shape, internal_dim, n_agents, use_radio, n_radio_actions).to(device)
    try:
        net.load_state_dict(strip_compile_prefix(state))
    except (RuntimeError, KeyError) as exc:
        print(f"[backfill] {level_name_of(level)}/{alg}: checkpoint doesn't match current "
              f"architecture ({exc}); skipping the trained point.")
        return
    _maybe_log(net, cumulative, "trained checkpoint")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", required=True, help="level dir (e.g. levels/test_level)")
    ap.add_argument("--alg", default="ppo", choices=["ppo", "dqn", "sac"])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    backfill(args.level, args.alg, args.run, num_envs=args.num_envs,
             eval_episodes=args.eval_episodes, device=args.device)


if __name__ == "__main__":
    main()
