"""Human demonstration recorder for the ego-centric SAR environment.

Collects human demos to behavior-clone (BC) pre-train the decentralized_ego
agents, then supports the iterative record -> train -> record loop. One run:

* records one ``--level``, or -- when omitted -- sweeps ALL levels so every env
  reaches its per-agent-type quota; each episode hands the player a random agent
  (from those still short of quota) to control from its ego POV;
* auto-radios for the controlled agent via a heuristic: the instant a survivor
  enters its ego view it broadcasts, and otherwise sends a random message to
  another agent with ~0.1 probability per step (~1 every 10 frames), so the BC
  targets contain enough radio use for the policy to learn its radio head;
* drives the non-controlled teammates with the latest decentralized_ego policy
  checkpoint for the level (random fallback). Teammates also transmit radio:
  from the policy's radio head when a decentralized_ego_radio checkpoint is
  loaded, else the same heuristic -- so their shared location/tiles/POIs reach
  the controlled agent's ego view;
* records ``--frames-per-agent`` frames *per agent type, per level* (default
  2500) and *appends* them under ``experiments/results/<level>/<agent_type>/`` as
  immutable segments; a level is done once every agent type hits its quota;
* logs the mean/std team episodic return for the session to
  ``experiments/results/<level>/human_returns.jsonl`` so team performance can be
  tracked as the RL improves over the record -> train -> record loop.

The recorded ego observation is the uint8 crop the network consumes, so the same
files feed the trainers' ``--human-bc`` term directly. The player sees exactly
that ego crop -- the agent centered in its view-range ring, discovered tiles,
and teammate/POI info shared over the radio -- plus a text panel of the vectorized
observation (position, battery, view range, known POI/teammate coords), i.e. all
the same data an artificial agent sees.

Controls: WASD move, close the window to stop (partial data is still appended).
Radio is automatic.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.checkpoint_utils import (
    load_resume,
    resume_checkpoint_path,
    results_dir_for,
    strip_compile_prefix,
    variant_name,
)
from RL.human_data import (
    HUMAN_FIELDS,
    append_human_segment,
    append_return_stats,
    count_human_frames,
    level_name_of,
)

try:
    import pygame
except ImportError as exc:
    raise ImportError(
        "pygame is required for human_dataset.py. Install with `pip install pygame`."
    ) from exc


# Discrete move set shared with the trainers (index -> [dy, dx]).
ACTION_MAP = np.array(
    [
        [0.0, 0.0],   # 0 stay
        [-1.0, 0.0],  # 1 up
        [1.0, 0.0],   # 2 down
        [0.0, -1.0],  # 3 left
        [0.0, 1.0],   # 4 right
    ],
    dtype=np.float32,
)

SPOTTED_CHANNEL = 1                 # radio channel broadcast when a survivor is seen
RANDOM_MESSAGE_PROB = 0.1           # per-step chance of an idle peer message (~1 / 10 frames)
# Model-dict key that holds the policy weights, per algorithm (matches the
# trainers' module_state / ckpt_state calls).
POLICY_KEY = {"dqn": "q_network", "ppo": "agent", "sac": "actor"}


def _keyboard_move() -> np.ndarray:
    keys = pygame.key.get_pressed()
    dy = dx = 0.0
    if keys[pygame.K_w]:
        dy -= 1.0
    if keys[pygame.K_s]:
        dy += 1.0
    if keys[pygame.K_a]:
        dx -= 1.0
    if keys[pygame.K_d]:
        dx += 1.0
    return np.asarray([dy, dx], dtype=np.float32)


def _discretize_move(move_xy: np.ndarray) -> int:
    return int(np.argmin(np.sum((ACTION_MAP - move_xy[None, :2]) ** 2, axis=1)))


def _heuristic_radio(ego_spatial, step: int, n_agents: int, poi_channel: int) -> int:
    """Heuristic radio action for one agent from its ego crop.

    Broadcasts the instant a survivor is in the agent's ego view; otherwise, on
    each step, sends a random peer message with probability RANDOM_MESSAGE_PROB
    (~1 every 10 frames). The steady background chatter plus the on-sighting
    broadcast guarantee the BC targets contain plenty of non-silent radio
    actions, so a policy cloning them actually learns to use the radio head.
    Radio value 0 == silent, 1..n_agents-1 == share with a peer (decoded
    engine-side). Used for the human and for teammates when the loaded policy
    has no radio head.
    """
    if n_agents <= 1:
        return 0
    survivor_spotted = bool((ego_spatial[poi_channel] > 0).any().item())
    if survivor_spotted:
        return SPOTTED_CHANNEL
    if step > 0 and random.random() < RANDOM_MESSAGE_PROB:
        return random.randrange(1, n_agents)
    return 0


def _load_agent_names(agents_json: str) -> list[str]:
    with open(agents_json) as f:
        return list(json.load(f).keys())


class TeammatePolicy:
    """Latest decentralized_ego policy for a level, used to drive teammates.

    Only decentralized ego checkpoints are usable (the runner feeds each agent
    its own ego crop). ``has_radio`` is True when a ``decentralized_ego_radio``
    checkpoint was loaded, in which case ``act`` returns policy-chosen radio
    actions too; otherwise radio is left to the caller's heuristic.
    """

    def __init__(self, net, act_fn, has_radio: bool):
        self.net = net
        self.act_fn = act_fn
        self.has_radio = has_radio

    def act(self, spatial, internal):
        """Return (move, radio) per-agent int arrays of shape (n_agents,).

        ``radio`` is None when the policy has no radio head (caller supplies a
        heuristic). spatial: (1, n_agents, C, S, S) uint8; internal: (1, n_agents, D).
        """
        return self.act_fn(spatial, internal)


def _load_policy_state(level, alg, run_number, use_radio):
    """Return the decentralized_ego[_radio] policy state_dict for
    (level, alg, run, use_radio), or None. Prefers the rolling resume
    checkpoint, then the 100% weight snapshot."""
    import torch

    variant = variant_name(centralized=False, ego_view=True, use_radio=use_radio)
    results_dir = results_dir_for(level, alg)
    prefix = f"{alg}_{variant}_run_{run_number}"
    key = POLICY_KEY[alg]

    resume = load_resume(resume_checkpoint_path(results_dir, prefix), map_location="cpu")
    if resume is not None and key in resume.get("models", {}):
        return resume["models"][key]

    pct100 = os.path.join(results_dir, "checkpoints", f"{prefix}_pct100.pt")
    if os.path.exists(pct100):
        state = torch.load(pct100, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and key in state:
            return state[key]
    return None


def _current_checkpoint_id(level, alg, run_number):
    """Stable id for the teammate checkpoint currently driving the level, so
    recorded demos can be tagged with (and counted against) the checkpoint they
    were gathered for. Mirrors ``_build_teammate_policy``'s load order (radio
    variant preferred). Returns ``"none"`` when no checkpoint exists yet."""
    import torch

    key = POLICY_KEY[alg]
    for use_radio in (True, False):
        variant = variant_name(centralized=False, ego_view=True, use_radio=use_radio)
        results_dir = results_dir_for(level, alg)
        prefix = f"{alg}_{variant}_run_{run_number}"

        resume = load_resume(resume_checkpoint_path(results_dir, prefix), map_location="cpu")
        if resume is not None and key in resume.get("models", {}):
            return f"{prefix}_step{int(resume.get('cumulative_step', 0))}"

        pct100 = os.path.join(results_dir, "checkpoints", f"{prefix}_pct100.pt")
        if os.path.exists(pct100):
            return f"{prefix}_pct100"
    return "none"


def _build_teammate_policy(env, level, alg, run_number, device):
    """Build the teammate policy, preferring a radio-enabled checkpoint.

    Tries the ``decentralized_ego_radio`` variant first (teammates then choose
    both movement *and* radio); falls back to the move-only ``decentralized_ego``
    variant (teammates move by policy, radio by heuristic); returns None when no
    checkpoint exists (random movement, heuristic radio)."""
    n_agents = env.config.n_agents
    spatial_shape = env.map_spatial_shape           # (C, ego, ego)
    internal_dim = (n_agents, env.agent_internal_dim)
    n_radio_actions = n_agents

    for use_radio in (True, False):
        state = _load_policy_state(level, alg, run_number, use_radio)
        if state is None:
            continue

        if alg == "dqn":
            from RL.cleanrl_dqn import QNetwork
            net = QNetwork(spatial_shape, internal_dim, n_agents, centralized=False,
                           use_radio=use_radio, n_radio_actions=n_radio_actions)
            if use_radio:
                def act_fn(spatial, internal):
                    move, radio = net.get_action_radio(spatial, internal)
                    return move[0].cpu().numpy(), radio[0].cpu().numpy()
            else:
                def act_fn(spatial, internal):
                    return net.get_action(spatial, internal)[0].cpu().numpy(), None
        elif alg == "ppo":
            from RL.cleanrl_ppo import Agent
            net = Agent(spatial_shape, internal_dim, n_agents, centralized=False,
                        use_radio=use_radio, n_radio_actions=n_radio_actions)
            if use_radio:
                def act_fn(spatial, internal):
                    out = net.get_action_and_value(spatial, internal)
                    move, radio = out[0], out[4]
                    return move[0].cpu().numpy(), radio[0].cpu().numpy()
            else:
                def act_fn(spatial, internal):
                    action = net.get_action_and_value(spatial, internal)[0]
                    return action[0].cpu().numpy(), None
        elif alg == "sac":
            from RL.cleanrl_sac import Actor
            net = Actor(spatial_shape, internal_dim, n_agents, centralized=False,
                        use_radio=use_radio, n_radio_actions=n_radio_actions)
            if use_radio:
                def act_fn(spatial, internal):
                    out = net.get_action_radio(spatial, internal)
                    move, radio = out[0], out[3]
                    return move[0].cpu().numpy(), radio[0].cpu().numpy()
            else:
                def act_fn(spatial, internal):
                    action = net.get_action(spatial, internal)[0]
                    return action[0].cpu().numpy(), None
        else:
            raise ValueError(f"Unknown teammate algorithm: {alg}")

        try:
            net.load_state_dict(strip_compile_prefix(state))
        except (RuntimeError, KeyError) as exc:
            # Checkpoint architecture no longer matches (e.g. recorded before the
            # internal vector gained the relative-entity block). Skip it rather
            # than crash; teammates fall back to random movement + heuristic radio.
            print(f"[teammates] Incompatible {alg} checkpoint (retrain needed): {exc}. "
                  f"Falling back to random movement + heuristic radio.")
            return None
        net.to(device).eval()
        variant = variant_name(centralized=False, ego_view=True, use_radio=use_radio)
        print(f"[teammates] Loaded {alg} {variant} policy for '{level_name_of(level)}' "
              f"(radio={'policy' if use_radio else 'heuristic'}).")
        return TeammatePolicy(net, act_fn, has_radio=use_radio)

    print(f"[teammates] No decentralized_ego{{,_radio}} {alg} checkpoint for "
          f"'{level_name_of(level)}' run {run_number}; random movement + heuristic radio.")
    return None


def run_episode(env, controlled_agent, agent_type, teammate_policy, poi_channel,
                max_steps=0, record=False, step_delay_ms=100, progress_line=""):
    """Play one full episode, collecting the controlled agent's ego transitions.

    ``max_steps <= 0`` caps the episode at the controlled agent's battery life
    (its max battery), i.e. the number of steps it can actually act for; pass a
    positive value to force a fixed cap instead.

    Returns (data, frames, n_frames, episodic_return, quit_requested) where
    ``episodic_return`` is the per-agent summed reward over the episode (shape
    (n_agents,)), used to report team performance.
    """
    import torch

    env.reset()
    obs = env._get_obs_dict()
    n_agents = env.config.n_agents

    # Cap the episode at the controlled agent's battery life unless the caller
    # forced a fixed max_steps: an agent can't act past a dead battery, so this
    # matches recorded episode length to how long it can actually operate.
    if max_steps <= 0:
        max_steps = int(env.config.agent_max_batteries[controlled_agent])

    data = {f: [] for f in HUMAN_FIELDS}
    frames = []
    quit_requested = False
    episodic_return = np.zeros(n_agents, dtype=np.float64)

    # AFK gate: don't start recording until the human signals they're present.
    # Recording steps on a timer once running, so without this an unattended
    # recorder (e.g. an auto-started record phase, or a synced-back checkpoint
    # kicking off the next iteration) would log a whole episode of no-op frames.
    # Blocks at ~0% CPU until a keypress; closing the window still stops cleanly.
    ready_lines = [
        f"Ready to record: {agent_type} (agent {controlled_agent})",
        "Press any key to START this episode  |  close window to stop",
    ]
    if progress_line:
        ready_lines.append(progress_line)
    env.render_ego(controlled_agent, env_idx=0, info_lines=ready_lines)
    while True:
        event = pygame.event.wait(200)          # blocks; wakes to keep window live
        if event.type == pygame.QUIT:
            quit_requested = True
            break
        if event.type == pygame.KEYDOWN:
            break
        if event.type == pygame.NOEVENT:
            env.render_ego(controlled_agent, env_idx=0, info_lines=ready_lines)
    if quit_requested:
        return data, frames, 0, episodic_return, quit_requested

    step = 0
    done = False
    while not done and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                break
        if quit_requested:
            break

        raw_move = _keyboard_move()
        move_idx = _discretize_move(raw_move)

        ego_spatial = obs["spatial"][0, controlled_agent]      # (C, S, S) uint8
        ego_internal = obs["internal"][0, controlled_agent]    # (D,)

        # Human radio: broadcast the instant a survivor is in the ego view, else
        # a random peer message with probability RANDOM_MESSAGE_PROB per step.
        radio = _heuristic_radio(ego_spatial, step, n_agents, poi_channel)

        move_actions = np.zeros((1, n_agents, 2), dtype=np.float32)
        radio_actions = np.zeros((1, n_agents), dtype=np.int32)
        move_actions[0, controlled_agent] = ACTION_MAP[move_idx]
        radio_actions[0, controlled_agent] = radio

        # Teammates choose both movement and radio. Movement comes from the
        # policy (or random); radio comes from the policy when it has a radio
        # head, else the same heuristic on that teammate's own ego crop -- so
        # teammates actually transmit and their info reaches the human's view.
        team_move = team_radio = None
        if teammate_policy is not None:
            with torch.no_grad():
                team_move, team_radio = teammate_policy.act(obs["spatial"], obs["internal"])
        for a in range(n_agents):
            if a == controlled_agent:
                continue
            if team_move is not None:
                move_actions[0, a] = ACTION_MAP[int(team_move[a])]
            else:
                move_actions[0, a] = np.random.uniform(-1.0, 1.0, size=(2,))
            if team_radio is not None:
                radio_actions[0, a] = int(team_radio[a])
            else:
                radio_actions[0, a] = _heuristic_radio(
                    obs["spatial"][0, a], step, n_agents, poi_channel
                )

        next_obs, reward, terminated, truncated, _ = env.step(move_actions, radio_actions)
        done = bool(terminated[0].item() or truncated[0].item())
        episodic_return += reward[0].cpu().numpy()

        data["obs_spatial"].append(ego_spatial.cpu().numpy())
        data["obs_internal"].append(ego_internal.cpu().numpy())
        data["actions_move"].append(np.int64(move_idx))
        data["actions_raw"].append(
            np.asarray([raw_move[0], raw_move[1], float(radio)], dtype=np.float32)
        )
        data["radio"].append(np.int64(radio))
        data["rewards"].append(np.float32(reward[0, controlled_agent].item()))
        data["dones"].append(np.float32(1.0 if done else 0.0))
        data["next_obs_spatial"].append(next_obs["spatial"][0, controlled_agent].cpu().numpy())
        data["next_obs_internal"].append(next_obs["internal"][0, controlled_agent].cpu().numpy())
        data["controlled_agent"].append(np.int64(controlled_agent))

        obs = next_obs

        # Show exactly what the controlled agent perceives: its ego crop with
        # discovered tiles, view-range ring, radio-shared teammate/POI info, and
        # a text panel of the vectorized obs (position, battery, known coords).
        info_lines = [
            f"controlling: {agent_type} (agent {controlled_agent})",
            f"episode step {step + 1}  team return={episodic_return.sum():.2f}",
        ]
        if progress_line:
            info_lines.append(progress_line)
        env.render_ego(controlled_agent, env_idx=0, info_lines=info_lines)
        if record and imageio is not None:
            frame_data = pygame.surfarray.array3d(pygame.display.get_surface())
            frames.append(frame_data.swapaxes(0, 1))

        pygame.time.wait(step_delay_ms)
        step += 1

    return data, frames, len(data["obs_spatial"]), episodic_return, quit_requested


def _stack_bucket(field_lists: dict) -> dict:
    out = {}
    for name, values in field_lists.items():
        if not values:
            continue
        out[name] = np.stack(values, axis=0) if np.ndim(values[0]) > 0 else np.asarray(values)
    return out


def collect_for_level(level, args, device) -> bool:
    """Record up to ``args.frames_per_agent`` frames for EVERY agent type of one
    level (env), then save + log. Prior on-disk demos for the current checkpoint
    count toward each quota, so this tops up only what's missing. Returns True if
    the user requested quit (closed the window)."""
    ego_view = not args.no_ego
    ego_size = args.ego_size if ego_view else None

    env = SARBatchedGridEnv(
        num_envs=1,
        map_png=os.path.join(level, "level.png"),
        tiles_json=os.path.join(level, "tiles.json"),
        agents_json=os.path.join(level, "agents.json"),
        survivors_json=os.path.join(level, "survivors.json"),
        mode="decentralized",
        requires_state=True,          # needed for the global (pov=-1) render
        device=device,
        ego_view=ego_view,
        ego_size=args.ego_size,
    )

    agent_names = _load_agent_names(os.path.join(level, "agents.json"))
    poi_channel = env.config.n_tiles + 1

    teammate_policy = None
    if not args.no_teammate_policy:
        teammate_policy = _build_teammate_policy(env, level, args.teammate_alg, args.teammate_run, device)

    # Demos are tagged with (and counted against) the current teammate checkpoint,
    # so re-running the recorder tops up only the agents that still lack data for
    # THIS checkpoint -- once an agent hits its quota it drops out of the pool.
    checkpoint_id = ("none" if args.no_teammate_policy
                     else _current_checkpoint_id(level, args.teammate_alg, args.teammate_run))

    target = args.frames_per_agent
    # One bucket per agent type (== agent name). Seed each bucket's count with the
    # frames already on disk for the current checkpoint so prior sessions count.
    collected = {
        name: count_human_frames(level, name, ego_size=ego_size, checkpoint=checkpoint_id)
        for name in agent_names
    }
    existing = count_human_frames(level, ego_size=ego_size)
    print(f"Level '{level_name_of(level)}' ({env.config.width}x{env.config.height}), "
          f"agents={agent_names}, ego={'off' if args.no_ego else args.ego_size}. "
          f"Recorded (matching): {existing} frames; for current checkpoint "
          f"'{checkpoint_id}': {collected}.")
    print("Controls: WASD move, close window to stop. Radio is automatic.")

    buckets = defaultdict(lambda: {f: [] for f in HUMAN_FIELDS})
    gif_buckets = defaultdict(list)
    episode_returns = []            # per-episode per-agent summed reward, (n_agents,)
    total = 0
    quit_requested = False

    def _needs_more():
        return [name for name in agent_names if collected[name] < target]

    while _needs_more() and not quit_requested:
        # Control the agent whose type is furthest from its quota (ties random),
        # so every bucket fills instead of only the ones randomly picked.
        deficit = max(target - collected[name] for name in _needs_more())
        candidates = [a for a in range(env.config.n_agents)
                      if target - collected[agent_names[a]] == deficit]
        controlled_agent = random.choice(candidates)
        agent_type = agent_names[controlled_agent]

        progress = "  ".join(f"{name}:{collected[name]}/{target}" for name in agent_names)
        progress_line = f"'{level_name_of(level)}' frames -> {progress}"
        print(f"Episode: controlling agent_{controlled_agent} ({agent_type}); {progress}")

        data, frames, n, ep_return, quit_requested = run_episode(
            env,
            controlled_agent=controlled_agent,
            agent_type=agent_type,
            teammate_policy=teammate_policy,
            poi_channel=poi_channel,
            max_steps=args.max_steps,
            record=args.record,
            step_delay_ms=args.step_delay_ms,
            progress_line=progress_line,
        )
        for f in HUMAN_FIELDS:
            buckets[agent_type][f].extend(data[f])
        if args.record:
            gif_buckets[agent_type].extend(frames)
        collected[agent_type] += n
        total += n
        if n > 0:
            episode_returns.append(ep_return)

    # --- Team episodic-return summary (tracks performance as the RL improves) ---
    if episode_returns:
        returns_arr = np.stack(episode_returns, axis=0)          # (n_episodes, n_agents)
        per_agent_mean = returns_arr.mean(axis=0)
        per_agent_std = returns_arr.std(axis=0)
        team = returns_arr.sum(axis=1)                            # (n_episodes,)
        print(f"\nEpisodic return over {len(episode_returns)} episodes "
              f"(teammates: {args.teammate_alg} run {args.teammate_run}):")
        for a, name in enumerate(agent_names):
            print(f"  agent {a} ({name}): mean={per_agent_mean[a]:.3f} std={per_agent_std[a]:.3f}")
        print(f"  TEAM (sum over agents): mean={team.mean():.3f} std={team.std():.3f}")
        stats = {
            "level": level_name_of(level),
            "n_episodes": len(episode_returns),
            "teammate_alg": args.teammate_alg,
            "teammate_run": args.teammate_run,
            "checkpoint": checkpoint_id,
            "agent_names": agent_names,
            "per_agent_return_mean": [float(x) for x in per_agent_mean],
            "per_agent_return_std": [float(x) for x in per_agent_std],
            "team_return_mean": float(team.mean()),
            "team_return_std": float(team.std()),
            "frames_this_session": total,
        }
        log_path = append_return_stats(level, stats)
        print(f"  logged -> {log_path}")

    if total == 0:
        print(f"Level '{level_name_of(level)}': quota already met; nothing to record.")
        return quit_requested

    meta = {"ego_size": ego_size, "ego_view": ego_view, "level": level_name_of(level),
            "checkpoint": checkpoint_id}
    for agent_type, field_lists in buckets.items():
        stacked = _stack_bucket(field_lists)
        if not stacked:
            continue
        seg_dir = append_human_segment(level, agent_type, stacked, meta=meta)
        n = len(stacked["obs_spatial"])
        cumulative = count_human_frames(level, agent_type, ego_size=ego_size)
        print(f"Saved {n} frames -> {seg_dir} (bucket '{agent_type}' total: {cumulative})")
        if args.record and imageio is not None and gif_buckets[agent_type]:
            gif_path = os.path.join(seg_dir, "replay.gif")
            imageio.mimsave(gif_path, gif_buckets[agent_type], fps=10)
            print(f"  wrote {gif_path}")

    print(f"Level '{level_name_of(level)}' now has "
          f"{count_human_frames(level, ego_size=ego_size)} matching frames.")
    return quit_requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default=None,
                        help="Level dir path (e.g. levels/test_level). If omitted, "
                             "sweep ALL levels until each has the per-agent quota.")
    parser.add_argument("--frames-per-agent", type=int, default=2500,
                        help="Frames to collect PER agent type this session; the "
                             "session ends once every type reaches this quota.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max steps per episode. 0 (default) caps each episode "
                             "at the controlled agent's battery life; a positive "
                             "value forces a fixed cap.")
    parser.add_argument("--ego-size", type=int, default=32,
                        help="Ego window side length (match the trainer's --ego-size).")
    parser.add_argument("--no-ego", action="store_true",
                        help="Record the full-map per-agent FOV instead of an ego crop.")
    parser.add_argument("--record", action="store_true", help="Save a replay GIF per bucket.")
    parser.add_argument("--teammate-alg", default="ppo", choices=["ppo", "dqn", "sac"])
    parser.add_argument("--teammate-run", type=int, default=1)
    parser.add_argument("--no-teammate-policy", action="store_true")
    parser.add_argument("--step-delay-ms", type=int, default=132)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    import torch

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    available = sorted(
        os.path.join("levels", d) for d in os.listdir("levels")
        if os.path.isdir(os.path.join("levels", d))
        and os.path.exists(os.path.join("levels", d, "level.png"))
    )
    if not available:
        raise SystemExit("No levels found under levels/.")

    # A given --level records just that env; otherwise ensure EVERY level (env)
    # reaches the per-agent-type quota, so the whole dataset is covered before
    # training. Order is fixed (sorted) so the coverage sweep is reproducible.
    if args.level:
        if not os.path.isdir(args.level):
            raise SystemExit(f"Level dir not found: {args.level}. Available: {available}")
        levels = [args.level]
    else:
        levels = available
        print(f"No --level given: ensuring {args.frames_per_agent} frames per agent type "
              f"for all {len(levels)} levels.")

    device = "cpu"
    pygame.init()
    for level in levels:
        print(f"\n===== Level '{level_name_of(level)}' "
              f"({levels.index(level) + 1}/{len(levels)}) =====")
        quit_requested = collect_for_level(level, args, device)
        if quit_requested:
            print("Quit requested (window closed); stopping the coverage sweep.")
            break


if __name__ == "__main__":
    main()
