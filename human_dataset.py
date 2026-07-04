"""Human demonstration recorder for the ego-centric SAR environment.

Collects human demos to behavior-clone (BC) pre-train the decentralized_ego
agents, then supports the iterative record -> train -> record loop. One run:

* loads a random level (unless ``--level``) and, each episode, hands the player a
  random agent to control from its ego POV;
* auto-radios for the controlled agent via a heuristic: the instant a survivor
  enters its ego view it broadcasts, and every 20 steps it also sends a random
  message to another agent;
* drives the non-controlled teammates with the latest decentralized_ego policy
  checkpoint for the level (random fallback). Teammates also transmit radio:
  from the policy's radio head when a decentralized_ego_radio checkpoint is
  loaded, else the same heuristic -- so their shared location/tiles/POIs reach
  the controlled agent's ego view;
* records 2500 frames (``--frames-per-run``) and *appends* them under
  ``experiments/results/<level>/<agent_type>/`` as immutable segments -- run it
  4x to reach the 10k pre-training target.

The recorded ego observation is the uint8 crop the network consumes, so the same
files feed the trainers' ``--human-bc`` term directly. The player sees exactly
that ego crop -- the agent centered in its view-range ring, discovered tiles,
and teammate/POI info shared over the radio -- i.e. what an artificial agent sees.

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
from RL.human_data import HUMAN_FIELDS, append_human_segment, count_human_frames, level_name_of

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
RANDOM_MESSAGE_PERIOD = 20
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

    Broadcasts the instant a survivor is in the agent's ego view; otherwise
    sends a random peer message every RANDOM_MESSAGE_PERIOD steps. Radio value
    0 == silent, 1..n_agents-1 == share with a peer (decoded engine-side). Used
    for the human and for teammates when the loaded policy has no radio head.
    """
    if n_agents <= 1:
        return 0
    survivor_spotted = bool((ego_spatial[poi_channel] > 0).any().item())
    if survivor_spotted:
        return SPOTTED_CHANNEL
    if step > 0 and step % RANDOM_MESSAGE_PERIOD == 0:
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

        net.load_state_dict(strip_compile_prefix(state))
        net.to(device).eval()
        variant = variant_name(centralized=False, ego_view=True, use_radio=use_radio)
        print(f"[teammates] Loaded {alg} {variant} policy for '{level_name_of(level)}' "
              f"(radio={'policy' if use_radio else 'heuristic'}).")
        return TeammatePolicy(net, act_fn, has_radio=use_radio)

    print(f"[teammates] No decentralized_ego{{,_radio}} {alg} checkpoint for "
          f"'{level_name_of(level)}' run {run_number}; random movement + heuristic radio.")
    return None


def run_episode(env, controlled_agent, frame_budget, teammate_policy, poi_channel,
                max_steps=250, record=False, step_delay_ms=132):
    """Play one episode, collecting up to ``frame_budget`` ego transitions."""
    import torch

    env.reset()
    obs = env._get_obs_dict()
    n_agents = env.config.n_agents

    data = {f: [] for f in HUMAN_FIELDS}
    frames = []
    quit_requested = False

    step = 0
    done = False
    while not done and step < max_steps and len(data["obs_spatial"]) < frame_budget:
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
        # a random peer message every RANDOM_MESSAGE_PERIOD steps.
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
        # discovered tiles, view-range ring, and radio-shared teammate/POI info.
        env.render_ego(controlled_agent, env_idx=0)
        if record and imageio is not None:
            frame_data = pygame.surfarray.array3d(pygame.display.get_surface())
            frames.append(frame_data.swapaxes(0, 1))

        pygame.time.wait(step_delay_ms)
        step += 1

    return data, frames, len(data["obs_spatial"]), quit_requested


def _stack_bucket(field_lists: dict) -> dict:
    out = {}
    for name, values in field_lists.items():
        if not values:
            continue
        out[name] = np.stack(values, axis=0) if np.ndim(values[0]) > 0 else np.asarray(values)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default=None,
                        help="Level dir path (e.g. levels/test_level). Random if omitted.")
    parser.add_argument("--frames-per-run", type=int, default=2500)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--ego-size", type=int, default=32,
                        help="Ego window side length (match the trainer's --ego-size).")
    parser.add_argument("--no-ego", action="store_true",
                        help="Record the full-map per-agent FOV instead of an ego crop.")
    parser.add_argument("--record", action="store_true", help="Save a replay GIF per bucket.")
    parser.add_argument("--teammate-alg", default="dqn", choices=["dqn", "ppo", "sac"])
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
    level = args.level or random.choice(available)
    if not os.path.isdir(level):
        raise SystemExit(f"Level dir not found: {level}. Available: {available}")

    ego_view = not args.no_ego
    ego_size = args.ego_size if ego_view else None

    device = "cpu"
    pygame.init()
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

    existing = count_human_frames(level, ego_size=ego_size)
    print(f"Level '{level_name_of(level)}' ({env.config.width}x{env.config.height}), "
          f"agents={agent_names}, ego={'off' if args.no_ego else args.ego_size}. "
          f"Already recorded (matching): {existing} frames.")
    print("Controls: WASD move, close window to stop. Radio is automatic.")

    buckets = defaultdict(lambda: {f: [] for f in HUMAN_FIELDS})
    gif_buckets = defaultdict(list)
    total = 0
    quit_requested = False

    while total < args.frames_per_run and not quit_requested:
        controlled_agent = random.randrange(env.config.n_agents)
        agent_type = agent_names[controlled_agent]
        remaining = args.frames_per_run - total
        print(f"Episode: controlling agent_{controlled_agent} ({agent_type}); "
              f"{total}/{args.frames_per_run} frames.")

        data, frames, n, quit_requested = run_episode(
            env,
            controlled_agent=controlled_agent,
            frame_budget=remaining,
            teammate_policy=teammate_policy,
            poi_channel=poi_channel,
            max_steps=args.max_steps,
            record=args.record,
            step_delay_ms=args.step_delay_ms,
        )
        for f in HUMAN_FIELDS:
            buckets[agent_type][f].extend(data[f])
        if args.record:
            gif_buckets[agent_type].extend(frames)
        total += n

    if total == 0:
        print("No frames recorded; nothing to save.")
        return

    meta = {"ego_size": ego_size, "ego_view": ego_view, "level": level_name_of(level)}
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


if __name__ == "__main__":
    main()
