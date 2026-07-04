"""Human demonstration recorder for the ego-centric SAR environment.

Collects human demos to behavior-clone (BC) pre-train the decentralized_ego
agents, then supports the iterative record -> train -> record loop. One run:

* loads a random level (unless ``--level``) and, each episode, hands the player a
  random agent to control from its ego POV;
* auto-radios teammates via a heuristic: the instant a survivor enters the
  controlled agent's ego view it broadcasts, and every 20 steps it also sends a
  random message to another agent;
* drives the non-controlled teammates with the latest decentralized_ego policy
  checkpoint for the level (random fallback);
* records 2500 frames (``--frames-per-run``) and *appends* them under
  ``experiments/results/<level>/<agent_type>/`` as immutable segments -- run it
  4x to reach the 10k pre-training target.

The recorded ego observation is the uint8 crop the network consumes, so the same
files feed the trainers' ``--human-bc`` term directly. The player sees the global
map (ego-crop rendering is unsupported); the *recorded* obs is still the ego crop.

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


def _load_agent_names(agents_json: str) -> list[str]:
    with open(agents_json) as f:
        return list(json.load(f).keys())


class TeammatePolicy:
    """Latest decentralized_ego policy for a level, used to drive teammates.

    Only decentralized ego checkpoints are usable (the runner feeds each agent
    its own ego crop). Falls back to random when none is found. Radio teammates
    are not modeled -- teammates stay silent.
    """

    def __init__(self, net, act_fn):
        self.net = net
        self.act_fn = act_fn

    def act(self, spatial, internal) -> np.ndarray:
        # spatial: (1, n_agents, C, S, S) uint8 ; internal: (1, n_agents, D)
        return self.act_fn(spatial, internal)[0].cpu().numpy()


def _load_policy_state(level, alg, run_number):
    """Return the decentralized_ego policy state_dict for (level, alg, run), or
    None. Prefers the rolling resume checkpoint, then the 100% weight snapshot."""
    import torch

    variant = variant_name(centralized=False, ego_view=True, use_radio=False)
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
    state = _load_policy_state(level, alg, run_number)
    if state is None:
        print(f"[teammates] No decentralized_ego {alg} checkpoint for '{level_name_of(level)}' "
              f"run {run_number}; using random teammate actions.")
        return None

    n_agents = env.config.n_agents
    spatial_shape = env.map_spatial_shape           # (C, ego, ego)
    internal_dim = (n_agents, env.agent_internal_dim)

    if alg == "dqn":
        from RL.cleanrl_dqn import QNetwork
        net = QNetwork(spatial_shape, internal_dim, n_agents, centralized=False)

        def act_fn(spatial, internal):
            return net.get_action(spatial, internal)
    elif alg == "ppo":
        from RL.cleanrl_ppo import Agent
        net = Agent(spatial_shape, internal_dim, n_agents, centralized=False)

        def act_fn(spatial, internal):
            action, _, _, _ = net.get_action_and_value(spatial, internal)
            return action
    elif alg == "sac":
        from RL.cleanrl_sac import Actor
        net = Actor(spatial_shape, internal_dim, n_agents, centralized=False)

        def act_fn(spatial, internal):
            action, _, _ = net.get_action(spatial, internal)
            return action
    else:
        raise ValueError(f"Unknown teammate algorithm: {alg}")

    net.load_state_dict(strip_compile_prefix(state))
    net.to(device).eval()
    print(f"[teammates] Loaded {alg} decentralized_ego policy for '{level_name_of(level)}'.")
    return TeammatePolicy(net, act_fn)


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

        # Heuristic radio: broadcast the instant a survivor is in the ego view,
        # else a random message to another agent every RANDOM_MESSAGE_PERIOD steps.
        survivor_spotted = bool((ego_spatial[poi_channel] > 0).any().item())
        if survivor_spotted:
            radio = SPOTTED_CHANNEL
        elif step > 0 and step % RANDOM_MESSAGE_PERIOD == 0 and n_agents > 1:
            radio = random.randrange(1, n_agents)  # 1..n_agents-1 == "share with a peer"
        else:
            radio = 0

        move_actions = np.zeros((1, n_agents, 2), dtype=np.float32)
        radio_actions = np.zeros((1, n_agents), dtype=np.int32)
        move_actions[0, controlled_agent] = ACTION_MAP[move_idx]
        radio_actions[0, controlled_agent] = radio

        if teammate_policy is not None:
            with torch.no_grad():
                team_actions = teammate_policy.act(obs["spatial"], obs["internal"])
            for a in range(n_agents):
                if a != controlled_agent:
                    move_actions[0, a] = ACTION_MAP[int(team_actions[a])]
        else:
            for a in range(n_agents):
                if a != controlled_agent:
                    move_actions[0, a] = np.random.uniform(-1.0, 1.0, size=(2,))

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

        env.render(-1, env_idx=0)  # global map (ego-crop rendering is unsupported)
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
