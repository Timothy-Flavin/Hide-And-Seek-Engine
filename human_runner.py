import os
import random
import time
import argparse  # Added for CLI args

import numpy as np
import imageio  # Added for GIF creation

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv

try:
    import pygame
except ImportError as exc:
    raise ImportError(
        "pygame is required for human_runner.py. Install with `pip install pygame`."
    ) from exc


def _keyboard_action() -> np.ndarray:
    keys = pygame.key.get_pressed()
    dy = 0.0
    dx = 0.0
    radio = 0.0

    if keys[pygame.K_w]:
        dy -= 1.0
    if keys[pygame.K_s]:
        dy += 1.0
    if keys[pygame.K_a]:
        dx -= 1.0
    if keys[pygame.K_d]:
        dx += 1.0

    if keys[pygame.K_1]:
        radio = 1.0
    elif keys[pygame.K_2]:
        radio = 2.0
    elif keys[pygame.K_3]:
        radio = 3.0

    return np.asarray([dy, dx, radio], dtype=np.float32)


def run_episode(
    env: SARBatchedGridEnv,
    controlled_agent: int,
    max_steps: int = 2000,
    record: bool = False,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    obs, _ = env.reset()
    done = False

    states_spatial, states_internal = [], []
    obs_spatial, obs_internal = [], []
    actions, rewards = [], []
    next_states_spatial, next_states_internal = [], []
    next_obs_spatial, next_obs_internal = [], []

    frames = []  # To store GIF frames

    step = 0
    while not done and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return {}, []

        action = np.zeros((1, env.config.n_agents, 3), dtype=np.float32)
        action[0, controlled_agent] = _keyboard_action()

        for a in range(env.config.n_agents):
            if a != controlled_agent:
                action[0, a, :2] = np.random.uniform(-1.0, 1.0, size=(2,)).astype(
                    np.float32
                )
                action[0, a, 2] = float(np.random.randint(0, 4))

        state = env.get_state()
        move_actions = action[..., :2]
        radio_actions = action[..., 2]
        next_obs, reward, terminated, truncated, _ = env.step(
            move_actions, radio_actions
        )
        next_state = env.get_state()

        # Data collection
        states_spatial.append(state["spatial"][0].cpu().numpy())
        states_internal.append(state["internal"][0].cpu().numpy())
        obs_spatial.append(obs["spatial"][0, controlled_agent].cpu().numpy())
        obs_internal.append(obs["internal"][0, controlled_agent].cpu().numpy())
        actions.append(action[0, controlled_agent].copy())
        rewards.append(
            np.asarray(reward[0, controlled_agent].cpu().numpy(), dtype=np.float32)
        )
        next_states_spatial.append(next_state["spatial"][0].cpu().numpy())
        next_states_internal.append(next_state["internal"][0].cpu().numpy())
        next_obs_spatial.append(next_obs["spatial"][0, controlled_agent].cpu().numpy())
        next_obs_internal.append(
            next_obs["internal"][0, controlled_agent].cpu().numpy()
        )

        obs = next_obs
        done = bool(terminated[0].item() or truncated[0].item())

        # Render
        env.render(controlled_agent, env_idx=0)
        # env.radio_render()

        # Capture frame for GIF if recording is enabled
        if record:
            # Grab the pixels from the active pygame display
            frame_data = pygame.surfarray.array3d(pygame.display.get_surface())
            # Pygame uses (width, height, rgb), imageio needs (height, width, rgb)
            frames.append(frame_data.swapaxes(0, 1))

        pygame.time.wait(132)
        step += 1

    data = {
        "states_spatial": np.stack(states_spatial, axis=0),
        "states_internal": np.stack(states_internal, axis=0),
        "obs_spatial": np.stack(obs_spatial, axis=0),
        "obs_internal": np.stack(obs_internal, axis=0),
        "actions": np.stack(actions, axis=0),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states_spatial": np.stack(next_states_spatial, axis=0),
        "next_states_internal": np.stack(next_states_internal, axis=0),
        "next_obs_spatial": np.stack(next_obs_spatial, axis=0),
        "next_obs_internal": np.stack(next_obs_internal, axis=0),
    }
    return data, frames


def save_episode(data: dict[str, np.ndarray], frames: list[np.ndarray], folder: str):
    os.makedirs(folder, exist_ok=True)
    for name, arr in data.items():
        np.save(os.path.join(folder, f"{name}.npy"), arr)

    if frames:
        gif_path = os.path.join(folder, "replay.gif")
        print(f"Encoding GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=10)  # Adjust FPS to match your 132ms wait


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", action="store_true", help="Record a GIF of the episode"
    )
    args = parser.parse_args()

    pygame.init()
    env = SARBatchedGridEnv(
        num_envs=1,
        map_png="levels/test_level/level.png",
        tiles_json="levels/test_level/tiles.json",
        agents_json="levels/test_level/agents.json",
        survivors_json="levels/test_level/survivors.json",
        requires_state=True,
    )

    print("Controls: WASD move, 1/2/3 radio channels, close window to exit")

    episode_idx = 0
    while True:
        controlled_agent = random.randrange(env.config.n_agents)
        print(
            f"Starting episode {episode_idx} with controlled agent_{controlled_agent}"
        )

        data, frames = run_episode(
            env, controlled_agent=controlled_agent, record=args.record
        )

        if not data:
            break

        user_name = input("Save episode name (blank to skip): ").strip()
        if user_name:
            out_dir = os.path.join("saved_human_behavior", user_name)
            save_episode(data, frames, out_dir)
            print(f"Saved data and GIF to {out_dir}")
        else:
            print("Skipped saving")

        episode_idx += 1


if __name__ == "__main__":
    main()
