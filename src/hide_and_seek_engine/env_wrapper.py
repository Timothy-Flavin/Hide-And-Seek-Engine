import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from hide_and_seek_engine.sar_loader import (
    SARConfig,
    load_sar_config,
)
import importlib

# Import the C++ extension from the installed package location
try:
    cpp_engine = importlib.import_module("hide_and_seek_engine.cpp_engine")
except ImportError:
    cpp_engine = None

from PIL import Image


def build_terrain_tensor_from_png(
    map_png_path: str,
    config: SARConfig,
    num_envs: int,
    map_size: int = 32,
) -> np.ndarray:

    img = Image.open(map_png_path).convert("RGB")
    if img.size != (map_size, map_size):
        img = img.resize((map_size, map_size), Image.Resampling.NEAREST)
    rgb = np.asarray(img, dtype=np.int32)

    n_tiles = len(config.terrain_names)
    out = np.zeros((num_envs, n_tiles + 1, map_size, map_size), dtype=np.float32)

    palette = config.terrain_rgb
    altitudes = config.terrain_altitudes

    flat_rgb = rgb.reshape(-1, 3)
    tile_indices = np.zeros(flat_rgb.shape[0], dtype=np.int32)
    for idx, px in enumerate(flat_rgb):
        diffs = palette - px
        tile_indices[idx] = int(np.argmin(np.sum(diffs * diffs, axis=1)))

    tile_grid = tile_indices.reshape(map_size, map_size)
    altitude_grid = altitudes[tile_grid]

    for t in range(n_tiles):
        out[:, t, :, :] = (tile_grid == t).astype(np.float32)
    out[:, n_tiles, :, :] = altitude_grid.astype(np.float32)
    return out


class SARBatchedGridEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs,
        map_png,
        tiles_json,
        agents_json,
        survivors_json,
        mode="centralized",
        combine_agent_layers=False,
    ):
        self.config = load_sar_config(tiles_json, agents_json, survivors_json, map_png)
        self.num_envs = num_envs
        self.map_area = self.config.width * self.config.height

        # 1. Determine Tensor Dimensions based on Observation Mode
        if mode == "centralized":
            # Channels: tiles + 1(altitude) + 1(POI) + 1(Observed) + agents
            self.channels = self.config.n_tiles + 1 + 1 + 1 + self.config.n_agents
        else:
            # Decentralized channels logic based on DecentralizedPartialObsStrides
            self.channels = (
                self.config.n_tiles + 1 + 1 + 1 + 1 + (self.config.n_agents - 1)
            )

        self.stride = self.channels * self.map_area
        self.tensor_size = self.num_envs * self.stride

        # 2. Allocate pinned PyTorch tensor
        self.state_tensor = torch.zeros(
            self.tensor_size, dtype=torch.float32, pin_memory=True
        )

        # 3. Instantiate C++ Engine
        self.env = cpp_engine.BatchedEnvironment(
            self.num_envs,
            42,  # Seed
            self.config.width,
            self.config.height,
            self.config.supports_walking,
            self.config.supports_aquatic,
            self.config.supports_flying,
            self.config.is_blocking,
            self.config.type_map,
            self.config.altitude_map,
            self.config.agent_speed_map,
            self.config.saveable_map,
            self.config.initial_agent_pos,
            self.config.initial_poi_pos,
            self.state_tensor.data_ptr(),  # Pass raw memory address
            True,  # Cooperative rewards
            0.05,  # reward_new_tile
            2.0,  # reward_found
            20.0,  # reward_saved
            250,  # max_frames
        )

    def _build_offsets(self):
        f = self.flat_map_size
        c = self.terrain_channels
        n = self.n_agents
        p = self.n_pois
        off = 0
        self.sl_terrain_altitude = slice(off, off + c * f)
        off += c * f
        self.sl_global_unsaved_pois = slice(off, off + f)
        off += f
        self.sl_global_obs_mask = slice(off, off + f)
        off += f
        self.sl_global_agent_layers = slice(off, off + n * f)
        off += n * f
        self.sl_local_poi_layers = slice(off, off + n * f)
        off += n * f
        self.sl_local_agent_layers = slice(off, off + n * n * f)
        off += n * n * f
        self.sl_local_obs_mask = slice(off, off + n * f)
        off += n * f
        self.sl_agent_positions = slice(off, off + n * 2)
        off += n * 2
        self.sl_agent_deploy = slice(off, off + n)
        off += n
        self.sl_agent_stuck = slice(off, off + n)
        off += n
        self.sl_agent_view = slice(off, off + n)
        off += n
        self.sl_agent_battery = slice(off, off + n)
        off += n
        self.sl_poi_positions = slice(off, off + p * 2)
        off += p * 2
        self.sl_poi_found = slice(off, off + p)
        off += p
        self.sl_poi_saved = slice(off, off + p)

    def _extract_local_obs(self, state_row: np.ndarray) -> dict[str, np.ndarray]:
        terrain_altitude = state_row[self.sl_terrain_altitude].reshape(
            self.terrain_channels, self.map_size, self.map_size
        )
        local_poi = state_row[self.sl_local_poi_layers].reshape(
            self.n_agents, self.map_size, self.map_size
        )
        local_obs = state_row[self.sl_local_obs_mask].reshape(
            self.n_agents, self.map_size, self.map_size
        )
        local_agents = state_row[self.sl_local_agent_layers].reshape(
            self.n_agents, self.n_agents, self.map_size, self.map_size
        )

        spatial = np.zeros(
            (
                self.n_agents,
                self.terrain_channels + 1 + 1 + self.n_agents,
                self.map_size,
                self.map_size,
            ),
            dtype=np.float32,
        )
        spatial[:, : self.terrain_channels] = terrain_altitude[None, :, :, :]
        spatial[:, self.terrain_channels] = local_poi
        spatial[:, self.terrain_channels + 1] = local_obs
        spatial[:, self.terrain_channels + 2 :] = local_agents

        pos = state_row[self.sl_agent_positions].reshape(self.n_agents, 2)
        deploy = state_row[self.sl_agent_deploy]
        stuck = state_row[self.sl_agent_stuck]
        view = state_row[self.sl_agent_view]
        battery = state_row[self.sl_agent_battery]
        internal = np.stack(
            [deploy, stuck, view, battery, pos[:, 0], pos[:, 1]], axis=-1
        ).astype(np.float32)
        return {"spatial": spatial, "internal": internal}

    def _extract_global_state(self, state_row: np.ndarray) -> dict[str, np.ndarray]:
        terrain_altitude = state_row[self.sl_terrain_altitude].reshape(
            self.terrain_channels, self.map_size, self.map_size
        )
        unsaved = state_row[self.sl_global_unsaved_pois].reshape(
            self.map_size, self.map_size
        )
        obs = state_row[self.sl_global_obs_mask].reshape(self.map_size, self.map_size)
        global_agents = state_row[self.sl_global_agent_layers].reshape(
            self.n_agents, self.map_size, self.map_size
        )
        spatial = np.concatenate(
            [terrain_altitude, unsaved[None, :, :], obs[None, :, :], global_agents],
            axis=0,
        ).astype(np.float32)

        pos = state_row[self.sl_agent_positions].reshape(self.n_agents, 2)
        deploy = state_row[self.sl_agent_deploy]
        stuck = state_row[self.sl_agent_stuck]
        view = state_row[self.sl_agent_view]
        battery = state_row[self.sl_agent_battery]
        poi_pos = (
            state_row[self.sl_poi_positions].reshape(self.n_pois, 2)
            if self.n_pois
            else np.zeros((0, 2), dtype=np.float32)
        )
        poi_found = (
            state_row[self.sl_poi_found]
            if self.n_pois
            else np.zeros((0,), dtype=np.float32)
        )
        poi_saved = (
            state_row[self.sl_poi_saved]
            if self.n_pois
            else np.zeros((0,), dtype=np.float32)
        )

        agent_internal = np.stack(
            [deploy, stuck, view, battery, pos[:, 0], pos[:, 1]], axis=-1
        ).astype(np.float32)
        if self.n_pois:
            poi_internal = np.concatenate(
                [poi_pos, poi_found[:, None], poi_saved[:, None]], axis=1
            ).astype(np.float32)
            internal = np.concatenate(
                [agent_internal.reshape(-1), poi_internal.reshape(-1)], axis=0
            )
        else:
            internal = agent_internal.reshape(-1)
        return {"spatial": spatial, "internal": internal.astype(np.float32)}

    def _stack_actor_obs(self):
        spatial = []
        internal = []
        for e in range(self.num_envs):
            obs = self._extract_local_obs(self.state_tensor[e].cpu().numpy())
            spatial.append(obs["spatial"])
            internal.append(obs["internal"])
        return {
            "spatial": torch.from_numpy(np.stack(spatial, axis=0)),
            "internal": torch.from_numpy(np.stack(internal, axis=0)),
        }

    def reset(self, seed=None, options=None):
        self.env.reset()
        self._last_known_agent.fill(-1)
        self._known_survivor.fill(False)
        self._last_known_survivor.fill(-1)
        obs = self._stack_actor_obs()
        if self.device != "cpu":
            obs = {k: v.to(self.device) for k, v in obs.items()}
        return obs, {}

    def reset_env(self, env_idx: int):
        self.env.reset_single(int(env_idx))
        self._last_known_agent[env_idx].fill(-1)
        self._known_survivor[env_idx].fill(False)
        self._last_known_survivor[env_idx].fill(-1)
        obs = self._extract_local_obs(self.state_tensor[env_idx].cpu().numpy())
        if self.device != "cpu":
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
        return obs

    def _normalize_actions(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        if isinstance(actions, dict):
            move = np.asarray(actions["move"], dtype=np.float32)
            radio = np.asarray(actions["radio"], dtype=np.float32)
            radio = radio[..., None]
            out = np.concatenate([move, radio], axis=-1)
        else:
            out = np.asarray(actions, dtype=np.float32)

        if out.shape != (self.num_envs, self.n_agents, 3):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.n_agents, 3)} but got {out.shape}"
            )

        out[:, :, :2] = np.clip(out[:, :, :2], -1.0, 1.0)
        out[:, :, 2] = np.clip(np.round(out[:, :, 2]), 0, 3)
        return out

    def step(self, actions):
        hybrid = self._normalize_actions(actions)
        rewards_np, terminated_np = self.env.step(hybrid)
        obs = self._stack_actor_obs()
        rewards = torch.from_numpy(rewards_np)
        terminated = torch.from_numpy(terminated_np)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.device != "cpu":
            obs = {k: v.to(self.device) for k, v in obs.items()}
            rewards = rewards.to(self.device)
            terminated = terminated.to(self.device)
            truncated = truncated.to(self.device)
        return obs, rewards, terminated, truncated, {}

    def state(self):
        global_states = [
            self._extract_global_state(self.state_tensor[e].cpu().numpy())
            for e in range(self.num_envs)
        ]
        spatial = torch.from_numpy(
            np.stack([g["spatial"] for g in global_states], axis=0)
        )
        internal = torch.from_numpy(
            np.stack([g["internal"] for g in global_states], axis=0)
        )
        if self.device != "cpu":
            spatial = spatial.to(self.device)
            internal = internal.to(self.device)
        return {"spatial": spatial, "internal": internal}

    def get_action_mask(self):
        return self.env.get_action_mask()

    def radio_render(self):
        self.env.radio_render()

    def _ensure_pygame(self):
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for rendering. Install with `pip install pygame`."
            )
        if self._pygame_screen is None:
            pygame.init()
            side = self.map_size * self._pygame_tile_px
            self._pygame_screen = pygame.display.set_mode((side, side))

    def _draw_grid(self, surface, base_colors: np.ndarray):
        tile = self._pygame_tile_px
        for y in range(self.map_size):
            for x in range(self.map_size):
                pygame.draw.rect(
                    surface,
                    tuple(int(c) for c in base_colors[y, x]),
                    pygame.Rect(x * tile, y * tile, tile, tile),
                )

    def _state_row(self, env_idx: int = 0) -> np.ndarray:
        return self.state_tensor[env_idx].cpu().numpy()

    def render(self, env_idx: int = 0):
        self._ensure_pygame()
        row = self._state_row(env_idx)

        global_obs = row[self.sl_global_obs_mask].reshape(self.map_size, self.map_size)
        agent_pos = row[self.sl_agent_positions].reshape(self.n_agents, 2)
        poi_pos = (
            row[self.sl_poi_positions].reshape(self.n_pois, 2)
            if self.n_pois
            else np.zeros((0, 2), dtype=np.float32)
        )
        poi_saved = (
            row[self.sl_poi_saved] if self.n_pois else np.zeros((0,), dtype=np.float32)
        )

        base_colors = self.config.terrain_rgb[self._tile_id_grid].astype(np.int32)
        dimmed = (base_colors // 2).astype(np.int32)
        mask = global_obs > 0.5
        final = dimmed.copy()
        final[mask] = base_colors[mask]

        self._draw_grid(self._pygame_screen, final)

        tile = self._pygame_tile_px
        for p in range(self.n_pois):
            py = int(np.clip(poi_pos[p, 0], 0, self.map_size - 1))
            px = int(np.clip(poi_pos[p, 1], 0, self.map_size - 1))
            color = (
                (255, 255, 255)
                if poi_saved[p] > 0.5
                else tuple(int(c) for c in self.config.survivor_rgb[p])
            )
            pygame.draw.circle(
                self._pygame_screen,
                color,
                (px * tile + tile // 2, py * tile + tile // 2),
                max(2, tile // 3),
            )

        for a in range(self.n_agents):
            ay = int(np.clip(agent_pos[a, 0], 0, self.map_size - 1))
            ax = int(np.clip(agent_pos[a, 1], 0, self.map_size - 1))
            color = tuple(int(c) for c in self.config.agent_rgb[a])
            pygame.draw.rect(
                self._pygame_screen,
                color,
                pygame.Rect(ax * tile + 2, ay * tile + 2, tile - 4, tile - 4),
            )

        pygame.display.flip()
        return pygame.surfarray.array3d(self._pygame_screen).transpose(1, 0, 2)

    def render_pov(self, agent_idx: int, env_idx: int = 0):
        self._ensure_pygame()
        if agent_idx < 0 or agent_idx >= self.n_agents:
            raise ValueError(f"agent_idx must be in [0, {self.n_agents - 1}]")

        row = self._state_row(env_idx)
        local_obs = row[self.sl_local_obs_mask].reshape(
            self.n_agents, self.map_size, self.map_size
        )[agent_idx]
        local_agents = row[self.sl_local_agent_layers].reshape(
            self.n_agents, self.n_agents, self.map_size, self.map_size
        )[agent_idx]
        local_poi = row[self.sl_local_poi_layers].reshape(
            self.n_agents, self.map_size, self.map_size
        )[agent_idx]

        poi_pos = (
            row[self.sl_poi_positions].reshape(self.n_pois, 2)
            if self.n_pois
            else np.zeros((0, 2), dtype=np.float32)
        )
        poi_saved = (
            row[self.sl_poi_saved] if self.n_pois else np.zeros((0,), dtype=np.float32)
        )

        base_colors = self.config.terrain_rgb[self._tile_id_grid].astype(np.int32)
        dimmed = (base_colors // 2).astype(np.int32)
        known_mask = local_obs > 0.5
        final = dimmed.copy()
        final[known_mask] = base_colors[known_mask]
        self._draw_grid(self._pygame_screen, final)

        tile = self._pygame_tile_px

        agent_pos = row[self.sl_agent_positions].reshape(self.n_agents, 2)

        for other in range(self.n_agents):
            if other == agent_idx:
                py = int(np.clip(agent_pos[other, 0], 0, self.map_size - 1))
                px = int(np.clip(agent_pos[other, 1], 0, self.map_size - 1))
                self._last_known_agent[env_idx, agent_idx, other] = np.asarray(
                    [py, px], dtype=np.int32
                )
            else:
                channel = local_agents[other]
                ys, xs = np.where(channel > 0.5)
                if len(ys) > 0:
                    oy = int(ys[0])
                    ox = int(xs[0])
                    self._last_known_agent[env_idx, agent_idx, other] = np.asarray(
                        [oy, ox], dtype=np.int32
                    )

            ky, kx = self._last_known_agent[env_idx, agent_idx, other]
            if ky >= 0 and kx >= 0:
                color = tuple(int(c) for c in self.config.agent_rgb[other])
                pygame.draw.rect(
                    self._pygame_screen,
                    color,
                    pygame.Rect(
                        int(kx) * tile + 2, int(ky) * tile + 2, tile - 4, tile - 4
                    ),
                )

        for p in range(self.n_pois):
            py = int(np.clip(poi_pos[p, 0], 0, self.map_size - 1))
            px = int(np.clip(poi_pos[p, 1], 0, self.map_size - 1))
            if local_poi[py, px] > 0.5:
                self._known_survivor[env_idx, agent_idx, p] = True
                self._last_known_survivor[env_idx, agent_idx, p] = np.asarray(
                    [py, px], dtype=np.int32
                )

            if not self._known_survivor[env_idx, agent_idx, p]:
                continue

            ky, kx = self._last_known_survivor[env_idx, agent_idx, p]
            if ky < 0 or kx < 0:
                continue
            color = (
                (255, 255, 255)
                if poi_saved[p] > 0.5
                else tuple(int(c) for c in self.config.survivor_rgb[p])
            )
            pygame.draw.circle(
                self._pygame_screen,
                color,
                (int(kx) * tile + tile // 2, int(ky) * tile + tile // 2),
                max(2, tile // 3),
            )

        pygame.display.flip()
        return pygame.surfarray.array3d(self._pygame_screen).transpose(1, 0, 2)

    def close(self):
        if self._pygame_screen is not None and PYGAME_AVAILABLE:
            pygame.display.quit()
            pygame.quit()
            self._pygame_screen = None


class SARParallelPettingZooEnv(ParallelEnv):
    metadata = {"name": "sar_parallel_v0", "render_modes": ["human"]}

    def __init__(self, **kwargs):
        if not PETTINGZOO_AVAILABLE:
            raise ImportError(
                "PettingZoo is required. Install with `pip install pettingzoo`."
            )
        self._batched = SARBatchedGridEnv(num_envs=1, **kwargs)
        self.possible_agents = [f"agent_{i}" for i in range(self._batched.n_agents)]
        self.agents = self.possible_agents[:]

    def observation_space(self, agent):
        return spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self._batched.single_observation_space["spatial"].shape[1:],
                    dtype=np.float32,
                ),
                "internal": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._batched.actor_internal_dim,),
                ),
            }
        )

    def action_space(self, agent):
        return self._batched.single_action_space

    @property
    def state_space(self):
        return self._batched.single_state_space

    def state(self):
        st = self._batched.state()
        return {
            "spatial": st["spatial"][0].cpu().numpy(),
            "internal": st["internal"][0].cpu().numpy(),
        }

    def reset(self, seed=None, options=None):
        obs, _ = self._batched.reset(seed=seed, options=options)
        spatial = obs["spatial"][0].cpu().numpy()
        internal = obs["internal"][0].cpu().numpy()
        self.agents = self.possible_agents[:]
        observations = {
            agent: {"spatial": spatial[i], "internal": internal[i]}
            for i, agent in enumerate(self.agents)
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        action_tensor = np.zeros((1, self._batched.n_agents, 3), dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            a = actions.get(agent, None)
            if a is None:
                continue
            if isinstance(a, dict):
                action_tensor[0, i, :2] = np.asarray(a["move"], dtype=np.float32)
                action_tensor[0, i, 2] = float(a["radio"])
            else:
                action_tensor[0, i] = np.asarray(a, dtype=np.float32)

        obs, rewards, terminated, truncated, _ = self._batched.step(action_tensor)
        done = bool(terminated[0].item())

        spatial = obs["spatial"][0].cpu().numpy()
        internal = obs["internal"][0].cpu().numpy()

        observations = {
            agent: {"spatial": spatial[i], "internal": internal[i]}
            for i, agent in enumerate(self.possible_agents)
        }
        rewards_out = {
            agent: float(rewards[0, i].item())
            for i, agent in enumerate(self.possible_agents)
        }
        terminations = {agent: done for agent in self.possible_agents}
        truncations = {
            agent: bool(truncated[0].item()) for agent in self.possible_agents
        }
        infos = {
            agent: {
                "action_mask": self._batched.get_action_mask()[0, i].astype(np.int8)
            }
            for i, agent in enumerate(self.possible_agents)
        }

        if done:
            self.agents = []

        return observations, rewards_out, terminations, truncations, infos

    def render(self):
        return self._batched.render(env_idx=0)

    def close(self):
        self._batched.close()


BatchedGridEnv = SARBatchedGridEnv
