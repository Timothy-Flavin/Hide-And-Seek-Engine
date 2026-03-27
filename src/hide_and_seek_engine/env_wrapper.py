import ctypes
import json
import os
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from . import _core as _core_partial

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pettingzoo import ParallelEnv

    PETTINGZOO_AVAILABLE = True
except ImportError:
    ParallelEnv = object
    PETTINGZOO_AVAILABLE = False


FeatureType = _core_partial.FeatureType


@dataclass
class SARConfig:
    terrain_names: list[str]
    terrain_rgb: np.ndarray
    terrain_altitudes: np.ndarray
    tile_supports_walking: np.ndarray
    tile_supports_aquatic: np.ndarray
    tile_supports_flying: np.ndarray
    tile_blocking: np.ndarray
    tile_name_to_id: dict[str, int]
    agent_names: list[str]
    agent_rgb: np.ndarray
    agent_class_to_id: dict[str, int]
    agent_specs: np.ndarray
    initial_agent_positions: np.ndarray
    survivor_names: list[str]
    survivor_rgb: np.ndarray
    poi_specs: np.ndarray
    initial_poi_positions: np.ndarray


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_named_list(payload: Any, key_hint: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        out = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            if "name" not in item:
                item = {**item, "name": f"{key_hint}_{i}"}
            out.append(item)
        return out

    if isinstance(payload, dict):
        for key in ("items", "agents", "survivors", "tiles", key_hint):
            val = payload.get(key)
            if isinstance(val, list):
                return _as_named_list(val, key_hint)

        out = []
        for name, item in payload.items():
            if not isinstance(item, dict):
                continue
            out.append({"name": str(name), **item})
        if out:
            return out

    raise ValueError(f"Could not find list payload for {key_hint}")


def _normalize_start_coord(
    raw: list[float] | tuple[float, float], map_size: int
) -> np.ndarray:
    pos = np.asarray(raw, dtype=np.float32)
    if pos.shape != (2,):
        return np.asarray([map_size // 2, map_size // 2], dtype=np.float32)
    if float(np.max(np.abs(pos))) <= 1.0:
        return np.clip(pos * (map_size - 1), 0.0, float(map_size - 1)).astype(
            np.float32
        )
    return np.clip(pos, 0.0, float(map_size - 1)).astype(np.float32)


def load_sar_config(
    tiles_json: str,
    agents_json: str,
    survivors_json: str,
    num_envs: int,
    map_size: int = 32,
) -> SARConfig:
    tiles_payload = _as_named_list(_load_json(tiles_json), "tiles")
    agents_payload = _as_named_list(_load_json(agents_json), "agents")
    survivors_payload = _as_named_list(_load_json(survivors_json), "survivors")

    terrain_names = [
        str(t.get("name", f"tile_{i}")) for i, t in enumerate(tiles_payload)
    ]
    terrain_rgb = np.asarray(
        [t.get("rgb", [0, 0, 0]) for t in tiles_payload], dtype=np.int32
    )
    terrain_altitudes = np.asarray(
        [float(t.get("altitude", 0.5)) for t in tiles_payload], dtype=np.float32
    )

    alt_min = float(np.min(terrain_altitudes)) if len(terrain_altitudes) else 0.0
    alt_max = float(np.max(terrain_altitudes)) if len(terrain_altitudes) else 1.0
    denom = alt_max - alt_min if abs(alt_max - alt_min) > 1e-8 else 1.0
    terrain_altitudes = (terrain_altitudes - alt_min) / denom

    tile_supports_walking = np.asarray(
        [int(bool(t.get("supports_walking", True))) for t in tiles_payload],
        dtype=np.int32,
    )
    tile_supports_aquatic = np.asarray(
        [int(bool(t.get("supports_aquatic", False))) for t in tiles_payload],
        dtype=np.int32,
    )
    tile_supports_flying = np.asarray(
        [int(bool(t.get("supports_flying", True))) for t in tiles_payload],
        dtype=np.int32,
    )
    tile_blocking = np.asarray(
        [int(bool(t.get("blocking", False))) for t in tiles_payload], dtype=np.int32
    )
    tile_name_to_id = {name: i for i, name in enumerate(terrain_names)}

    agent_names = [
        str(a.get("name", f"agent_{i}")) for i, a in enumerate(agents_payload)
    ]
    while len(agent_names) < 4:
        agent_names.append(f"agent_{len(agent_names)}")

    agent_class_names = sorted(
        {
            str(a.get("name", a.get("class", f"class_{i}")))
            for i, a in enumerate(agents_payload)
        }
    )
    while len(agent_class_names) < 4:
        agent_class_names.append(f"class_{len(agent_class_names)}")
    agent_class_to_id = {name: idx for idx, name in enumerate(agent_class_names)}

    n_tiles = len(tiles_payload)
    spec_width = 10 + n_tiles
    agent_specs = np.zeros((4, spec_width), dtype=np.float32)
    agent_rgb = np.zeros((4, 3), dtype=np.int32)
    initial_agent_positions = np.zeros((num_envs, 4, 2), dtype=np.float32)

    default_agent = {
        "flying": False,
        "aqueous": False,
        "walking": True,
        "altitude_min": 0.0,
        "altitude_max": 1.0,
        "base_speed": 1.0,
        "base_view": 3.0,
        "battery": 100.0,
        "deployment_delay": 0.0,
        "rgb": [255, 0, 0],
    }

    for i in range(min(4, len(agents_payload))):
        a = {**default_agent, **agents_payload[i]}
        terrain_multipliers = a.get("terrain_speed", {}) or {}

        class_name = str(a.get("name", a.get("class", f"class_{i}")))
        if class_name not in agent_class_to_id:
            agent_class_to_id[class_name] = len(agent_class_to_id)

        agent_specs[i, 0] = float(bool(a.get("flying", False)))
        agent_specs[i, 1] = float(bool(a.get("aqueous", False)))
        agent_specs[i, 2] = float(bool(a.get("walking", True)))
        agent_specs[i, 3] = float(a.get("altitude_min", 0.0))
        agent_specs[i, 4] = float(a.get("altitude_max", 1.0))
        agent_specs[i, 5] = float(a.get("base_speed", 1.0))
        agent_specs[i, 6] = float(a.get("base_view", 3.0))
        agent_specs[i, 7] = float(a.get("battery", 100.0))
        agent_specs[i, 8] = float(a.get("deployment_delay", 0.0))
        agent_specs[i, 9] = float(agent_class_to_id[class_name])

        for t_idx, t_name in enumerate(terrain_names):
            agent_specs[i, 10 + t_idx] = float(terrain_multipliers.get(t_name, 1.0))

        agent_rgb[i] = np.asarray(a.get("rgb", [255, 0, 0]), dtype=np.int32)

        pos = _normalize_start_coord(
            a.get("start", [map_size // 2, map_size // 2]), map_size
        )
        for e in range(num_envs):
            initial_agent_positions[e, i, :] = pos

    for i in range(len(agents_payload), 4):
        agent_specs[i, 2] = 1.0
        agent_specs[i, 4] = 1.0
        agent_specs[i, 5] = 1.0
        agent_specs[i, 6] = 3.0
        agent_specs[i, 7] = 100.0
        agent_specs[i, 9] = float(i)
        agent_specs[i, 10:] = 1.0
        agent_rgb[i] = np.asarray([255, 0, 0], dtype=np.int32)
        for e in range(num_envs):
            initial_agent_positions[e, i, :] = np.asarray(
                [map_size // 2, map_size // 2], dtype=np.float32
            )

    survivor_names = [
        str(p.get("name", f"survivor_{i}")) for i, p in enumerate(survivors_payload)
    ]
    survivor_rgb = np.asarray(
        [p.get("rgb", [255, 215, 0]) for p in survivors_payload], dtype=np.int32
    )

    poi_specs = np.zeros((len(survivors_payload), 2), dtype=np.float32)
    initial_poi_positions = np.zeros(
        (num_envs, len(survivors_payload), 2), dtype=np.float32
    )
    for p_idx, p in enumerate(survivors_payload):
        allowed = p.get("allowed_savers", [])
        mask = 0
        for c in allowed:
            cname = str(c)
            if cname in agent_class_to_id:
                mask |= 1 << agent_class_to_id[cname]

        poi_specs[p_idx, 0] = float(bool(p.get("moves", True)))
        poi_specs[p_idx, 1] = float(mask)

        pos = _normalize_start_coord(
            p.get("start", [map_size // 2, map_size // 2]), map_size
        )
        for e in range(num_envs):
            initial_poi_positions[e, p_idx, :] = pos

    return SARConfig(
        terrain_names=terrain_names,
        terrain_rgb=terrain_rgb,
        terrain_altitudes=terrain_altitudes,
        tile_supports_walking=tile_supports_walking,
        tile_supports_aquatic=tile_supports_aquatic,
        tile_supports_flying=tile_supports_flying,
        tile_blocking=tile_blocking,
        tile_name_to_id=tile_name_to_id,
        agent_names=agent_names,
        agent_rgb=agent_rgb,
        agent_class_to_id=agent_class_to_id,
        agent_specs=agent_specs,
        initial_agent_positions=initial_agent_positions,
        survivor_names=survivor_names,
        survivor_rgb=survivor_rgb,
        poi_specs=poi_specs,
        initial_poi_positions=initial_poi_positions,
    )


def build_terrain_tensor_from_png(
    map_png_path: str,
    config: SARConfig,
    num_envs: int,
    map_size: int = 32,
) -> np.ndarray:
    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for PNG map loading. Install with `pip install Pillow`."
        )

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
        num_envs: int,
        map_png: str,
        tiles_json: str,
        agents_json: str,
        survivors_json: str,
        map_size: int = 32,
        device: str = "cpu",
        seed: int = 42,
        cooperative_rewards: bool = True,
        reward_new_tile: float = 0.05,
        reward_found: float = 2.0,
        reward_saved: float = 20.0,
    ):
        self.num_envs = int(num_envs)
        self.map_size = int(map_size)
        self.device = device
        self.n_agents = 4
        self._map_png = map_png

        self.config = load_sar_config(
            tiles_json=tiles_json,
            agents_json=agents_json,
            survivors_json=survivors_json,
            num_envs=self.num_envs,
            map_size=self.map_size,
        )
        terrain_tensor = build_terrain_tensor_from_png(
            map_png_path=map_png,
            config=self.config,
            num_envs=self.num_envs,
            map_size=self.map_size,
        )

        self._tile_id_grid = np.argmax(terrain_tensor[0, :-1], axis=0).astype(np.int32)

        if _core_partial is None:
            raise ImportError(
                "Compiled extension `hide_and_seek_engine._core` is not available. "
                "Build and install the package (see README)."
            )

        self.env = _core_partial.BatchedEnvironment(
            self.num_envs,
            seed,
            terrain_tensor,
            self.config.agent_specs,
            self.config.poi_specs,
            self.config.initial_agent_positions,
            self.config.initial_poi_positions,
            self.config.tile_supports_walking,
            self.config.tile_supports_aquatic,
            self.config.tile_supports_flying,
            self.config.tile_blocking,
            cooperative_rewards,
            reward_new_tile,
            reward_found,
            reward_saved,
        )

        self.terrain_channels = int(self.env.get_terrain_channels())
        self.n_pois = int(self.env.get_num_pois())
        self.stride = int(self.env.get_stride())
        self.flat_map_size = int(self.env.get_flat_map_size())

        self._build_offsets()

        local_channels = self.terrain_channels + 1 + 1 + self.n_agents
        global_channels = self.terrain_channels + 1 + 1 + self.n_agents
        self.actor_internal_dim = 6
        self.critic_internal_dim = self.n_agents * 6 + self.n_pois * 4

        self.single_observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.n_agents, local_channels, self.map_size, self.map_size),
                    dtype=np.float32,
                ),
                "internal": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_agents, self.actor_internal_dim),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        self.num_envs,
                        self.n_agents,
                        local_channels,
                        self.map_size,
                        self.map_size,
                    ),
                    dtype=np.float32,
                ),
                "internal": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_envs, self.n_agents, self.actor_internal_dim),
                    dtype=np.float32,
                ),
            }
        )

        self.single_state_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(global_channels, self.map_size, self.map_size),
                    dtype=np.float32,
                ),
                "internal": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.critic_internal_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.single_action_space = spaces.Dict(
            {
                "move": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "radio": spaces.Discrete(4),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.n_agents, 3),
            dtype=np.float32,
        )

        super().__init__()

        ptr, size_bytes = self.env.get_memory_view()
        float_count = size_bytes // 4
        ctypes_array = (ctypes.c_float * float_count).from_address(ptr)
        np_array = np.ctypeslib.as_array(ctypes_array)
        self._raw_tensor = torch.from_numpy(np_array)
        self.state_tensor = self._raw_tensor.view(self.num_envs, self.stride)

        self._pygame_screen = None
        self._pygame_tile_px = 20
        self._last_known_agent = np.full(
            (self.num_envs, self.n_agents, self.n_agents, 2), -1, dtype=np.int32
        )
        self._known_survivor = np.zeros(
            (self.num_envs, self.n_agents, self.n_pois), dtype=bool
        )
        self._last_known_survivor = np.full(
            (self.num_envs, self.n_agents, self.n_pois, 2), -1, dtype=np.int32
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
