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
    tile_requires_aquatic: np.ndarray
    tile_requires_flying: np.ndarray
    agent_class_to_id: dict[str, int]
    agent_specs: np.ndarray
    initial_agent_positions: np.ndarray
    poi_specs: np.ndarray
    initial_poi_positions: np.ndarray


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(payload: Any, key_hint: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "agents", "survivors", "tiles", key_hint):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    raise ValueError(f"Could not find list payload for {key_hint}")


def load_sar_config(
    tiles_json: str,
    agents_json: str,
    survivors_json: str,
    num_envs: int,
    map_size: int = 32,
) -> SARConfig:
    tiles_payload = _as_list(_load_json(tiles_json), "tiles")
    agents_payload = _as_list(_load_json(agents_json), "agents")
    survivors_payload = _as_list(_load_json(survivors_json), "survivors")

    terrain_names = [str(t.get("name", f"tile_{i}")) for i, t in enumerate(tiles_payload)]
    terrain_rgb = np.asarray([t.get("rgb", [0, 0, 0]) for t in tiles_payload], dtype=np.int32)
    terrain_altitudes = np.asarray([float(t.get("altitude", 0.5)) for t in tiles_payload], dtype=np.float32)
    alt_min = float(np.min(terrain_altitudes)) if len(terrain_altitudes) else 0.0
    alt_max = float(np.max(terrain_altitudes)) if len(terrain_altitudes) else 1.0
    denom = alt_max - alt_min if abs(alt_max - alt_min) > 1e-8 else 1.0
    terrain_altitudes = (terrain_altitudes - alt_min) / denom

    tile_requires_aquatic = np.asarray(
        [int(bool(t.get("requires_aquatic", False))) for t in tiles_payload], dtype=np.int32
    )
    tile_requires_flying = np.asarray(
        [int(bool(t.get("requires_flying", False))) for t in tiles_payload], dtype=np.int32
    )

    agent_class_names = sorted({str(a.get("class", f"class_{i}")) for i, a in enumerate(agents_payload)})
    agent_class_to_id = {name: idx for idx, name in enumerate(agent_class_names)}

    n_tiles = len(tiles_payload)
    spec_width = 9 + n_tiles
    agent_specs = np.zeros((4, spec_width), dtype=np.float32)
    initial_agent_positions = np.zeros((num_envs, 4, 2), dtype=np.float32)

    default_agent = {
        "flying": False,
        "aqueous": False,
        "altitude_min": 0.0,
        "altitude_max": 1.0,
        "base_speed": 1.0,
        "base_view": 3.0,
        "battery": 100.0,
        "deployment_delay": 0.0,
    }

    for i in range(min(4, len(agents_payload))):
        a = {**default_agent, **agents_payload[i]}
        terrain_multipliers = a.get("terrain_speed", {}) or {}
        agent_specs[i, 0] = float(bool(a.get("flying", False)))
        agent_specs[i, 1] = float(bool(a.get("aqueous", False)))
        agent_specs[i, 2] = float(a.get("altitude_min", 0.0))
        agent_specs[i, 3] = float(a.get("altitude_max", 1.0))
        agent_specs[i, 4] = float(a.get("base_speed", 1.0))
        agent_specs[i, 5] = float(a.get("base_view", 3.0))
        agent_specs[i, 6] = float(a.get("battery", 100.0))
        agent_specs[i, 7] = float(a.get("deployment_delay", 0.0))
        agent_specs[i, 8] = float(agent_class_to_id[str(a.get("class", f"class_{i}"))])

        for t_idx, t_name in enumerate(terrain_names):
            agent_specs[i, 9 + t_idx] = float(terrain_multipliers.get(t_name, 1.0))

        pos = np.asarray(a.get("start", [map_size // 2, map_size // 2]), dtype=np.float32)
        for e in range(num_envs):
            initial_agent_positions[e, i, :] = pos

    poi_specs = np.zeros((len(survivors_payload), 2), dtype=np.float32)
    initial_poi_positions = np.zeros((num_envs, len(survivors_payload), 2), dtype=np.float32)
    for p_idx, p in enumerate(survivors_payload):
        allowed = p.get("allowed_savers", [])
        mask = 0
        for c in allowed:
            if c in agent_class_to_id:
                mask |= 1 << agent_class_to_id[c]
        poi_specs[p_idx, 0] = float(bool(p.get("moves", True)))
        poi_specs[p_idx, 1] = float(mask)
        pos = np.asarray(p.get("start", [map_size // 2, map_size // 2]), dtype=np.float32)
        for e in range(num_envs):
            initial_poi_positions[e, p_idx, :] = pos

    return SARConfig(
        terrain_names=terrain_names,
        terrain_rgb=terrain_rgb,
        terrain_altitudes=terrain_altitudes,
        tile_requires_aquatic=tile_requires_aquatic,
        tile_requires_flying=tile_requires_flying,
        agent_class_to_id=agent_class_to_id,
        agent_specs=agent_specs,
        initial_agent_positions=initial_agent_positions,
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
        raise ImportError("Pillow is required for PNG map loading. Install with `pip install Pillow`.")

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

        self.env = _core_partial.BatchedEnvironment(
            self.num_envs,
            seed,
            terrain_tensor,
            self.config.agent_specs,
            self.config.poi_specs,
            self.config.initial_agent_positions,
            self.config.initial_poi_positions,
            self.config.tile_requires_aquatic,
            self.config.tile_requires_flying,
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
                    shape=(self.num_envs, self.n_agents, local_channels, self.map_size, self.map_size),
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
                "radio": spaces.Discrete(2),
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

    def _build_offsets(self):
        f = self.flat_map_size
        c = self.terrain_channels
        n = self.n_agents
        p = self.n_pois
        off = 0
        self.sl_terrain_alt = slice(off, off + c * f)
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
        terrain_alt = state_row[self.sl_terrain_alt].reshape(
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
        spatial[:, : self.terrain_channels] = terrain_alt[None, :, :, :]
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
        terrain_alt = state_row[self.sl_terrain_alt].reshape(
            self.terrain_channels, self.map_size, self.map_size
        )
        unsaved = state_row[self.sl_global_unsaved_pois].reshape(self.map_size, self.map_size)
        obs = state_row[self.sl_global_obs_mask].reshape(self.map_size, self.map_size)
        global_agents = state_row[self.sl_global_agent_layers].reshape(
            self.n_agents, self.map_size, self.map_size
        )
        spatial = np.concatenate(
            [terrain_alt, unsaved[None, :, :], obs[None, :, :], global_agents], axis=0
        ).astype(np.float32)

        pos = state_row[self.sl_agent_positions].reshape(self.n_agents, 2)
        deploy = state_row[self.sl_agent_deploy]
        stuck = state_row[self.sl_agent_stuck]
        view = state_row[self.sl_agent_view]
        battery = state_row[self.sl_agent_battery]
        poi_pos = state_row[self.sl_poi_positions].reshape(self.n_pois, 2) if self.n_pois else np.zeros((0, 2), dtype=np.float32)
        poi_found = state_row[self.sl_poi_found] if self.n_pois else np.zeros((0,), dtype=np.float32)
        poi_saved = state_row[self.sl_poi_saved] if self.n_pois else np.zeros((0,), dtype=np.float32)

        agent_internal = np.stack(
            [deploy, stuck, view, battery, pos[:, 0], pos[:, 1]], axis=-1
        ).astype(np.float32)
        if self.n_pois:
            poi_internal = np.concatenate([poi_pos, poi_found[:, None], poi_saved[:, None]], axis=1).astype(np.float32)
            internal = np.concatenate([agent_internal.reshape(-1), poi_internal.reshape(-1)], axis=0)
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
        obs = self._stack_actor_obs()
        if self.device != "cpu":
            obs = {k: v.to(self.device) for k, v in obs.items()}
        return obs, {}

    def reset_env(self, env_idx: int):
        self.env.reset_single(int(env_idx))
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
        spatial = torch.from_numpy(np.stack([g["spatial"] for g in global_states], axis=0))
        internal = torch.from_numpy(np.stack([g["internal"] for g in global_states], axis=0))
        if self.device != "cpu":
            spatial = spatial.to(self.device)
            internal = internal.to(self.device)
        return {"spatial": spatial, "internal": internal}

    def get_action_mask(self):
        return self.env.get_action_mask()


class SARParallelPettingZooEnv(ParallelEnv):
    metadata = {"name": "sar_parallel_v0", "render_modes": []}

    def __init__(self, **kwargs):
        if not PETTINGZOO_AVAILABLE:
            raise ImportError("PettingZoo is required. Install with `pip install pettingzoo`.")
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
                    dtype=np.float32,
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
        truncations = {agent: bool(truncated[0].item()) for agent in self.possible_agents}
        infos = {agent: {"action_mask": self._batched.get_action_mask()[0, i].astype(np.int8)} for i, agent in enumerate(self.possible_agents)}

        if done:
            self.agents = []

        return observations, rewards_out, terminations, truncations, infos


BatchedGridEnv = SARBatchedGridEnv
