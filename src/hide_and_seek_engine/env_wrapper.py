import numpy as np
import torch
from hide_and_seek_engine.sar_loader import load_sar_config
import importlib

# Import the C++ extension
try:
    cpp_engine = importlib.import_module("hide_and_seek_engine.cpp_engine")
except ImportError:
    cpp_engine = None
    print("Warning: C++ Engine not found. Please compile the bindings.")


class SARBatchedGridEnv:
    def __init__(
        self,
        num_envs,
        map_png,
        tiles_json,
        agents_json,
        survivors_json,
        mode="decentralized",
        requires_state=True,
        device="cpu",
    ):
        self.config = load_sar_config(tiles_json, agents_json, survivors_json, map_png)
        self.num_envs = num_envs
        self.device = device
        self.requires_state = requires_state

        mode_map = {"decentralized": 0, "centralized": 1, "no_obs": 2}
        self.mode_val = mode_map.get(mode.lower(), 0)

        # 1. Determine Tensor Dimensions
        # Spatial Channels: Tiles + Altitude + POI + Observed Mask + Agents
        self.spatial_channels = self.config.n_tiles + 3 + self.config.n_agents

        # Internal channels per agent: y, x, battery, view_range, deploy, stuck
        self.agent_internal_dim = 6

        # 2. Allocate contiguous PyTorch memory for zero-copy C++ updates
        # Pinned memory ensures fast transfer if the user later shifts these to a GPU
        if self.mode_val == 0:  # DECENTRALIZED
            self.obs_spatial = torch.zeros(
                (
                    self.num_envs,
                    self.config.n_agents,
                    self.spatial_channels,
                    self.config.height,
                    self.config.width,
                ),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
            self.obs_internal = torch.zeros(
                (self.num_envs, self.config.n_agents, self.agent_internal_dim),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
        else:  # CENTRALIZED or NO_OBS
            self.obs_spatial = torch.zeros(
                (
                    self.num_envs,
                    self.spatial_channels,
                    self.config.height,
                    self.config.width,
                ),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
            self.obs_internal = torch.zeros(
                (self.num_envs, self.config.n_agents, self.agent_internal_dim),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()

        if self.requires_state:
            self.state_spatial = torch.zeros(
                (
                    self.num_envs,
                    self.spatial_channels,
                    self.config.height,
                    self.config.width,
                ),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
            self.state_internal = torch.zeros(
                (self.num_envs, self.config.n_agents * 6 + self.config.n_pois * 4),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
        else:
            self.state_spatial = torch.empty(0)
            self.state_internal = torch.empty(0)

        # 3. Instantiate C++ Engine passing data_ptr()
        if cpp_engine is not None:
            self.env = cpp_engine.BatchedEnvironment(
                self.num_envs,
                42,  # Sim Seed
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
                self.obs_spatial.data_ptr() if self.mode_val != 2 else 0,
                self.obs_internal.data_ptr() if self.mode_val != 2 else 0,
                self.state_spatial.data_ptr() if self.requires_state else 0,
                self.state_internal.data_ptr() if self.requires_state else 0,
                self.requires_state,
                True,  # Cooperative rewards
                0.05,  # Reward: new tile
                2.0,  # Reward: found
                20.0,  # Reward: saved
                250,  # Max frames
                self.mode_val,
            )

    def _get_obs_dict(self):
        """Returns the in-place updated tensors."""
        return {
            "spatial": (
                self.obs_spatial.to(self.device)
                if self.device != "cpu"
                else self.obs_spatial
            ),
            "internal": (
                self.obs_internal.to(self.device)
                if self.device != "cpu"
                else self.obs_internal
            ),
        }

    def reset(self):
        self.env.reset()
        return self._get_obs_dict(), {}

    def reset_env(self, env_idx: int):
        self.env.reset_env(int(env_idx))
        return self._get_obs_dict(), {}

    def step(self, move_actions, radio_actions):
        """
        Takes move_actions [num_envs, n_agents, 2] and radio_actions [num_envs, n_agents]
        """
        # Ensure correct formatting for the pybind arrays
        move_act = np.asarray(move_actions, dtype=np.float32)
        radio_act = np.asarray(radio_actions, dtype=np.int32)

        # Advance the environment. The C++ function directly writes to our spatial/internal memory!
        rewards_np, terminated_np, truncated_np = self.env.step(move_act, radio_act)

        # Convert engine returns to tensors
        rewards = torch.from_numpy(rewards_np)
        terminated = torch.from_numpy(terminated_np)
        truncated = torch.from_numpy(truncated_np)

        if self.device != "cpu":
            rewards = rewards.to(self.device)
            terminated = terminated.to(self.device)
            truncated = truncated.to(self.device)

        return self._get_obs_dict(), rewards, terminated, truncated, {}

    def get_state(self):
        if not self.requires_state:
            return None
        return {
            "spatial": (
                self.state_spatial.to(self.device)
                if self.device != "cpu"
                else self.state_spatial
            ),
            "internal": (
                self.state_internal.to(self.device)
                if self.device != "cpu"
                else self.state_internal
            ),
        }
