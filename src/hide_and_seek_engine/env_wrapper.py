import numpy as np
import torch
from hide_and_seek_engine.sar_loader import load_sar_config
import importlib
import pygame

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
        self.render_initialized = False

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
        move_act = np.asarray(move_actions, dtype=np.float32).flatten()
        radio_act = np.asarray(radio_actions, dtype=np.int32).flatten()

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

    def _init_renderer(self):
        """Just-in-time initialization of Pygame and cached drawing variables."""
        if getattr(self, "render_initialized", False):
            return
        pygame.init()
        self._pygame_tile_px = 16
        window_width = self.config.width * self._pygame_tile_px
        window_height = self.config.height * self._pygame_tile_px
        self._pygame_screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("SAR Batched Environment Viewer")

        # Pull colors safely populated by the updated loader
        self._terrain_colors = self.config.terrain_rgb
        self._agent_colors = self.config.agent_rgb
        self._survivor_colors = self.config.survivor_rgb

        # Pre-cache the true map layout as an RGB array [H, W, 3]
        type_map_grid = np.array(self.config.type_map).reshape(
            self.config.height, self.config.width
        )
        self._base_map_rgb = self._terrain_colors[type_map_grid].astype(np.uint8)

        # Cut RGB values in half for undiscovered tiles
        self._dimmed_map_rgb = (self._base_map_rgb // 2).astype(np.uint8)

        self.render_initialized = True

    def _extract_agent_positions(self, env_idx):
        """Grabs the N x 2 array of [y, x] floats from the vectorized internal state."""
        if self.requires_state:
            # Internal State layout: [E, A*6 + P*4]. Agents are the first A*6 block.
            internal = self.state_internal[env_idx].cpu().numpy()
            agent_data = internal[: self.config.n_agents * 6].reshape(
                self.config.n_agents, 6
            )
            return agent_data[:, :2]

        # Fallback if requires_state=False
        if self.mode_val != 0:
            internal = self.obs_internal[env_idx].cpu().numpy()
            return internal[:, :2]
        else:
            print(
                "Warning: Cannot extract all true agent positions when requires_state=False in Decentralized mode."
            )
            return np.zeros((self.config.n_agents, 2))

    def render(self, pov=-1, env_idx=0):
        self._init_renderer()

        if pov == -1 and not self.requires_state:
            print(
                "Warning: Cannot render true state (pov=-1) because requires_state=False. Skipping."
            )
            return

        if pov >= self.config.n_agents:
            print(f"Warning: Requested POV {pov} exceeds agent count. Skipping.")
            return

        # Fetch the shared true positions (avoids layer scanning)
        agent_pos = self._extract_agent_positions(env_idx)

        if pov == -1:
            self._render_true_state(env_idx, agent_pos)
        elif self.mode_val == 0:
            self._render_decentralized_pov(env_idx, pov, agent_pos)
        else:
            self._render_centralized_obs(env_idx, agent_pos)

    def _render_true_state(self, env_idx, agent_pos):
        spatial = self.state_spatial[env_idx].cpu().numpy()
        idx_poi = self.config.n_tiles + 1
        idx_obs = self.config.n_tiles + 2

        observed_mask = spatial[idx_obs] > 0.5
        poi_mask = spatial[idx_poi] > 0.5

        rgb_grid = np.where(
            observed_mask[..., None], self._base_map_rgb, self._dimmed_map_rgb
        )
        self._draw_to_screen(rgb_grid, poi_mask, agent_pos)

    def _render_centralized_obs(self, env_idx, agent_pos):
        spatial = self.obs_spatial[env_idx].cpu().numpy()
        idx_poi = self.config.n_tiles + 1
        idx_obs = self.config.n_tiles + 2

        observed_mask = spatial[idx_obs] > 0.5
        poi_mask = spatial[idx_poi] > 0.5

        rgb_grid = np.where(
            observed_mask[..., None], self._base_map_rgb, self._dimmed_map_rgb
        )
        self._draw_to_screen(rgb_grid, poi_mask, agent_pos)

    def _render_decentralized_pov(self, env_idx, pov, agent_pos):
        spatial = self.obs_spatial[env_idx, pov].cpu().numpy()
        idx_poi = self.config.n_tiles + 1
        idx_obs = self.config.n_tiles + 2

        observed_mask = spatial[idx_obs] > 0.5
        poi_mask = spatial[idx_poi] > 0.5

        rgb_grid = np.where(
            observed_mask[..., None], self._base_map_rgb, self._dimmed_map_rgb
        )
        self._draw_to_screen(rgb_grid, poi_mask, agent_pos)

    def _draw_to_screen(self, rgb_grid, poi_mask, agent_pos):
        import pygame

        pygame.event.pump()

        # 1. Base map
        surface = pygame.surfarray.make_surface(rgb_grid.transpose(1, 0, 2))
        scaled_surface = pygame.transform.scale(
            surface,
            (
                self.config.width * self._pygame_tile_px,
                self.config.height * self._pygame_tile_px,
            ),
        )
        self._pygame_screen.blit(scaled_surface, (0, 0))

        tile = self._pygame_tile_px

        # 2. POIs (Since POI mask is a single flattened layer, np.where is practically free here)
        poi_ys, poi_xs = np.where(poi_mask)
        for y, x in zip(poi_ys, poi_xs):
            center = (x * tile + tile // 2, y * tile + tile // 2)
            pygame.draw.circle(
                self._pygame_screen, (255, 255, 255), center, max(2, tile // 3)
            )

        # 3. Agents (Pulled directly from internal float arrays!)
        for a in range(self.config.n_agents):
            y, x = int(agent_pos[a, 0]), int(agent_pos[a, 1])

            # Simple bounds check
            if 0 <= y < self.config.height and 0 <= x < self.config.width:
                color = tuple(int(c) for c in self._agent_colors[a])
                rect = pygame.Rect(x * tile + 2, y * tile + 2, tile - 4, tile - 4)
                pygame.draw.rect(self._pygame_screen, color, rect)

        pygame.display.flip()
