import time
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
        init_mode="parallel_first_touch",
        ego_view=False,
        ego_size=32,
    ):
        self.config = load_sar_config(tiles_json, agents_json, survivors_json, map_png)
        self.num_envs = num_envs
        self.device = device
        self.requires_state = requires_state
        self.render_initialized = False
        self.init_mode_val = 0 if init_mode == "parallel_first_touch" else 1

        # Ego-centric obs: the spatial obs tensor becomes a fixed
        # ego_size x ego_size window centered on each agent instead of the full
        # HxW map. State (requires_state) is unaffected and stays global.
        self.ego_view = bool(ego_view)
        self.ego_size = int(ego_size)
        if self.ego_view and self.ego_size <= 0:
            raise ValueError("ego_size must be > 0 when ego_view=True")

        self.steps_taken = 0
        self.t_py_flatten = 0.0
        self.t_cpp_step = 0.0
        self.t_py_tensor = 0.0
        self.t_py_obs = 0.0

        mode_map = {"decentralized": 0, "centralized": 1, "no_obs": 2}
        self.mode_val = mode_map.get(mode.lower(), 0)

        # 1. Determine Tensor Dimensions
        # Spatial Channels: Tiles + Altitude + POI + Observed Mask + Agents
        self.spatial_channels = self.config.n_tiles + 3 + self.config.n_agents

        # Internal channels per agent: y, x, battery, view_range, deploy, stuck
        self.agent_internal_dim = 6

        # 2. Allocate contiguous PyTorch memory for zero-copy C++ updates.
        # Spatial obs footprint. In ego mode the per-agent window replaces the
        # full map, and centralized obs becomes per-agent (each agent gets its
        # own crop of the shared observed map).
        if self.ego_view:
            obs_h = obs_w = self.ego_size
        else:
            obs_h, obs_w = self.config.height, self.config.width

        # Whether the spatial obs tensor carries a per-agent dimension.
        obs_per_agent = (self.mode_val == 0) or self.ego_view

        # Spatial obs is stored as uint8 (4x less memory than float32): every
        # channel is binary (255=on) except altitude, which the C++ side
        # quantizes to a byte. Consumers cast with `.float() / 255.0` (see
        # OBS_UINT8_SCALE and the encoder's cast-on-consume). The `internal`
        # vector and `state` tensor stay float32.
        if obs_per_agent and self.mode_val != 2:  # DECENTRALIZED, or ego CENTRALIZED
            self.obs_spatial = torch.empty(
                (
                    self.num_envs,
                    self.config.n_agents,
                    self.spatial_channels,
                    obs_h,
                    obs_w,
                ),
                dtype=torch.uint8,
                pin_memory=True,
            ).contiguous()
        else:  # non-ego CENTRALIZED or NO_OBS
            self.obs_spatial = torch.empty(
                (
                    self.num_envs,
                    self.spatial_channels,
                    obs_h,
                    obs_w,
                ),
                dtype=torch.uint8,
                pin_memory=True,
            ).contiguous()

        self.obs_internal = torch.empty(
            (self.num_envs, self.config.n_agents, self.agent_internal_dim),
            dtype=torch.float32,
            pin_memory=True,
        ).contiguous()

        # Per-sample spatial obs shape (excludes the leading num_envs dim).
        # Runners should read this instead of assuming (C, H, W): in ego
        # centralized mode it is (A, C, S, S).
        self.obs_spatial_shape = tuple(self.obs_spatial.shape[1:])
        # Shape of a single agent's spatial map slab (C, H, W) or (C, S, S).
        # This is what a per-agent CNN encoder consumes; it is stable across
        # modes and is the value runners should feed to their encoders.
        self.map_spatial_shape = (self.spatial_channels, obs_h, obs_w)
        # Whether obs_spatial carries a leading per-agent dimension.
        self.obs_is_per_agent = obs_per_agent and self.mode_val != 2

        if self.requires_state:
            self.state_spatial = torch.empty(
                (
                    self.num_envs,
                    self.spatial_channels,
                    self.config.height,
                    self.config.width,
                ),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
            self.state_internal = torch.empty(
                (self.num_envs, self.config.n_agents * 6 + self.config.n_pois * 4),
                dtype=torch.float32,
                pin_memory=True,
            ).contiguous()
        else:
            self.state_spatial = torch.empty(0)
            self.state_internal = torch.empty(0)

        self.rewards = torch.empty(
            (self.num_envs, self.config.n_agents),
            dtype=torch.float32,
            pin_memory=True,
        ).contiguous()
        self.terminated = torch.empty(
            (self.num_envs,),
            dtype=torch.bool,
            pin_memory=True,
        ).contiguous()
        self.truncated = torch.empty(
            (self.num_envs,),
            dtype=torch.bool,
            pin_memory=True,
        ).contiguous()

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
                self.config.agent_view_ranges,
                self.config.saveable_map,
                self.config.initial_agent_pos,
                self.config.initial_poi_pos,
                self.config.agent_max_batteries,
                self.config.agent_flags,
                self.config.agent_max_altitudes,
                self.obs_spatial.data_ptr() if self.mode_val != 2 else 0,
                self.obs_internal.data_ptr() if self.mode_val != 2 else 0,
                self.state_spatial.data_ptr() if self.requires_state else 0,
                self.state_internal.data_ptr() if self.requires_state else 0,
                self.rewards.data_ptr(),
                self.terminated.data_ptr(),
                self.truncated.data_ptr(),
                self.requires_state,
                True,  # Cooperative rewards
                0.05,  # Reward: new tile
                2.0,  # Reward: found
                20.0,  # Reward: saved
                250,  # Max frames
                self.mode_val,
                self.init_mode_val,
                self.ego_view,
                self.ego_size,
            )

    def _get_obs_dict(self):
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
        t0 = time.perf_counter()

        move_act = np.asarray(move_actions, dtype=np.float32).reshape(-1)
        radio_act = np.asarray(radio_actions, dtype=np.int32).reshape(-1)

        t1 = time.perf_counter()
        self.env.step(move_act, radio_act)

        t2 = time.perf_counter()
        rewards = self.rewards
        terminated = self.terminated
        truncated = self.truncated

        if self.device != "cpu":
            rewards = rewards.to(self.device)
            terminated = terminated.to(self.device)
            truncated = truncated.to(self.device)

        t3 = time.perf_counter()
        obs = self._get_obs_dict()
        t4 = time.perf_counter()

        self.steps_taken += 1
        self.t_py_flatten += t1 - t0
        self.t_cpp_step += t2 - t1
        self.t_py_tensor += t3 - t2
        self.t_py_obs += t4 - t3

        if self.steps_taken % 10000 == 0:
            print(f"[Python Timing after {self.steps_taken} steps]")
            print(f"  py_flatten_actions: {self.t_py_flatten:.4f} s")
            print(f"  cpp_engine_step: {self.t_cpp_step:.4f} s")
            print(f"  py_tensor_conversion: {self.t_py_tensor:.4f} s")
            print(f"  py_get_obs: {self.t_py_obs:.4f} s")
            print()
            self.t_py_flatten = 0.0
            self.t_cpp_step = 0.0
            self.t_py_tensor = 0.0
            self.t_py_obs = 0.0

        return obs, rewards, terminated, truncated, {}

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

    def _init_renderer(self, grid_w=None, grid_h=None, tile_px=8):
        """Create (or resize) the pygame window.

        ``grid_w``/``grid_h`` default to the full map; the ego renderer passes
        the ego window side so the display matches the crop the agent consumes.
        The window is (re)created whenever the requested grid or tile size
        changes, so the same env can render either the global map or an ego crop.
        """
        grid_w = self.config.width if grid_w is None else int(grid_w)
        grid_h = self.config.height if grid_h is None else int(grid_h)

        already = getattr(self, "render_initialized", False)
        same = (
            already
            and getattr(self, "_render_grid_w", None) == grid_w
            and getattr(self, "_render_grid_h", None) == grid_h
            and getattr(self, "_pygame_tile_px", None) == tile_px
        )
        if same:
            return

        pygame.init()
        self._pygame_tile_px = tile_px
        self._render_grid_w = grid_w
        self._render_grid_h = grid_h
        window_width = grid_w * self._pygame_tile_px
        window_height = grid_h * self._pygame_tile_px
        self._pygame_screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("SAR Batched Environment Viewer")

        self._terrain_colors = self.config.terrain_rgb
        self._agent_colors = self.config.agent_rgb
        self._survivor_colors = self.config.survivor_rgb

        type_map_grid = np.array(self.config.type_map).reshape(
            self.config.height, self.config.width
        )
        self._base_map_rgb = self._terrain_colors[type_map_grid].astype(np.uint8)

        self._dimmed_map_rgb = (self._base_map_rgb // 2).astype(np.uint8)

        self.render_initialized = True

    def _extract_agent_positions(self, env_idx):
        if self.requires_state:
            internal = self.state_internal[env_idx].cpu().numpy()
            agent_data = internal[: self.config.n_agents * 6].reshape(
                self.config.n_agents, 6
            )
            return agent_data[:, :2]

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

        agent_pos = self._extract_agent_positions(env_idx)

        if pov == -1:
            self._render_true_state(env_idx, agent_pos)
        elif self.mode_val == 0:
            self._render_decentralized_pov(env_idx, pov, agent_pos)
        else:
            self._render_centralized_obs(env_idx, agent_pos)

    def render_ego(self, pov, env_idx=0, tile_px=16):
        """Render exactly the ego-centric observation agent ``pov`` consumes.

        This draws the uint8 ego crop the network receives -- the agent sits at
        the center, undiscovered tiles are shown as fog, and everything visible
        (terrain, survivors, teammates) is whatever has accumulated in the
        agent's observation buffer, including tiles/POIs shared over the radio.
        The agent's circular view range is overlaid as a ring.

        Falls back to the full-map POV render when the env is not in ego mode.
        """
        if not self.ego_view:
            self.render(pov, env_idx)
            return
        if pov >= self.config.n_agents:
            print(f"Warning: Requested ego POV {pov} exceeds agent count. Skipping.")
            return

        S = self.ego_size
        self._init_renderer(grid_w=S, grid_h=S, tile_px=tile_px)

        spatial = self.obs_spatial[env_idx, pov].cpu().numpy()  # (C, S, S) uint8
        n_tiles = self.config.n_tiles
        idx_poi = n_tiles + 1
        idx_obs = n_tiles + 2
        idx_me = n_tiles + 3          # MY_LOCATION channel
        idx_others = n_tiles + 4      # OTHER_LOCATIONS channels start here

        observed = spatial[idx_obs] > 0
        tile_argmax = spatial[:n_tiles].argmax(axis=0)
        terrain = self._terrain_colors[tile_argmax].astype(np.uint8)  # (S, S, 3)
        fog = np.array([12, 12, 18], dtype=np.uint8)
        rgb_grid = np.where(observed[..., None], terrain, fog).astype(np.uint8)

        poi_mask = spatial[idx_poi] > 0

        # Map the packed OTHER_LOCATIONS channels back to real agent indices
        # (they are the agents other than pov, in ascending order).
        others = [a for a in range(self.config.n_agents) if a != pov]
        agent_layers = [(spatial[idx_me] > 0, pov)]
        for k, a in enumerate(others):
            agent_layers.append((spatial[idx_others + k] > 0, a))

        view_range = float(self.obs_internal[env_idx, pov, 3].item())
        self._draw_ego_to_screen(rgb_grid, poi_mask, agent_layers, view_range)

    def _draw_ego_to_screen(self, rgb_grid, poi_mask, agent_layers, view_range):
        import pygame
        pygame.event.pump()

        S = self.ego_size
        tile = self._pygame_tile_px
        surface = pygame.surfarray.make_surface(rgb_grid.transpose(1, 0, 2))
        scaled_surface = pygame.transform.scale(surface, (S * tile, S * tile))
        self._pygame_screen.blit(scaled_surface, (0, 0))

        # View-range ring around the agent, which is fixed at the crop center.
        half = S // 2
        center = (half * tile + tile // 2, half * tile + tile // 2)
        if view_range and view_range > 0:
            pygame.draw.circle(
                self._pygame_screen, (255, 255, 0), center,
                int(round(view_range * tile)), 1
            )

        poi_ys, poi_xs = np.where(poi_mask)
        for y, x in zip(poi_ys, poi_xs):
            c = (x * tile + tile // 2, y * tile + tile // 2)
            pygame.draw.circle(self._pygame_screen, (255, 255, 255), c, max(2, tile // 3))

        for mask, a in agent_layers:
            ys, xs = np.where(mask)
            color = tuple(int(c) for c in self._agent_colors[a])
            for y, x in zip(ys, xs):
                rect = pygame.Rect(x * tile + 2, y * tile + 2, tile - 4, tile - 4)
                pygame.draw.rect(self._pygame_screen, color, rect)

        pygame.display.flip()

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

        poi_ys, poi_xs = np.where(poi_mask)
        for y, x in zip(poi_ys, poi_xs):
            center = (x * tile + tile // 2, y * tile + tile // 2)
            pygame.draw.circle(
                self._pygame_screen, (255, 255, 255), center, max(2, tile // 3)
            )

        for a in range(self.config.n_agents):
            y, x = int(agent_pos[a, 0]), int(agent_pos[a, 1])

            if 0 <= y < self.config.height and 0 <= x < self.config.width:
                color = tuple(int(c) for c in self._agent_colors[a])
                rect = pygame.Rect(x * tile + 2, y * tile + 2, tile - 4, tile - 4)
                pygame.draw.rect(self._pygame_screen, color, rect)

        pygame.display.flip()