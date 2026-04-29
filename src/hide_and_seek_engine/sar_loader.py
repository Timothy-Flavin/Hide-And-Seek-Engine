import json
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class SARConfig:
    n_tiles: int
    n_agents: int
    n_pois: int
    width: int
    height: int

    supports_walking: list[int]
    supports_aquatic: list[int]
    supports_flying: list[int]
    is_blocking: list[int]

    type_map: list[int]
    altitude_map: list[float]
    agent_speed_map: list[float]
    agent_view_ranges: list[float]
    agent_max_batteries: list[float]
    agent_flags: list[int]
    agent_max_altitudes: list[float]
    saveable_map: list[int]
    initial_agent_pos: list[float]
    initial_poi_pos: list[float]

    terrain_rgb: np.ndarray
    agent_rgb: np.ndarray
    survivor_rgb: np.ndarray


def load_sar_config(
    tiles_json: str, agents_json: str, survivors_json: str, map_png: str
) -> SARConfig:
    with open(tiles_json) as f:
        tiles_data = json.load(f)
    with open(agents_json) as f:
        agents_data = json.load(f)
    with open(survivors_json) as f:
        survivors_data = json.load(f)

    # 1. Tile Properties
    tile_names = list(tiles_data.keys())
    n_tiles = len(tile_names)

    supports_walking = [
        int(tiles_data[t].get("supports_walking", False)) for t in tile_names
    ]
    supports_aquatic = [
        int(tiles_data[t].get("supports_aquatic", False)) for t in tile_names
    ]
    supports_flying = [
        int(tiles_data[t].get("supports_flying", False)) for t in tile_names
    ]
    is_blocking = [bool(tiles_data[t].get("blocking", False)) for t in tile_names]

    tile_colors = np.array([tiles_data[t]["rgb"] for t in tile_names], dtype=np.int32)
    tile_altitudes = [float(tiles_data[t]["altitude"]) for t in tile_names]

    # 2. Image Map Parsing
    img = Image.open(map_png).convert("RGB")
    width, height = img.size
    flat_rgb = np.asarray(img).reshape(-1, 3)

    type_map = []
    altitude_map = []

    # Nearest neighbor color mapping
    for px in flat_rgb:
        dist = np.sum((tile_colors - px) ** 2, axis=1)
        t_id = int(np.argmin(dist))
        type_map.append(t_id)
        altitude_map.append(tile_altitudes[t_id])

    # 3. Agent Properties
    agent_names = list(agents_data.keys())
    n_agents = len(agent_names)
    agent_speed_map = []
    initial_agent_pos = []
    agent_view_ranges = []
    agent_max_batteries = []
    agent_flags = []
    agent_max_altitudes = []
    agent_colors = np.array(
        [agents_data[a].get("rgb", [255, 0, 0]) for a in agent_names], dtype=np.int32
    )

    for a_name in agent_names:
        agent = agents_data[a_name]
        speeds = agent.get("terrain_speed", {})
        agent_view_ranges.append(float(agent.get("base_view", 5.0)))
        for t_name in tile_names:
            agent_speed_map.append(float(speeds.get(t_name, 1.0)))
            
        agent_max_batteries.append(float(agent.get("battery", 250.0)))
        
        start = agent.get("start", [0.5, 0.5])
        initial_agent_pos.extend(
            [float(start[0] * (height - 1)), float(start[1] * (width - 1))]
        )
        
        # Determine bitmask flags for C++ optimization
        can_fly = bool(agent.get("flying", False))
        can_swim = bool(agent.get("aqueous", False))
        can_walk = bool(agent.get("walking", False))
        
        flag = 0
        if can_walk: flag |= 2 # Bit 1: walk
        if can_fly: flag |= 4  # Bit 2: fly
        if can_swim: flag |= 8 # Bit 3: swim
        agent_flags.append(flag)
        
        agent_max_altitudes.append(float(agent.get("altitude_max", 1.0)))

    # 4. POI (Survivor) Properties
    survivor_names = list(survivors_data.keys())
    n_pois = len(survivor_names)
    saveable_map = []
    initial_poi_pos = []
    survivor_colors = np.array(
        [survivors_data[s].get("rgb", [255, 255, 255]) for s in survivor_names],
        dtype=np.int32,
    )

    for s_name in survivor_names:
        survivor = survivors_data[s_name]
        allowed = survivor.get("allowed_savers", [])

        # Generates an array of booleans determining if each agent can save this POI
        for a_name in agent_names:
            saveable_map.append(bool(a_name in allowed))

        start = survivor.get("start", [0.5, 0.5])
        initial_poi_pos.extend(
            [float(start[0] * (height - 1)), float(start[1] * (width - 1))]
        )

    return SARConfig(
        n_tiles=n_tiles,
        n_agents=n_agents,
        n_pois=n_pois,
        width=width,
        height=height,
        supports_walking=supports_walking,
        supports_aquatic=supports_aquatic,
        supports_flying=supports_flying,
        is_blocking=is_blocking,
        type_map=type_map,
        altitude_map=altitude_map,
        agent_speed_map=agent_speed_map,
        agent_view_ranges=agent_view_ranges,
        agent_max_batteries=agent_max_batteries,
        agent_flags=agent_flags,
        agent_max_altitudes=agent_max_altitudes,
        saveable_map=saveable_map,
        initial_agent_pos=initial_agent_pos,
        initial_poi_pos=initial_poi_pos,
        terrain_rgb=tile_colors,
        agent_rgb=agent_colors,
        survivor_rgb=survivor_colors,
    )