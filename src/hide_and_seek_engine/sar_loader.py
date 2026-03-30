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

    supports_walking: list[bool]
    supports_aquatic: list[bool]
    supports_flying: list[bool]
    is_blocking: list[bool]

    type_map: list[int]
    altitude_map: list[float]
    agent_speed_map: list[float]
    saveable_map: list[bool]
    initial_agent_pos: list[float]
    initial_poi_pos: list[float]


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
        tiles_data[t].get("supports_walking", False) for t in tile_names
    ]
    supports_aquatic = [
        tiles_data[t].get("supports_aquatic", False) for t in tile_names
    ]
    supports_flying = [tiles_data[t].get("supports_flying", False) for t in tile_names]
    is_blocking = [tiles_data[t].get("blocking", False) for t in tile_names]
    tile_colors = np.array([tiles_data[t]["rgb"] for t in tile_names])
    tile_altitudes = [tiles_data[t]["altitude"] for t in tile_names]

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
        altitude_map.append(float(tile_altitudes[t_id]))

    # 3. Agent Properties
    agent_names = list(agents_data.keys())
    n_agents = len(agent_names)
    agent_speed_map = []
    initial_agent_pos = []

    for a_name in agent_names:
        agent = agents_data[a_name]
        speeds = agent.get("terrain_speed", {})
        for t_name in tile_names:
            agent_speed_map.append(float(speeds.get(t_name, 1.0)))

        # Denormalize starting coords
        start = agent.get("start", [0.5, 0.5])
        initial_agent_pos.extend(
            [float(start[0] * (height - 1)), float(start[1] * (width - 1))]
        )

    # 4. POI (Survivor) Properties
    survivor_names = list(survivors_data.keys())
    n_pois = len(survivor_names)
    saveable_map = []
    initial_poi_pos = []

    for s_name in survivor_names:
        survivor = survivors_data[s_name]
        allowed = survivor.get("allowed_savers", [])
        for a_name in agent_names:
            saveable_map.append(a_name in allowed)

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
        saveable_map=saveable_map,
        initial_agent_pos=initial_agent_pos,
        initial_poi_pos=initial_poi_pos,
    )
