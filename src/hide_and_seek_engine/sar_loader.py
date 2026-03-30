import importlib

# Import the C++ extension from the installed package location
try:
    cpp_engine = importlib.import_module("hide_and_seek_engine.cpp_engine")
except ImportError:
    cpp_engine = None
import ctypes
import json
import os
from dataclasses import dataclass
from typing import Any
import numpy as np

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


if cpp_engine is not None:
    FeatureType = cpp_engine.FeatureType
else:
    FeatureType = None


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
