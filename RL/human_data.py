"""Human demonstration dataset: append-only storage + a BC minibatch source.

Companion to ``human_dataset.py`` (the recorder) and the CleanRL trainers'
``--human-bc`` term. The human plays the ego-centric env, so each recorded
transition is a single agent's ego observation -- exactly the per-agent input the
*decentralized* (shared-head) networks consume, which is why BC is only wired up
for those.

Storage layout (mirrors the RL results layout one bucket deeper, keyed by which
agent the human controlled that episode):

    experiments/results/<level>/<agent_type>/segment_XXXX/<field>.npy
                                             /segment_XXXX/meta.json
                                             /frames.json          (running total)

Each recording session appends a new immutable ``segment_XXXX`` -- previously
collected frames are never rewritten, so 4 runs of 2.5k accumulate to the 10k
pre-training target. ``obs_spatial``/``next_obs_spatial`` are stored as uint8
(the env's compressed obs); the networks cast on read (``cast_obs``).
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch

RESULTS_ROOT = "experiments/results"

# Per-transition fields. obs_* are the ego crop the controlled agent saw (uint8);
# actions_move is the discrete index (ACTION_MAP) used as the BC target;
# actions_raw keeps the human's raw [dy, dx, radio]; radio is the emitted channel.
HUMAN_FIELDS = [
    "obs_spatial",
    "obs_internal",
    "actions_move",
    "actions_raw",
    "radio",
    "rewards",
    "dones",
    "next_obs_spatial",
    "next_obs_internal",
    "controlled_agent",
]

# Minimal field set a decentralized learner needs for a BC pass.
BC_FIELDS = ["obs_spatial", "obs_internal", "actions_move"]


def level_name_of(level: str) -> str:
    """Basename of a level path (``levels/test_level`` -> ``test_level``)."""
    return os.path.basename(os.path.normpath(level))


def human_bucket_dir(level: str, agent_type: str) -> str:
    return os.path.join(RESULTS_ROOT, level_name_of(level), agent_type)


# --------------------------------------------------------------------------- #
# Append-only segments
# --------------------------------------------------------------------------- #
def append_human_segment(level: str, agent_type: str, data: dict, meta: dict | None = None) -> str:
    """Append ``data`` (equal-length arrays) as a new segment; return its dir."""
    if not data:
        raise ValueError("Refusing to write an empty human segment")
    lengths = {name: len(np.asarray(arr)) for name, arr in data.items()}
    n = next(iter(lengths.values()))
    if any(v != n for v in lengths.values()):
        raise ValueError(f"All human fields must share length; got {lengths}")

    bucket = human_bucket_dir(level, agent_type)
    os.makedirs(bucket, exist_ok=True)
    seg_idx = len(sorted(glob.glob(os.path.join(bucket, "segment_*"))))
    seg_dir = os.path.join(bucket, f"segment_{seg_idx:04d}")
    os.makedirs(seg_dir, exist_ok=True)

    for name, arr in data.items():
        np.save(os.path.join(seg_dir, f"{name}.npy"), np.asarray(arr))

    meta_out = {"frames": int(n), "timestamp": time.time()}
    if meta:
        meta_out.update(meta)
    with open(os.path.join(seg_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    _refresh_frame_index(bucket)
    return seg_dir


def _read_meta(seg_dir: str) -> dict:
    path = os.path.join(seg_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _refresh_frame_index(bucket: str) -> int:
    segs = sorted(glob.glob(os.path.join(bucket, "segment_*")))
    total = sum(int(_read_meta(s).get("frames", 0)) for s in segs)
    with open(os.path.join(bucket, "frames.json"), "w") as f:
        json.dump({"total_frames": total, "segments": len(segs)}, f, indent=2)
    return total


def count_human_frames(level: str, agent_type: str | None = None, ego_size: int | None = None) -> int:
    base = os.path.join(RESULTS_ROOT, level_name_of(level))
    if not os.path.isdir(base):
        return 0
    buckets = [agent_type] if agent_type else [
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    ]
    total = 0
    for bucket in buckets:
        for seg in glob.glob(os.path.join(base, bucket, "segment_*")):
            meta = _read_meta(seg)
            if ego_size is not None and meta.get("ego_size") != ego_size:
                continue
            total += int(meta.get("frames", 0))
    return total


def load_human_dataset(
    level: str,
    fields: list[str] | None = None,
    agent_types: list[str] | None = None,
    ego_size: int | None = None,
) -> dict:
    """Concatenate recorded segments for a level.

    ``ego_size`` (when given) keeps only segments whose stored ego crop matches,
    so a BC run never mixes 32x32 and full-map demos. Returns ``{field: ndarray}``.
    """
    base = os.path.join(RESULTS_ROOT, level_name_of(level))
    if not os.path.isdir(base):
        return {}
    fields = fields or HUMAN_FIELDS
    if agent_types is None:
        agent_types = sorted(
            d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
        )

    acc: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    for bucket in agent_types:
        for seg in sorted(glob.glob(os.path.join(base, bucket, "segment_*"))):
            if ego_size is not None and _read_meta(seg).get("ego_size") != ego_size:
                continue
            for f in fields:
                path = os.path.join(seg, f"{f}.npy")
                if os.path.exists(path):
                    acc[f].append(np.load(path))

    return {f: np.concatenate(v, axis=0) for f, v in acc.items() if v}


# --------------------------------------------------------------------------- #
# BC minibatch source
# --------------------------------------------------------------------------- #
class BCBatcher:
    """Holds ego human transitions on CPU and yields device minibatches.

    ``obs_spatial`` is kept as uint8 (the network's ``cast_obs`` handles the
    float conversion, exactly as in RL rollouts).
    """

    def __init__(self, spatial: np.ndarray, internal: np.ndarray, actions: np.ndarray):
        self.spatial = torch.as_tensor(np.asarray(spatial))          # uint8 [N,C,S,S]
        self.internal = torch.as_tensor(np.asarray(internal), dtype=torch.float32)
        self.actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        self.n = int(self.spatial.shape[0])
        self.spatial_shape = tuple(self.spatial.shape[1:])

    def sample(self, batch_size: int, device):
        idx = torch.randint(0, self.n, (min(batch_size, self.n),))
        return (
            self.spatial[idx].to(device),
            self.internal[idx].to(device),
            self.actions[idx].to(device),
        )


def load_bc_batcher(level: str, expected_spatial_shape=None, ego_size: int | None = None):
    """Build a :class:`BCBatcher` from a level's human data, or return None.

    ``expected_spatial_shape`` (the trainer's per-agent ``map_spatial_shape``) is
    checked so a shape mismatch (e.g. wrong ego size or level) is caught early
    with a clear message rather than a cryptic runtime error.
    """
    data = load_human_dataset(level, fields=BC_FIELDS, ego_size=ego_size)
    if not data or "obs_spatial" not in data or len(data["obs_spatial"]) == 0:
        return None
    batcher = BCBatcher(data["obs_spatial"], data["obs_internal"], data["actions_move"])
    if expected_spatial_shape is not None and tuple(batcher.spatial_shape) != tuple(expected_spatial_shape):
        print(
            f"[human-bc] recorded obs shape {batcher.spatial_shape} != model input "
            f"{tuple(expected_spatial_shape)}; skipping BC. (Record with matching "
            f"--ego-size / config.)"
        )
        return None
    return batcher
