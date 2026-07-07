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
collected frames are never rewritten, so repeated sessions accumulate toward the
pre-training target (the recorder collects a per-agent-type quota per session).
``obs_spatial``/``next_obs_spatial`` are stored as uint8 (the env's compressed
obs); the networks cast on read (``cast_obs``). Per-session team episodic-return
summaries are appended to ``<level>/human_returns.jsonl`` (see
``append_return_stats``).
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
# rewards is the CONTROLLED agent's own reward; team_rewards is the summed-over-
# agents cooperative reward (both kept: individual for credit, team for the shared
# objective). terminated and truncated are recorded SEPARATELY (truncation is a
# time-limit that should bootstrap the value; termination is a real end); dones
# (= terminated | truncated) is retained for backward compatibility.
HUMAN_FIELDS = [
    "obs_spatial",
    "obs_internal",
    "actions_move",
    "actions_raw",
    "radio",
    "rewards",
    "team_rewards",
    "dones",
    "terminated",
    "truncated",
    "next_obs_spatial",
    "next_obs_internal",
    "controlled_agent",
]

# Minimal field set a decentralized learner needs for a BC pass. ``radio`` is the
# human's emitted radio action, cloned by the radio head when --use-radio is set.
BC_FIELDS = ["obs_spatial", "obs_internal", "actions_move", "radio"]

# Field set a decentralized learner needs for an offline *Q* pass (DQN CQL):
# the full single-agent ego transition. ``team_rewards`` (summed-over-agents
# cooperative reward) matches the online DQN target (which bootstraps against the
# summed reward), so offline and online backups are directly comparable;
# ``terminated`` (a real episode end, not a time-limit truncation) is the correct
# bootstrap mask. Older segments may predate these fields, so the loader falls
# back to ``rewards`` / ``dones`` respectively.
OFFLINE_FIELDS = [
    "obs_spatial", "obs_internal", "actions_move", "radio",
    "team_rewards", "rewards", "terminated", "dones",
    "next_obs_spatial", "next_obs_internal",
]


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

    # Anti-aliasing sanity check: catch the class of recorder bug where the ego
    # observation was appended as a VIEW into the engine's reused buffer, so every
    # frame collapses to the episode's final observation. A healthy multi-frame
    # segment has many distinct observations; ~1 distinct row means aliasing.
    obs = data.get("obs_internal")
    if obs is not None and n > 4:
        arr = np.asarray(obs).reshape(n, -1)
        if len(np.unique(arr, axis=0)) <= 1:
            raise ValueError(
                f"Refusing to write an aliased human segment: all {n} frames of "
                f"obs_internal are identical (the recorder appended a view into the "
                f"engine's reused obs buffer instead of a copy). Snapshot the obs "
                f"with .clone() before appending."
            )

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


def append_return_stats(level: str, stats: dict) -> str:
    """Append one episodic-return summary to the level's returns log (JSON lines).

    Written by the recorder after each session so team performance can be tracked
    over the record -> train -> record loop. Returns the log path.
    """
    base = os.path.join(RESULTS_ROOT, level_name_of(level))
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "human_returns.jsonl")
    entry = {"timestamp": time.time(), **stats}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return path


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


def count_human_frames(level: str, agent_type: str | None = None, ego_size: int | None = None,
                       checkpoint: str | None = None) -> int:
    """Total recorded frames for a level, optionally restricted to one bucket,
    a matching ``ego_size``, and/or a matching ``checkpoint`` tag (the teammate
    checkpoint the demos were recorded against). ``checkpoint`` lets the recorder
    count only the demos gathered for the *current* policy checkpoint."""
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
            if checkpoint is not None and meta.get("checkpoint") != checkpoint:
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

    def __init__(self, spatial: np.ndarray, internal: np.ndarray, actions: np.ndarray,
                 radio: np.ndarray | None = None):
        self.spatial = torch.as_tensor(np.asarray(spatial))          # uint8 [N,C,S,S]
        self.internal = torch.as_tensor(np.asarray(internal), dtype=torch.float32)
        self.actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        # Optional radio BC target (the human's emitted radio action per frame).
        self.radio = None if radio is None else torch.as_tensor(np.asarray(radio), dtype=torch.long)
        self.n = int(self.spatial.shape[0])
        self.spatial_shape = tuple(self.spatial.shape[1:])

    def sample(self, batch_size: int, device):
        """Return (spatial, internal, move, radio); radio is None when unavailable."""
        idx = torch.randint(0, self.n, (min(batch_size, self.n),))
        radio = None if self.radio is None else self.radio[idx].to(device)
        return (
            self.spatial[idx].to(device),
            self.internal[idx].to(device),
            self.actions[idx].to(device),
            radio,
        )


def load_bc_batcher(level: str, expected_spatial_shape=None, ego_size: int | None = None,
                    expected_internal_dim: int | None = None):
    """Build a :class:`BCBatcher` from a level's human data, or return None.

    ``expected_spatial_shape`` (the trainer's per-agent ``map_spatial_shape``) and
    ``expected_internal_dim`` (the trainer's ``agent_internal_dim``) are checked so
    a shape mismatch -- e.g. wrong ego size, or demos recorded before the internal
    vector gained the relative-entity block -- is caught early with a clear message
    rather than a cryptic runtime error.
    """
    data = load_human_dataset(level, fields=BC_FIELDS, ego_size=ego_size)
    if not data or "obs_spatial" not in data or len(data["obs_spatial"]) == 0:
        return None
    n = len(data["obs_spatial"])
    # Only clone radio when a per-frame radio target aligns with the obs.
    radio = data.get("radio")
    if radio is not None and len(radio) != n:
        radio = None
    batcher = BCBatcher(data["obs_spatial"], data["obs_internal"], data["actions_move"], radio=radio)
    if expected_spatial_shape is not None and tuple(batcher.spatial_shape) != tuple(expected_spatial_shape):
        print(
            f"[human-bc] recorded obs shape {batcher.spatial_shape} != model input "
            f"{tuple(expected_spatial_shape)}; skipping BC. (Record with matching "
            f"--ego-size / config.)"
        )
        return None
    internal_dim = int(batcher.internal.shape[-1])
    if expected_internal_dim is not None and internal_dim != int(expected_internal_dim):
        print(
            f"[human-bc] recorded internal dim {internal_dim} != model input "
            f"{int(expected_internal_dim)}; skipping BC. (Re-record: the internal "
            f"vector changed, e.g. added the relative-entity block.)"
        )
        return None
    return batcher


def load_per_agent_bc_batchers(level: str, n_agents: int, expected_spatial_shape=None,
                               ego_size: int | None = None, expected_internal_dim: int | None = None):
    """Build one :class:`BCBatcher` per controlled-agent index for per-agent-nets
    training, keyed by agent index. Splits the level's demos on the recorded
    ``controlled_agent`` field so each agent's own net clones only its own demos.
    Returns ``{agent_idx: BCBatcher}`` (agents with no demos are omitted); empty
    dict when there is no data or a shape/internal-dim mismatch (same checks as
    :func:`load_bc_batcher`)."""
    data = load_human_dataset(level, fields=BC_FIELDS + ["controlled_agent"], ego_size=ego_size)
    if not data or "obs_spatial" not in data or len(data["obs_spatial"]) == 0:
        return {}
    n = len(data["obs_spatial"])
    ca = data.get("controlled_agent")
    if ca is None or len(ca) != n:
        print(f"[human-bc] per-agent-nets needs the 'controlled_agent' field but it "
              f"is missing/misaligned for '{level}'; falling back to pooled BC.")
        return {}
    ca = np.asarray(ca).reshape(-1).astype(int)
    radio_all = data.get("radio")
    has_radio = radio_all is not None and len(radio_all) == n

    out: dict[int, BCBatcher] = {}
    for a in range(n_agents):
        idx = np.where(ca == a)[0]
        if idx.size == 0:
            continue
        radio_a = radio_all[idx] if has_radio else None
        batcher = BCBatcher(data["obs_spatial"][idx], data["obs_internal"][idx],
                            data["actions_move"][idx], radio=radio_a)
        if expected_spatial_shape is not None and tuple(batcher.spatial_shape) != tuple(expected_spatial_shape):
            print(f"[human-bc] recorded obs shape {batcher.spatial_shape} != model input "
                  f"{tuple(expected_spatial_shape)}; skipping per-agent BC.")
            return {}
        internal_dim = int(batcher.internal.shape[-1])
        if expected_internal_dim is not None and internal_dim != int(expected_internal_dim):
            print(f"[human-bc] recorded internal dim {internal_dim} != model input "
                  f"{int(expected_internal_dim)}; skipping per-agent BC.")
            return {}
        out[a] = batcher
    return out


# --------------------------------------------------------------------------- #
# Stale-demo guard
# --------------------------------------------------------------------------- #
# The recorded internal dim can MATCH the model (66 == 66) yet the demos still be
# unusable: if they were recorded against an older engine build, the entire
# relative-entity block (the internal-vector tail that encodes the discovered/
# shared survivor & teammate positions the policy navigates by) is ALL ZERO in
# every demo frame even though the current env populates it. Training on that is
# worse than useless: the offline action is not a function of the (goal-less)
# observation, so BC injects a near-random gradient into the shared encoder (and
# the offline-Q obs is off-distribution).
#
# The guard identifies the relative-entity region as the internal columns PAST the
# always-on base features (position/battery/view-range/... which the env populates
# every step), then flags the demos as stale only if they populate NONE of that
# region while the env does. This deliberately tolerates sparse coverage: a fresh
# recording may leave some entity slots empty (the human never encountered that
# entity), so a per-column "any missing" test would false-positive -- what marks a
# stale recording is the WHOLE entity region being dead.

def _env_internal_population_frac(env, num_envs, n_agents, internal_dim, device,
                                  probe_steps=200):
    """Per-column fraction of agent-steps in which the live env populates each
    internal-vector column, over a short random rollout. Saves/restores the
    torch/numpy/python RNG and resets the env afterward, so it does not perturb
    training determinism any more than the existing post-chunk eval does."""
    import random

    rng_torch = torch.get_rng_state()
    rng_np = np.random.get_state()
    rng_py = random.getstate()
    counts = np.zeros(int(internal_dim), dtype=np.float64)
    total = 0
    try:
        env.reset()
        for _ in range(probe_steps):
            move = np.random.uniform(-1.0, 1.0, size=(num_envs, n_agents, 2)).astype(np.float32)
            radio = np.random.randint(0, n_agents, size=(num_envs, n_agents)).astype(np.int32)
            obs, _, terminations, truncations, _ = env.step(move, radio)
            it = obs["internal"].reshape(-1, int(internal_dim)).detach().cpu().numpy()
            counts += (it != 0).sum(axis=0)
            total += it.shape[0]
            if bool(terminations.any()) or bool(truncations.any()):
                env.reset()
    finally:
        torch.set_rng_state(rng_torch)
        np.random.set_state(rng_np)
        random.setstate(rng_py)
        env.reset()
    return counts / total if total else counts


def assert_demos_match_env(env, demo_internal, num_envs, n_agents, device, *,
                           label="human-bc", probe_steps=200, live_threshold=0.05,
                           always_on=0.9):
    """Raise SystemExit if the demos are stale relative to the current env's
    observation: the whole relative-entity region of the internal vector is zero
    across every demo frame while the live env populates it. ``demo_internal`` is
    the [N, D] recorded internal array/tensor. Returns quietly otherwise (including
    when the entity region is only sparsely covered -- that is expected)."""
    demo = np.asarray(demo_internal)
    demo_nz = set(int(c) for c in np.where((demo != 0).any(axis=0))[0])
    internal_dim = demo.shape[-1]
    live_frac = _env_internal_population_frac(
        env, num_envs, n_agents, internal_dim, device, probe_steps=probe_steps,
    )
    # The relative-entity region is everything past the last always-on base feature
    # (position/battery/... which the env fills every step). Conditionally-zero base
    # fields like deployment_remaining sit at low indices, so they never fall in it.
    base = np.where(live_frac > always_on)[0]
    entity_start = int(base.max()) + 1 if base.size else 0
    entity_cols = {int(c) for c in np.where(live_frac > live_threshold)[0] if c >= entity_start}
    if entity_cols and demo_nz.isdisjoint(entity_cols):
        raise SystemExit(
            f"[{label}] STALE DEMOS: the env populates the relative-entity block "
            f"(internal columns {sorted(entity_cols)}) but EVERY one is zero across all "
            f"recorded demo frames. These demos were recorded against an older observation "
            f"build (before that block was populated); the offline term cannot see the "
            f"goal/entity information the policy uses, so it would corrupt training rather "
            f"than help. Re-record demos with the current engine (human_dataset.py) -- and "
            f"delete the old segment_* dirs first so stale frames are not mixed in -- then "
            f"re-run."
        )


# --------------------------------------------------------------------------- #
# Offline-transition minibatch source (DQN offline-Q / CQL)
# --------------------------------------------------------------------------- #
class TransitionBatcher:
    """Holds full ego human *transitions* on CPU and yields device minibatches.

    Unlike :class:`BCBatcher` (obs -> action only), this keeps the reward, next
    observation and terminal flag needed for an offline Bellman backup. Both
    ``spatial`` and ``next_spatial`` stay uint8 (the network's ``cast_obs``
    handles the float conversion, exactly as in RL rollouts). ``reward`` is the
    summed-over-agents team reward (matching the online DQN target) and ``done``
    is the true-termination mask (truncations still bootstrap).
    """

    def __init__(self, spatial, internal, next_spatial, next_internal,
                 actions, reward, done, radio=None):
        self.spatial = torch.as_tensor(np.asarray(spatial))                       # uint8 [N,C,S,S]
        self.internal = torch.as_tensor(np.asarray(internal), dtype=torch.float32)
        self.next_spatial = torch.as_tensor(np.asarray(next_spatial))             # uint8 [N,C,S,S]
        self.next_internal = torch.as_tensor(np.asarray(next_internal), dtype=torch.float32)
        self.actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        # reward/done kept as [N,1] float columns (matches the DQN buffer layout).
        self.reward = torch.as_tensor(np.asarray(reward), dtype=torch.float32).reshape(-1, 1)
        self.done = torch.as_tensor(np.asarray(done), dtype=torch.float32).reshape(-1, 1)
        self.radio = None if radio is None else torch.as_tensor(np.asarray(radio), dtype=torch.long)
        self.n = int(self.spatial.shape[0])
        self.spatial_shape = tuple(self.spatial.shape[1:])

    def sample(self, batch_size: int, device):
        """Return (spatial, internal, next_spatial, next_internal, move, radio,
        reward[B,1], done[B,1]); radio is None when unavailable."""
        idx = torch.randint(0, self.n, (min(batch_size, self.n),))
        radio = None if self.radio is None else self.radio[idx].to(device)
        return (
            self.spatial[idx].to(device),
            self.internal[idx].to(device),
            self.next_spatial[idx].to(device),
            self.next_internal[idx].to(device),
            self.actions[idx].to(device),
            radio,
            self.reward[idx].to(device),
            self.done[idx].to(device),
        )


def load_transition_batcher(level: str, expected_spatial_shape=None, ego_size: int | None = None,
                            expected_internal_dim: int | None = None):
    """Build a :class:`TransitionBatcher` from a level's human data, or return None.

    Same shape/internal-dim validation as :func:`load_bc_batcher`. Prefers the
    ``team_rewards`` / ``terminated`` fields, falling back to ``rewards`` /
    ``dones`` for segments recorded before those were stored.
    """
    data = load_human_dataset(level, fields=OFFLINE_FIELDS, ego_size=ego_size)
    if (not data or "obs_spatial" not in data or len(data["obs_spatial"]) == 0
            or "next_obs_spatial" not in data):
        return None
    n = len(data["obs_spatial"])

    reward = data.get("team_rewards")
    if reward is None or len(reward) != n:
        reward = data.get("rewards")
    done = data.get("terminated")
    if done is None or len(done) != n:
        done = data.get("dones")
    if reward is None or done is None:
        print(f"[human-cql] human data for '{level}' lacks reward/terminal fields; skipping offline-Q.")
        return None

    radio = data.get("radio")
    if radio is not None and len(radio) != n:
        radio = None

    batcher = TransitionBatcher(
        data["obs_spatial"], data["obs_internal"],
        data["next_obs_spatial"], data["next_obs_internal"],
        data["actions_move"], reward, done, radio=radio,
    )
    if expected_spatial_shape is not None and tuple(batcher.spatial_shape) != tuple(expected_spatial_shape):
        print(
            f"[human-cql] recorded obs shape {batcher.spatial_shape} != model input "
            f"{tuple(expected_spatial_shape)}; skipping offline-Q. (Record with matching "
            f"--ego-size / config.)"
        )
        return None
    internal_dim = int(batcher.internal.shape[-1])
    if expected_internal_dim is not None and internal_dim != int(expected_internal_dim):
        print(
            f"[human-cql] recorded internal dim {internal_dim} != model input "
            f"{int(expected_internal_dim)}; skipping offline-Q. (Re-record: the internal "
            f"vector changed, e.g. added the relative-entity block.)"
        )
        return None
    return batcher
