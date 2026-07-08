"""Post-hoc 'remaining squares' terminal reward for recorded human demos.

Companion to the C++ env change that, when an episode ends because the mission is
COMPLETE (all entities saved -- not battery death or timeout), credits the still-
undiscovered squares' dense reward split equally across agents. This script applies
the SAME correction retroactively to demos recorded before that change.

Per recorded segment we split the concatenated frames into episodes (a fresh env
reset bumps the controlled agent's deployment_remaining back to its max battery, so
any increase in that column marks a new episode's first real frame). For each
episode that is:

    * terminal (its last frame has terminated == 1), AND
    * shorter than max_frames (250, i.e. not a timeout/truncation), AND
    * shorter than the longest agent battery (i.e. not an all-battery-death),

all entities were necessarily found and saved, so the reward still owed is

    remaining = num_squares*0.05 + n_found*2 + n_saved*20 - cumulative_team_reward

(with n_found == n_saved == num_entities for a completed mission). This equals the
env's undiscovered_remaining*0.05 terminal bonus. We add ``remaining`` to the
terminal frame's ``team_rewards`` and ``remaining / n_agents`` to its ``rewards``
(the controlled agent's equal share), matching the env's equal split.

Idempotent: a processed segment gets ``remaining_squares_applied: true`` in its
meta.json and its original team_rewards/rewards are backed up to ``*.orig.npy``;
re-running skips it. Dry-run by default; pass --apply to write.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np

from RL.human_data import RESULTS_ROOT, level_name_of

REWARD_NEW_TILE = 0.05
REWARD_FOUND = 2.0
REWARD_SAVED = 20.0
MAX_FRAMES = 250
DEP_COL = 4  # controlled agent's deployment_remaining in the internal vector


def _level_params(level_dir: str):
    """(num_squares, num_entities, n_agents, longest_battery) for a level dir."""
    import json as _json
    with open(os.path.join(level_dir, "agents.json")) as f:
        agents = _json.load(f)
    with open(os.path.join(level_dir, "survivors.json")) as f:
        survivors = _json.load(f)
    from PIL import Image
    w, h = Image.open(os.path.join(level_dir, "level.png")).size
    longest_battery = max(float(a.get("battery", 0.0)) for a in agents.values())
    return w * h, len(survivors), len(agents), longest_battery


def _episode_start(dep: np.ndarray, team: np.ndarray, term: np.ndarray,
                   trunc: np.ndarray, f: int) -> int:
    """Start index of the episode that ends at terminal frame ``f``.

    The episode begins right after the previous episode's END. A previous episode
    ends at index k < f when it is:
      * a real done (``term[k]`` or ``trunc[k]`` == 1), or
      * a controlled-agent battery death: its deployment_remaining hit ~0
        (``dep[k] <= 1``) on a frame that carries only a small movement/tile
        reward. The stale-obs first frame of a FRESH episode ALSO shows dep ~0
        (it copies the previous agent's dead battery), but it carries the large
        initial field-of-view discovery reward -- so a low-dep frame with a big
        reward is a start, NOT a death boundary, and is kept in the episode.
    Two independent lower bounds on the start are combined by taking the LATEST
    (tightest) of the two, since either signal alone is incomplete:
      * previous-END boundary: the nearest real done or battery-death before ``f``
        (catches back-to-back quick missions whose small battery drain leaves no
        big deployment reset between them);
      * deployment RESET: the nearest large jump UP in ``dep`` before ``f`` (catches
        a first episode with no preceding done/death, where no END boundary exists).
    Returns 0 when neither precedes ``f``.
    """
    end_boundary = -1
    for k in range(f - 1, -1, -1):
        if term[k] > 0.5 or trunc[k] > 0.5 or (dep[k] <= 1.0 and team[k] < 1.0):
            end_boundary = k
            break
    reset_start = 0
    for k in range(f, 0, -1):
        if dep[k] - dep[k - 1] > 50.0:
            # k is the first real step of a fresh episode; the stale-obs frame k-1
            # (large initial-FOV reward) is that episode's true first frame.
            prev = k - 1
            reset_start = (k if (term[prev] > 0.5 or trunc[prev] > 0.5) else prev)
            break
    return max(end_boundary + 1, reset_start)


def process_segment(seg_dir: str, num_squares: int, num_entities: int, n_agents: int,
                    longest_battery: float, apply: bool) -> dict:
    """Return a summary dict; if ``apply``, write the corrected arrays + meta."""
    meta_path = os.path.join(seg_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    if meta.get("remaining_squares_applied"):
        return {"seg": seg_dir, "skipped": "already applied"}

    def _load(name):
        p = os.path.join(seg_dir, f"{name}.npy")
        return np.load(p) if os.path.exists(p) else None

    team = _load("team_rewards")
    rew = _load("rewards")
    term = _load("terminated")
    trunc = _load("truncated")
    obs_int = _load("obs_internal")
    if team is None or term is None or obs_int is None:
        return {"seg": seg_dir, "skipped": "missing team_rewards/terminated/obs_internal"}

    team = team.astype(np.float64).copy()
    rew = None if rew is None else rew.astype(np.float64).copy()
    term = np.asarray(term).astype(np.float64)
    trunc = (np.zeros_like(term) if trunc is None else np.asarray(trunc).astype(np.float64))
    dep = np.asarray(obs_int)[:, DEP_COL].astype(np.float64)

    theoretical = num_squares * REWARD_NEW_TILE + num_entities * (REWARD_FOUND + REWARD_SAVED)
    edits = []
    warnings = []
    for f in np.where(term > 0.5)[0]:
        f = int(f)
        # Mission-complete vs all-battery-death: `terminated` fires for BOTH
        # (all_saved || all_out_of_battery). The distinguisher is a completing save
        # on the terminal frame -- a mission ends the instant the last entity is
        # saved (+reward_saved), while a battery death carries only movement/tile
        # reward. This is boundary-independent, so it is the PRIMARY gate.
        mission_complete = team[f] >= REWARD_SAVED - 1e-6
        s = _episode_start(dep, team, term, trunc, f)
        length = f - s + 1
        # Length cross-check: a genuine early completion is shorter than both the
        # time limit and the longest battery (otherwise the longest-battery agent
        # would still be alive and the mission not yet ended by all-battery-death).
        length_ok = (length < MAX_FRAMES and length < longest_battery)
        if not mission_complete:
            continue
        if not length_ok:
            warnings.append(f"terminal@{f}: save on terminal but length {length} "
                            f">= longest_battery/max_steps; skipped")
            continue
        cumulative = float(team[s:f + 1].sum())
        # Validate the detected episode boundary before crediting (never corrupt on
        # a boundary error -- skip and warn instead). A completed mission found AND
        # saved every entity, so its episode reward must be at least
        # num_entities*(found+saved) (multiple saves can share one frame, so count
        # by reward magnitude, not frames), and cannot exceed the theoretical max
        # (which would mean the span bled into an adjacent episode).
        min_complete = num_entities * (REWARD_FOUND + REWARD_SAVED)
        remaining = theoretical - cumulative
        if cumulative > theoretical + 1e-6:
            warnings.append(f"ep[{s}:{f}] cumulative {cumulative:.2f} > theoretical "
                            f"{theoretical:.2f} (span bled into another episode); skipped")
            continue
        if cumulative < min_complete - 1e-6:
            warnings.append(f"ep[{s}:{f}] cumulative {cumulative:.2f} < {min_complete:.2f} "
                            f"(missing found/saved reward; start too late); skipped")
            continue
        if remaining <= 1e-6:
            continue                                   # fully explored; nothing owed
        edits.append({"start": int(s), "end": int(f), "len": int(length),
                      "cumulative": round(cumulative, 3), "remaining": round(remaining, 3)})
        if apply:
            team[f] += remaining
            if rew is not None:
                rew[f] += remaining / n_agents

    result = {"seg": seg_dir, "n_frames": int(len(team)),
              "n_terminal": int((term > 0.5).sum()),
              "n_terminal_complete": len(edits),
              "total_added": round(sum(x["remaining"] for x in edits), 3),
              "edits": edits, "warnings": warnings}
    if apply and edits:
        # Back up originals once, then overwrite in place, then flag in meta.
        for name, arr in (("team_rewards", team), ("rewards", rew)):
            if arr is None:
                continue
            src = os.path.join(seg_dir, f"{name}.npy")
            bak = os.path.join(seg_dir, f"{name}.orig.npy")
            if os.path.exists(src) and not os.path.exists(bak):
                shutil.copy2(src, bak)
            np.save(src, arr.astype(np.float32))
        meta["remaining_squares_applied"] = True
        meta["remaining_squares_added"] = result["total_added"]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=None,
                    help="Level dirs (default: every levels/* with a level.png).")
    ap.add_argument("--apply", action="store_true",
                    help="Write the corrected arrays (default: dry-run report only).")
    args = ap.parse_args()

    if args.levels:
        level_dirs = args.levels
    else:
        level_dirs = sorted(
            os.path.join("levels", d) for d in os.listdir("levels")
            if os.path.isdir(os.path.join("levels", d))
            and os.path.exists(os.path.join("levels", d, "level.png"))
        )

    grand_added = 0.0
    grand_eps = 0
    for level_dir in level_dirs:
        lname = level_name_of(level_dir)
        base = os.path.join(RESULTS_ROOT, lname)
        if not os.path.isdir(base):
            continue
        num_squares, num_entities, n_agents, longest_battery = _level_params(level_dir)
        buckets = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
        theoretical = num_squares * REWARD_NEW_TILE + num_entities * (REWARD_FOUND + REWARD_SAVED)
        print(f"\n=== {lname}: {num_squares} squares, {num_entities} entities, "
              f"{n_agents} agents, longest_battery={longest_battery:g}, "
              f"complete-episode target reward={theoretical:.2f} ===")
        for bucket in buckets:
            segs = sorted(glob.glob(os.path.join(base, bucket, "segment_*")))
            for seg in segs:
                r = process_segment(seg, num_squares, num_entities, n_agents,
                                    longest_battery, args.apply)
                if r.get("skipped"):
                    print(f"  [{bucket}/{os.path.basename(seg)}] skipped: {r['skipped']}")
                    continue
                grand_added += r["total_added"]
                grand_eps += r["n_terminal_complete"]
                if r["n_terminal_complete"] or r.get("warnings"):
                    print(f"  [{bucket}/{os.path.basename(seg)}] terminals={r['n_terminal']} "
                          f"mission-complete={r['n_terminal_complete']} added={r['total_added']:.2f}")
                    for ed in r["edits"]:
                        print(f"        ep[{ed['start']}:{ed['end']}] len={ed['len']} "
                              f"cumulative={ed['cumulative']:.2f} -> +{ed['remaining']:.2f}")
                    for wmsg in r.get("warnings", []):
                        print(f"        !! {wmsg}")

    mode = "APPLIED" if args.apply else "DRY-RUN (no files written; pass --apply to write)"
    print(f"\n{mode}: {grand_eps} mission-complete episodes, "
          f"total reward added = {grand_added:.2f}")


if __name__ == "__main__":
    main()
