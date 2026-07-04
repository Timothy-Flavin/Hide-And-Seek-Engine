# Ego-Centric Observations

Opt-in feature that replaces the full `H x W` spatial map view with a fixed
`ego_size x ego_size` window **centered on each agent**. Enable it at env
construction:

```python
env = SARBatchedGridEnv(
    ..., mode="decentralized", ego_view=True, ego_size=32,
)
```

## What it changes

- **Spatial obs only.** The `internal` (per-agent vector) obs and the global
  `state` tensor (`requires_state=True`) are unchanged — the state stays global,
  which is what a centralized critic wants.
- **Shape.** The spatial obs tensor becomes per-agent in every obs mode:
  - decentralized: `(E, A, C, ego, ego)` (was `(E, A, C, H, W)`)
  - centralized:   `(E, A, C, ego, ego)` — each agent gets its own crop of the
    single shared observed map (was `(E, C, H, W)`).
  - `NO_OBS`: unaffected (no spatial obs).
- Out-of-bounds cells (agent near the map edge) are **zero-padded**.
- `ego_view=False` (default) is byte-for-byte the old behavior.

## Performance / memory model

The full-map fog-of-war accumulation still happens exactly as before, but into a
**reused internal C++ buffer** (`ego_full_obs`) instead of the shared tensor.
Each step, one crop copy per agent (`memcpy` per row, per channel) writes the
window into the shared pinned tensor. This is the single, unavoidable copy the
design accepts; no `new`/`malloc` happens on the hot path — every buffer is
pre-sized in the constructor and reused. Zero-copy is retained for the
`internal` and `state` tensors.

## Reading shapes in a runner (do this instead of hardcoding)

Do **not** assume `(C, H, W)`. The wrapper exposes:

| attribute                 | meaning                                                        |
|---------------------------|----------------------------------------------------------------|
| `env.map_spatial_shape`   | single-agent map slab `(C, H, W)` or `(C, ego, ego)` — feed to the CNN encoder |
| `env.obs_spatial_shape`   | full per-sample obs shape incl. any leading agent dim — use for replay/rollout buffers |
| `env.obs_is_per_agent`    | `True` if the spatial obs carries a leading agent dimension    |

`cleanrl_dqn.py`, `cleanrl_ppo.py`, `cleanrl_sac.py` were updated to derive
shapes from these attributes and to accept `--ego-view` / `--ego-size` flags,
e.g.:

```bash
python RL/cleanrl_dqn.py --no-centralized --ego-view --ego-size 32
```

## Caveat: centralized + ego

Ego output is inherently per-agent (there is no single center for a shared map),
so ego mode should be run through the **decentralized** code path
(`--no-centralized`). The centralized network heads flatten a single
`(C, H, W)` map and do not match the per-agent `(A, C, ego, ego)` ego tensor. If
you want an ego-centric centralized critic, feed the still-global `state` tensor
(`requires_state=True`) to the critic and the per-agent ego crops to the actors.
