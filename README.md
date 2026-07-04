# Hide-And-Seek Engine (MARL POMDP Simulator)

| Example wilderness SAR | Coast Guard Monitoring | Neighborhood Watch | Warehouse Fire |
|---|---|---|---|
| <img src="saved_human_behavior/example/replay.gif?raw=true" alt="Example wilderness SAR" width="200"> | <img src="saved_human_behavior/island/replay.gif?raw=true" alt="Coast Guard Monitoring" width="200"> | <img src="saved_human_behavior/neighborhood/replay.gif?raw=true" alt="Neighborhood Watch" width="200"> | <img src="saved_human_behavior/warehouse/replay.gif?raw=true" alt="Warehouse Fire" width="200"> |

## Cite this using
```
@misc{flavin2026highthroughputcomputeefficientpomdphideandseekengine,
      title={A High-Throughput Compute-Efficient POMDP Hide-And-Seek-Engine (HASE) for Multi-Agent Operations}, 
      author={Timothy Flavin and Sandip Sen},
      year={2026},
      eprint={2604.27162},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2604.27162}, 
}
```

A C++/PyBind11 environment engine for large-scale Multi-Agent Reinforcement Learning (MARL) in partially observable grid-worlds. Designed for high-throughput, batched simulation with PyTorch.

-----

## Core Usage & API

The primary entry point for PyTorch-based RL is the `SARBatchedGridEnv`. It natively returns batched tensors.

```python
from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
import numpy as np

env = SARBatchedGridEnv(
    num_envs=128,
    map_png="levels/test_level/level.png",
    tiles_json="levels/test_level/tiles.json",
    agents_json="levels/test_level/agents.json",
    survivors_json="levels/test_level/survivors.json",
    mode="centralized",
    requires_state=True,
    # Optional ego-centric observations:
    # ego_view=True, ego_size=32,
)

obs, info = env.reset()

# Sample random continuous movement and discrete radio actions
move_actions = np.random.uniform(-1.0, 1.0, size=(128, env.config.n_agents, 2))
radio_actions = np.random.randint(0, env.config.n_agents, size=(128, env.config.n_agents))

next_obs, rewards, terminated, truncated, info = env.step(move_actions, radio_actions)

if env.requires_state:
    global_state = env.get_state()
```

### Action Space

The action space is a hybrid continuous/discrete space defined per agent. When passing actions to `env.step(move_actions, radio_actions)`, you provide two tensors:

1.  **Movement (`move_actions`)**: A 2D continuous vector `[dy, dx]` bounded in `[-1.0, 1.0]`. 
2.  **Radio (`radio_actions`)**: A discrete integer from `0` to `n_agents`. Transmitting an agent's own ID acts as the no-op (no transmission).

**Shapes:**
* `move_actions`: `[num_envs, n_agents, 2]`
* `radio_actions`: `[num_envs, n_agents]`

**Example Action Tensor (1 env, 2 agents):**
```python
# Agent 0: Moves South-East [1.0, 1.0], transmits to Agent 1 (action = 1)
# Agent 1: Moves North [-1.0, 0.0], stays silent on radio (transmits own ID: action = 1)
move_actions = np.array([[[1.0, 1.0], [-1.0, 0.0]]], dtype=np.float32)
radio_actions = np.array([[1, 1]], dtype=np.int32)
```

### Observation Space (Actor Input)

The `obs` returned by `reset()` and `step()` is a dictionary separating spatial feature maps from internal vector states:

  * `obs["spatial"]`: Shape depends on the `mode` and `ego_view` (see Observation Modes below). The `C = n_tiles + 3 + n_agents` channels are: one-hot tile-type layers (`n_tiles`), altitude, survivor (person-of-interest) layer, observed/FOV mask, and one location layer per agent.
  * `obs["internal"]`: Shape `[num_envs, n_agents, 6]`. Per-agent vector data in order `[y, x, battery, view_range, deploy_remaining, stuck]`.

### State Space (Critic Input)

If `requires_state=True`, `env.get_state()` returns the global, unmasked environment state for centralized training (e.g., CTDE):

  * `state["spatial"]`: `[num_envs, C_global, H, W]`
  * `state["internal"]`: `[num_envs, flattened_agent_and_survivor_dim]`

### Observation Modes

The observation footprint is controlled by three initialization parameters: `mode` (`"centralized"`, `"decentralized"`, or `"no_obs"`), `requires_state` (`True`/`False`), and the ego-centric options `ego_view` (`bool`) and `ego_size` (`int`).

| Mode | `requires_state` | `obs["spatial"]` Shape | Behavior / Use Case |
| :--- | :--- | :--- | :--- |
| `"centralized"` | `False` | `[E, C, H, W]` | Agents share a single, merged fog-of-war observation tensor. Useful for joint-action policies. |
| `"centralized"` | `True` | `[E, C, H, W]` | Same as above, but `env.state()` is enabled to fetch the global CTDE state for critics. |
| `"decentralized"` | `False` | `[E, A, C, H, W]` | Each agent $A$ receives strictly its own local FOV observation. |
| `"decentralized"` | `True` | `[E, A, C, H, W]` | Standard CTDE setup. Decentralized execution for actors, global state available for the central critic. |
| `"no_obs"` | `False` | *Empty/None* | Blind agents. Useful for debugging or relying entirely on communication/internal states. |
| `"no_obs"` | `True` | *Empty/None* | Blind actors, but global state is still tracked and returned for debugging/value estimation or MDP Joint action learning. |

*(Note: `E` = num\_envs, `A` = n\_agents, `C = n_tiles + 3 + n_agents` = spatial feature channels, `H` = height, `W` = width).*

#### Ego-centric observations (`ego_view=True`)

By default the spatial observation spans the entire `H×W` map. Setting `ego_view=True` instead returns a fixed `ego_size × ego_size` window **centered on each agent** (out-of-bounds cells are zero-padded), so the input size is independent of the map size. This is an opt-in flag; the default behavior is byte-for-byte unchanged.

```python
env = SARBatchedGridEnv(..., mode="decentralized", ego_view=True, ego_size=32)
```

* Ego observations are always **per-agent** (even in `"centralized"` mode each agent gets its own crop of the shared observed map), so the shape is `[E, A, C, ego_size, ego_size]`.
* The global fog-of-war accumulation is kept in a reused internal C++ buffer; each step performs a single crop copy per agent into the shared pinned tensor (the internal (vector) and `state` tensors remain full zero-copy).
* Convenience attributes for building networks: `env.map_spatial_shape` (single-agent `(C, H, W)` or `(C, ego, ego)`), `env.obs_spatial_shape` (full per-sample shape incl. any agent dim), and `env.obs_is_per_agent`.

-----

## Installation

```bash
pip install -e .
```

Optional rendering/input dependencies:

```bash
pip install pygame pillow pettingzoo imageio
```

-----

## Benchmark Experiments (Reproducing the Results)

The benchmark trains **3 MARL algorithms** (PPO, DQN, SAC) under **4 observation/communication configurations** on **4 environments** for **5 seeds** each, at **5,000,000 environment frames** per run on a single GPU. That is `3 × 4 × 4 × 5 = 240` training runs, all launched by one script (`RL/run_experiments.sh`).

### Environments

Every level ships as an image + three JSON files (see *Level File Formats*). All four levels use **3 heterogeneous agents** and **2–3 survivors** ("persons of interest") with per-survivor rescue constraints.

| Level | Map (W×H) | Tile types | Agents (3) | Survivors |
| :--- | :--- | :--- | :--- | :--- |
| `test_level` | 32×32 | 5 — grass, water, mountain, peak, forest | human, robodog, drone | kid, adult (2) |
| `island_level` | 64×64 | 6 — deep_ocean, plains, dense_jungle, grasslands, mountains, volcano_peak | coast_guard_cutter, surveillance_drone, patrol_helicopter | smuggler_vessel, cargo_drop_site, clandestine_port (3) |
| `warehouse_level` | 64×64 | 6 — exterior_ground, warehouse_floor, shrubbery, wall, road, fire | surveillance_drone, security_guard, firefighter | fleeing_arsonist, trapped_worker, hazardous_materials (3) |
| `neighborhood_level` | 128×128 | 4 — grass, house, woods, street | watch_drone, patrol_volunteer, police_officer | suspicious_prowler, lost_pet, unfamiliar_vehicle (3) |

**Agent heterogeneity.** Each environment mixes an aerial agent (flying, wide view, fast, terrain-independent), a ground/amphibious agent, and a mobility-restricted ground agent. Agents differ in `base_speed`, `base_view`, `battery`, altitude limits, and per-terrain speed multipliers. Movement is dynamically gated by tile support (walk/fly/swim) and altitude, and view range scales with altitude. For example, in `test_level`: **human** (walk + swim, view 5, battery 150), **robodog** (walk only, view 3, battery 100), **drone** (fly, view 7, battery 200).

**Survivors.** Each survivor defines `allowed_savers` (which agent types may rescue it) and whether it `moves`, which forces role assignment and cooperation.

### Reward & episode structure

Rewards are **cooperative** (shared / VDN-summed across agents):

| Event | Reward |
| :--- | :--- |
| Newly discovered tile (exploration) | `+0.05` |
| Survivor found (spotted within view) | `+2.0` |
| Survivor rescued (within rescue range by an allowed saver) | `+20.0` |

Episodes **truncate** at 250 frames and **terminate** early when all survivors are rescued or all agent batteries are depleted. On construction the engine applies a randomized burn-in so episodes begin from diverse states.

### Action space used by the benchmark

Movement is a continuous `[dy, dx]`, but the RL agents select from a **discrete 5-action** movement set (stay, N, S, E, W) mapped to `[dy, dx]`. The **radio** action is a discrete target `∈ {0 … n_agents−1}`; sending your own ID is the no-op, any other ID broadcasts that agent's discovered tiles/survivor knowledge to that peer.

### The four configurations

| Config | Runner flags | Description |
| :--- | :--- | :--- |
| Centralized | `--centralized` | Single shared fog-of-war map observation; per-agent policy heads. |
| Decentralized | `--no-centralized` | Per-agent local FOV over the full map (radio disabled). |
| Decentralized + Ego | `--no-centralized --ego-view --ego-size 32` | Per-agent 32×32 ego-centric crop (radio disabled). |
| Decentralized + Ego + Radio | `--no-centralized --ego-view --ego-size 32 --use-radio` | Ego crop plus a **trainable per-agent radio head** that learns which peer to share observations with. |

In the first three configurations the radio is fixed to the no-op; only the fourth learns a radio policy.

### Model configurations

All three algorithms share the same observation encoder (`MixedObservationEncoder`): a **3-layer CNN** over the spatial tensor (→ 128) plus a **2-layer MLP** over the internal vector (→ 32), fused to a **256-d** feature. Policies/critics are multi-agent with per-agent heads and **Value-Decomposition (VDN)** additive team value; the dueling/Q advantages are **mean-centered** for V/A identifiability. In the radio configuration each network emits a single widened head of size `(move + radio)` that is split zero-copy into the two action factors (one forward pass, no extra kernels).

Shared across algorithms: **5M frames**, **128 parallel envs**, **γ = 0.99**, run numbers = seeds **1–5**, GPU (`--cuda`, on by default).

| Hyperparameter | PPO | DQN | SAC (discrete) |
| :--- | :--- | :--- | :--- |
| Learning rate | 2.5e-4 (annealed) | 2.5e-4 | policy 3e-4 / Q 3e-4 |
| Rollout / update | `num_steps` 128, 4 minibatches, 4 epochs | `train_frequency` 128 | `train_frequency` 128 |
| Replay buffer | — (on-policy) | 100k | 100k |
| Batch size | 128×128 rollout | 1024 | 1024 |
| Discount γ / GAE λ | 0.99 / 0.95 | 0.99 | 0.99 |
| Target network | — | hard update every 2048 (τ = 1.0) | soft τ = 0.005, every update |
| Exploration | entropy coef 0.01 | ε: 1.0 → 0.05 over 20% of training | autotuned α (target-entropy scale 0.5) |
| Warmup | — | `learning_starts` 1000 | `learning_starts` 4096 |
| Other | clip 0.2, vf 0.5, max-grad-norm 0.5 | dueling + VDN | dueling + VDN |

### Running the benchmark

```bash
bash RL/run_experiments.sh
```

**Seed is the outermost loop:** the script runs every environment/model/config once for seed 1, then seed 2, and so on — so after the first pass you have exercised all 48 combinations and can confirm everything works before waiting on all 5 seeds. Runs continue on error and a failure summary is printed after each seed pass; per-level combined plots are regenerated after every seed.

Override any axis via environment variables, e.g. a single quick verification pass:

```bash
SEEDS=1 bash RL/run_experiments.sh          # one seed, all envs/models/configs
FRAMES=1000000 ALGS="ppo" bash RL/run_experiments.sh
```

Results are written per level to `experiments/results/<level_name>/` as
`<alg>_<config>_episodic_returns_run_<seed>.npy`, and one
`combined_learning_curves.png` per level (mean ± standard error over seeds) is produced by `RL/plot_results.py`.

-----

## Engine Architecture

  * **Memory Layout**: Utilizes Data-Oriented Design (DOD) with cache-aligned, bit-packed memory slabs for the environment state. A 256-byte stride padding is used to prevent false sharing across threads.
  * **Execution Model**: Relies on OpenMP for environment parallelization and NUMA-aware allocations. Serial C++ and GPU execution steps are prioritized to simplify debugging and maintain code maintainability, intentionally avoiding complex asynchronous thread-overlapping while preserving high throughput.
  * **PyTorch Integration**: Pinned memory tensors are mutated directly in C++ and instantly available to PyTorch/GPUs via DMA, avoiding Python-side serialization or GIL contention.

-----

## Level File Formats

Environment generation is data-driven via a mapping of images to JSON properties.

### `tiles.json`

Each tile definition:

  * `rgb`: `[r, g, b]` (for PNG mapping)
  * `altitude`: float
  * `supports_walking`, `supports_flying`, `supports_aquatic`: bool
  * `blocking`: bool

### `agents.json`

Each agent definition:

  * `flying`, `aqueous`, `walking`: bool
  * `altitude_min`, `altitude_max`, `base_speed`, `base_view`, `battery`, `deployment_delay`
  * `rgb`
  * `terrain_speed` (dict by tile name)
  * `start`: `[y, x]` (map coords or normalized)

### `survivors.json`

Each survivor definition:

  * `allowed_savers`: list of agent names
  * `moves`: bool
  * `rgb` (optional)
  * `start` (optional)

### `level.png`

A standard PNG image. Each pixel is matched to the nearest tile `rgb` defined in `tiles.json`.

-----

## Rendering & Visualization

  * **Global Map**: `env.render(env_idx=0)` (Undiscovered tiles dimmed, saved survivors white)
  * **Agent POV**: `env.render_pov(agent_idx=0, env_idx=0)` (Shows last-known positions and radio knowledge)
  * **Radio Trace**: `env.radio_render()` (Prints event logs for transmissions)

-----

## PettingZoo API Compatibility

A wrapper is provided for drop-in compatibility with standard multi-agent frameworks.

```python
from hide_and_seek_engine.env_wrapper import SARParallelPettingZooEnv

pz_env = SARParallelPettingZooEnv(
    map_png="levels/test_level/level.png",
    tiles_json="levels/test_level/tiles.json",
    agents_json="levels/test_level/agents.json",
    survivors_json="levels/test_level/survivors.json",
)

obs, infos = pz_env.reset()
actions = {
    agent: {"move": [0.0, 1.0], "radio": 1}
    for agent in pz_env.agents
}
obs, rewards, terminations, truncations, infos = pz_env.step(actions)
```

-----

## Utilities

### Test & Benchmark Suite

Run unit checks, stress tests, and FPS measurements across parallel batches:

```bash
python env_spec.py --steps 10000 --envs 1 2 4 8 --skip-render
```

### Human Data Recorder

Collect SARSA tuples by controlling an agent manually:

```bash
python human_runner.py --record
```

  * **Controls**: `W`, `A`, `S`, `D` (move), `1`, `2`, `3` (radio).
  * Data is written to `saved_human_behavior/<name>/` as raw `.npy` buffers alongside a `replay.gif`.
