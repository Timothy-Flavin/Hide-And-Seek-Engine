# Hide-And-Seek Engine (MARL POMDP Simulator)

| Example wilderness SAR | Coast Guard Monitoring | Neighborhood Watch | Warehouse Fire |
|---|---|---|---|
| <img src="saved_human_behavior/example/replay.gif?raw=true" alt="Example wilderness SAR" width="200"> | <img src="saved_human_behavior/island/replay.gif?raw=true" alt="Coast Guard Monitoring" width="200"> | <img src="saved_human_behavior/neighborhood/replay.gif?raw=true" alt="Neighborhood Watch" width="200"> | <img src="saved_human_behavior/warehouse/replay.gif?raw=true" alt="Warehouse Fire" width="200"> |


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
    seed=42,
)

obs, info = env.reset()

# Sample random continuous movement and discrete radio actions
move_actions = np.random.uniform(-1.0, 1.0, size=(128, env.config.n_agents, 2))
radio_actions = np.random.randint(0, env.config.n_agents, size=(128, env.config.n_agents))

next_obs, rewards, terminated, truncated, info = env.step(move_actions, radio_actions)

if env.requires_state:
    global_state = env.state()
```

Got it. That makes much more sense for a targeted communication protocol—using the agent's own ID as the logical silence/no-op is a clean way to handle it. 

Here is the corrected **Action Space** section for the README with the updated shapes and a revised code example to reflect the target-ID logic.

***

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

  * `obs["spatial"]`: Shape depends on the `mode` (see Observation Modes below). Contains local terrain, altitude, survivor locations, FOV masks, and agent layers.
  * `obs["internal"]`: Shape `[num_envs, n_agents, 6]`. Contains vector data: `[deploy_remaining, stuck, view_range, battery, y, x]`.

### State Space (Critic Input)

If `requires_state=True`, `env.state()` returns the global, unmasked environment state for centralized training (e.g., CTDE):

  * `state["spatial"]`: `[num_envs, C_global, H, W]`
  * `state["internal"]`: `[num_envs, flattened_agent_and_survivor_dim]`

### Observation Modes

The environment configuration is controlled by a 3x2 matrix of parameters during initialization: `mode` (`"centralized"`, `"decentralized"`, or `"no_obs"`) and `requires_state` (`True` or `False`).

| Mode | `requires_state` | `obs["spatial"]` Shape | Behavior / Use Case |
| :--- | :--- | :--- | :--- |
| `"centralized"` | `False` | `[E, C_local, H, W]` | Agents share a single, merged spatial observation tensor. Useful for joint-action policies. |
| `"centralized"` | `True` | `[E, C_local, H, W]` | Same as above, but `env.state()` is enabled to fetch the global CTDE state for critics. |
| `"decentralized"` | `False` | `[E, A, C_local, H, W]` | Each agent $A$ receives strictly its own local FOV observation. |
| `"decentralized"` | `True` | `[E, A, C_local, H, W]` | Standard CTDE setup. Decentralized execution for actors, global state available for the central critic. |
| `"no_obs"` | `False` | *Empty/None* | Blind agents. Useful for debugging or relying entirely on communication/internal states. |
| `"no_obs"` | `True` | *Empty/None* | Blind actors, but global state is still tracked and returned for debugging/value estimation or MDP Joint action learning. |

*(Note: `E` = num\_envs, `A` = n\_agents, `C_local` = local feature channels, `H` = height, `W` = width).*

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
