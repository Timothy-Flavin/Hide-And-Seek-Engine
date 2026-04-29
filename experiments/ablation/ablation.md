### Ablation Study: C++ Engine Optimization for SARBatchedGridEnv

We conducted a performance ablation study comparing 5 distinct environment versions to evaluate our optimizations for step-time latency and parallel scaling (FPS).

#### 1. Baseline
*   **Architecture:** C++ internal state as a contiguous float array converted via `torch.to()`, using OpenMP threading.
*   **Reference:** `speedtest_tensor_state.png`

#### 2. Zero-copy Cache Aligned
*   **Architecture:** State is compressed to a dense cache-aligned `uint8` array per environment. A single sweep copies values into a pinned PyTorch tensor shared at initialization.
*   **Reference:** `speedtest_results_baseline.png`

#### 3. Diff Sweep
*   **Architecture:** The full state sweep is replaced by a "diff" sweep, where only updated variables are sent to the tensor representation.
*   **Reference:** `speedtest_results_before_pin.png`

#### 4. Static Shared Arrays
*   **Architecture:** Rewards, terminated, and truncated signals become static shared arrays instead of being instantiated per-step and packaged as Python tuples.
*   **Reference:** `speedtest_results_final.png`

#### 5. Gymnasium Async
*   **Architecture:** No C++ multi-threading; uses Gymnasium async vectorized environments. Note that while other versions scale to 256 parallel envs, Gymnasium caps out at 32 due to VRAM limits.
*   **Reference:** `speedtest_results_gymnasium.png`

#### 6. NUMA & False-Sharing Localization
*   **Architecture:** Addresses deep performance bottlenecks on high-core-count processors (e.g., AMD EPYC). Memory buffers are spaced out using 256-byte cache alignment to defeat aggressive spatial prefetchers. Pinned tensor memory is initialized inside the parallel OpenMP loop to leverage OS first-touch policies, locking memory pages directly to the localized NUMA nodes of the threads utilizing them. Additionally, np.random was taking roughly 2/3 the runtime and was replaced with:
```
rng = np.random.default_rng()

def _random_local_actions(num_envs: int, n_agents: int) -> tuple[np.ndarray, np.ndarray]:
    # Generating directly into the correct dtype saves a copy/cast cycle
    move_actions = rng.random((num_envs, n_agents, 2), dtype=np.float32)
    # Scale and shift: [0, 1) -> [-1, 1)
    move_actions = move_actions * 2.0 - 1.0
    radio_actions = rng.integers(0, n_agents, size=(num_envs, n_agents), dtype=np.int32)
    return move_actions, radio_actions
```
Finally, output arrays (rewards, terminated, truncated) use padded, thread-local buffers to completely eliminate dense-array L1/L2 cache false-sharing invalidation storms leading to 2.8M fps on the laptop and 6M on the Epic CPUs as opposed to 1.75M and 1.4M respectively in the single agent cases before. Additionally, the removal of all memory overlap allows 12M+ fps on the Epic server and 5M on the laptop with 1024 envs. 
*   **Reference:** `speedtest_results_localmem.png`
