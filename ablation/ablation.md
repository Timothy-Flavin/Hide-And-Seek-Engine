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
