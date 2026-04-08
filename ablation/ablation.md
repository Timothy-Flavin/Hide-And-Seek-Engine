Here is an organized ablation study summarizing the performance improvements across the three major development phases, based on our iteration and commit history. 

### Ablation Study: C++ Engine Optimization for SARBatchedGridEnv

This study tracks the impact of two critical optimizations on the step-time latency and parallel scaling (FPS) of the simulated environment.

#### 1. Baseline speedtest_results_main.png (Full Map Traversal)
*   **Architecture:** The C++ `step()` loop computed movement, raycasted line-of-sight, and resolved agent-PoI interactions. Then, `fill_torch_obs` and `fill_torch_state` looped over the entire `map_area` to check and push masking updates into the Pinned PyTorch spatial memory. At the end of the step, C++ instantiated new Numpy arrays for `rewards`, `terminated`, and `truncated`, returning a `py::tuple` which Python converted to PyTorch tensors.
*   **Bottleneck:** Memory bandwith limitations. Looping `map_area` for every environment and agent frame-over-frame resulted in cache thrashing. `fill_torch_obs` alone accounted for ~60-70% of the total C++ runtime.
*   **Performance Profile:** 
    *   ~22,000 FPS (at 256 Envs, 10 Agents)
    *   ~36,000 FPS (at 4 Envs, 10 Agents)

#### 2. Optimization 1: Inline Observation Updates speedtest_results_before_pin.png
*   **Intervention:** Eliminated the exhaustive loop in `fill_torch_obs/state`. Because the underlying Pinned PyTorch arrays are persistent across steps, we moved the memory pointer updates (`spat_base[...] = 1.0f`) directly into the `resolve_local_interactions` and `execute_radio` subroutines. Writes now *only* trigger exactly when a tile flips from undiscovered to discovered. 
*   **Impact:** The `fill_torch_obs` time collapsed from ~3.28s down to effectively zero. C++ execution scaled beautifully, no longer constrained by map size.
*   **Performance Profile:**
    *   ~380,000+ FPS (at 256 Envs, 10 Agents) — a **~17x improvement**.
    *   *New Bottleneck:* Profiling revealed that `py_tensor_conversion` (converting the returned Numpy `py::tuple` to PyTorch tensors) now accounted for an unacceptably high ~30-33% of the remaining total runtime.

#### 3. Optimization 2: Zero-Copy Void Returns speedtest_results.png
*   **Intervention:** Pre-allocated `rewards`, `terminated` (`torch.bool`), and `truncated` tensors during Python `__init__`. Passed their raw memory addresses (`.data_ptr()`) directly into the C++ `BatchedEnvironment` constructor. The C++ `step()` method now writes individual reward values and flags straight into these memory addresses and returns `void`. 
*   **Impact:** Completely bypassed Python object creation and deep copying. Python overhead for tensor conversion dropped from **~0.260 seconds** per 10,000 steps to just **0.005 seconds**.
*   **Performance Profile:**
    *   ~55,000 FPS (at 4 Envs, 10 Agents) — a **~50% improvement** over Phase 1 for low batch-counts where Python overhead was most dominant. 
    *   Extremely stable scaling with almost negligible Python footprint; the workload is now heavily isolated to required C++ raycasting mathematics.

### Conclusion
By aggressively targeting memory iteration overlap (Inline Array Masking) and Python<->C++ data boundaries (Zero-Copy Pointers), the environment pipeline shifted from a poorly-scaling memory-bound architecture to a highly efficient compute-bound simulation capable of ~400k+ global steps per second.