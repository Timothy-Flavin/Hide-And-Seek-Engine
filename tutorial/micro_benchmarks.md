# Micro-Benchmarks and Profiling Log

This document records the empirical micro-benchmarks gathered during the C++ engine optimization process. It serves as a historical reference for why certain standard C++ or Loop Nest Optimization (LNO) techniques were either kept or explicitly reverted.

## 1. Baseline Performance
* **Throughput:** ~3,700,000 FPS (Frames Per Second)

## 2. Proxy Bits and Heap Allocations (`std::vector<bool>` $\rightarrow$ `uint8_t`)
* **Experiment:** The C++ standard library specializes `std::vector<bool>` to pack booleans as proxy bits, which introduces bitwise shifting latency during read access. We swapped pass-by-value `std::vector<bool>` parameters with `const std::vector<uint8_t>&`. Concurrently, we removed `std::string` initializations and concatenations from the `execute_radio` hot loop to halt heap fragmentation.
* **Measurement:** Throughput increased to **> 4,100,000 FPS**.
* **Verdict:** **Kept.** Byte-aligned read/writes and complete removal of dynamic heap allocations in hot loops yielded a massive ~400k FPS gain.

## 3. Memory Striding & Loop Nest Optimization (LNO)
* **Experiment:** PyTorch allocates its tensors in a channel-first layout `[envs, channels, height, width]`. Classical architecture advice dictates flipping the nested C++ loops so the innermost loop continuously iterates across the contiguous spatial memory (`width`). 
* **Measurement:** **Massive FPS Drop.**
* **Analysis:** Natively conforming to PyTorch's layout required looping sequentially through `n_tiles` (channels) externally and scanning the grid internally for every single channel. This forced the engine to redundantly evaluate the entire spatial map `n_tiles` times.
* **Verdict:** **Reverted.** The CPU pipeline bottleneck introduced by redundantly re-scanning the map completely destroyed the expected cache-locality benefits. We kept the "sparse scattered memory assignments" approach, as the L1 cache handles sparse writes much better than the CPU handles $10\times$ redundant logical evaluations.

## 4. Branchless Execution vs. CPU Branch Predictor
* **Experiment:** To mitigate conditional branch stalls, we implemented branchless assignments when filling the PyTorch observation tensors (e.g., casting boolean visibility checks to floats `(0.0 or 1.0)` and multiplying them directly into the spatial state tensor to avoid `if(has_seen)` blocks).
* **Measurement:** **Decreased FPS.**
* **Analysis:** The visibility matrix is overwhelmingly `false` early in the episode, and states are strictly monotonic (once a tile is seen, it stays seen). Modern CPU branch predictors achieve near-perfect efficiency in skipping the conditional block altogether. 
* **Verdict:** **Reverted.** Branchless execution forced the CPU to perform redundant floating-point multiplications and zero-writes to memory, actively burning memory bandwidth. Letting the CPU branch predictor skip the writes was empirically faster.

## 5. Localized Reward Accumulators (False Sharing Guards)
* **Experiment:** Attempted to isolate `individual_rewards` accumulations by scoping them into a temporary local-thread variable (`local_reward`) during the `step()` function, flushing to the global array at the very end. The theory was to prevent physical hardware false-sharing on the cache line.
* **Measurement:** **Decreased FPS.**
* **Analysis:** The `EnvironmentArena` already mathematically enforces strict 64-byte spacing between environments, rendering cross-thread false sharing physically impossible by design. 
* **Verdict:** **Reverted.** The manual scoping actually constrained the compiler, disabling its native ability to vectorize standard unaliased assignments. The engine retained the direct array assignments.

## 6. Manual Base Pointer Cache Alignment
* **Experiment:** While the `EnvironmentArena` calculated each environment's stride to be a multiple of 64 bytes, the `std::vector<uint8_t>` default allocator only guarantees 8- or 16-byte alignment. If the base origin pointer was off-grid, every subsequent environment block physically straddled two 64-byte cache lines, inadvertently causing False Sharing across cores. We manually over-allocated the memory by 63 bytes and snapped the `aligned_base` pointer tightly to the nearest 64-byte hardware boundary (`(ptr + 63) & ~63`).
* **Measurement:** Throughput increased to **> 4,677,511.5 FPS**.
* **Verdict:** **Kept.** Perfecting the fundamental memory origin alignment stopped the OpenMP threads from triggering any residual L1 cache invalidations across the CPU silicon, securing an additional ~500k+ FPS performance leap.