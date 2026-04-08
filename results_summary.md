# Performance Benchmarking Results

## Phase 1: Pure C++ MARL Environment Benchmarks
**Objective**: Isolate and measure raw environment steps per second.
**Setup**: `speedtest.py --mode centralized --requires_state True --init_mode parallel_first_touch`

1. **Verification Logging (Task 1.1)**:
   Added `sched_getcpu()` inside a `#pragma omp critical` block on the first simulation step. The generated `thread_affinity.log` confirms successfully retrieving the physical CPU ID for each OpenMP thread.
2. **First-Touch Allocators (Task 1.2)**:
   Refactored `EnvironmentArena` to support `--init-mode` command-line arguments to toggle `parallel_first_touch` and `serial`. First-touch initialized memory parallelly per env stride length to properly distribute pages across NUMA nodes.
3. **Static Loop Scheduling (Task 1.3)**:
   Ensured static schedule bounds exactly match between memory initialization loops and the physics step loop blocks.
4. **Thread Affinity Matrix (Task 1.4)**:
   Tested baseline versus `OMP_PLACES=cores OMP_PROC_BIND=true`. Output from EPYC architecture displays high sensitivities across boundaries.

## Phase 2: PyTorch + C++ Integration Benchmarks
**Objective**: Measure end-to-end RL loop throughput.
**Setup**: 100,000 steps testing using `cleanrl_ppo.py`.

1. **PyTorch Native First-Touch (Task 2.1)**:
   Removed `torch.zeros()` initializing tensors in favor of `torch.empty(..., pin_memory=True)` inside Python buffer allocations for Native fast pinned memory mappings into the C++ domain.
2. **Wait Policy & Thread Limit Matrix (Task 2.2)**:
   - **Control (Default)**: 15,537 SPS, Avg GPU Utilization 13.08%
   - **Test A (torch.set_num_threads(1))**: 15,800 SPS, Avg GPU Utilization 13.72%
   - **Test B (OMP_WAIT_POLICY=passive)**: 15,830 SPS, Avg GPU Utilization 20%
   - **Test C (Combined)**: **17,055 SPS**, Avg GPU Utilization 17.27%
   *Conclusion*: The combination of restricting PyTorch inter-op threads (`set_num_threads(1)`) while setting `OMP_WAIT_POLICY=passive` provided the highest throughput, reducing context-switch thrashing and preventing CPU starvation which blocks GPU feed availability.
3. **EPYC Specific NUMA-Aware PCIe Affinity Binding (Task 2.3)**:
   - Evaluated using `taskset` mirroring Python processes directly to the GPU's local NUMA block (CPUs 16-31, 48-63) versus Misaligned NUMA block (CPUs 0-15, 32-47).
   - **Aligned Performance**: 16,656 SPS
   - **Misaligned Performance**: 16,389 SPS
   *Conclusion*: Binding to the corresponding NUMA CPUs closest to the PCIe root complex avoids Infinity Fabric latency overheads, granting higher steps per second.
