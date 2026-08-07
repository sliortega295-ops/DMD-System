# Nsight Systems / NVTX profiling

The profiling code in this repository is part of the experiment harness, not the steady-state runtime.

## Capture

- `scripts/distillschedule/run_nsys_locked.sh` acquires a node-level GPU lock, checks that the node is idle, records immutable source/environment metadata, captures a named `cudaProfilerApi` range, and validates that the resulting report contains both NVTX and CUDA-kernel data.
- `scripts/distillschedule/nsys_profile_command.sh` is the inner-capture wrapper used when the target launcher must own `torchrun`/multi-process setup.

## Analysis

- `scripts/distillschedule/analyze_w16_pool_nsys.py` reports per-device compute/NCCL busy unions and idle fractions for the world16 Critic pool.
- Additional Step15-A and full-world16 timeline/Gantt analyzers can be added under this profiling surface without coupling them to the model adapter.

## Evidence policy

Profiler output is diagnostic evidence. Formal speedups come from separate unprofiled matched runs. Raw `.nsys-rep`, exported `.sqlite`, generated plots, logs, and large trace artifacts are intentionally excluded from Git and should be stored as experiment artifacts.

A valid profile is expected to preserve:

1. exact workload/checkpoint/operation-trace identity;
2. explicit NVTX phase labels;
3. source commit and launch command;
4. report SHA-256;
5. a separate claim boundary for profiler overhead.
