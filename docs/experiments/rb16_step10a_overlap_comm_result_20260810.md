# Iteration 298: async ZeRO-3 prefetch stream

**Status:** Measured bounded negative; boundary identity PASS.

The sole change from Iteration 297 was `overlap_comm=true`, which places ZeRO-3 parameter prefetch and gradient partition work on dedicated CUDA streams.

| Metric | strict-v4 baseline | sync window | async window |
|---|---:|---:|---:|
| Generator Step10-A median | 123.115506 s | 115.489513 s | 120.465448 s |
| world max | 129.003622 s | 120.909677 s | 124.963757 s |
| wave0 rollout | 20.675323 s | 19.275301 s | 20.041639 s |
| wave0 backward | 38.368997 s | 36.082621 s | 37.008138 s |
| wave1 rollout | 21.036667 s | 21.213110 s | 20.544714 s |
| wave1 backward | 39.909053 s | 37.354568 s | 41.349006 s |

The async candidate is only 2.15% faster than the original baseline and 4.31% slower than the synchronous window. Node0 GPU peaks rose from 31,762 MiB to 33,314--33,514 MiB. Node0 cgroup peak was 358,040,174,592 bytes; no OOM occurred. All 8x2 trace/x0/gradient/loss identities remained exact.

Thus the dedicated stream does not convert this workload into useful overlap: contention/stream synchronization and extra in-flight buffers outweigh hidden communication, particularly in the second checkpoint backward. Step10-A ZeRO-3 persistence/window/overlap tuning is closed as a bounded negative and remains above the 85--105 s target.
