# Iteration 295: pinned transition with legacy G3 redistribution

**Status:** Measured bounded negative; correctness PASS.

The strict-v4 run `rb16_f5_iter295_pinned64x2_g3legacy_correctness_20260810_001` changed only the G3 bucket size from 128 MiB to legacy per-tensor operation while retaining bounded pinned64x2 H2D.

| Metric | Iter293 baseline | Iter295 | Change |
|---|---:|---:|---:|
| transition | 49.839245 s | 47.881013 s | -3.93% |
| target H2D max | 33.168865 s | 28.955584 s | -12.70% |
| G3 broadcast wall max | 10.868068 s | 14.097394 s | +29.72% |
| Step10-A | 124.911580 s | 124.918693 s | neutral diagnostic |
| Critic phase | 165.794245 s | 170.207090 s | +2.66% |
| Full5 | 341.819356 s | 346.689093 s | +1.42% |

The materializer remained bounded to 256 MiB pinned memory per switching rank and required 17.180--28.932 s. Node cgroup peaks were 363,020,435,456 and 172,666,871,808 bytes; OOM delta was zero.

All 16 rank records passed exact trace/input/version/queue checks and reached G3/C15/EMA15. This trial refutes the hypothesis that simply undoing the bucket policy would cross the transition gate. Across Iterations 294--295, bounded pinned H2D improves target materialization but does not robustly reduce the full transition by >=10%. The critical path remains concurrent node-0 target-model materialization plus variable rendezvous wait. Further bucket-size tuning is stopped.
