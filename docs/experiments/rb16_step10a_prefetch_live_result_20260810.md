# Iteration 297: one-block prefetch / two-block live Step10-A

**Status:** Measured bounded negative for the 10% gate; boundary identity PASS.

The candidate retained 2M persistence and changed only:

- `stage3_prefetch_bucket_size=402,653,184` elements (1.146 frozen blocks);
- `stage3_max_live_parameters=805,306,368` elements (2.292 blocks);
- reuse distance, allgather bucket, `overlap_comm=false`, graph and math unchanged.

| Metric | strict-v4 baseline | Iter297 | Change |
|---|---:|---:|---:|
| Generator Step10-A median | 123.115506 s | 115.489513 s | -6.19% |
| world max | 129.003622 s | 120.909677 s | -6.27% |
| wave0 rollout | 20.675323 s | 19.275301 s | -6.77% |
| wave0 backward | 38.368997 s | 36.082621 s | -5.96% |
| wave1 rollout | 21.036667 s | 21.213110 s | +0.84% |
| wave1 backward | 39.909053 s | 37.354568 s | -6.40% |

This is only 1.15% faster than persistence alone. All trace/x0/gradient/loss boundary identities matched exactly, G2->G3 occurred once, no OOM occurred, node0 peak remained 31,762 MiB/GPU, and node0 cgroup peak was 359,993,237,504 bytes.

Pinned DeepSpeed source inspection closes the causal question: with `overlap_comm=false`, the ZeRO-3 parameter coordinator assigns prefetch all-gathers to the default stream. The enlarged window can coalesce/look ahead but cannot overlap those collectives with model compute. Therefore Iter297 is the final synchronous-window negative; the next one-variable candidate enables the already existing DeepSpeed communication stream while keeping the bounded window.
