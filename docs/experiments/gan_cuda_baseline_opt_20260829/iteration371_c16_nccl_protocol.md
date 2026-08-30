# Iteration 371: C16 NCCL protocol canary

## Question and boundary

The C15 profile exposed `6.874140 s` of NCCL versus non-NCCL compute and used
`RING_LL` for every large FSDP all-gather. C16 therefore asked one bounded
question before touching the model workload: can a forced NCCL protocol reduce
both the representative large all-gather and its compute-overlap window?

The formal C15 GAN workload was **NOT RUN** in this iteration. The canary kept
the same two nodes, world16, Torch `2.10.0+cu128`, NCCL `2.27.5`, representative
`44.8 MiB/rank` BF16 all-gather shards, a `128 MiB` FP32 reduce-scatter input,
and six real-shape BF16 GEMMs in the overlap window.

## Result

Run: `iter371_gan_cuda_c16_nccl_protocol_canary_r1_20260830`.

All 16 ranks passed the collective value gates under all three protocols. There
were 48 rank/protocol receipts, zero CUDA/cgroup OOM increments, no residual
worker, and no held GPU lock.

| Protocol | all-gather median | reduce-scatter median | compute median | overlap-window median |
|---|---:|---:|---:|---:|
| NCCL default | 14.119908 ms | 2.584603 ms | 17.699867 ms | 19.176956 ms |
| Simple | 14.172094 ms | 2.569480 ms | 17.778084 ms | 19.226858 ms |
| LL128 | 17.006500 ms | 2.804402 ms | 17.725687 ms | 19.421629 ms |

Relative to default:

- `Simple`: all-gather `+0.370%`, reduce-scatter `-0.585%`, overlap window
  `+0.260%`.
- `LL128`: all-gather `+20.443%`, reduce-scatter `+8.504%`, overlap window
  `+1.276%`.

## Decision

**REJECTED_CANARY_PERFORMANCE_GATE.** Neither forced protocol improves both the
large all-gather and overlap window by the preregistered 5%. Running a formal
world16 model arm would repeat a bounded negative. Return to C15 and select the
next candidate from GAN-specific compute or source-qualified launch gaps.

The formal fastest result remains `131.562027140 s`; the requested 30%-faster
threshold remains `121.110138103 s`.
