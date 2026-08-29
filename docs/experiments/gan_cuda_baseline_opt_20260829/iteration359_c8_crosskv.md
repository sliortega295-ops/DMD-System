# Iteration 359: exact Generator rollout cross-attention K/V reuse

## Contract

- Parent performance substrate: C5 world16 FSDP2 at
  `e75c80e95198690ee070a8d7f77c6db9bfb85ce9`, formal Full5
  `142.295341636 s`.
- Candidate source: `9cee963a55175aad02d94da105f32ad0690451d4`.
- Frozen workload: GAN-on logical steps 1000--1004, world16 FSDP2,
  B1/global-B16, Score packing off, pinned local FA2, modulation chunk 2048,
  optimizer dispatch warmup, exact residual/modulated-norm fusions, Critic
  selective Flash checkpoint stride 2, CUDA Generator/Critic residuals, CUDA
  EMA, no parameter offload/streaming/model swap, G1/C5/EMA1 and final
  G201/C1005/EMA801.
- Authorized change: inside one Generator no-grad rollout, cache each block's
  exact cross-attention encoder K/V after its first use and reuse it for later
  DiT calls in that same rollout. The cache never crosses a rollout, Generator
  version, or encoder storage/version identity. It removes no DiT, block,
  attention, FSDP collective, loss, or optimizer call.

## Validation

- Focused source/runtime tests: 24/24 PASS in the server's lightweight test
  gate. A broader invocation under the node's system Torch 2.1 installation is
  not admissible evidence for this Torch 2.10 workload because it lacks the
  required `CheckpointPolicy` and pinned Triton APIs.
- Validate-only:
  `iter359_gan_cuda_c8_crosskv_validate_r1_20260829`, 16 ranks, PASS.
  Summary SHA-256:
  `3b31761f261e45708e596d48c68cccfe7356833b438ec8e50a1a221d6ce5d120`.
- Correctness/capacity:
  `iter359_gan_cuda_c8_crosskv_correctness_r1_20260829`, 16/16 PASS,
  claimable correctness, no reported errors. Summary SHA-256:
  `ee87bf4d99647498e0ceedf98182903663a99a3cf6b6117c8d0d7df2ee7da172`.
- Every rank reports 200 cache fills and 880 cache reuses over the five
  rank-qualified rollouts. Correctness mode also executes the uncached B1
  reference path, so its `180.845526592 s` diagnostic wall is deliberately not
  used as performance evidence.

The authoritative JSON remains in remote persistent storage:

`/share/project/shuchenweng/yongyan_liu/DMD-System_gan_cuda_baseline_20260829/artifacts/runs/<RUN_ID>/summary.json`

## Formal timing

- Run: `iter359_gan_cuda_c8_crosskv_timing_r1_20260829`.
- Status: PASS, 16 ranks, `claimable_timing=true`, no reported errors.
- Matched unprofiled Full5: **`141.235767135 s`**.
- Summary SHA-256:
  `403fdca286a2deb3504bf72eaea34b75579dfdf8c34ebe74fde09e2b3f96035e`.
- Improvement versus C5: `1.059574501 s` (`0.744630%`).
- Improvement versus fresh C0 (`173.014483004 s`): `31.778715869 s`
  (`18.367662%`, `1.225005x`).
- Requested 30%-reduction threshold: `121.110138103 s`.
- Remaining gap: `20.125629032 s`; C8 still needs another `14.249669%`
  reduction from its own wall.

## Decision

**PROMOTED_INTERMEDIATE_PASS_SMALL_GAIN.** The cache is exact, bounded to a
single rollout, correctness-qualified, and measurably faster, so C8 becomes
the fastest substrate. Its isolated gain is only 0.745%; it does not justify
further broad context caching without a new source-level invariant and profile
evidence. B2/B5 rollout batching, Real-Score packing, node-HSDP, and stride-1
checkpoint retention remain rejected or capacity-bounded prior paths and must
not be repeated by default.
