# C1 iteration preflight

- Gate: port the previously validated exact Helios compute-fusion substrate,
  then run frozen GAN-on correctness before timing.
- Live source before candidate: `c01cb7781c3c6feaec4e7a3f9128b45a263d5102`;
  baseline math source `55d8884cf394ecb2ddd9b762844466a82055b983`.
- Live resources: both assigned nodes expose eight idle A100-SXM4-40GB GPUs;
  no live CUDA worker was present and cgroup `oom/oom_kill` counters were zero.
- Environment: reuse the validated Torch 2.10 archive at
  `/share/project/shuchenweng/yongyan_liu/DistillSchedule_envs/torch210_20260824/`
  rather than constructing a new training environment.
- Frozen baseline and workload: see `draft.md`; formal denominator is
  `173.014483004 s`, target `121.110138103 s`.
- Matching prior evidence: No-GAN Iterations 353-367 exercised the same Helios
  LoRA/residual/modulation code paths with exact fail-closed receipts. The new
  GAN profile independently shows the corresponding high-count add, mul,
  cast, copy, norm, and materialization kernels.
- Rejected/superseded paths: no blind FSDP prefetch (NCCL is about 87% hidden),
  no CPU-offload repair (D2H is effectively absent), no B2/packing/topology
  change, and no reuse of the historical No-GAN timing as a GAN claim.
- Falsifiable question: does the exact fusion substrate preserve all GAN-on
  receipts and reduce the matched unprofiled Full5 below 173.014483004 s?
- Decision: **GO**, limited to the exact fusion implementation and its
  correctness/timing gates. A contract drift or fallback count rejects C1.

## C1.1 packed-B3 preflight

- Current gate: repair only the implementation coverage gap exposed by C1;
  timing remains forbidden until a fresh correctness run has zero fallback.
- Live source: `c4e0f6901f47c5c8ad1681e16309a506e4bd5ec7`, clean and identical on
  the local worktree and both remote nodes at preflight time.
- Live resources: all 16 assigned A100 GPUs are idle, no training/profiling
  worker is live, and both cgroups report `oom=0`, `oom_kill=0`.
- Matching prior evidence: C1 run
  `iter349_gan_cuda_fusions_correctness_r1_20260829` completed all ranks but
  correctly failed the zero-fallback gate. Every rank recorded exactly 800
  residual-grad and 800 modulated-norm `NON_B1_HIDDEN` fallbacks.
- Root cause: the frozen workload is logically B1/rank but its official GAN-D
  Score forward is intentionally packed physical B3 (fake, real, R1). This is
  part of the frozen compute inventory and must not be split or removed.
- Rejected path: relaxing the fallback count or silently using the eager path
  for packed B3 after observing the result. Either would invalidate the
  preregistered exact-fusion candidate.
- Authorized hypothesis: the existing exact pointwise math is batch-general;
  extending batch-aware gate/component/table addressing to B3 can cover the
  official call without changing its values, gradients, order, or count.
- New falsifiable question: do distinct per-batch table/component values match
  the source bitwise for B3 forward and all required gradients, followed by a
  16/16 world16 receipt with zero fallback?
- Decision: **GO**, limited to exact B1/B3 addressing, focused parity tests,
  and one fresh correctness run. No timing or other optimization is combined.

## C4 modulation/residual chunk-2048 preflight

- Current gate: C3 is the fastest correctness-qualified substrate at
  `143.379703687 s`, still `22.269565584 s` above the requested threshold.
- Live source before C4: `5f399a0f4faf4df0a38e7a46255299e00992da71`;
  both remote execution trees are clean at C3 code HEAD `86d819813a800181ba5294b2dfa72a09a2f41442`.
- Live resources: both assigned nodes expose eight idle A100-SXM4-40GB GPUs;
  node locks are not held, no live CUDA worker exists, and both cgroups remain
  at `oom=0`, `oom_kill=0`.
- Authorized hypothesis: the pinned Helios source implements token-independent
  modulation, gated residual, and cross-normalization in source-default chunks
  of 512 tokens.  C3 receipts record tens of thousands of such chunks.  Raising
  only this already-existing launch-grouping control to 2048 may reduce Python,
  autograd, launch, cast, and temporary-materialization overhead.
- Frozen contract: C3 source assets, world16 FSDP2 topology, GAN-on steps
  1000--1004, trace/RNG, B1/global-B16, physical Score inventory, local FA2,
  losses, precision, updates, optimizer/EMA order, residual residency, and
  formal timer boundaries remain unchanged.
- Matching prior evidence: the same mechanism was the largest measured
  low-risk launch-granularity gain in the Wan1.3B path.  It has not previously
  been measured on this GAN-on Helios execution graph, so this is a new bounded
  question rather than reuse of a timing claim.
- Rejected/superseded paths: no segment/checkpoint policy, no HSDP, no
  all-gather prefetch, no Score/rollout packing, and no GAN physical-call
  regrouping are combined with C4.
- Falsifiable question: do the pinned real Helios B1/B3 residual and
  modulated-norm paths preserve output/gradient parity for 512 versus 2048,
  followed by 16/16 world16 correctness with identical inventory and zero OOM,
  and does one matched unprofiled Full5 improve over C3?
- Decision: **GO**, conditional on the bounded CUDA canary and focused tests.
  Any parity, receipt, inventory, or capacity failure rejects C4 before timing.
