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
