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

## Post-C4 repeated-failure audit and C5 preflight

- Current gate: C4 is the fastest correctness-qualified substrate at
  `143.206713110 s`, still `22.096575007 s` above the requested 30%-reduction
  threshold.
- Live source before C5: `92de9f07af81a9e7ff96de755aafdcf4dadadf2b`,
  clean.  At the renewed audit both assigned nodes expose eight idle
  A100-SXM4-40GB GPUs, no live CUDA worker, and cgroup `oom=0`, `oom_kill=0`.
- Frozen contract: C4 source/assets, world16 FSDP2, GAN-on steps 1000--1004,
  trace/RNG, B1/global-B16, official physical Score/DiT inventory, local FA2,
  losses, BF16 precision, updates, optimizer/EMA ordering, CUDA residency and
  formal timer boundaries.
- Matching B2 evidence: commits `3f8d16f` and `a240f58` implement
  schedule-compatible grouping, metadata normalization, and per-row B1 parity.
  `iter355_helios14b_hsdp_compatible_b2_correctness_r2_20260824` then failed
  before update on all 15 preserved ranks: normalized-L2 about `0.058--0.071`
  and cosine about `0.9977--0.9984`, versus limits `0.005` and `0.9999`.
  This matches the proposed physical batching mechanism and magnitude; B2 and
  B5 are therefore **NO-GO** under the repeated-failure rule.
- Matching packing evidence: `iter298_nogan_B_correctness_20260815` already
  rejected Real-Score packing under the same strict gate.  It is not retried.
- Authorized C5 hypothesis: keep every physical call unchanged, but use the
  previously validated selective-checkpoint mechanism (`28ab629` through
  `ad9e16b`) on Critic blocks `0,2,...,38`.  Only the exact pinned FA2 result is
  saved in the original forward and reused during checkpoint replay.  The
  official GAN-D B3 graph is materially new capacity/execution evidence.
- Rejected/superseded paths: no B2/B5, no Score packing, no block checkpoint
  bypass, no KV cache, no HSDP, no parameter prefetch/offload, and no relaxed
  numerical or inventory gate.
- New falsifiable question: can selective FA2 replay on 20 Critic blocks pass
  exact per-rank save/reuse receipts and the frozen 16-rank GAN correctness and
  capacity gates, then reduce one matched unprofiled Full5 below C4?
- Decision: **GO** for C5 only.  Any missing dispatcher match, count mismatch,
  fallback, parity/inventory failure, or OOM rejects C5 before timing.

## Iteration 356 C5 diagnostic-profile preflight

- Current gate: C5 is correctness-qualified and is the fastest formal arm at
  `142.295341636 s`, but it remains `21.185203533 s` above the requested
  `121.110138103 s` threshold.  This iteration may only diagnose C5; its
  profiled wall is not a new performance denominator.
- Source identity: local orchestration/docs HEAD is
  `7613dcedcbdca1428c0a484894bf2e421ac5f2c8`; the exact clean execution tree
  on both nodes is runtime HEAD
  `e75c80e95198690ee070a8d7f77c6db9bfb85ce9`.  The only later local commit is
  the C5 result document, so no runtime diff is omitted from the remote tree.
- Live substrate: both assigned nodes expose eight idle A100-SXM4-40GB GPUs,
  no live CUDA worker owns a device, the node lock is not held, and both
  cgroups report `oom=0`, `oom_kill=0`.  PID-1-owned defunct `nsys-launcher`
  entries have no GPU context, listener, or lock and are retained as harmless
  historical zombies rather than killed.
- Environment: reuse `/tmp/distillschedule-helios-official-torch210`, restored
  from the validated persistent Torch-2.10 archive; reuse the same pinned local
  FA2 extension.  Reuse `/usr/local/bin/nsys` because the same binary produced
  the valid Iteration-351 trace on this exact CUDA/Torch substrate.  Analyze
  large SQLite reports in place over SSH using the already-provisioned
  `/tmp/lyy-experiments/DMD-System/gan_cuda_baseline_opt_20260829/nsys-ai-0.3.0`
  environment; do not create another environment or download the reports.
- Frozen contract: GAN-on steps 1000--1004, world16 FSDP2, B1/global-B16,
  Score packing off, pinned local FA2, chunk 2048, C1 exact fusions, C2
  optimizer warmup, Critic selective replay stride 2, CUDA residuals/EMA,
  trace/RNG/loss/precision/update order/inventory and exact Full5 range.
- Matching prior evidence and rejected paths: B2/B5 and Score packing retain
  their bounded parity failures; C5 capacity evidence rejects expanding the
  replay set to stride 1; Iteration 351 shows exposed NCCL alone cannot close
  the remaining gap.  None is retried in this diagnostic.
- Falsifiable question: after C2--C5, which phase/kernel/host or communication
  mechanism now owns the largest *exposed* critical-path time, and does it
  support one concrete single-variable candidate with enough potential to
  justify a fresh correctness/timing iteration?
- Decision: **GO** for one immutable node-outer world16 diagnostic profile,
  remote `nsys-ai doctor/analyze/evidence`, overlap-aware phase attribution,
  and kernel-design candidate selection.  Any missing report, SQLite, range
  receipt, rank, or parser-health gate makes the profile invalid.

## C6 node-HSDP topology preflight

- Current gate: Iteration 356 is a valid diagnostic of C5.  Formal C5 remains
  `142.295341636 s`; the profile contributes no replacement timing claim.
- Matching evidence: median Full5 compute union is `128.836834 s`, NCCL union
  `60.529270 s`, compute/NCCL overlap `53.169919 s`, and NCCL exposed versus
  compute `7.354369 s`.  No-GAN node-HSDP previously provided only a bounded
  `~1.24 s` control improvement, so C6 is not preregistered as sufficient for
  the 30% goal.
- Live substrate at preregistration: both remote trees are clean at `e75c80e`;
  each assigned node exposes eight idle A100-SXM4-40GB GPUs, the node lock is
  free, and cgroup `oom=0`, `oom_kill=0`.
- Authorized hypothesis: node-local shard groups may shorten parameter
  all-gathers and reduce cross-node parameter-materialization contention while
  the replica dimension preserves global gradient semantics.
- Frozen contracts: steps1000--1004, 16 distinct B1 rank inputs/global-B16,
  trace/RNG, physical Score/DiT inventory, losses, BF16, G1/C5/EMA1,
  G201/C1005/EMA801, CUDA residuals/EMA, local FA2, chunk2048, optimizer warmup,
  exact fusions, selective Critic replay stride2 and timer boundary.
- Rejected paths: B2/B5, Score packing, persistent GAN pipeline, denser
  checkpoint retention, prefetch and any simultaneous kernel change remain
  excluded.
- New falsifiable question: does C5 with `(replicate=2, shard=8)` fit the GAN
  graph, pass all 16-rank state/inventory/trace/capacity gates, then improve one
  unprofiled Full5?
- Decision: **GO** for validate-only, one correctness/capacity run, and one
  timing only after correctness PASS.

## C6.1 bounded construction recovery

- C6 validate passed with 16 ranks and the exact HSDP policy.  Its first real
  run did not enter model computation: after the Generator was sharded, the
  Critic loader called `model.to(device)` before `fully_shard`.  Every rank
  reached about `38.27 GiB` allocated and failed a `144 MiB` allocation.
- This matches the documented pre-FSDP full-model CUDA materialization
  mechanism.  It is not evidence that the persistent HSDP workload OOMs.
- Authorized repair: node-HSDP only, keep checkpoint restore and LoRA injection
  on CPU, then call the existing 40-block bottom-up FSDP2 policy.  PyTorch
  `fully_shard` moves each managed group to its CUDA mesh device and shards it
  immediately.  World16 construction stays unchanged.
- Frozen runtime contract: no parameter offload, no streaming, no model swap,
  no timer-boundary change, and the exact C6 topology/math/inventory remain.
- Stop condition: any post-construction topology mismatch, parameter/state hash
  mismatch, steady-state CUDA OOM, or correctness failure rejects C6; no
  allocator flag or additional memory mechanism may be added automatically.
- Decision: **GO** for focused tests and one fresh correctness run with a new
  immutable run ID.

## C7 bounded NCU preflight

- Current gate: C6.1 is rejected at `144.716706512 s`; C5 world16 FSDP remains
  the fastest formal arm at `142.295341636 s`.  The requested target remains
  `121.110138103 s`.
- Live evidence: Iteration 356 measures `128.836834 s` compute union.  Its
  largest non-NCCL family is the production BF16 GEMM
  `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn`, with
  `226.94062 GPU-s` aggregated over the node0 eight-GPU report.  This aggregate
  is not wall time and is used only to choose a representative kernel.
- Repeated-failure audit: node-HSDP has just regressed formal wall; B2/B5 and
  Score packing retain their bounded parity failures; denser retained
  checkpoint state has no new capacity evidence.  None is retried.
- Environment: reuse `/tmp/distillschedule-helios-official-torch210` and the
  existing `/opt/nvidia/nsight-compute/2023.2.1/ncu`.  A one-kernel A100 canary
  completed ten replay passes and produced a valid report, proving profiler
  permission before an expensive run.  No environment is rebuilt.
- Authorized action: add an instrumentation-only launcher that inherits every
  current P0 worker argument, profiles node0 device0 only, filters one exact
  demangled kernel inside one exact block NVTX range, and leaves the other 15
  ranks executing normally.  Raw NCU evidence stays on remote shared storage.
- Frozen contract: C5 world16 FSDP, GAN steps1000--1004, B1/global-B16,
  rank-qualified trace/RNG, physical inventory, local FA2, chunk2048, exact
  fusions, optimizer warmup, selective Critic replay stride2, CUDA residuals
  and EMA, losses, updates and precision.
- Falsifiable question: is the dominant GEMM already near A100 Tensor/SM
  throughput, requiring a structural compute reduction, or is there measured
  kernel-level headroom for an exact implementation change?
- Decision: **GO diagnostic-only**.  The NCU wall cannot update formal timing.
  Missing range receipts, wrong kernel/range, empty report, replay failure, or
  any workload/OOM gate failure invalidates the capture.

## C8 Generator no-grad cross-attention K/V preflight

- Current gate: C7 proves the dominant Critic BF16 GEMM is already
  Tensor-compute bound (`88.322976%` Tensor/SM throughput, `25.233768%` DRAM),
  while C5 remains `21.185203533 s` above target.  A marginal replacement of
  that GEMM is not selected.
- Matching prior code: commits `89b7efa` and `945a9f3` implement source-pinned
  per-rollout encoder K/V reuse and B1 reference parity.  No preserved real-GPU
  timing or failure artifact exists for this mechanism, and it was restricted
  to No-GAN admission rather than rejected on GAN semantics.
- Authorized hypothesis: after G201 commits, the Generator version and one
  rollout's text conditioning are invariant across its six physical DiT calls.
  Reusing each block's exact pre-norm cross-attention encoder K/V inside that
  one rollout removes repeated projection compute without removing query,
  attention, output projection, block, DiT, or FSDP calls.
- Frozen contract: exact C5 source/assets, GAN steps1000--1004, five distinct
  B1 rank-qualified rollouts/global-B16, trace/RNG/loss/physical inventory,
  world16 FSDP, local FA2, chunk2048, checkpoint stride2, G1/C5/EMA1,
  G201/C1005/EMA801, CUDA residency and timer boundary.
- Strengthened fail-closed gate: cache reuse requires the exact same encoder
  storage pointer, offset, tensor version, dtype, shape, stride and device;
  cache clears after every logical rollout.  Correctness must compare every
  candidate rollout to the uncached C5 B1 path under the existing multi-metric
  threshold and report exact 40 fills plus 200 reuses per rollout.
- Rejected combinations: no B2/B5, Score packing, HSDP, denser checkpoint
  retention, TP/PP, kernel replacement or cross-rollout reuse is combined.
- New falsifiable question: does the existing exact cache mechanism admit the
  real GAN-on rollout trace, pass all five per-rank B1 parity and lifecycle
  gates on 16 ranks, fit capacity, and reduce one unprofiled C5 Full5?
- Decision: **GO** for a source-only port, focused tests, validate-only,
  correctness/capacity, then one formal timing only after PASS.
