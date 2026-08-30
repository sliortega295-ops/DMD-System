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

## C9 Generator no-grad history K/V preflight

- Current gate: C8 is the fastest correctness-qualified arm at
  `141.235767135 s`, but remains `20.125629032 s` above the requested
  `121.110138103 s` threshold.  C8's cross-attention encoder K/V reuse is
  exact but saves only `1.059574501 s`; broadening that same cache without a
  separate invariant is rejected.
- Live substrate: the local branch is clean at the C8 result commit, both
  remote execution trees are clean at runtime `9cee963`, all sixteen assigned
  A100-SXM4-40GB devices are idle with no CUDA process, and both cgroups retain
  `oom=0`, `oom_kill=0`.  Reuse the validated Torch-2.10 environment and pinned
  local FA2 asset; create no replacement environment.
- Matching prior code: commit `5dc7e5c` wraps the source-pinned Helios
  `HeliosAttnProcessor` cache that already implements exact restrict-self-
  attention history K/V and history-attention-output reuse.  No real-GPU
  timing or documented failure for that candidate exists.  Its old No-GAN
  admission restriction is an untested gate, not a GAN semantic result.
- Source invariant: within one no-grad rollout the history latents, indices,
  Generator version, and restrict-self-attention history prefix stay fixed
  while only the current noisy latent/timestep advance.  The pinned upstream
  processor caches the first call's history K/V and history attention output,
  still executes current-token Q/K/V, attention, output projection, every
  block, every DiT call, and every FSDP collective.
- Authorized hypothesis: add the existing history cache to C8 for Generator
  no-grad B1 rollouts only.  Clear all 40 processor caches at every rollout
  boundary, audit exact fill/reuse/miss counts, and in correctness mode compare
  every cached rollout against the same uncached C8/C5 B1 reference.
- Frozen contract: C8 GAN steps1000--1004, five distinct rank-qualified B1
  rollouts/global-B16, world16 FSDP2, trace/RNG/loss/physical call inventory,
  G1/C5/EMA1, G201/C1005/EMA801, local FA2, chunk2048, selective Critic replay
  stride2, CUDA residuals/EMA, cross-attention cache scope and timer boundary.
- Rejected/superseded paths: B2/B5, Real-Score packing, node-HSDP, stride-1
  retained checkpoint state, TP/PP, parameter offload/prefetch, and direct
  tuning of the C7 Tensor-bound GEMM are not retried or combined.
- Falsifiable question: does exact same-rollout history reuse pass five
  uncached B1 parity checks and all 16-rank lifecycle/capacity/state gates, then
  improve one unprofiled Full5 over C8?
- Decision: **GO for C9 only**. Any unexpected cache miss, mutation/leak,
  NAViT path, output-parity failure, physical-inventory drift, or OOM rejects
  C9 before timing.

## C9 iteration-end failure handoff

- Source-only validation passed at `31d28f77f4dec393e2bfff82058f260f50ab4ca3`:
  compile/shell gates passed, the focused admissible subset was `30 passed`,
  and `iter360_gan_cuda_c9_historykv_validate_r1_20260829` returned 16/16
  validate receipts with summary SHA-256
  `8cac3a46e6a8af6cbf1ff64b806f8b8dc533dd0daa79326723383e6cfff9e1b4`.
- The first real bounded verification,
  `iter360_gan_cuda_c9_historykv_correctness_r1_20260829`, failed closed on
  all ranks at the first Generator no-grad self-attention call, before a
  complete rollout or update.  The exact error is
  `cache-enabled Helios processor escaped restrict-self-attention`; CUDA and
  cgroup OOM deltas remained zero.
- Root cause: the frozen Generator and Score config assets omit
  `restrict_self_attn`, whose pinned `HeliosTransformer3DModel` default is
  `False`.  That single config value is passed to all 40 blocks.  Therefore
  the upstream history-prefix cache is not part of this workload's attention
  graph; enabling restrict-self-attention to make it applicable would change
  the frozen attention math.
- Validated recovery: keep C8 and leave the optional history cache disabled.
  The C9 implementation is reverted from the active substrate after preserving
  its commit and failed run evidence.  Do not retry C9 unless the official
  workload itself materially changes to restrict-self-attention.
- Rejected paths: silently skipping all non-restrict processors yields zero
  reuse and no candidate; changing the model config is a semantic-contract
  violation.  Formal C9 timing is **NOT_RUN**.

## C10 LoRA-B beta-add epilogue preflight

- Current gate: C8 remains the fastest valid arm at `141.235767135 s`, still
  `20.125629032 s` above the requested threshold.  C9 is semantically
  inapplicable and has been reverted; the active local and remote trees are
  clean at `065c92913b2e57d1be21a43272969780fa536e08`.
- Live substrate: both owned A100 nodes were rechecked on 2026-08-30 UTC; all
  sixteen GPUs had no compute processes.  Reuse the validated
  `/tmp/distillschedule-helios-official-torch210` runtime and pinned FA2 asset;
  no environment construction or large artifact transfer is authorized.
- Matching prior evidence: C5 node0 profile reports `439,436` BF16 vector-add
  launches.  C8 receipts report `21,962` Generator and `4,908` Critic scale-1
  LoRA epilogues per rank, so `214,960` node0 additions have an executable
  source owner.  In contrast, the exact world16 FSDP2 explicit-next prefetch
  test regressed from `118.267264657 s` to `118.737632053 s`; it is rejected.
- Authorized hypothesis: replace only the supported PEFT `LoRA-B` followed by
  BF16 scale-1 addition with a beta-add GEMM using the already computed base
  result.  The LoRA-A GEMM, LoRA-B weights, base output, adapter state, and
  autograd gradients remain present.  No model call or optimizer operation is
  removed.
- Frozen contract: C8 GAN steps1000--1004, B1/global-B16, world16 FSDP2,
  trace/RNG/loss/physical call counts, BF16 policy, local FA2, chunk2048,
  Critic selective replay stride2, exact cross-attention K/V scope,
  G1/C5/EMA1, final G201/C1005/EMA801, CUDA residency, and timer boundaries.
- Mandatory canary: on one locked owned A100, compare source `linear + add`
  against the candidate for registered B1/B3 representative LoRA shapes.  Gate
  output, base-result/input/LoRA-A/LoRA-B gradients with the existing P0
  multi-metric thresholds; record bitwise status and warm CUDA-event latency.
- Promotion gate: canary parity PASS and candidate faster; focused tests PASS;
  validate-only 16/16; correctness/capacity 16/16 with exact new receipt counts
  and zero fallback/OOM; only then one unprofiled Full5 faster than C8.
- Rejected combinations: no FSDP prefetch, HSDP, B2/B5, Score packing,
  checkpoint stride1, history cache, TP/PP, host offload, or other kernel is
  combined.
- New falsifiable question: can the LoRA-B beta-add GEMM preserve the frozen
  output/gradient gate while eliminating a meaningful fraction of the
  source-qualified BF16 add launches and improve C8 formal wall time?
- Decision: **GO for the bounded one-GPU canary only**.  World16 remains
  `NOT_RUN` until that gate passes.

## C10 iteration-end failure handoff

- Run `iter361_gan_cuda_c10_lora_beta_canary_r1_20260830` acquired the same
  node lock used by formal launches, used node0 GPU0 only, and left cgroup
  `oom=0`, `oom_kill=0` with no residual GPU process.
- Parity: all B1/B3 attention/FFN outputs pass the unchanged P0 gate; all four
  input, LoRA-A, LoRA-B, and base-result gradients are bitwise identical.
- Failure signature: warm forward speedups are `1.0171x`, `0.9971x`,
  `1.0081x`, and `0.9950x`.  The beta-add epilogue is not uniformly faster and
  its dominant no-grad B1 benefit is only about one percent.  Two B3 shapes
  regress, violating the registered promotion gate.
- Root cause: cuBLAS already executes the standalone low-rank LoRA-B GEMM and
  subsequent bandwidth add efficiently; folding beta into GEMM does not reduce
  the dominant GEMM work and can select a slightly worse B3 path.  Large B3
  training gains do not compensate the much more frequent no-grad B1 calls
  under the frozen workload.
- Evidence: report SHA-256
  `9e4e9ce951fd5c890fdfe0ab2e853a1baee95c4123f3b6a790f2757b9c5aab07`.
- Recovery: retain C8 at `141.235767135 s`.  Do not implement or launch C10
  world16 without materially different kernel evidence; formal C10 timing is
  **NOT_RUN**.

## C11 explicit modulated-norm backward preflight

- Current gate: C10 is closed by its canary; C8 remains the active substrate.
- Matching evidence: the chunk2048 formal receipts contain about `4,720`
  modulated-norm backward chunks per rank.  The pinned source rebuilds an
  autograd graph and invokes `torch.autograd.grad` once per chunk, mapping
  directly to repeated LayerNorm forward/backward, add, multiply, cast, copy,
  and reduction families in the valid C5 profile.
- Authorized hypothesis: preserve the installed exact forward and saved tensor
  set; replace only backward graph reconstruction with the algebraically same
  FP32 LayerNorm backward and scale/shift component gradients.  Preserve the
  source chunk order and BF16 accumulation boundaries for PhaseLocal inputs.
- Frozen contract: identical to C8/C10; no workload, call-count, topology,
  attention, checkpoint, cache, optimizer, residency, or timer change.
- Mandatory canary: one locked node0 A100, B1 and B3, width5120, chunk2048,
  two-segment PhaseLocal timestep projection.  Compare output, hidden gradient,
  and both timestep-input gradients with the unchanged P0 metrics; record
  bitwise status and warm source/candidate forward+backward CUDA-event time.
- Promotion gate: both shapes pass parity and improve latency before any
  repository implementation or world16 launch.  Subsequent implementation is
  default-off/source-pinned and still requires focused tests, validate-only,
  16/16 correctness/capacity, exact receipts, and one faster formal timing.
- Decision: **GO for the bounded C11 canary only**.

## C11 canary result and implementation GO

- `iter362_gan_cuda_c11_modnorm_backward_canary_r2_20260830` used one locked
  node0 A100 with the validated Torch 2.10 runtime and pinned local FA2 mapping.
- Output and hidden gradients are bitwise equal.  Both PhaseLocal timestep
  gradients pass the unchanged P0 gate; worst observed metrics are normalized
  L2 `0.002174`, cosine `0.9999976`, and max-absolute `1.0`.
- Warm forward-plus-backward latency improves `4.341077 -> 3.164160 ms` for B1
  and `10.837334 -> 7.937024 ms` for B3 (`1.37195x`/`1.36542x`).
- The preserved `r1` failure omitted `LOCAL_KERNELS` before model execution. It
  is an environment-integration failure and does not authorize a second
  mechanism variant; `r2` is the one bounded C11 mechanism verification.
- Evidence SHA-256:
  `36008126ce3567551474431d5ab3c81af679afca0c31e2c53f08fe0f5592e31c`.
- Decision: **GO** for the preregistered default-off implementation and its
  focused/world16 gates.  No second optimization may be combined.

## C11 iteration-end handoff

- Validate-only, correctness, and timing all PASS at
  `f3c5ed480001fb0007914de25264edd8439e9f50`; exact run IDs and summary hashes
  are recorded in `iteration363_c11_modnorm_backward.md`.
- Failure signature: none.  All 16 ranks end at G201/C1005/EMA801 with exact
  G1/C5/EMA1 update counts; explicit backward executes 1,120 calls and 4,720
  PhaseLocal chunks per rank with zero fallback.
- Formal result: `138.462140212 s`, an isolated `1.963828%` gain over C8 and a
  cumulative `19.970781%` reduction from C0.
- Recovery/next constraint: C11 is the validated substrate.  Do not retry
  explicit FSDP prefetch, C10 beta-add, HSDP, B2/B5, Score packing, stride1, or
  history cache without materially new evidence.  The next iteration must ask
  a new falsifiable question capable of addressing the remaining
  `17.352002109 s` gap.

## C11 post-fix diagnostic profile preflight

- Current gate: C11 is the fastest correctness-qualified arm at
  `138.462140212 s`; the requested threshold remains `121.110138103 s`.
- Live substrate (2026-08-30 UTC): both owned A100 nodes are clean at exact
  workload HEAD `f3c5ed480001fb0007914de25264edd8439e9f50`; all 16 GPUs have
  no compute process, both cgroups remain at `oom=0`, `oom_kill=0`, the pinned
  Torch 2.10/CUDA 12.8 environment imports successfully, and the Helios source
  SHA remains `5d3ffe37...65a886` on both nodes.
- Authorized action: one diagnostic-only node-outer Nsight Systems capture of
  the exact C11 GAN-on workload, followed by in-place remote `nsys-ai`
  analysis. Raw reports and SQLite remain on the remote evidence filesystem;
  only compact derived results may be committed.
- Frozen contract: GAN-on steps 1000--1004, world16 FSDP2, B1/global-B16,
  Score packing off, CUDA residuals and EMA, local FA2, chunk2048, Critic
  selective Flash replay stride2, Generator no-grad cross-KV reuse, exact
  modulated-norm backward, trace/RNG/loss/physical inventory, G1/C5/EMA1 and
  final G201/C1005/EMA801. The profiled wall is not a formal denominator.
- Matching evidence and rejected paths: C5's valid profile predates C8/C11 and
  therefore cannot quantify the post-fix kernel mix. It already bounds exposed
  NCCL and copy-only work; HSDP, explicit FSDP prefetch, B2/B5, packing,
  checkpoint stride1, history-KV reuse, and LoRA beta-add remain rejected.
- New falsifiable question: after C8 and C11 removed cross-attention projection
  work and modulated-norm autograd reconstruction, which source-qualified
  compute or host-launch family owns enough post-fix critical-path time to
  justify the next exact-semantics candidate?
- Decision: **GO** for run
  `iter364_gan_cuda_c11_profile_p1_20260830`; implementation and formal timing
  of a new candidate remain `NOT_RUN` until the profile identifies a new
  bounded mechanism.
