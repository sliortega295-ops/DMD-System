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

## C11 profile result and C12 GO decision

- `iter364_gan_cuda_c11_profile_p1_20260830` is a valid diagnostic capture:
  16/16 exact range receipts, two non-empty node reports/SQLite exports, and
  `nsys-ai doctor` PASS on both nodes.  The median rank-local range contains
  `126.112 s` compute union, `61.524 s` NCCL union with `54.422 s` hidden by
  compute, only `7.121 s` exposed NCCL, `0.501 s` exposed copies, and
  `5.903 s` with no traced GPU activity.  The GPU model was not exported, so
  there is no MFU claim.
- Source-qualified evidence: rank0 has `10.416 s` of synchronization API wall,
  but that includes required FSDP/framework synchronization.  The bounded
  application-owned subset is one Generator-update `empty_cache`, one
  Generator-update post-rollout `cuda.synchronize`, and five no-grad rollout
  post-rollout `cuda.synchronize` calls.  Generator update alone contains a
  median `4.500 s` no-traced complement.
- Current gate: C11 remains the formal denominator at `138.462140212 s`;
  profiled timing does not update it.  Repeated model compute is still the
  dominant cost, and exposed NCCL/copy alone cannot close the remaining
  `17.352002109 s` gap.
- Authorized C12 hypothesis: behind one explicit default-off P0 policy, omit
  exactly those six in-window rollout synchronizations and the one
  Generator-update allocator flush.  Keep the outer Full5 start/end
  synchronization, PyTorch dependency ordering, every tensor/call check, and
  all frozen workload semantics.
- Falsifiable question: does this isolated host-launch cleanup retain 16-rank
  G1/C5/EMA1, final G201/C1005/EMA801, trace/inventory/parity and zero OOM,
  while reducing one unprofiled Full5 below C11?
- Rejected/superseded paths remain unchanged: no B2/B5, Score packing, HSDP,
  explicit FSDP prefetch, stride1 retained state, history-KV, beta-add, or
  semantic relaxation may be combined.
- Decision: **GO** for C12 focused tests, validate-only, correctness/capacity,
  and then one timing only if every preceding gate passes.

## C12 iteration-end handoff

- Code `7811b20e967d0807987094f22ab3f8647357d8f1` passes focused static tests,
  controller dry-run, world16 validate-only, correctness/capacity and one
  unprofiled timing.  The initial system-Python test invocation exposed only a
  mismatched historical Triton CUDA package; rerunning the same focused suite
  without that non-training CUDA environment gives `33 passed, 36 skipped`.
  Real correctness then exercises the pinned Torch 2.10/Triton runtime.
- All 16 correctness and timing receipts report exactly zero Generator
  allocator flushes, zero Generator post-rollout synchronizations, zero Critic
  rollout synchronizations, and preserved outer Full5 boundary
  synchronization.  Updates and final versions remain G1/C5/EMA1 and
  G201/C1005/EMA801; maximum allocated/reserved are `24.9634/38.3125 GiB`,
  with zero CUDA/cgroup OOM delta.
- Formal result: `138.176082544 s`, only `0.286057668 s` (`0.206596%`) faster
  than C11.  This proves the source-owned explicit fences were mostly waiting
  on work that was already required, rather than creating a large recoverable
  critical-path bubble.  Do not extrapolate rank0's total synchronization API
  wall into removable time.
- C12 becomes the fastest valid substrate, but remains `17.065944441 s` above
  the 30% target.  The next candidate must target repeated compute or a larger
  source-qualified launch/materialization family; no further broad sync
  removal is authorized without new evidence.

## C13 Generator no-grad static-context reuse preflight

- Current gate: C12 at `7811b20e967d0807987094f22ab3f8647357d8f1`
  is the fastest correctness-qualified substrate at `138.176082544 s`; the
  requested threshold remains `121.110138103 s`.
- Live substrate revalidation: the local worktree is clean apart from this
  preregistration, both remote worktrees are clean at exact C12, all 16 owned
  A100s have no compute process, both node locks are acquirable, both cgroups
  remain at `oom=0`/`oom_kill=0`, and the previously validated Torch 2.10
  environment and pinned Helios transformer SHA are unchanged.
- Matching profile evidence: C11/C12 remains GPU-compute-bound.  The five
  Generator no-grad rollouts have median wall/compute of
  `31.913/28.993 s`.  C8 removed only repeated per-block encoder K/V
  projections and saved `1.059574501 s`; its source-pinned scope still calls
  `process_input_hidden_states()` and `condition_embedder()` on every physical
  DiT invocation.
- Source-qualified repeated work: within one immutable B1 rollout the history
  latents, history/current indices, text embedding and model version do not
  change.  The pinned transformer nevertheless repeats the three history
  Conv3d patch paths, history/current RoPE construction, history concatenation,
  and the PixArt text projection for every DiT call.  The current C8 receipts
  prove more than one call per rollout (`200` K/V fills and `880` reuses per
  rank over five scopes), so this is recurring rather than one-time setup.
- Authorized C13 change: inside the existing Generator-no-grad one-rollout
  scope only, cache the exact first-call history-patch prefix, full RoPE tensor,
  structural shape lists, and projected text tensor.  Subsequent calls still
  execute the current-latent patch, every DiT block/attention/FFN, every FSDP
  collective and the complete variable-exit schedule.  Cache entries are
  CUDA-resident, identity/version pinned, cleared after each rollout, and the
  implementation is pinned to the exact Helios source SHA.
- Frozen baseline/contracts: GAN-on steps 1000--1004, world16 FSDP2,
  B1/global-B16, Score packing off, trace/RNG/noise/history/indices, losses,
  precision, model/LoRA versions, physical DiT/attention inventory,
  G1/C5/EMA1, final G201/C1005/EMA801, C12 residency and timer boundaries all
  remain unchanged.  No cross-rollout cache, batching, topology, prefetch,
  offload, checkpoint-policy or logging change may be combined.
- Matching rejected/superseded paths: B2/B5, Score packing, HSDP, explicit
  FSDP prefetch, stride1 Flash retention, restrict-self-attention history K/V,
  LoRA beta-add, broad synchronization removal and direct tuning of the
  Tensor-bound dominant GEMM remain closed.
- New falsifiable question: can exact per-rollout static-context reuse pass a
  five-row uncached B1 output reference on every rank, report the expected
  fill/reuse/clear inventory with zero fallback or mutation, retain all frozen
  correctness/capacity gates, and reduce one unprofiled Full5 below C12?
- Decision: **GO** for C13 alone: focused CPU/CUDA tests, validate-only,
  16-rank correctness/capacity, then one formal timing only after all gates
  pass.

### C13 pre-launch source-contract correction

- No C13 GPU workload had started when the pinned `run_generator` and
  `inference_with_trajectory_stage2` source was re-read.  The rollout has one
  immutable history/text identity but traverses three current-latent pyramid
  layouts.  The original one-layout mock would therefore have failed closed at
  a normal stage transition and was not admitted to validate/correctness.
- This does not authorize a broader optimization.  The corrected cache keeps
  one exact history patch prefix, one history RoPE prefix and one projected
  text tensor for the existing rollout scope.  It computes the current patch
  on every call and keys only current-token RoPE by exact post-patch
  `(B,T,H,W,tokens,dtype,device,layout)`, expecting three layouts from the
  rank-qualified trace.
- Static history/index/text storage pointer, offset, version, dtype, shape,
  stride and device remain fail-closed.  Every DiT/attention/FFN/FSDP call and
  the five uncached B1 parity rows remain mandatory.  Decision remains **GO**
  for the same C13 hypothesis after focused tests.

## C13 iteration-end handoff

- Exact code `02517f1bbdb3a1fb3d5b184305fdc6f4d48e2a3e` passes focused tests,
  validate-only, world16 correctness/capacity and one unprofiled timing.
- Correctness is 16/16 PASS with G1/C5/EMA1, final G201/C1005/EMA801, five
  bitwise uncached B1 rollout references per rank and zero cache lifecycle,
  identity, mutation, layout, CUDA OOM or cgroup OOM failure.
- Formal Full5 is `138.243935311 s`, a `0.067852767 s` (`0.049106%`)
  regression against C12. The exact history/text/RoPE preparation is not a
  material Full5 bottleneck after C8/C11/C12 and its bookkeeping offsets the
  removed work.
- Decision: **REJECT C13 for performance and return to C12**. Preserve C13 as
  bounded negative evidence; do not stack it into the next candidate. The
  fastest substrate remains `138.176082544 s`, `17.065944441 s` above target.

## C14 fused BF16 LayerNorm plus modulation preflight

- Current gate and substrate: C12 remains the fastest valid workload at
  `138.176082544 s`; C13 is disabled and retained only as negative evidence.
- Matching evidence: the valid C11 profile records `20.52632 s` of native
  LayerNorm kernels and `10.33442 s` of the custom modulation-materialization
  kernel across node0's eight GPUs. The pinned `_TembAwareChunkedNorm` source
  performs a BF16-to-FP32 materialization, affine-free width-5120 LayerNorm,
  then the separate fused modulation epilogue for every registered tile.
- Authorized first step: add only a standalone Triton forward kernel and one
  locked-A100 B1/B3 canary. The kernel must preserve the source FP32 formula,
  epsilon, component/table additions, BF16 output and existing C11 backward.
  It may not change chunk size, model code, checkpointing, FSDP, or the formal
  workload before the canary passes.
- Frozen contracts: GAN steps1000--1004, world16 FSDP2, B1/global-B16, local
  FA2, chunk2048, trace/RNG/loss/precision, DiT/attention/FFN counts,
  G1/C5/EMA1, final G201/C1005/EMA801, CUDA residual/EMA residency, and the C12
  formal timer. No C13 cache, packing, batching, topology or prefetch change.
- Rejected paths remain closed: B2/B5, Score packing, HSDP, explicit FSDP
  prefetch, stride1 saved state, history-KV, LoRA beta-add, broad sync removal,
  static rollout context and direct dominant-GEMM tuning.
- Falsifiable canary: both B1 and B3 width5120/chunk2048 outputs must pass the
  unchanged P0 multi-metric thresholds and warm forward latency must improve by
  at least `1.15x` for both. Failure rejects C14 before world16.
- Decision: **GO for the bounded C14 canary only**.

### C14 canary result

- `iter367_gan_cuda_c14_fusedln_canary_r2_20260830` runs on one locked A100 in
  the validated Torch 2.10/Triton 3.6 environment and passes without OOM.
  Report SHA-256:
  `91ffb5704ec76069ed5215785eeb57aaf0e3804085f342de1fdd663a7207a72e`.
- B1/B3 output normalized-L2 is `9.7813e-6`/`1.1158e-5`, cosine is
  `1.0`/`1.000000119`, and both pass the unchanged P0 thresholds.
- Warm source/candidate latency is `0.231240/0.041779 ms` for B1 and
  `0.661422/0.106926 ms` for B3: `5.5348x` and `6.1858x`.
- The retained r1 run failed before kernel launch because the canary imported a
  nonexistent parity alias. The minimal harness fix did not change the kernel
  or test contract.
- Decision: **GO** for default-off C14 runtime integration, focused tests,
  validate-only, correctness/capacity and one timing only after all gates pass.

### C15 canary result

- `iter368_gan_cuda_c15_fusedbackward_canary_r5_20260830` runs on one locked
  A100 using the validated Torch 2.10/Triton 3.6 environment.  Report SHA-256:
  `55e21b69eb062509005f089eb03a3e266f75f409c84025ee88dbfdd60afe9d99`.
- All B1/B3 PhaseLocal/Tensor3d/Tensor4d and multi-chunk PhaseLocal hidden and
  component gradients pass the unchanged P0 thresholds.  Production B1/B3
  hidden-gradient normalized-L2 is `1.848e-5`/`1.991e-5`.
- Warm production PhaseLocal forward-plus-backward latency is
  `1.432627/0.660890 ms` for B1 and `3.675955/1.260442 ms` for B3, or
  `2.1677x`/`2.9164x`; CUDA/cgroup OOM deltas are zero.
- The preserved r1-r4 failures are bounded environment/canary-harness issues:
  missing the already-validated local FA2 mapping, creating the evidence root
  before the clean-tree gate, and two correctness-only Tensor4d stride gates.
  Neither changed the production PhaseLocal kernel math.
- Decision: **GO** for default-off C15 integration and the normal world16
  gates; no Full5 claim exists until those gates and one unprofiled timing pass.

## C14 iteration-end handoff and C15 GO decision

- C14 exact HEAD: `e391bb43f811929d59fb33c3cd36cdc2ee9a6fca`.
- Validate-only, correctness and formal timing all PASS. All 16 ranks preserve
  G1/C5/EMA1, final G201/C1005/EMA801, exact C12 workload identity, zero
  fallback and no admitted OOM. Formal Full5 is `136.084681777 s`.
- Root-cause conclusion: the repeated forward LayerNorm/materialization chain
  was measurable and C14 removes enough of it for a `2.091400767 s` gain, but
  it does not dominate Full5. C14 becomes the validated substrate.
- Matching next evidence: the same valid profile records `7.119277 s` of
  native LayerNorm grad-input kernels across node0's eight GPUs, owned by the
  1,120 explicit-backward calls and 4,720 chunks per rank. Existing C11 code
  still launches native LayerNorm backward plus separate cast, product,
  reduction and store work for every tile.
- Frozen contracts: C14 world16 GAN steps1000--1004, B1/global-B16,
  trace/RNG/noise/loss/precision, physical calls, G/C/EMA updates, local FA2,
  selective checkpoint-save stride 2, CUDA residency and formal timer.
- Rejected paths remain closed: B2/B5, packing, HSDP, explicit FSDP prefetch,
  stride1 saved state, history KV, LoRA beta-add, broad sync removal, static
  rollout context and dominant GEMM tuning.
- New falsifiable question: can deterministic source-pinned backward kernels
  pass all registered B1/B3 layouts and improve warm forward+backward by at
  least `1.15x`, before any world16 implementation?
- Decision: **GO for C15 canary only**. Failure returns to C14 without further
  invasive repair.

## C15 iteration-end handoff

- Exact integrated HEAD: `c2ccd577b4e94a1ea64cf538cd7626373206ebfa`.
- Validate, correctness and formal timing runs are respectively
  `iter369_gan_cuda_c15_fusedbackward_validate_r1_20260830`,
  `iter369_gan_cuda_c15_fusedbackward_correctness_r1_20260830`, and
  `iter369_gan_cuda_c15_fusedbackward_timing_r1_20260830`; all PASS.
- Frozen contract remained world16 FSDP2 GAN steps1000--1004, B1/global-B16,
  Score packing off, GPU residuals/EMA, local FA2, unchanged trace/RNG/losses,
  G1/C5/EMA1 and G201/C1005/EMA801.
- Every rank records 1,120 fused calls, 4,720 chunks, 5,840 hidden-gradient and
  5,840 component-gradient launches, with zero fallback. No CUDA or cgroup OOM
  occurred.
- Formal Full5 is `131.562027140 s`, improving C14 by `4.522654637 s`
  (`3.323411995%`). C15 is the validated substrate.
- The original `121.110138103 s` target is not met. Remaining gap is
  `10.451889037 s`; next falsifiable question must come from a current C15
  profile, not a repeated or more invasive C15 repair.
- Decision: **GO for one diagnostic-only C15 Nsight capture**, followed by one
  preregistered single-variable candidate. Profile wall time is never a formal
  denominator.

## C19 bounded FlashAttention NCU diagnostic preflight

- **Current gate:** C18 is the fastest correctness-qualified substrate at
  `129.441379467 s`, still `8.331241364 s` above the registered
  `121.110138103 s` target.  C19 is diagnostic-only and cannot replace either
  formal timing number.
- **Live substrate:** local orchestration HEAD is `d91c3727`; both remote
  execution trees are clean at C18 runtime HEAD `823a999f`, all 16 assigned
  A100-SXM4-40GB GPUs are idle, both cgroups remain at `oom=0/oom_kill=0`, and
  the validated Torch 2.10 plus pinned local-FA2 assets remain available.
- **Matching evidence:** the valid C15 profile remains representative of C18's
  unchanged model kernels.  On device 0, FA2 forward contributes
  `14.359417 s` of kernel sum and its dominant dense signature contributes
  `10.180233 s` across 820 launches.  The exact kernel has
  `grid=(87,1,40)`, `block=128`, `registers/thread=255`, and its longest
  instances are source-correlated to Critic block-forward ranges.  FA2
  backward contributes another roughly `13 s/GPU` of kernel sum.  These are
  GPU-kernel sums, not removable wall time.
- **Source audit:** the pinned A100 dispatcher already selects the dense
  `flash_attn_func` when no NAViT mask exists and the varlen entry only when a
  real mask is passed.  No source-qualified redundant dispatch or flag was
  found.  Naive projection fusion is rejected because the installed
  `fuse_projections()` copies only base Q/K/V weights and would bypass PEFT
  LoRA semantics.
- **Repeated-failure audit:** B2/B5, Real-Score packing, HSDP, explicit FSDP
  prefetch, forced NCCL protocols, static-context caching, simple LoRA
  beta-add, stride-1 retained Flash state and dominant-GEMM tuning remain
  closed.  This diagnostic does not retry any of them.
- **Authorized action:** first bring the NCU launcher up to the current C18
  policy flags without changing the worker math.  Then profile exactly one
  occurrence of the exact dominant FA2 forward kernel on node0 GPU0 inside
  one exact `FSDP_BLOCK_FORWARD/critic/block_32` range while the other 15 ranks
  execute the frozen workload normally.  Keep raw evidence remote.
- **Frozen contract:** GAN-on steps 1000--1004, world16 FSDP2, B1/global-B16,
  Score packing off, trace/RNG/loss/call inventory, local FA2, chunk2048,
  Critic Flash retention stride2, Generator cross-KV cache, all exact fused
  kernels, GPU residuals/EMA, scalar-log/free-memory elision and G/C/EMA
  update/version receipts remain unchanged.
- **Falsifiable question:** is this dominant FA2 forward already near A100
  SM/Tensor throughput, or do NCU roofline, warp-stall, occupancy and memory
  metrics identify a specific source-compatible tuning with enough measured
  headroom to justify one new canary?
- **Decision:** **GO_DIAGNOSTIC_ONLY** after focused tests and a locked NCU
  permission canary.  Missing 16-rank receipts, a wrong range/kernel, empty
  report, replay error, OOM or contract mismatch invalidates C19.

### C19 result and C20 profile-only GO decision

- Run `iter374_gan_cuda_c18_ncu_flashfwd_p1_20260830` completes at exact HEAD
  `f4f005435d49228eccd2238166edee1fdbc77366`. All 16 range receipts exist;
  the one-kernel NCU report and details CSV are nonempty and hash-pinned; both
  nodes have zero new CUDA/cgroup OOM and no residual worker/profiler process.
- The target dense FA2 kernel reports `11.581504 ms`, `70.363613%` SM/Tensor
  elapsed throughput, `2.865665%` DRAM throughput, `12.359914%` achieved
  occupancy, 255 registers/thread and 65.536 KiB dynamic shared memory. It is
  compute/tensor constrained with fixed-latency/math-pipe stalls; memory-
  bandwidth tuning is rejected.
- The pinned source exposes no equivalent launch-tuning flag. SDPA, B2/B5,
  Score packing, HSDP, FSDP prefetch, forced NCCL protocols, static-context
  caching, simple LoRA beta-add and stride-1 retained Flash state remain
  rejected or superseded paths. No attention implementation change is
  authorized by C19.
- **New falsifiable question:** after C17 removed 20 scalar fences/rank and C18
  removed four official Generator allocator/GC flushes/rank, what exact
  compute, exposed-NCCL, copy and no-traced budgets remain in each C18 phase?
- **Authorized scope:** one exact C18 node-outer Nsight capture and remote-only
  analysis. No workload, kernel, topology, trace, update, timing boundary or
  formal result may change. Missing node report, rank range, kernel device or
  parser health makes the diagnostic fail closed.
- **Decision:** **GO_DIAGNOSTIC_ONLY for the C18 post-fix profile**. Choose at
  most one C20 optimization only after that evidence exists.

## C20 result and C21 GO decision

- C20 run `iter375_gan_cuda_c18_profile_p1_20260830` is valid diagnostic
  evidence: 16/16 range receipts, two nonempty reports/SQLite, no parser error,
  no OOM, and an exact-range median of `130.093331865 s`.
- The formal denominator remains C18 `129.441379467 s`.  C20 reports Full5
  medians of `119.306060 s` compute union, `57.804130 s` NCCL union,
  `50.703842 s` compute/NCCL overlap, `6.947460 s` NCCL exposed vs compute,
  `0.376638 s` copy exposed and `3.470532 s` no traced GPU activity.
- C17/C18 achieved their registered mechanism: Score/DMD/GAN-G wall is
  `2.691273 s` lower than C15 while its compute union is unchanged.  Reopening
  scalar-log/free-memory repair is rejected.
- The five exact no-grad Generator rollouts remain `30.986322 s`, including
  `20.466938 s` NCCL union and `2.543321 s` exposed NCCL.  B2/B5, HSDP,
  explicit prefetch and protocol forcing are prior bounded negatives.
- **New falsifiable question:** can the already-supported FSDP2 eval behavior
  retain all Generator full parameters after the first B1 rollout, reuse them
  for the other four, and explicitly restore all 41 FSDP units before Critic
  without exceeding 40-GiB HBM or changing any logical/physical call?
- **Decision:** **GO for C21 capacity/correctness first**.  Full persistence is
  one bounded candidate.  OOM, a non-restored module, or any inventory drift
  rejects it; there is no automatic partial-block fallback in this iteration.

## C21 result and C22 diagnostic-only GO decision

- C21 passes validate-only and the second correctness run at exact HEAD
  `9b528b356b9351fe18e1e1247c2bd786bdc4a5ef`. The first correctness run is
  retained as an integration-only failure caused by a stale two-value hardened
  wrapper; its bounded receipt-propagation fix changed no workload operation.
- All 16 ranks report 41 fully materialized Generator FSDP units during the
  five no-grad rollouts, zero after explicit reshard, G201/C1005/EMA801 and
  zero CUDA/cgroup OOM.
- Formal unprofiled C21 is `125.814004444 s`, improving C18 by
  `3.627375023 s` (`2.8023%`) and C0 by `47.200478560 s` (`27.2812%`,
  `1.37516x`). The registered 30%-reduction target remains
  `4.703866341 s` away.
- Peak rollout receipt memory is about `37.006 GiB` allocated and
  `38.594 GiB` reserved, so no broader persistence experiment is authorized
  without a new capacity proof.
- **New falsifiable question:** after removing repeated Generator-rollout
  materialization, which exact Score, Critic, allocator/host or compute phase
  owns a source-qualified removable budget of at least 4.70 seconds?
- **Decision:** **GO_DIAGNOSTIC_ONLY for one exact C21 Nsight capture**. Keep
  the C21 formal denominator frozen; select no next optimization until the new
  phase attribution exists.

## C22 result and C23 allocator-cache GO decision

- C22 run `iter377_gan_cuda_c21_profile_p1_20260830` passes: 16/16 exact
  range receipts, two nonempty reports/SQLite, devices 0--7 on both nodes,
  zero unresolved kernel names/parser errors/OOM, and no GPU worker residue.
- The formal denominator remains `125.814004444 s`. The diagnostic exact-range
  median is `127.053260181 s`, with `116.565187 s` non-NCCL compute union,
  `38.595025 s` NCCL union, `33.228303 s` overlap, `5.337442 s` NCCL exposed,
  `0.352561 s` copy exposed and `4.899652 s` no traced GPU activity.
- C21 achieved its intended mechanism: the no-grad rollout NCCL union drops
  from `20.466938 s` to `1.104142 s`; rollouts 1--4 contain zero NCCL.
- A new source-qualified allocator cost remains. Median `cudaFree` overlap
  with GPU-inactive intervals is `2.1232 s`. Representative rank 0 shows the
  persistence pre-flush as a `1.258138 s` envelope and the post-reshard flush
  as `0.327522 s`. These are the two calls in
  `fsdp2_nograd_persistence.py`, not an inferred model operation.
- **Authorized hypothesis:** retain the already-freed Generator parameter
  storage in PyTorch's allocator cache across the bounded persistence entry
  and exit. Do not retain an unsharded parameter, skip explicit reshard, or
  change a model/collective call.
- **Falsifiable question:** can the same C21 workload reuse cached blocks for
  Generator materialization and subsequent Critic work, keep all 41 units
  restored, pass 16/16 correctness without OOM, and reduce one formal Full5?
- **Decision:** **GO_CAPACITY_FIRST for C23**. Any CUDA/cgroup OOM, missing
  reshard receipt, or inventory drift rejects no-flush mode and returns to
  C21. Timing is allowed only after correctness/capacity passes.

## C23 result and C24 allocator-fragmentation GO decision

- C23 exact HEAD `fce8bcf011171c2e3a337ab8929d62d4e313f061` passes
  119 focused tests, validate-only and a 16/16 correctness/capacity run. The
  correctness receipt has G201/C1005/EMA801, zero pre/post cache flushes,
  41/0 unsharded-unit lifecycle and 41 explicit reshards.
- The separate formal timing attempt is not capacity-safe. Ranks 3, 4, 11 and
  12 fail a 292-MiB allocation with about 36.43 GiB allocated, 1.91 GiB
  reserved-but-unallocated, and only 196--202 MiB driver-free. It is retained
  as `iter378_gan_cuda_c23_persist_noflush_timing_r1_20260830`; no partial
  latency is claimable.
- **Matching prior substrate:** the repository's earlier GAN launcher and
  Route-A plans already use PyTorch `expandable_segments:True`; this is not a
  new allocator implementation or a workload change.
- **New falsifiable question:** can process-start expandable CUDA segments
  make the exact C23 cache layout service the 292-MiB LoRA allocation while
  preserving every C21/C23 receipt and avoiding OOM?
- **Decision:** **GO for one bounded C24 recovery**. Validate-only and a fresh
  matched correctness run are required before timing. A repeated OOM closes
  the no-flush path and returns to C21; no further allocator patch escalation.

## C24 result and allocator-path closeout

- C24 exact HEAD `5655a771c16a31e2ae0fa0e142914f8b18b58241` passes 119
  focused tests, validate-only, 16/16 correctness/capacity and one unprofiled
  formal timing.  All ranks retain the exact C23 no-flush lifecycle, explicit
  41-to-0 reshard contract and G201/C1005/EMA801; CUDA/cgroup OOM deltas are
  zero.
- `expandable_segments:True` resolves the observed C23 allocation failure, but
  formal Full5 is `127.619803944 s`, a `1.805799500 s` (`1.4353%`)
  regression from C21 `125.814004444 s`.
- **Decision:** reject C24 and return to C21.  C23/C24 now form a bounded
  negative result for allocator-cache no-flush: one arm OOMs and the supported
  fragmentation recovery is slower.  Do not retry or escalate allocator
  changes without materially new evidence.

## C25 iteration preflight

- **Current gate:** post-C24 return to C21; one kernel canary is authorized,
  while world16 is not yet authorized.
- **Authorized hypothesis:** fuse only the non-NaViT guidance-cross-attention
  suffix BF16 add and history concatenation into one exact custom autograd
  operation, including direct full-hidden gradient routing.
- **Frozen baseline and contracts:** C21 GAN steps1000--1004, world16 FSDP2,
  B1/global-B16, local FA2, exact trace/RNG/loss/call/update/version/timer,
  complete C15/C17/C18 fusions and Generator 41-unit rollout persistence.
- **Matching evidence:** C22 node0 device0 records 15,860 cat-copy launches
  (`1.4589 GPU-s`) and 1,320 split-copy launches (`1.5597 GPU-s`), while the
  pinned Helios transformer SHA is
  `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`.
  These are selection evidence, not a wall-time claim.
- **Rejected/superseded paths:** allocator no-flush C23/C24, B2/B5, HSDP,
  explicit FSDP prefetch, forced NCCL protocols, static-context cache,
  restrict-self-attention history K/V, simple LoRA beta-add, stride-1 Flash
  checkpoint retention, and direct attention replacement remain closed.
- **New falsifiable question:** can the exact B1/B3 output and gradients be
  preserved while warm forward-plus-backward improves by at least `1.15x`,
  before one independently gated world16 run?
- **Decision:** **GO_CANARY_ONLY**.  Any parity failure, unsupported layout,
  source mismatch, fallback, or sub-threshold shape rejects C25 before world16.

### C25 canary gate result

- `iter380_gan_cuda_c25_history_current_canary_r1_20260830` passes on one
  owned A100: B1/B3 output, full-hidden gradient and attention gradient are
  bitwise equal; forward-plus-backward improves `1.5882x`/`2.0946x`.
- The exact pinned Helios block method rewrites successfully and 116 focused
  tests pass in the recorded Torch 2.10 environment.
- **Decision:** **GO_WORLD16_GATES** for this candidate only.  Validate-only
  and fresh 16/16 correctness/capacity must pass with live calls, registered
  B1/B3 shapes and zero fallback before one unprofiled timing.

### C25 world16 result

- Exact HEAD `d865cdc0e5d7ee99fbe6102a5965de08c8c81b9d` passes validate-only,
  16/16 correctness/capacity, and one unprofiled formal timing.
- Every timing rank reports 2,360 fused calls (1,120 grad, 1,240 no-grad), all
  expected B1/B3 shapes, zero reference fallback, G201/C1005/EMA801, and the
  unchanged 41 fully-unsharded to zero-at-exit lifecycle.  CUDA/cgroup OOM
  deltas are zero.
- Formal Full5 is `125.258735931 s`, improving C21 by `0.555268513 s`
  (`0.4413%`) and C0 by `27.6022%`.  It remains `4.148597828 s` above the
  registered 30%-reduction target.
- **Decision:** promote C25 as the new intermediate denominator; do not infer
  end-to-end savings from isolated kernel-sum or canary speedup.  A subsequent
  candidate requires new measured evidence and the exact C25 contract.

## C26 exact paired-RoPE canary preflight

- **Current gate:** C25 is the fastest correctness-qualified substrate at
  `125.258735931 s`; only one locked-A100 kernel canary is authorized before
  another world16 launch.
- **Authorized hypothesis:** replace the two consecutive self-attention
  query/key calls to the pinned Triton RoPE implementation with one paired
  kernel that reads the shared rotary table once and writes two independent
  outputs.  The paired backward performs the same inverse rotation for both
  incoming gradients.
- **Frozen baseline and contracts:** exact C25 GAN steps 1000--1004, world16
  FSDP2 B1/global-B16, Score packing off, trace/RNG/loss/model-call inventory,
  local FA2, checkpointing, all prior exact fusions, 41-unit Generator
  persistence, G1/C5/EMA1, final G201/C1005/EMA801, and timer boundaries.
- **Matching evidence:** C22 node0 records `46,720` production
  `_apply_rope_transposed_kernel` launches totaling `12.83587 GPU-s` across
  eight ranks.  This reconciles exactly to two launches for each of 40 blocks
  across 73 application DiT forwards per rank.  The pinned source also
  materializes the identical cos/sin views independently for query and key.
  These are GPU-kernel sums and selection evidence, not removable wall time.
- **Rejected or superseded paths:** C23/C24 allocator no-flush, B2/B5, Score
  packing, HSDP, explicit prefetch, NCCL protocol forcing, static-context
  cache, history K/V cache, LoRA beta-add, stride-1 Flash retention, direct
  attention replacement, and further C25 repair remain closed.
- **New falsifiable question:** for production B1/B3 width-128 head layouts,
  can paired query/key RoPE preserve both outputs and both input gradients
  bitwise while reducing warm forward-plus-backward time by at least `1.15x`
  with zero fallback?
- **Decision:** **GO_CANARY_ONLY**.  World16 integration is forbidden unless
  both shapes pass the unchanged parity and latency gate.

### C26 direct-paired failure and bounded repair

- `iter382_gan_cuda_c26_paired_rope_canary_r1_20260830` is an immutable
  environment-integration failure before CUDA work: the diagnostic launcher
  omitted the already validated offline `LOCAL_KERNELS` mapping.  It is not a
  candidate verdict and the shared environment was not modified.
- `iter382_gan_cuda_c26_paired_rope_canary_r2_20260830` uses the validated
  mapping and measures `1.9358x`/`1.8789x` B1/B3 forward-plus-backward
  speedups, but **fails** all four bitwise output/gradient gates.  A compact
  follow-up finds deterministic differences in about `1.0e-5` of elements,
  with production-shape maximum BF16 absolute difference `0.03125`.
- **Matching mechanism:** the direct combined Triton program changes codegen
  and BF16 rounding relative to the separately launched pinned Helios kernel;
  this is not trace noise or nondeterminism.  The fast result is therefore not
  eligible for world16 integration.
- **One permitted bounded repair:** materialize the identical FP32 cosine and
  sine views once, then invoke the exact installed Helios arithmetic kernel
  separately for Q and K in forward/backward.  This retains the production
  codegen while removing only redundant frequency staging.  The original
  bitwise and `1.15x` gates remain unchanged.  Failure closes C26; no further
  arithmetic or tolerance repair is authorized.

### C26 bounded-repair result

- `iter384_gan_cuda_c26_shared_freq_canary_r3_20260830` passes bitwise Q/K
  output and input-gradient parity for both production shapes, proving that
  reusing the installed arithmetic kernel restores the frozen contract.
- Warm forward-plus-backward is only `1.0085x` for B1 and `1.0087x` for B3,
  far below the preregistered `1.15x` promotion threshold.
- **Decision:** **REJECT_C26_NO_WORLD16**.  The fast single-launch form changes
  BF16 results; the exact form has negligible isolated gain.  Do not integrate
  either implementation or spend a world16 run on this path.  C25 remains the
  authoritative denominator at `125.258735931 s`.

## C27 fused frozen cross-attention LayerNorm preflight

- **Current gate:** C25 remains the fastest correctness-qualified substrate at
  `125.258735931 s`, `4.148597828 s` above the registered 30%-reduction target.
  C26 is closed and no distributed launch is authorized before a locked-A100
  kernel canary passes.
- **Authorized hypothesis:** replace only the frozen affine `norm2` tile used
  by `_ChunkedFrozenNorm` with one BF16-input/FP32-math/BF16-output Triton
  forward and one hidden-gradient-only Triton backward.  The operation keeps
  the exact LayerNorm epsilon, affine weight/bias, width 5120, chunking and
  frozen-parameter semantics while eliminating the explicit BF16-to-FP32
  tensor, FP32 output, BF16 cast/copy and native-autograd reconstruction.
- **Frozen baseline and contracts:** exact C25 GAN steps 1000--1004, world16
  FSDP2, B1/global-B16, Score packing off, trace/RNG/loss/physical calls,
  checkpoint policy, local FA2, prior exact fusions, 41-unit Generator
  persistence, G1/C5/EMA1, final G201/C1005/EMA801 and timer boundaries.
- **Matching evidence:** the pinned Helios source at SHA-256
  `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`
  routes every frozen affine cross-attention `norm2` through
  `_ChunkedFrozenNorm`.  Its forward explicitly executes
  `hidden.float() -> Flash LayerNorm FP32 output -> BF16 cast -> copy_`; its
  backward reconstructs that graph per chunk.  C22 device 0 records 8,760
  `layernorm_forward` launches (`0.520950048 GPU-s`) and 1,800
  `layernorm_backward` launches (`0.171421406 GPU-s`).  Copy/cast kernel totals
  are only broad upper bounds until the canary and runtime receipt identify
  this helper's exact ownership; none are treated as removable wall time.
- **Matching prior result:** C14/C15 prove that the same width-5120
  FP32-LayerNorm arithmetic can pass the unchanged multi-metric P0 contract and
  produce large isolated forward/backward gains, but those kernels implement
  affine-free modulated `norm1`/`norm3`, not this frozen affine `norm2` path.
- **Rejected or superseded paths:** C23/C24 allocator no-flush, B2/B5, Score
  packing, HSDP, explicit FSDP prefetch, NCCL forcing, static-context and
  history-KV caches, LoRA beta-add, stride-1 Flash retention, direct attention
  replacement and C26 paired RoPE remain closed.
- **New falsifiable question:** for production B1/B3, width-5120, 2,048-token
  tiles with nontrivial FP32 affine parameters, do output and hidden gradient
  pass the unchanged P0 contract with zero fallback while warm
  forward-plus-backward improves by at least `1.15x` for both batches?
- **Decision:** **GO_CANARY_ONLY**.  Any source mismatch, unsupported dtype or
  layout, fallback, parity failure, or sub-threshold batch closes C27 before a
  world16 policy is added.

### C27 canary result and user pause

- `iter386_gan_cuda_c27_frozen_crossnorm_canary_r3_20260830` passes the
  preregistered P0 output/hidden-gradient gate and measures `2.9772x` B1 and
  `6.8783x` B3 warm forward-plus-backward speedup on one locked A100.
- The two earlier immutable run IDs are launcher/result-path failures, not
  kernel verdicts; neither changes the successful r3 evidence.
- C27 is eligible for a future default-off world16 integration, but the user
  explicitly paused optimization before that integration.  Therefore the
  formal denominator stays C25 `125.258735931 s`; no end-to-end C27 gain is
  claimed.
- Both nodes were checked after the pause: 16/16 A100s are idle, there are no
  compute processes, both locks are free, and cgroup OOM counters remain zero.
