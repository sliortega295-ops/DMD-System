# DistillSchedule Research Log

## Iteration 0

### Question

Can the existing Helios Stage-3 implementation be measured safely and reproducibly on the available
2x8 A100-40GB cluster, and what is the first correctness or feasibility blocker?

### Hypothesis

Before steady-state profiling, the upstream path fails because it materializes Generator and Score
simultaneously and may also retain an unintended Generator graph during the Critic rollout.

### Importance

These facts determine whether the paper should focus on activation scheduling, model residency, or
correctness-preserving component partitioning.

### Evidence Needed

- Immutable code, weight, and environment manifests.
- Controlled cold-start memory trace.
- Isolated Generator and Score residency measurements.
- requires_grad and Generator-gradient state around Critic backward.
- Actual H2D/D2H and inter-node transport measurements.

### Refutation Condition

The upstream path starts within 40 GiB without hidden sharding, and Critic-only execution neither builds
nor retains any Generator graph.

### Actions

- Created a clean research worktree from Helios commit 8f2a2faa.
- Reconstructed the logical and physical-state graphs from the public source.
- Added a standalone phase tracer, transfer benchmark, distributed benchmark,
  and semantic-gradient probe.
- Measured NUMA-local H2D/D2H, eight-GPU intra-node all-reduce, and
  topology-affine one-GPU-per-node all-reduce.

### Commands

Representative commands and all raw-input hashes are recorded in
`artifacts/manifests/iteration_000_measurements.json`.  Raw logs remain under
`artifacts/logs/` and are intentionally not committed.

### Results

- **Confirmed by source/static checkpoint headers:** simultaneous Generator +
  Score active state is at least 62.6 GiB/GPU before activations and optimizer
  state; default ZeRO-2 does not shard those parameters.
- **Measured:** pinned large-buffer H2D/D2H is about 24.4 GiB/s.  Pageable D2H
  is only 9.24--9.27 GiB/s.  A 31.3 GiB whole-backbone transfer therefore has
  an estimated 1.29 s pinned or 3.38 s pageable D2H floor.
- **Measured:** eight-GPU NVSwitch all-reduce reaches 130.33 GB/s algorithm
  bandwidth at 1 GiB.  A topology-affine cross-node GPU2/`mlx5_2` run reaches
  19.41 GB/s at 1 GiB with GPUDirect RDMA; a non-affine GPU0 run reaches only
  9.59 GB/s at 256 MiB.  The full 16-GPU communicator reaches 25.62 GB/s
  algorithm bandwidth (48.04 GB/s bus bandwidth) at 256 MiB/rank.
- **Measured on tiny CPU and CUDA reproductions, not full Helios:** an inner
  `torch.set_grad_enabled(True)` rebuilds a Generator graph inside an outer
  `torch.no_grad()`.  The unbounded CUDA case leaves nonzero Generator
  gradient (L2 0.3778); an explicit detach leaves no materialized Generator
  gradient.
- **Measured:** phase-tracer tests pass (6/6), and the A100 NVTX/CUDA-event
  smoke emitted allocator and stream fields without instrumentation errors.
- **Not Run:** actual 14B cold-start OOM and actual DeepSpeed Critic gradient
  pollution.

### Interpretation

The first hypothesis is strongly supported statically but not yet closed by a
protected 14B run.  Activation checkpointing cannot fix the 62.6 GiB
simultaneous parameter request.  Current hand-written whole-model swapping is
also potentially expensive enough to dominate a short step, particularly
because parameters are not placed in a pinned pool.  Network placement is a
first-class planner input: HCA affinity doubled observed cross-node bandwidth.

The Critic finding is a real source-semantics hazard, but the tiny reproduction
does **not** prove that the distributed Helios run pollutes Generator `.grad`.
That claim remains open until instrumented at the actual backward boundary.

### Files Changed

- README_AUTORESEARCH.md
- research_log.md
- docs/00_hardware_feasibility.md
- docs/01_source_of_truth.md
- docs/02_execution_graph.md
- artifacts/graphs/*.json
- artifacts/manifests/iteration_000_measurements.json
- runtime/trace.py
- scripts/distillschedule/{bench_dist,bench_transfer,semantic_grad_probe,summarize_iteration0}.py
- tests/test_trace.py

### Failure or Limitation

The system Python lacks Stage-3 dependencies.  Existing assets contain a full
Helios-Base transformer but not Helios-Distilled/`transformer_ode`, and no
training/demo data.  The first distributed benchmark hung due to NCCL barrier
device inference; preserving the failed run exposed and fixed the benchmark,
not the workload.

### Next Highest-Value Question

Can one physical 14B backbone plus its rank-256 adapter be loaded and profiled
in isolation on an A100-40GB without relying on the infeasible two-backbone
cold-start path?

### Next Command

Finish the private environment and immutable asset manifest, then run a
single-rank, protected Score load with host RSS and CUDA peak instrumentation.

## Iteration 1

### Question

Can one real Helios 14B physical Score plus its rank-256 Critic adapter reside
on an A100-40GB, and does the hand-written whole-module offload behave like the
contiguous PCIe bandwidth model?

### Hypothesis

The isolated Score will fit narrowly, but less than 8 GiB will remain for
activations and runtime state.  Whole-module `.to(cpu/cuda)` will be materially
slower than a contiguous transfer and will not fully release CUDA reservations.

### Importance

This separates the two-backbone cold-start bug from steady-state one-model
feasibility and tests whether residency/prefetch, rather than checkpointing
alone, is a credible optimization target.

### Evidence Needed

- Exact real-model parameter, trainable, and dtype counts.
- Peak host RSS while loading/casting.
- CUDA allocated, reserved, and driver-free memory at residency boundaries.
- Direct whole-module D2H/H2D timings.
- A pinned, immutable environment/asset manifest.

### Refutation Condition

The Score alone OOMs before any activation, or its module transfer time tracks
the contiguous-copy lower bound and offload returns the reservation to the
driver without explicit allocator action.

### Actions

- Built an ephemeral private venv overlay on a read-only Torch 2.4.1 parent;
  installed the checkpoint-matching Diffusers 0.37 API and Helios probe deps.
- Added a protected single-process model-load probe that disables remote
  kernel fetch only during import, records RSS/cgroup/CUDA state, and never
  enters the two-backbone training path.
- Loaded the real six-shard Helios-Base model, froze it, attached the exact
  rank-256 Critic LoRA, materialized it on A100, and measured a full
  GPU->CPU->GPU module round trip.

### Commands

The successful load used commit `bf7a5da1436c367e97eb0f1afba9fbac55ce4957`;
the round trip used `20a8353a2a31b99267bbde7e38736a7c5db58ff2`.
Commands, compact results, and raw hashes are in
`artifacts/manifests/iteration_001_real_score_load.json`.

### Results

- **Measured, real Helios Score:** 14,312,427,584 frozen Base elements plus
  1,246,314,496 Critic-LoRA elements across 1,913 parameter tensors.
- **Measured:** final parameter bytes 31.36 GiB; CUDA allocated 31.48 GiB,
  reserved 31.50 GiB, and driver free 7.41 GiB.  The isolated model fits.
- **Measured:** the protected load/cast peaked at 82.89 GiB process RSS; host
  cast was 27.9--39.6 s across repeated runs.
- **Measured:** initial module H2D was 12.87 s.  A steady module D2H was
  18.28 s and the following H2D was 1.33 s.
- **Measured:** after D2H, PyTorch allocated memory was zero, but reserved
  memory stayed at 31.50 GiB and driver free memory stayed at 7.31 GiB.
- **Not Run:** Real/Fake Score forward, Critic backward, optimizer state, EMA,
  or a true Helios-Distilled Generator.

### Interpretation

The hypothesis is verified.  One physical backbone fits, while the source's
simultaneous two-backbone request cannot.  More importantly, a component's
residency cannot be represented by a binary CPU/GPU flag: allocated,
same-process reusable reservation, and driver-free memory differ by over
31 GiB after offload.  The contiguous PCIe model also underpredicts exposed
module movement by up to an order of magnitude because tensor allocation,
page faults, and 1,913-tensor granularity are part of the real path.

This strengthens a shared allocator/residency-pool direction.  It does not yet
show an end-to-end speedup, and asynchronous prefetch is useful only if a long
independent compute window can hide these measured costs.

### Files Changed

- scripts/distillschedule/probe_model_load.py
- scripts/distillschedule/summarize_iteration1.py
- artifacts/manifests/helios_probe_env_node0.json
- artifacts/manifests/iteration_001_real_score_load.json
- docs/00_hardware_feasibility.md
- research_log.md

### Failure or Limitation

The system Torch 2.1 failed with Diffusers 0.36 (`torch.xpu` API mismatch).
Diffusers 0.36 also lacked `apply_lora_scale`; checkpoint metadata identified
0.37 as the matching API.  Two early protected probes failed on probe-only
argument/import bugs and are retained.  The successful loader explicitly
requested `low_cpu_mem_usage=True`, whereas the public custom loader defaults
to `False`; this is not an unmodified cold-start result.

### Next Highest-Value Question

Can a real, minimum-shape Real/Fake Score forward execute within the remaining
7.41 GiB, and what activation/workspace peak does one transformer call add?

### Next Command

Extend the protected probe with a source-derived minimum valid input and run a
single no-grad forward before adding Critic backward or any Runtime policy.

## Iteration 2

### Question

Can a real Helios 14B physical Score execute its minimum valid Real/Fake
no-grad forward within the 7.41 GiB remaining on one A100-40GB?

### Hypothesis

A minimum no-history transformer call will fit because parameters dominate at
this shape, but this will not establish feasibility at Stage-3 video shapes or
with packed history attention.

### Importance

An OOM even at the minimum shape would force immediate sharding. A successful
minimum call instead lets the next experiment isolate how latent/history size
consumes the narrow remaining memory and where the sharding boundary lies.

### Evidence Needed

- A forward through the real `HeliosTransformer3DModel` and exact Base weights.
- Both adapter-disabled Real Score and adapter-enabled Fake Score modes.
- Input/output shape and dtype, forward time, baseline allocation, and peak.
- Preserved failure evidence and explicit limits on kernel/checkpoint fidelity.

### Refutation Condition

The minimum no-grad call OOMs, changes the output contract, or requires more
than the available 7.41 GiB before any realistic history is introduced.

### Actions

- Extended the protected 14B loader with source-signature-derived synthetic
  inputs and Real/Fake adapter modes.
- Ran the minimum input `[1,16,1,2,2]` plus one 4096-wide text token, with no
  short/mid/long history tensors.
- Preserved an output-wrapper incompatibility from the first run, then used
  `return_dict=False`, which is the existing Helios `compute_kl_grad` path.
- Added a hashed iteration manifest and updated the hardware gate.

### Commands

The successful probe used commit
`8cc46118a5f84b4f7ac25275baaf908fc7c12111`. Parsed arguments, environment,
raw logs, and SHA-256 hashes are recorded in
`artifacts/manifests/iteration_002_minimal_score_forward.json`. The original
shell line was not captured verbatim; future probes will record `sys.argv` and
selected environment fields in the raw JSON rather than reconstructing it.

### Results

- **Measured, real 14B physical Score:** both Real and Fake mode returned a
  `[1,16,1,2,2]` BF16 tensor without OOM.
- **Measured:** resident baseline was 31.4751 GiB CUDA allocated; the maximum
  observed allocation was 31.4843 GiB.
- **Measured:** the cold, first Real call took 538.73 ms and added 9.41 MiB to
  peak allocation.
- **Measured:** the second, warm Fake call took 127.89 ms and added 1.28 MiB
  relative to its already elevated baseline.
- **Measured:** initial Real and Fake output checksums are equal because the
  architecture-real but untrained Critic LoRA has zero-initialized output
  projections.
- **Not Run:** representative video/history shape, varlen FlashAttention,
  backward, optimizer, EMA, or a full Stage-3 iteration.

### Interpretation

The hypothesis is verified only at the minimum no-history boundary. Parameter
residency, not activation, dominates this artificial call, but the result says
nothing yet about real packed histories. The 4.1x Real/Fake timing difference
is order/warm-up confounded and must not be presented as an adapter result.
The next high-value step is therefore not more minimum-shape timing: it is to
recover the exact Stage-3 latent and history shapes and drive a representative
synthetic forward with the real model.

### Files Changed

- scripts/distillschedule/summarize_iteration2.py
- artifacts/manifests/iteration_002_minimal_score_forward.json
- docs/00_hardware_feasibility.md
- research_log.md

### Failure or Limitation

The local assets do not contain a trained Critic LoRA or Helios-Distilled
Generator. The offline kernel shim supports this no-mask SDPA path but is not a
valid performance substitute for Helios packed-history attention. The first
run's `Transformer2DModelOutput(logits=...)` failure was a stable Diffusers
wrapper mismatch after computation, not a model-forward failure.

### Next Highest-Value Question

What are the actual Stage-3 training latent, per-section history, packed index,
and text shapes, and at which realistic shape does an isolated Score exhaust
the remaining A100 memory?

### Next Command

Trace `dataloader_dmd.py` through `_generator_loss`, `gt_all_data`, and
`compute_kl_grad`; encode the derived contracts into a shape manifest before
running the next expensive 14B probe.

## Iteration 3

### Question

What exact tensor/token contract reaches Real/Fake Score in the default
Stage-3-Post configuration, and can each no-grad call fit on one A100-40GB?

### Hypothesis

The default one-section GT-history call will fit narrowly, but the enabled
rank-256 Critic adapter will consume materially more temporary memory than
adapter-disabled Real Score and leave too little headroom for Critic training
state.

### Importance

The minimum probe could not distinguish parameter pressure from realistic
sequence pressure. This experiment establishes the actual no-grad Score
boundary and determines whether the next priority is attention/activation
reduction or distributed Critic gradients/optimizer state.

### Evidence Needed

- Source-derived target, history, index, text, and patch-token shapes.
- A reproducible, hashed shape-contract artifact.
- Separate first-call Real and Fake measurements with exact Base weights.
- Allocated/reserved/driver-free memory at residency and forward boundaries.
- Explicit distinction between real architecture and missing trained adapter.

### Refutation Condition

Either default-shape call OOMs, or Fake Score has no material peak-memory
increase over Real Score and leaves enough memory for unsharded Critic
training state.

### Actions

- Traced `dataloader_dmd.prepare_stage1_latent` through
  `prepare_stage1_clean_input_from_latents`, `train_helios` noise creation,
  `compute_kl_grad`, and transformer patching.
- Added a deterministic shape-contract generator and tests.
- Extended the protected probe with GT-history tensors/indices, exact argv and
  environment capture, seed control, and optional warm-up support.
- Ran default-shape Real and Fake Score once each in separate clean processes.
- Preserved a runner-only failure caused by assuming `/usr/bin/time`; reran
  with Bash's available `time` builtin.

### Commands

Exact argv, selected environment, commit, raw logs, and hashes are recorded in
`artifacts/manifests/iteration_003_stage3_score_forward.json`. Both successful
runs used commit `04e4b59968ab7ec75391af52b37031fc66a294fb` and synthetic tensors
from `iteration_003_stage3_shape_contract.json`.

### Results

- **Source-Derived:** default target is `[1,16,9,48,80]`; short/mid/long
  histories are `[1,16,{2,2,16},48,80]`; text is `[1,512,4096]`.
- **Source-Derived:** patching yields 8,640 target tokens and 2,400 history
  tokens, for a dense 11,040-token default self-attention sequence.
- **Measured, Real Score:** 2.825 s, 3.393 GiB incremental peak, 34.8767 GiB
  total allocated peak, 36.7227 GiB reserved, 2.1644 GiB driver-free after.
- **Measured, Fake Score:** 3.557 s, 4.160 GiB incremental peak, 35.6432 GiB
  total allocated peak, 37.6523 GiB reserved, 1.2347 GiB driver-free after.
- **Measured:** both output shapes are `[1,16,9,48,80]` BF16 and both calls
  complete without OOM.
- **Not Run:** Critic backward, optimizer step, ZeRO buckets, Generator, EMA,
  or end-to-end Stage-3.

### Interpretation

The hypothesis is verified. No-grad Score attention/temporaries are large but
not themselves the A100 failure boundary. The enabled Critic adapter adds
about 0.77 GiB to peak allocation and leaves only 1.24 GiB driver-free. The
next bottleneck is therefore expected at Critic gradient/bucket/optimizer
materialization, not at the no-grad Real/Fake call.

Two paper-scope corrections follow from source evidence. First, default
Stage-3-Post has exactly one DMD section, so multi-section scheduling cannot be
claimed as the source of a default-config speedup; it must be validated on the
self-forcing/multi-section branch or a second workload. Second, default
non-NaViT attention passes no varlen mask: history indices encode temporal
position, but do not by themselves create a packed scheduling boundary.

### Files Changed

- scripts/distillschedule/derive_stage3_shapes.py
- scripts/distillschedule/probe_model_load.py
- scripts/distillschedule/summarize_iteration3.py
- tests/test_stage3_shape_contract.py
- artifacts/manifests/iteration_003_stage3_shape_contract.json
- artifacts/manifests/iteration_003_stage3_score_forward.json
- docs/00_hardware_feasibility.md
- docs/01_source_of_truth.md
- research_log.md

### Failure or Limitation

The public `demo_data/ultravideo-long` files and trained Critic adapter are
missing. Inputs are synthetic but shape/architecture exact. The repository's
optional FlashAttention-2 package was not fetched; the default no-mask call
used PyTorch SDPA, so timings are A100 feasibility measurements rather than
upstream-kernel-equivalent throughput. Each timing is a first forward.

### Next Highest-Value Question

Does a default-shape Critic forward/backward fit with checkpointing on one GPU,
and if not, is the default eight-GPU ZeRO-2 plan sufficient once gradient and
communication-bucket residency are measured?

### Next Command

Add a protected checkpointed Critic-loss/backward mode, preserve the expected
single-GPU OOM boundary, then run the same contract with the smallest real
ZeRO-2 group that can complete.

## Iteration 4

### Question

Does the default Stage-3 Critic training call fit on one A100-40GB with
gradient checkpointing, and is allocator fragmentation rather than capacity
the immediate failure mechanism?

### Hypothesis

The minimum probe will materialize the exact 4.64 GiB FP32 Critic-LoRA
gradient set, while the default shape will fail at the training-state boundary.
If expandable_segments:True makes the default call fit, the boundary is
primarily allocator fragmentation; otherwise it is a real
parameter-plus-forward-temporary capacity problem.

### Importance

This distinguishes three very different next actions: allocator tuning,
gradient/optimizer sharding, or reducing parameter/forward temporary
residency. It also tests whether the upstream ZeRO-2 choice can address the
first A100-40GB failure rather than assuming that any multi-GPU launch will.

### Evidence Needed

- A successful checkpointed backward at a controlled minimum shape.
- Exact materialized gradient tensor/element/byte counts.
- A default-contract training attempt with the OOM site and CUDA peaks.
- A same-contract allocator falsification run.
- Clear phase attribution: forward, loss, or backward.

### Refutation Condition

The minimum backward does not materialize the expected Critic gradients, the
default shape completes, or expandable segments alone convert the failure to
success.

### Actions

- Added a protected grad-enabled Fake-Score loss/backward mode with exact
  gradient statistics and deterministic seed.
- Ran the minimum no-history checkpointed forward/backward.
- Ran the source-derived default Stage-3 contract with checkpointing.
- Repeated the failed default contract under
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
- Preserved raw JSON/logs and generated a hash-backed Iteration-4 manifest.

### Commands

Exact argv, environment, commit, exception text, raw logs, and hashes are in
artifacts/manifests/iteration_004_critic_backward_boundary.json. The first
two runs used 861eebdf5966dcd3d5c735699c8f1cdf4bd02ed3; the allocator
falsification used fc9b647c2e7eb1b3399d02b9885a7380d1637134.

### Results

- **Measured, minimum shape:** forward 694.32 ms; backward 1,055.74 ms;
  36.1732 GiB peak allocated.
- **Measured, exact gradients:** 812 tensors, 1,246,314,496 FP32 elements,
  4,985,257,984 bytes (4.6429 GiB), exactly matching the static trainable
  Critic-LoRA count.
- **Measured, default allocator:** default-contract checkpointed forward OOMs
  before producing output or constructing loss; max allocated reaches
  37.7725 GiB and the failed PEFT LoRA branch requests another 584 MiB.
- **Measured, expandable allocator:** the same forward again OOMs at the same
  584 MiB allocation; max allocated reaches 38.3890 GiB and backward is not
  reached.
- **Not Run:** optimizer state/step, DeepSpeed buckets, actual eight-GPU
  ZeRO-2, Generator, EMA, and full Stage-3.

### Interpretation

The minimum result validates the probe and the expected 4.64 GiB gradient
surface. The default-shape part of the hypothesis is stronger than expected:
the immediate boundary is not gradient or optimizer materialization, but the
gradient-enabled checkpointed forward itself. The allocator-only hypothesis
is refuted.

The upstream ZeRO-2 plan partitions optimizer and gradient state but retains
replicated parameters, so it cannot by itself remove this measured
parameter-plus-forward-activation/temporary boundary. Running unchanged
eight-GPU ZeRO-2 would add communication/bucket state without reducing the
failing forward footprint. The next experiment should instead test the
repository's existing FFN/QKV tiled-linear primitive as a temporary-reduction
baseline, then move to ZeRO-3/FSDP if that primitive fails or violates
distributed gradient semantics.

### Files Changed

- scripts/distillschedule/probe_model_load.py
- scripts/distillschedule/summarize_iteration4.py
- artifacts/manifests/iteration_004_critic_backward_boundary.json
- docs/00_hardware_feasibility.md
- docs/01_source_of_truth.md
- research_log.md

### Failure or Limitation

This is an isolated Score/Critic probe with exact Base weights and exact
architecture but a randomly initialized Critic adapter. Synthetic inputs obey
the exact source-derived contract; demo data are absent. PyTorch SDPA replaces
the unavailable optional FlashAttention-2 package. No optimizer, DeepSpeed
wrapper, Generator, EMA, or parameter update ran, so this is a feasibility
boundary rather than Stage-3 throughput.

### Next Highest-Value Question

Can Helios's already-checked-in tiled-linear primitive reduce the real
default-shape FFN temporary enough to complete forward/backward, while
preserving exact outputs and parameter gradients in an unwrapped baseline?

### Next Command

Add a protected --tiled-linear-shards probe mode, first verify the primitive
against ordinary forward/backward on a small deterministic module, then run
the default Stage-3 Critic contract with four tiles and inspect both peak
memory and complete gradient coverage.

## Iteration 5

### Question

Can an existing Helios memory primitive or a real distributed sharding backend
cross the default-shape Critic training boundary on A100-40GB, and which
mechanism should define the Runtime's first hardware backend?

### Hypothesis

Four-way tiled FFNs will reduce the failing temporary enough to finish the
single-GPU Critic call. If that is refuted, shortening the six-way modulation
lifetime should complete forward, and ZeRO-2 should then be sufficient for
backward gradient state.

### Importance

This separates a local activation-lifetime problem from a replicated-parameter
problem. The answer determines whether DistillSchedule should prioritize
automatic kernel/lifetime selection, ZeRO-2 placement, or true parameter
sharding.

### Evidence Needed

- Numeric and hook semantics of the checked-in tiled-linear primitive.
- Default-shape single-GPU runs with tiled FFNs and controlled allocator state.
- A semantics-preserving modulation-lifetime experiment.
- Real two/eight-GPU ZeRO-2 initialization, forward, and backward.
- Real ZeRO-3 forward, backward, and optimizer step.
- Peak allocated/reserved memory and exact failure phase for every negative.

### Refutation Condition

Tiled FFNs do not cross the forward boundary; phase-local modulation still
cannot complete forward; or tuned ZeRO-2 remains unable to run the default
shape. The positive sharding hypothesis is refuted if ZeRO-3 also cannot
complete a real default-shape Critic update.

### Actions

- Added CPU/CUDA semantic probes for the repository's tiled-linear primitive.
- Added protected model-probe switches for four/eight-way tiled FFNs.
- Implemented an opt-in, component-wise phase-local modulation materialization
  with bitwise component tests.
- Ran single-GPU minimum and default-shape tiled/lifetime variants.
- Implemented a rank-local, hashable DeepSpeed Critic probe with serialized
  host loading, configurable ZeRO stage/buckets, and CUDA/RSS snapshots.
- Ran two- and eight-GPU ZeRO-2 probes.
- Ran two-GPU ZeRO-3 minimum, default forward/backward, and optimizer-step
  probes.
- Diagnosed and removed a mixed-dtype persistent-parameter all-gather failure.

### Commands

Exact distributed argv/configuration, per-rank environment, source commit,
exceptions, raw logs, and hashes are recorded in
artifacts/manifests/iteration_005_sharding_and_lifetime.json.

### Results

- **Measured, tiled semantics:** CPU/CUDA output, input gradients, and parameter
  gradients match; CUDA max parameter-gradient difference is 7.15e-7.
- **Measured, tiled hazard:** parameter hooks fire once per tile and a
  parameter-only autograd edge is lost.
- **Measured, tiled real model:** four/eight tiles still OOM in default forward
  on a 1.26 GiB modulation allocation.
- **Measured, phase-local lifetime:** default single-GPU forward completes in
  3.633 s, but backward OOMs on a 216 MiB attention-LoRA allocation.
- **Measured, ZeRO-2 two GPUs:** minimum configuration OOMs during engine
  initialization while cloning a 2.32 GiB partition.
- **Measured, ZeRO-2 eight GPUs minimum:** success; 1.277 s forward, 0.712 s
  backward, 32.759 GiB maximum allocated.
- **Measured, ZeRO-2 eight GPUs default:** even with tuned 64 MiB-equivalent
  buckets and phase-local modulation, forward OOMs on a 584 MiB FFN-LoRA
  allocation; backward is not reached.
- **Measured, ZeRO-3 two GPUs default:** forward 4.054 s, backward 8.578 s,
  AdamW step 0.713 s; backward peak is 35.132 GiB allocated and 36.646 GiB
  reserved.
- **Measured:** ZeRO-3 engine-ready allocation is 20.362 GiB versus 32.015 GiB
  for eight-GPU ZeRO-2.
- **Measured negative:** a 100k persistent-parameter threshold fails after
  optimizer step with a BF16/FP32 all-gather dtype mismatch; threshold zero
  succeeds.
- **Not Run:** Generator, EMA, complete Stage-3, trained adapter, second node,
  optional FlashAttention-2, and steady-state repetitions.

### Interpretation

The original tiled-first hypothesis is refuted. Tiled computation exposes both
a different 1.26 GiB lifetime hotspot and unsafe distributed hook semantics.
Phase-local materialization proves that activation lifetime is real, but it is
not the decisive complete solution.

The stronger finding is that default Critic feasibility is controlled by
parameter residency. ZeRO-2 remains replicated regardless of GPU count and
fails in forward even after gradient/bucket tuning. ZeRO-3 is the first tested
configuration that completes default forward, backward, and optimizer step,
and it needs only two A100-40GBs for this isolated component.

This revises the Runtime direction: parameter-sharded physical state must be a
first-class backend, while tiled/lifetime choices are secondary local
feasibility actions. It also reveals a shared-state requirement absent from a
simple cost model: mixed-dtype frozen/LoRA parameters constrain which ZeRO-3
states may be persisted or coalesced.

### Files Changed

- helios/modules/transformer_helios.py
- scripts/distillschedule/probe_model_load.py
- scripts/distillschedule/probe_tiled_linear_semantics.py
- scripts/distillschedule/probe_deepspeed_critic.py
- scripts/distillschedule/summarize_iteration5.py
- tests/test_tiled_linear_semantics.py
- tests/test_modulation_lifetime.py
- artifacts/manifests/iteration_005_sharding_and_lifetime.json
- docs/00_hardware_feasibility.md
- docs/01_source_of_truth.md
- docs/03_profile_report.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

The successful call is an isolated Score/Critic update with real Base weights,
exact architecture, and exact source-derived shapes, but random Critic adapter
values and synthetic data. It is not a complete Stage-3 iteration and does not
validate convergence. Optional FlashAttention-2 was unavailable. Timings are
first-call feasibility measurements. The successful optimizer step returned
without error, but no before/after parameter digest was recorded.

### Next Highest-Value Question

Can a structure-equivalent multi-component workload reproduce the measured
shared-state, update-frequency, gradient-boundary, and sharding decisions, so
that Runtime planning can be implemented and ablated without the missing
Helios-Distilled Generator?

### Next Command

Implement the Level-3 mini workload with Generator, shared Real/Fake Score,
Critic LoRA, EMA, nested no-grad/selected steps, and Full-Graph versus cached
input-gradient/replay modes; first assert loss/gradient/update equivalence,
then profile Naive, hand-written, and budget-aware schedules.

## Iteration 6

### Question

Can a structure-equivalent multi-component workload reproduce the measured shared-state,
frequency, nested-axis, and gradient-boundary decisions, and does automatic selection
beat a fixed gradient strategy?

### Hypothesis

Full Graph, Cache Input Gradient, and selected-step Generator Replay can preserve the
same update semantics. Replay will lower peak memory whenever a retained Generator
graph overlaps Score/GAN computation, and a profile-driven planner can select it only
when the hardware budget requires it.

### Importance

The real Helios-Distilled Generator asset is absent, so this Level-3 gate determines
whether the Runtime abstractions are executable rather than merely documentary. It
also tests whether gradient-boundary selection is a plausible contribution beyond
Helios's existing cache-grad path.

### Evidence Needed

- One-step gradients, parameters, EMA, and loss/checksum equivalence.
- A graph that separates logical Real/Fake/Critic/GAN nodes from one Score state.
- Observable nested forward counts, selected gradient position, and update frequencies.
- GPU peak memory and end-to-end time for all three strategies.
- A branch ablation with GAN disabled, matching the official default branch property.
- Budget-specific planner decisions derived from measured profiles.

### Refutation Condition

Any strategy changes Generator/Critic/EMA updates; replay does not lower memory when a
differentiable auxiliary graph overlaps Generator state; or a fixed replay policy also
wins in the GAN-disabled DMD-like branch.

### Actions

- Implemented typed logical/physical Runtime IR and graph validation.
- Implemented a measured-profile budget planner and physical-state-aware action emitter.
- Implemented a Level-3 workload with Generator, shared Score modes, DMD, Critic, EMA,
  optional GAN/Reward, nested axes, continuous state, and update frequencies.
- Implemented Full Graph, Cache Input Gradient, and selected-step Generator Replay.
- Removed redundant replay-state clones after the first no-GAN profile exposed their
  artificial cost; reran all final profiles from a clean commit.
- Added deterministic semantic, frequency, sharing, and planner tests.
- Profiled GAN-enabled and GAN-disabled branches for 31 steady-state iterations and
  recorded synchronized phase traces.

### Commands

Exact commands, configuration, environment, commit, raw JSON, phase traces, logs, and
SHA-256 hashes are recorded in
`artifacts/manifests/iteration_006_gradient_boundary_runtime.json`.

### Results

- **Measured correctness:** 15 tests pass and 2 dependency-gated tests skip. One-step
  Generator gradients/updates, Critic gradients/updates, and EMA match across all three
  strategies; replay reconstructs the selected output exactly on CPU FP32.
- **Measured, GAN enabled:** Full Graph is 86.611 ms and 1.054381 GiB peak allocated;
  Cache Input Gradient is 86.613 ms and 1.054380 GiB; Replay is 89.035 ms and
  0.757108 GiB.
- **Measured tradeoff:** GAN-enabled Replay reduces peak allocated memory by 28.19%
  versus Full Graph at 2.80% median-time overhead.
- **Measured, GAN disabled:** Full Graph is 76.238 ms and 0.663358 GiB; Cache is
  76.080 ms and 0.663357 GiB; Replay is 78.949 ms and 0.725857 GiB.
- **Measured negative:** GAN-disabled Replay increases memory by 9.42% and time by
  3.56%; Cache and Full Graph differ by only 1,024 peak allocated bytes.
- **Measured forward dominance:** the two nested Generator rollouts consume 71.9–72.2%
  of traced phase CUDA time with GAN and about 82.0% without GAN.
- **Measured planner behavior:** the GAN plan uses Replay from 0.76 through 1.05 GiB
  and Full Graph at 1.055 GiB; the no-GAN plan selects Cache and never Replay for any
  feasible tested budget.
- **Not Run:** real Generator, full Stage-3, physical-sharing memory ablation, planner-
  controlled ZeRO-3, offload/prefetch, multi-node executor, or second workload.

### Interpretation

The semantic hypothesis is verified, but the fixed-replay hypothesis is refuted.
Gradient-boundary choice is genuinely conditional: replay is useful when a
Differentiable GAN graph and Generator graph would coexist, but harmful in the default
GAN-disabled DMD-like branch because replay state and extra compute replace no large
overlapping graph. Cache-grad is already present in Helios and shows no meaningful peak
reduction here.

The paper direction therefore narrows again. Gradient awareness is a necessary planner
input, not a standalone optimization. The stronger system problem is coordination of
shared physical state and parameter-sharded backends, with branch-aware gradient
boundaries as one decision. The mini workload provides an executable testbed for that
claim without pretending to be real Helios.

### Files Changed

- runtime/ir.py
- runtime/planner.py
- runtime/workloads/mini_distill.py
- examples/run_mini_distill.py
- scripts/distillschedule/plan_mini_distill.py
- scripts/distillschedule/summarize_iteration6.py
- tests/test_mini_distill.py
- tests/test_runtime_ir.py
- artifacts/graphs/mini_distill_graph.json
- artifacts/manifests/iteration_006_gradient_boundary_runtime.json
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This is a synthetic Level-3 architecture on one A100. It preserves scheduling structure,
not Helios numerical behavior or scale. All model states are resident and the planner
uses measured profiles rather than predicting unseen configurations. GPU performance
runs enable TF32; exact equivalence is gated separately on CPU FP32. The physical-
sharing estimate is implemented, but a duplicated-model memory baseline has not run.

### Next Highest-Value Question

How much memory and time does explicit physical Score sharing save relative to a naive
runtime that instantiates Real, Fake/Critic, and GAN as independent models, and can the
planner combine that state identity with a sharded/materialized backend rather than
only choosing a gradient boundary?

### Next Command

Add a naive duplicated-Score Level-3 baseline and a state-residency executor interface;
first compare exact outputs and actual parameter/peak memory, then attach the measured
ZeRO-3 Critic profile as a backend candidate without modifying the existing Helios
training path.

## Iteration 7

### Question

How much memory and time does physical Score sharing save relative to a naive runtime
that instantiates Real Score, Fake Score, Critic, and optional GAN as independent
models?

### Hypothesis

A logical-model-centric runtime will duplicate static Score parameters without changing
activation demand. A physical-state-aware runtime will preserve exact updates while
removing that duplication; the memory delta should equal the duplicate parameter bytes.

### Importance

The IR only matters if logical/physical identity changes a real schedule or feasibility
boundary. This experiment also distinguishes a necessary Runtime abstraction from a
novel optimization: Helios already implements shared Score state manually.

### Evidence Needed

- Shared and duplicated layouts with identical initial weights and loss semantics.
- Explicit adapter/head version synchronization for duplicated Fake/GAN replicas.
- Generator, Critic, and EMA update equivalence.
- Actual peak allocation and end-to-end time in activation-heavy, parameter-heavy, and
  GAN-disabled configurations.
- Comparison of measured peak delta with exact duplicate parameter bytes.

### Refutation Condition

Duplicated and shared layouts cannot preserve identical updates, physical sharing does
not lower measured peak allocation, or the peak delta is instead dominated by changed
activation behavior.

### Actions

- Added a duplicated Score layout with separate Real, Fake, Critic, and conditional GAN
  physical modules.
- Kept only the Critic replica trainable and synchronized adapter/GAN-head parameter
  versions to inference replicas after every Critic optimizer step.
- Extended the graph with distinct physical states and explicit version-transfer nodes.
- Added shared-versus-duplicated correctness tests.
- Profiled three configurations for 31 steady-state iterations each on one A100.

### Commands

Exact commands, clean source commit, configuration, raw profiles, traces, logs, and
SHA-256 hashes are in
`artifacts/manifests/iteration_007_physical_score_sharing.json`.

### Results

- **Measured correctness:** 18 tests pass and 2 dependency-gated tests skip. Shared and
  duplicated layouts match Generator gradients/updates, Critic gradients/updates, and
  EMA; inference replicas match the post-step Critic adapter/head exactly.
- **Measured, activation-heavy GAN:** shared is 86.494 ms / 1.054380 GiB; duplicated is
  86.594 ms / 1.208362 GiB. Sharing saves 12.74% peak allocation and 0.12% time.
- **Measured, parameter-heavy GAN:** shared is 85.381 ms / 1.132150 GiB; duplicated is
  85.927 ms / 1.935321 GiB. Sharing saves 41.50% peak allocation and 0.64% time.
- **Measured, GAN disabled:** shared is 76.243 ms / 0.663357 GiB; duplicated is
  76.410 ms / 0.766012 GiB. Sharing saves 13.40% peak allocation and 0.22% time.
- **Measured causal check:** peak-allocation deltas equal duplicate Score parameter
  bytes within 1,524 bytes in all configurations.
- **Measured sync cost:** synchronized adapter/head copies are 0.50–4.02 MiB per
  iteration; traced copy phases take 0.192–0.535 ms.
- **Estimated, Not Run:** mapping the measured 31.356-GiB real Score state to three
  default logical copies would add 62.712 GiB per rank; four GAN-enabled copies would
  add 94.068 GiB. Helios already avoids this counterfactual.

### Interpretation

The hypothesis is verified exactly: physical sharing changes residency, and the memory
saving is the removed parameter duplication rather than a change in activation
lifetime. Its throughput effect is negligible in these resident profiles.

This is a required Runtime abstraction but not a Helios speedup or standalone novelty.
The strong baseline already shares the Score backbone. DistillSchedule must demonstrate
value by automatically preserving this state identity while coordinating sharding,
residency, frequency, and conditional branches—not by comparing only with the naive
duplicated baseline.

### Files Changed

- runtime/ir.py
- runtime/workloads/mini_distill.py
- examples/run_mini_distill.py
- tests/test_mini_distill.py
- tests/test_runtime_ir.py
- artifacts/graphs/mini_distill_duplicated_graph.json
- artifacts/manifests/iteration_007_physical_score_sharing.json
- scripts/distillschedule/summarize_iteration7.py
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

All measurements are Level-3 synthetic workloads with resident vanilla-PyTorch models.
The duplicated baseline is intentionally naive and weaker than Helios. It synchronizes
only mutable adapter/head state because frozen backbones never change. The real-Helios
memory implication is an estimate, not a run. No offload, sharding executor, or
multi-node placement is involved.

### Next Highest-Value Question

Can the planner's abstract materialize/release actions control a backend-owned residency
state without violating shared aliases or parameter versions, and do frequency-aware
plans reduce exposed transfers relative to Helios's synchronous whole-model switching?

### Next Command

Implement a synchronous residency backend for the Level-3 workload with shared-alias
reference counting and version checks; measure resident versus frequency-aware offload
under real PCIe bandwidth before adding copy streams or asynchronous prefetch.

## Iteration 8

### Question

Can planner materialize/release actions control backend-owned physical residency without
breaking aliases or parameter versions, and does frequency-aware synchronous offload
provide a useful memory/throughput point?

### Hypothesis

Selected-step Replay creates a safe Generator release boundary that Cache Input
Gradient alone does not. A synchronous backend can exploit that boundary to lower peak
memory, but whole-model copies will remain exposed; update-frequency edges should avoid
unnecessary transitions on Critic-only iterations.

### Importance

Helios already performs synchronous whole-model CPU/GPU switching. Before implementing
copy streams or prefetch, the Runtime needs a correct backend and a measured lower bound
on how much synchronous switching costs. This also tests whether gradient boundaries
actually alter legal residency plans.

### Evidence Needed

- Alias reference counting and mutually exclusive adapter modes.
- Parameter-version validation across optimizer updates and replica synchronization.
- Resident versus time-multiplexed update equivalence on CUDA.
- Peak allocation, end-to-end time, transfer bytes/time, and action counts.
- Cache versus Replay under the same residency policy.
- Generator-frequency-one versus frequency-five transfer counts.

### Refutation Condition

Backend-controlled moves change updates; Replay cannot create a lower-memory legal
plan; Cache permits the same release; synchronous copies are mostly hidden; or frequency
conditions do not change actions.

### Actions

- Implemented a synchronous module residency backend with physical aliases, reference
  counts, adapter-mode conflicts, monotonic parameter versions, and measured events.
- Connected Generator and shared Score calls to backend acquisitions.
- Held Generator state through Full/Cache backward and exposed a release boundary only
  for Replay.
- Added optimizer-safe pre-offload gradient clearing and version updates.
- Added CPU unit tests and a CUDA resident-versus-time-multiplex integration test.
- Profiled four residency/gradient combinations on a parameter-heavy workload.
- Profiled Generator update every fifth iteration and generated budget-specific plans.

### Commands

Exact commands, source commit, configuration, per-transition records, raw profiles,
traces, logs, and SHA-256 hashes are in
`artifacts/manifests/iteration_008_synchronous_residency.json`.

### Results

- **Measured correctness:** 22 tests pass and 2 dependency-gated tests skip. The CUDA
  integration test gives bit-exact Generator, Critic, and EMA updates for resident and
  time-multiplexed Replay.
- **Measured, resident Cache:** 85.788 ms, 1.132150 GiB peak allocated.
- **Measured, resident Replay:** 88.253 ms, 0.921082 GiB; 18.64% lower peak for 2.87%
  time overhead.
- **Measured, synchronous multiplexed Cache:** 217.325 ms, 1.132150 GiB. Its live
  Generator graph produces 45 `retain_active` decisions, so swapping adds 153.33% time
  without reducing peak memory.
- **Measured, synchronous multiplexed Replay:** 301.675 ms, 0.653359 GiB. It saves
  42.29% peak versus resident Cache but increases median time by 251.65%.
- **Measured transfer exposure:** Cache copies 1.575 GiB and 143.63 ms per iteration;
  Replay copies 2.079 GiB and 212.85 ms. Copies account for 64.7% and 69.8% of
  end-to-end time respectively.
- **Measured bandwidth:** effective pageable module-copy bandwidth is 9.43–11.31 GiB/s,
  well below the earlier pinned-buffer microbenchmark.
- **Measured frequency effect:** Generator update every fifth step reduces Replay
  transfer events from 8.0 to 4.8 per iteration, bytes from 2.079 to 1.248 GiB, and
  transfer time from 212.85 to 128.18 ms: approximately 40% each.
- **Measured planner:** 0.66–0.80 GiB selects synchronous Replay; 0.93–1.10 GiB selects
  resident Replay; at 1.14 GiB it selects resident Cache.

### Interpretation

The legal-schedule hypothesis is verified, and the performance hypothesis is negative.
Replay is necessary to offload Generator before Score when a differentiable auxiliary
edge exists. Cache Input Gradient does not release the retained Generator graph, so a
state-only scheduler that ignores gradient lifetime performs useless copies and retains
the original peak.

Synchronous whole-model offload is a capacity fallback, not an efficient steady-state
plan: copy time dominates. Frequency awareness removes actions exactly when a branch is
absent, but cannot hide the remaining transfers. The next comparison should move the
small `x_fake`/`dL/dx` boundary tensors between dedicated Generator and Score GPU groups
instead of moving hundreds of MiB (and eventually 31 GiB) of weights.

### Files Changed

- runtime/residency.py
- runtime/workloads/mini_distill.py
- runtime/__init__.py
- examples/run_mini_distill.py
- tests/test_residency.py
- tests/test_residency_integration.py
- scripts/distillschedule/summarize_iteration8.py
- artifacts/manifests/iteration_008_synchronous_residency.json
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This is synchronous pageable module transfer in a Level-3 workload. It is not pinned,
asynchronous, sharded, or overlapped. CPU optimizer state is absent because the mini
workload uses SGD. The parameter-heavy model is about 0.52 GiB across Generator and
Score, not 14B. The frequency result mixes Generator-update and Critic-only iterations
and is reported with branch-specific means in the manifest. No multi-GPU execution is
involved.

### Next Highest-Value Question

Is it more efficient to keep Generator and Score resident on separate GPU groups and
transfer only the gradient-boundary tensors than to time-multiplex weights, including
when the boundary crosses the measured inter-node network?

### Next Command

Implement a two-rank Level-3 split executor: rank 0 owns Generator/EMA, rank 1 owns
Score/Critic, and NCCL sends `x_fake` plus cached `dL/dx`; first prove one-step equality,
then compare one-node NVLink and two-node latency, bytes, memory, and end-to-end time.

## Iteration 9

### Question

Is it more efficient to keep Generator and Score resident on dedicated GPU groups and
move only `x_fake`/`dL/dx_fake` than to time-multiplex whole model states, including
across the measured inter-node link?

### Hypothesis

Dedicated groups will preserve Replay semantics, reduce transfer volume by over an
order of magnitude, and beat synchronous whole-model offload. Cross-node transport will
not dominate for the source-sized latent boundary, but a sequential split may remain
slower than a one-GPU resident plan.

### Importance

Iteration 8 showed that synchronous pageable state movement consumes 69.8% of runtime.
If small gradient-boundary tensors can cross groups cheaply, the planner needs a
placement decision, not merely better `.to()` calls.

### Evidence Needed

A deterministic split-vs-single-process correctness test; local and two-node end-to-end
profiles; per-role memory; exact protocol bytes; isolated bidirectional P2P at both the
mini-workload and source-derived Stage-3 tensor sizes; comparisons with resident and
time-multiplexed Replay.

### Refutation Condition

Reject the placement hypothesis if updates diverge, boundary traffic approaches weight
traffic, or measured network latency dominates the split iteration. Reject universal
throughput benefit if resident execution remains faster or the extra GPU is essential.

### Actions

- Added Generator/EMA and shared Score/Critic role executors.
- Added a two-rank NCCL protocol with cached input-gradient return and fresh Critic fake.
- Added a deterministic split equivalence test.
- Ran the parameter-heavy split on two local A100s and on GPU2 of each node with
  `mlx5_2` GPUDirect RDMA.
- Added an isolated bidirectional P2P benchmark at exact target/history sizes.
- Compared against clean resident Cache, resident Replay, and synchronous Replay
  profiles from Iteration 8.

### Commands

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --output-dir artifacts/profile/mini_split_local_001 \
  --batch-size 2 --tokens 512 --hidden-size 2048 \
  --generator-depth 16 --score-depth 16 --adapter-rank 128 \
  --sections 3 --denoise-steps 4 --warmup 3 --iterations 31

# One process on each node, CUDA_VISIBLE_DEVICES=2 on both:
NCCL_IB_HCA=mlx5_2 NCCL_NET_GDR_LEVEL=2 NCCL_SOCKET_IFNAME=eth0 \
  torchrun --nnodes=2 --nproc-per-node=1 ... \
  scripts/distillschedule/probe_split_distill.py \
  --output-dir artifacts/profile/mini_split_crossnode_001 ...

PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/bench_p2p.py \
  --dtype bfloat16 --bytes 1105920 3563520 8232960 12427264 ...

PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache python \
  scripts/distillschedule/summarize_iteration9.py
```

Full arguments and launch environment are preserved in the rank JSON and raw logs.

### Results

- Test suite: 23 passed, 2 dependency-gated skips.
- Local split: 94.459 ms median, 0.622109 GiB maximum per-GPU peak.
- Cross-node split: 94.923 ms, identical memory, +0.491% versus local.
- Isolated 8 MiB P2P: 0.1067 ms local and 0.4412 ms cross-node.
- Local split versus synchronous Replay: 68.69% lower median time and 4.78% lower
  maximum per-GPU peak.
- Local split versus resident Replay: 7.03% slower, 32.46% lower maximum per-GPU peak,
  and one additional GPU.
- Local split versus resident Cache: 10.11% slower, 45.05% lower maximum per-GPU peak,
  and one additional GPU.
- Mini boundary: 24.000 MiB/iteration versus 2.079 GiB/iteration of weight copies,
  an 88.72x byte reduction.
- Source-derived Helios boundary: 8,232,960 BF16 bytes for two state sends plus one
  gradient; composed cross-node median is **Estimated** at 0.614 ms.
- Local/cross final parameter, EMA, loss checksums are exact; maximum recorded numeric
  delta is 1.12e-10 and replay error is zero.

### Interpretation

The main hypothesis is verified against synchronous offload, and network dominance is
refuted for the tested boundary. The universal throughput hypothesis is refuted:
dedicated groups spend a second GPU and remain slower than an already-resident
single-GPU plan because the roles are sequential. The planner should select among
resident, split, and offload based on capacity and GPU availability. The next potential
speed benefit must come from a parameter-version-safe pipeline, not from placement
alone.

### Files Changed

- runtime/workloads/split_distill.py
- scripts/distillschedule/probe_split_distill.py
- scripts/distillschedule/bench_p2p.py
- tests/test_split_distill.py
- scripts/distillschedule/summarize_iteration9.py
- artifacts/manifests/iteration_009_split_placement.json
- docs/00_hardware_feasibility.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This is Level 3, not a real 14B Generator/Score step. The mini protocol omits histories
and text; exact-size P2P covers their bytes separately. Blocking receive durations
include remote compute and are not reported as network latency. The roles are not
pipelined, and comparisons against one-GPU resident baselines have different GPU cost.

### Next Highest-Value Question

Can multiple microbatches safely overlap Generator rollout and Score work without
crossing Generator/Critic optimizer-version barriers, and does the overlap recover the
7-10% split overhead without changing gradients?

### Next Command

Implement a two-microbatch, one-update pipeline with explicit version barriers; first
compare loss/gradients/updates to a non-overlapped gradient-accumulation baseline, then
profile local and cross-node overlap, idle time, and end-to-end latency.

## Iteration 10

### Question

Can multiple microbatches overlap Generator rollout and Score work without crossing
Generator/Critic optimizer-version barriers, and does that overlap recover the
sequential split overhead?

### Hypothesis

A one-item wavefront can overlap `Generator(i+1)` with `Score(i)` while all
microbatches observe fixed Generator and Score versions. It should improve split
throughput with exact updates; one microbatch should show no benefit, and wider
wavefronts may trade memory for diminishing latency gains.

### Importance

Iteration 9 established that dedicated groups avoid weight swapping but execute
sequentially and remain 7-10% slower than resident one-GPU baselines. A valid pipeline
is the first mechanism that can use the otherwise idle second role without weakening
training semantics.

### Evidence Needed

An independent accumulated-update correctness test; clean local sequential/pipeline
profiles; the same comparison across nodes; per-role compute and blocking-wait time;
optimizer version counts; a 1/2/4/8 microbatch ablation for speed, throughput, and
memory.

### Refutation Condition

Refute the hypothesis if schedule results diverge, a microbatch observes a post-step
version, latency does not improve beyond the one-microbatch noise control, cross-node
transport erases the gain, or memory growth dominates the useful range.

### Actions

- Split Generator and Critic APIs into begin/backward/finish accumulation phases.
- Added one Generator and one Critic optimizer-version barrier per accumulated update.
- Added sequential and one-item-wavefront schedules to the two-rank probe.
- Added synchronized role-compute phase measurement distinct from blocking P2P waits.
- Added a two-microbatch single-process reference test.
- Ran local and cross-node two-microbatch comparisons and a local 1/2/4/8 scan.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --output-dir artifacts/profile/mini_pipeline_local_001/{sequential|pipeline} \
  --batch-size 2 --tokens 512 --hidden-size 2048 \
  --generator-depth 16 --score-depth 16 --adapter-rank 128 \
  --sections 3 --denoise-steps 4 --microbatches 2 \
  --schedule {sequential|pipeline} --warmup 3 --iterations 31

# Same arguments, one visible GPU per node on GPU2/mlx5_2:
NCCL_IB_HCA=mlx5_2 NCCL_NET_GDR_LEVEL=2 NCCL_SOCKET_IFNAME=eth0 \
  torchrun --nnodes=2 --nproc-per-node=1 ...

# Local ablation repeats both schedules for M in {1,4,8}; M=2 is the main run.
HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration10.py
```

Complete arguments, source commit, environment, per-rank events, and hashes are in the
manifest.

### Results

- Tests: 24 passed, 2 dependency-gated skips.
- Two-microbatch local sequential/pipeline: 183.386/162.196 ms; 11.55% latency
  reduction and 13.06% throughput increase.
- Two-microbatch cross-node sequential/pipeline: 184.280/163.073 ms; 11.51% reduction.
- Cross-node pipeline is 0.54% slower than local pipeline.
- Same-placement sequential/pipeline iteration results are bit exact for all 31 updates.
- All accumulated updates increment Generator, EMA, and Critic versions once, not once
  per microbatch.
- Local normalized role-compute occupancy proxy rises 48.94% -> 55.45% at M=2.
- Latency reduction for M=1/2/4/8 is -0.10/11.55/18.21/20.97%.
- Pipeline throughput for M=1/2/4/8 is 21.03/24.66/26.99/28.21 samples/s.
- Maximum per-GPU peak grows 0.624 -> 0.679 -> 0.788 -> 1.007 GiB.

### Interpretation

The version-safe pipeline hypothesis is verified at Level 3. The one-microbatch control
refutes a launch-order artifact, and the cross-node result refutes network erasure for
the tested latent. Parameter versions are now a necessary scheduling primitive: the
Runtime can overlap inside a fixed-version epoch but must not cross either optimizer
barrier. The speed benefit saturates while replay-state memory grows, exposing the next
falsifiable choice between forward-priority batching and streaming replay release.

### Files Changed

- runtime/workloads/split_distill.py
- scripts/distillschedule/probe_split_distill.py
- tests/test_split_distill.py
- scripts/distillschedule/summarize_iteration10.py
- artifacts/manifests/iteration_010_version_safe_pipeline.json
- docs/02_execution_graph.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This remains a Level-3 workload. Phase-time occupancy is not an Nsight GPU-utilization
measurement. The microbatch scan increases effective batch size. The pipeline uses
blocking P2P rather than separate CUDA streams. Its forward-priority boundary phase
retains all replay records and returned gradients until the Generator barrier, so peak
memory grows with wavefront width.

### Next Highest-Value Question

Can returned input gradients be consumed by streaming Generator replay before the
optimizer barrier, bounding replay-state memory while retaining the measured overlap?

### Next Command

Add a streaming-pipeline schedule that replays and releases microbatch `i` after
`Generator(i+1)` overlaps `Score(i)`, but defers the Generator optimizer/EMA step until
all microbatches finish; compare it with batched pipeline at M=4 and M=8.

## Iteration 11

### Question

Can each returned input gradient be consumed by Generator replay before the optimizer
barrier, bounding replay-state memory while retaining or improving pipeline overlap?

### Hypothesis

Sending `x_{i+1}` before replaying `i` lets Score compute `i+1` concurrently with
Generator replay `i`. The replay record and `dL_i/dx_i` can then be released without
advancing Generator parameters. This should preserve exact updates, remove the batched
O(M) replay/gradient lifetime, and avoid a throughput penalty.

### Importance

Iteration 10 gained up to 20.97% latency but raised peak memory from 0.624 to 1.007 GiB
as microbatch width grew. If gradient consumption and parameter update are separate
lifetime events, the Runtime can avoid that false memory/time tradeoff.

### Evidence Needed

A streaming-order unit test against the independent reference; clean M=1/2/4/8 local
profiles; bit-exact batched/streaming results; byte-level peak deltas; phase timing; and
an M=8 two-node validation.

### Refutation Condition

Refute the hypothesis if streaming changes gradients/updates, crosses a version barrier,
loses more throughput than its memory benefit, fails to remove the replay-state slope,
or behaves differently enough across nodes to erase the result.

### Actions

- Added `streaming_pipeline` to the split executor.
- Began Generator accumulation before the wavefront and deferred only optimizer/EMA.
- Sent the next fake before replaying the previous microbatch so Score and replay overlap.
- Released each replay record and returned gradient after backward.
- Extended the independent reference test with an interleaved streaming order.
- Measured M=1/2/4/8 locally and M=8 across GPU2/`mlx5_2`.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --output-dir artifacts/profile/mini_streaming_replay_scan_001/mb${M} \
  --batch-size 2 --tokens 512 --hidden-size 2048 \
  --generator-depth 16 --score-depth 16 --adapter-rank 128 \
  --sections 3 --denoise-steps 4 --microbatches ${M} \
  --schedule streaming_pipeline --warmup 3 --iterations 31

# M=8, one process per node on GPU2/mlx5_2 with the same model arguments:
NCCL_IB_HCA=mlx5_2 NCCL_NET_GDR_LEVEL=2 NCCL_SOCKET_IFNAME=eth0 \
  torchrun --nnodes=2 --nproc-per-node=1 ...

HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration11.py
```

### Results

- Tests: 24 passed, 2 dependency-gated skips.
- Batched and streaming iteration results are bit exact at M=1/2/4/8.
- Streaming versus batched peak saving at M=1/2/4/8: 0/8.51/72/200 MiB.
- Streaming versus batched latency reduction: 0.24/3.61/1.51/0.89%.
- At M=8, streaming reaches 562.202 ms, 28.46 samples/s, and 0.811570 GiB peak;
  batched pipeline is 567.240 ms, 28.21 samples/s, and 1.006882 GiB.
- Streaming M=8 is 21.67% faster than sequential accumulation.
- Beyond M=4, the removed batched replay/gradient slope is exactly 32 MiB per added
  microbatch. The common preloaded-input slope is 24 MiB/microbatch.
- Cross-node streaming M=8 is 564.163 ms (+0.35%), with exact final state/loss and
  maximum recorded numeric delta 1.22e-10.

### Interpretation

The hypothesis is verified. The version edge blocks parameter mutation but does not
block consumption of a completed gradient edge. Separating those events turns the
measured memory/time tradeoff into a Pareto improvement. This is a concrete nested-axis
scheduling opportunity not captured by model-level placement alone: microbatch `i`
releases continuous replay state while `i+1` remains inside the same model-version
epoch.

### Files Changed

- scripts/distillschedule/probe_split_distill.py
- tests/test_split_distill.py
- scripts/distillschedule/summarize_iteration11.py
- artifacts/manifests/iteration_011_streaming_replay.json
- docs/02_execution_graph.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This is Level 3 and the schedule remains hand-written. The probe preloads all input
microbatches, so total peak is not constant. Blocking P2P does not prove copy-stream
overlap. The real 14B Generator is still unavailable, so the 32 MiB slope cannot be
extrapolated numerically to Helios.

### Next Highest-Value Question

Can the Runtime synthesize the streaming wavefront from gradient lifetimes, physical
state identity, parameter-version barriers, and a memory budget rather than selecting a
hard-coded schedule?

### Next Command

Implement a small lifetime-aware wavefront planner over the Runtime IR, make the probe
consume its generated action order, and verify it selects streaming under the measured
M=4/M=8 budgets while retaining batched/sequential counterfactuals.

## Iteration 12

### Question

Can the Runtime synthesize and execute the streaming microbatch wavefront from explicit
gradient lifetimes, parameter-version barriers, measured phase costs, and a memory
budget instead of selecting the schedule by hand?

### Hypothesis

A small candidate enumerator plus liveness-aware memory model will reject batched
accumulation under tight M=4/M=8 budgets, emit the streaming action order, and preserve
the hand-written executor's exact updates with negligible planning overhead.

### Importance

If verified, gradient/version/nested-axis metadata changes an automatic runtime decision
rather than merely documenting a hand optimization. If refuted, the current IR is not
yet actionable and the paper claim must remain a workload-specific scheduling recipe.

### Evidence Needed

- generated candidate action orders and computed replay/gradient liveness;
- held-out predicted versus measured peaks;
- successful execution of the selected plan under the target budget;
- exact loss, gradient, optimizer, and EMA equivalence to manual streaming;
- unit counterfactual showing the planner can select a non-streaming policy;
- planner core latency.

### Refutation Condition

The emitted action order cannot be executed, changes parameter versions or updates,
misses peak memory by enough to violate the target budget, always returns streaming
regardless of costs, or costs materially more than one update.

### Actions

- Added gradient/version-gated `LifetimeAwareWavefrontPlanner` candidate enumeration.
- Added explicit boundary actions and a validating liveness state machine.
- Calibrated a three-term memory model from Iteration 11 M=1 and tensor sizes.
- Refactored the split probe to execute emitted actions in `planned` mode.
- Added malformed-graph, liveness, budget-switch, and execution-equivalence tests.
- Generated M=4/M=8 plans, executed the tight-budget choices, and compared against the
  manual streaming profiles.
- Measured planner core latency over 10,000 calls after 1,000 warmups.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/distillschedule/plan_wavefront.py \
  --microbatches 4 --budgets-gib 0.71 0.73 0.79 \
  --output artifacts/profile/wavefront_planner_001/m4_plans.json
HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/distillschedule/plan_wavefront.py \
  --microbatches 8 --budgets-gib 0.80 0.82 1.02 \
  --output artifacts/profile/wavefront_planner_001/m8_plans.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --output-dir artifacts/profile/mini_planned_wavefront_local_001/m4_budget073 \
  --batch-size 2 --tokens 512 --hidden-size 2048 --generator-depth 16 \
  --score-depth 16 --adapter-rank 128 --sections 3 --denoise-steps 4 \
  --microbatches 4 --schedule planned \
  --wavefront-plan artifacts/profile/wavefront_planner_001/m4_plans.json \
  --wavefront-budget-gib 0.73 --warmup 3 --iterations 31

# Repeat for M=8 with budget 0.82 GiB.
HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration12.py
```

### Results

- Tests: 30 passed.
- The planner requires both gradient and version edges and emits the exact action arrays
  consumed by the executor.
- Batched M=4/M=8 liveness is M replay records, M returned gradients, and M live edges;
  streaming is bounded at 2, 1, and 2 respectively.
- Peak prediction errors are -0.063%/-0.049% for batched M=4/M=8 and conservative
  +1.019%/+0.902% for streaming.
- At 0.73 GiB (M=4) and 0.82 GiB (M=8), only streaming fits. Execution succeeds at
  0.717820 and 0.811570 GiB.
- Planned M=4 is 291.329 ms versus manual 291.922 ms (-0.20%); planned M=8 is
  563.001 ms versus manual 562.202 ms (+0.14%). Peaks and all iteration results are
  identical.
- Planner median core latency is 0.182 ms at M=4 and 0.300 ms at M=8; p95 is
  0.190/0.310 ms.
- At high budgets all policies fit and measured streaming remains fastest. A synthetic
  cost counterfactual selects batched pipeline when it is faster and fits.

### Interpretation

The hypothesis is verified at Level 3. The minimum useful scheduling state is now
operational: physical identity controls capacity, gradient consume/release controls
activation lifetime, version barriers protect semantics, and a repeated axis exposes
an overlap window. The Runtime can derive the same Pareto-improving plan as the manual
implementation with a peak model accurate enough for the tested tight budgets.

### Files Changed

- runtime/wavefront.py
- runtime/__init__.py
- scripts/distillschedule/plan_wavefront.py
- scripts/distillschedule/probe_split_distill.py
- scripts/distillschedule/summarize_iteration12.py
- tests/test_wavefront_planner.py
- tests/test_split_distill.py
- artifacts/manifests/iteration_012_wavefront_planner.json
- docs/02_execution_graph.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This remains one Helios-shaped Level-3 workload. The memory coefficients are calibrated,
not statically inferred; the planner enumerates three templates rather than arbitrary
DAG schedules; and input microbatches remain preloaded. No real 14B Generator execution
or second-workload generalization is demonstrated.

### Next Highest-Value Question

Do the physical-state, gradient-lifetime, version-barrier, and repeated-axis abstractions
transfer to a structurally different image distillation workload without video sections
or continuous history, and does the planner still make a useful budget-dependent choice?

### Next Command

Implement a second image-DMD-style Level-3 workload with independent teacher/student
states, no section/history axis, a cross-component input-gradient edge, different
component update frequency, and the same planner/executor contract; then run
sequential, batched, streaming, and planned profiles with exact-update checks.

## Iteration 13

### Question

Do the physical-state, gradient-lifetime, version-barrier, and repeated-axis
abstractions transfer to a structurally different image distillation workload, and does
streaming retain its Helios-shaped memory/throughput benefit?

### Hypothesis

The wavefront semantics will transfer to DMD2-style image distillation, but independent
Real/Fake guidance states may move the global memory bottleneck away from Generator and
eliminate streaming's feasibility benefit.

### Importance

A second workload is required to distinguish generic runtime mechanisms from Helios
special cases. A negative result is particularly valuable because it constrains the
planner and prevents a universal streaming/replay claim.

### Evidence Needed

- pinned public DMD2 source paths for model identity, gradient, frequency, and update order;
- a distinct logical/physical graph without section/history, shared adapters, or EMA;
- exact batched/streamed updates;
- replicated latency and role-local peak measurements;
- frequency-ablation traffic and version counts;
- evidence identifying the global peak owner.

### Refutation Condition

The existing IR cannot express the DMD2 path, removing Helios-specific features breaks
semantic equivalence, or the supposed bottleneck shift is not visible in role-local
measurements.

### Actions

- Pinned `tianweiy/DMD2` at commit `8d8fa556...` in a clean read-only evidence checkout.
- Reconstructed a one-step image Generator, independent frozen Real teacher, trainable
  Fake guidance/GAN, detached DMD gradient, periodic Generator update, and cached
  pre-update fake guidance path.
- Generalized the terminal wavefront action so graphs without EMA emit
  `optimizer_update` rather than fabricating EMA state.
- Added role-local input residency and preserved DMD2's pre-update fake reuse.
- Exported a 9-node/13-edge graph and added exact-update tests.
- Ran three replicated policy profiles at M=4 and M=8 plus an update-frequency ablation.

### Commands

```bash
git -C /share/project/shuchenweng/external/DMD2 checkout --detach \
  8d8fa55633d47cfb81bbc7a892e7248f9518763f

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --workload image_dmd2 --batch-size 2 --channels 1024 --image-size 16 \
  --generator-depth 12 --guidance-depth 12 --generator-update-frequency 1 \
  --microbatches ${M} --schedule ${POLICY} --warmup 3 --iterations 31 ...

# Frequency ablation: M=8, streaming, generator-update-frequency=4, 32 iterations.
HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration13.py
```

### Results

- Tests: 33 passed.
- DMD2 source confirms independent frozen Real/trainable Fake networks, no-grad DMD
  gradient construction, detached guidance input, periodic Generator updates, and
  every-step guidance updates.
- All rank-0 iteration results are bit exact across all policies and replicates; replay
  maximum error is zero. No EMA state or version is fabricated.
- M=8 median-of-three: sequential 256.898 ms, pipeline 237.038 ms, streaming
  237.129 ms. Pipeline and streaming are effectively tied; both improve about 7.7%
  over sequential.
- Streaming saves 22 MiB on Generator at M=8 (0.248403 -> 0.226918 GiB), but guidance
  remains 0.406716 GiB for every policy. Global peak saving is exactly zero.
- M=4 streaming median-of-three is 140.474 ms versus 147.740 sequential, but ranges
  overlap and run-to-run variability prevents a strong ordering claim.
- Generator frequency 4 executes 8 Generator and 32 guidance updates in 32 measured
  iterations. Communication falls 37.50% (32.000 -> 20.000 MiB/iteration), not 75%.

### Interpretation

The hypothesis is verified. Gradient consume/release, version barriers, and a repeated
microbatch axis are generic beyond Helios. Shared adapters, EMA, video sections, and
continuous history are workload-specific. The performance benefit is topology-dependent:
locally reducing Generator lifetime does not alter feasibility when independent guidance
owns the peak. The current scalar memory model is therefore insufficient.

### Files Changed

- runtime/workloads/image_dmd2.py
- runtime/workloads/__init__.py
- runtime/wavefront.py
- scripts/distillschedule/probe_split_distill.py
- scripts/distillschedule/export_image_dmd2_graph.py
- scripts/distillschedule/summarize_iteration13.py
- tests/test_image_dmd2.py
- tests/test_wavefront_planner.py
- artifacts/graphs/image_dmd2_graph.json
- artifacts/manifests/iteration_013_image_dmd2_generalization.json
- docs/02_execution_graph.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- docs/06_related_work.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This is source-grounded Level 3, not execution of real DMD2 weights/data or a quality
experiment. Latency varies materially across runs, so only replicated medians/ranges are
reported. The external source was inspected but not executed. Automatic planning still
uses a single scalar peak and cannot yet express the guidance-owned floor.

### Next Highest-Value Question

Can a minimal role-indexed memory model correctly preserve Generator lifetime savings,
identify guidance as the global bottleneck, and select the fastest feasible policy
without inventing a global memory benefit?

### Next Command

Extend the lifetime model to estimate per-role peaks and maximum bottleneck role, feed it
three-replicate image policy costs, generate constrained/relaxed plans, execute the
selected plan, and compare its prediction with both role-local measurements.


## Iteration 14

### Question

Can a minimal role-indexed memory model preserve Generator lifetime savings, identify
guidance as the global bottleneck, and prevent false feasibility decisions?

### Hypothesis

A planner that predicts each physical role separately and takes their maximum will
match held-out role peaks within 1%, reject budgets below the independent-guidance
floor, and execute a feasible emitted plan without changing updates.

### Importance

If this fails, the Runtime still cannot use its physical graph to make capacity-safe
placements. If it succeeds, the image workload supplies a direct ablation against a
single scalar memory estimate.

### Evidence Needed

Role-owned input placement, M=1/M=4 calibration profiles, held-out M=8 role peaks,
constrained plans near the measured guidance floor, a Generator-only ablation, and exact
planned-versus-manual updates.

### Refutation Condition

Role predictions exceed 1% error, the bottleneck role is misidentified, the scalar and
role-aware planners make the same decision under a discriminating budget, or the emitted
plan changes training state.

### Actions

- Removed non-owner real samples from the Generator rank and non-owner noise/conditioning
  from the guidance rank.
- Added `RoleMemoryTerm` and `RoleAwareLifetimeMemoryModel`, per-role predictions, and
  bottleneck reporting while preserving the scalar API.
- Calibrated on M=1/M=4 and held out all M=8 role peaks.
- Generated 0.406/0.407/0.45 GiB plans from the three-replicate timing table.
- Ran the selected 0.407 GiB plan and compared actions, states, and peaks with manual
  pipeline execution.
- Ablated the guidance role at a 0.22 GiB budget and compared timing aggregations.

### Commands

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --workload image_dmd2 --microbatches ${M} --schedule ${POLICY} ...

python scripts/distillschedule/plan_image_dmd2_wavefront.py \
  --output artifacts/profile/image_dmd2_role_planner_002/plans.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --workload image_dmd2 --microbatches 8 --schedule planned \
  --wavefront-plan artifacts/profile/image_dmd2_role_planner_002/plans.json \
  --wavefront-budget-gib 0.407 ...

HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration14.py
```

### Results

- Tests: 34 passed.
- Held-out M=8 streaming Generator/guidance predictions are exact; batched Generator is
  conservatively high by 0.839%, and every guidance/global prediction is exact.
- At 0.406 GiB no policy fits; at 0.407 GiB all fit and pipeline is selected by the
  configured median-of-replicate cost.
- The planned action list is consumed exactly; all role-local iteration results match
  manual pipeline with maximum numeric delta zero. Peaks are identical.
- At 0.22 GiB the Generator-only ablation falsely selects streaming, while the role-aware
  planner correctly rejects every policy due to the 0.406716 GiB guidance peak.
- The speed winner is aggregation-sensitive: median-of-run-medians selects pipeline,
  while pooled iteration median selects streaming.

### Interpretation

The memory hypothesis is verified and the throughput sub-hypothesis is refuted. A
physical-role maximum is necessary for feasibility, but a point timing estimate is too
precise for overlapping noisy profiles. The planner is memory-correct and semantics-safe;
its current speed choice should be treated as a tie rather than a robust winner.

### Files Changed

- runtime/wavefront.py
- scripts/distillschedule/probe_split_distill.py
- scripts/distillschedule/plan_image_dmd2_wavefront.py
- scripts/distillschedule/summarize_iteration14.py
- tests/test_wavefront_planner.py
- artifacts/manifests/iteration_014_role_aware_planner.json
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

This remains Level 3. Coefficients are empirically calibrated, per-role budgets assume
homogeneous devices, and point-estimate timing cannot distinguish overlapping policies.
The 235.579 ms planned run lies inside historical variability and is not claimed as a
speedup.

### Next Highest-Value Question

Can an uncertainty-aware cost model avoid unstable policy claims while retaining a
deterministic, capacity-safe choice, and does a fresh confirmatory profile support its
reported tie or winner?

### Next Command

Add replicate distributions and a configurable uncertainty margin to the planner, make
overlapping policies explicit ties with a conservative secondary objective, then execute
a fresh selected plan and compare it against both candidates.


## Iteration 15

### Question

Does minimizing the worst of the three observed replicate medians stabilize the
pipeline-versus-streaming throughput decision?

### Hypothesis

An empirical worst-replicate objective will select streaming from the historical profile
and a fresh paired experiment will preserve that winner while retaining exact updates.

### Importance

Iteration 14 showed that the point-estimate winner changes with aggregation. If a simple
replicate-aware risk objective is stable, the planner can avoid overfitting sub-percent
median differences without a complex statistical model.

### Evidence Needed

A serialized replicate-aware plan, a preselected policy, fresh paired runs with order
alternation, exact state comparisons, and memory peaks.

### Refutation Condition

The fresh median, pooled median, or observed-worst ranking reverses; execution changes
updates; or the expected role-local memory behavior fails to repeat.

### Actions

- Added replicate samples and point/observed-worst selection modes to the Runtime.
- Generated a clean 0.407 GiB plan from the historical three-replicate measurements;
  observed-worst selected streaming instead of point-estimate pipeline.
- Ran five fresh pipeline/planned-streaming pairs, alternating launch order, with 31
  measured iterations after three warmups per run.
- Compared every rank's iteration results, planned action list, and role peaks.

### Commands

```bash
python scripts/distillschedule/plan_image_dmd2_wavefront.py \
  --cost-selection-mode observed_worst_case \
  --budgets-gib 0.406 0.407 0.45 \
  --output artifacts/profile/image_dmd2_uncertainty_planner_001/plans.json

# Five paired runs; order alternated by replicate.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py \
  --workload image_dmd2 --microbatches 8 --schedule planned \
  --wavefront-plan artifacts/profile/image_dmd2_uncertainty_planner_001/plans.json \
  --wavefront-budget-gib 0.407 --warmup 3 --iterations 31 ...

HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration15.py
```

### Results

- Tests: 36 passed.
- Historical observed-worst scores select streaming: 238.235 versus 244.413 ms.
- Fresh pipeline run medians are 231.430–236.881 ms; fresh planned streaming medians
  are 231.343–248.227 ms.
- Fresh median-of-run-medians is 236.608 ms pipeline versus 238.856 ms streaming; pooled
  medians are 235.544 versus 239.219 ms.
- Pipeline wins three paired runs; streaming wins two. The paired median delta is
  +1.975 ms for streaming.
- All updates and selected actions are exact. Streaming again saves 22 MiB on Generator
  but zero globally because guidance remains at 0.406716 GiB.

### Interpretation

The hypothesis is refuted. The “worst of three” objective merely replaces one unstable
point statistic with another. It picked streaming historically, but every reported fresh
aggregate—including observed worst—favors pipeline. The Runtime should expose ambiguity
or apply an explicit performance/memory preference; it should not imply statistical
robustness from a tiny empirical maximum.

### Files Changed

- runtime/wavefront.py
- runtime/__init__.py
- scripts/distillschedule/plan_image_dmd2_wavefront.py
- scripts/distillschedule/summarize_iteration15.py
- tests/test_wavefront_planner.py
- artifacts/manifests/iteration_015_replicate_aware_cost.json
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

Five pairs are still a small Level-3 sample, cluster frequency/contention is not fully
controlled, and no significance claim is made. `observed_worst_case` is transparent
empirical risk, not a confidence bound.

### Next Highest-Value Question

Can the planner represent an explicit throughput-equivalence band and memory Pareto
frontier, so it reports ambiguity rather than inventing a winner from noisy profiles?

### Next Command

Add a user-visible slowdown tolerance and nondominated candidate set, verify that memory
budgets still override the band, then execute the selected policy under historical and
fresh profiles and report whether the decision is invariant.


## Iteration 16

### Question

Can the planner expose pipeline/streaming ambiguity as a Pareto tradeoff and apply an
explicit slowdown preference instead of inventing a noise-derived winner?

### Hypothesis

Both policies will be nondominated; a 1% performance band will select streaming under
both historical and fresh profiles for lower Generator memory, while capacity filtering
will still reject every policy below the guidance floor.

### Importance

Iteration 15 refuted empirical-worst ranking. The Runtime still needs a deterministic
execution choice, but that choice must distinguish measured feasibility from a user's
performance-versus-headroom preference.

### Evidence Needed

Pareto sets, decisions across two independent cost profile sets and multiple tolerances,
a near-floor budget, and exact execution of the emitted plan.

### Refutation Condition

The policies are not both nondominated, the one-percent choice changes across profile
sets, memory preference bypasses capacity, or planned execution changes updates.

### Actions

- Added `performance_band_memory` selection and an explicit relative slowdown tolerance.
- Added per-role Pareto reporting and serialized band membership/selected slowdown.
- Compared historical three-replicate and fresh five-pair costs at 0%, 0.5%, 1%, and 2%.
- Scanned 0.406/0.407 GiB budgets and executed the one-percent 0.407 GiB plan.

### Commands

```bash
python scripts/distillschedule/compare_image_dmd2_tolerance_plans.py \
  --budgets-gib 0.406 0.407 \
  --relative-slowdown-tolerances 0 0.005 0.01 0.02 \
  --output artifacts/profile/image_dmd2_tolerance_planner_001/plans.json

python scripts/distillschedule/plan_image_dmd2_wavefront.py \
  --cost-selection-mode performance_band_memory \
  --relative-slowdown-tolerance 0.01 --budgets-gib 0.407 \
  --output artifacts/profile/image_dmd2_tolerance_execute_001/plans.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distillschedule/probe_split_distill.py --workload image_dmd2 \
  --microbatches 8 --schedule planned \
  --wavefront-plan artifacts/profile/image_dmd2_tolerance_execute_001/plans.json \
  --wavefront-budget-gib 0.407 ...
```

### Results

- Tests: 38 passed.
- Pipeline and streaming are both Pareto-nondominated in time and per-role memory.
- Zero tolerance selects pipeline under both cost profiles.
- One percent selects streaming under both, accepting 0.039% historical or 0.950% fresh
  median slowdown for 22 MiB lower Generator peak.
- At 0.5%, historical selects streaming but fresh selects pipeline; the preference is
  too close to profile drift to be invariant.
- At 0.406 GiB no candidate fits because guidance requires 0.406716 GiB.
- The executed one-percent plan is action-exact and update-exact; measured peaks match
  prior streaming exactly.

### Interpretation

The hypothesis is verified with a narrow claim. The planner can make an explicit
performance-memory tradeoff reproducible across these profile sets, but it has not
proved streaming faster or globally smaller. The 1% value is a declared preference;
0.5% sensitivity shows that profile uncertainty remains visible.

### Files Changed

- runtime/wavefront.py
- scripts/distillschedule/plan_image_dmd2_wavefront.py
- scripts/distillschedule/compare_image_dmd2_tolerance_plans.py
- scripts/distillschedule/summarize_iteration16.py
- tests/test_wavefront_planner.py
- artifacts/manifests/iteration_016_performance_memory_band.json
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/05_runtime_design.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

The tolerance is user supplied and role-peak sum is only a deterministic tie-break. The
22 MiB saving remains local to Generator and does not alter homogeneous global capacity.
This is still Level 3.

### Next Highest-Value Question

Do the accumulated Runtime mechanisms address the actual real-14B feasibility gap, or
have the Level-3 scheduling studies drifted away from the dominant ZeRO-3 parameter
materialization and real Generator-weight blocker?

### Next Command

Audit the Hardware Feasibility Gate against current artifacts, re-probe weight availability
and both hosts, then select the highest-value missing real-14B experiment rather than
adding another mini-planner policy.


## Iteration 17

### Question

Can the exact Helios-Distilled 14B Generator with the Stage-3 rank-256 LoRA be
materialized and execute the source-derived no-grad workload on one A100-40GB, or is
the real Generator itself already a mandatory sharding boundary?

### Hypothesis

The isolated BF16 Generator plus FP32 rank-256 LoRA will fit on one A100-40GB and
complete both a minimum forward and the default Stage-3 source-derived
`[1,16,9,48,80]` no-grad forward. Its resident footprint will be close to the measured
Fake Score footprint, leaving too little headroom to colocate the Score backbone.

### Importance

All prior real-14B measurements used Helios-Base as Score or a labeled Generator proxy.
This experiment closes the largest asset-derived Hardware Feasibility Gate and
determines whether Generator/Score grouping is optional or required on A100-40GB.

### Evidence Needed

A byte- and digest-verified `Helios-Distilled/transformer_ode` shard set, exact
Stage-3 Generator LoRA target/parameter counts, clean-source load snapshots, and
minimum/default-shape forward time and peak allocated/reserved memory.

### Refutation Condition

The verified model cannot materialize, the minimum forward OOMs, or the default
no-grad forward OOMs under the protected one-GPU probe.

### Actions

- Queried the ModelScope primary asset API and downloaded only the configured
  `Helios-Distilled/transformer_ode` component into the private asset directory.
- Verified config, index, and all six shards against remote byte sizes and SHA-256
  digests using four concurrent readers.
- Loaded the exact Distilled backbone, attached the source-equivalent rank-256
  Generator LoRA, and measured clean-source minimum and default-shape no-grad calls.
- Compared the Generator result with the accepted exact-shape Fake Score profile and
  reran the Runtime test suite.

### Commands

```bash
# Remote identity metadata; payload acquisition used the corresponding resolve URLs.
PYTHONPATH=. python - <<'PY'
from scripts.distillschedule.inspect_assets import modelscope_files
print(modelscope_files("BestWishYsh/Helios-Distilled", "transformer_ode"))
PY

# Exact shard verification used expected Size/Sha256 values from the API.
python - <asset-root> <remote-manifest> <verification-output>

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. \
PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
/tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/probe_model_load.py \
  --model-root /share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled \
  --subfolder transformer_ode --role generator --adapter generator --rank 256 \
  --dtype bf16 --loader-mode low_cpu --forward-modes generator \
  --latent-shape 1 2 2 --text-tokens 1 \
  --output artifacts/profile/generator_forward_minimal_a100_001.json

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. \
PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
/tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/probe_model_load.py \
  --model-root /share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled \
  --subfolder transformer_ode --role generator --adapter generator --rank 256 \
  --dtype bf16 --loader-mode low_cpu --forward-modes generator \
  --latent-shape 9 48 80 --history-sizes 16 2 1 --text-tokens 512 \
  --output artifacts/profile/generator_forward_stage3_contract_a100_001.json

HF_HUB_OFFLINE=1 PYTHONPATH=. pytest -q
python scripts/distillschedule/summarize_iteration17.py
```

### Results

- All eight requested files match remote size and SHA-256; the six shards encode
  14,312,427,584 F32 backbone elements (57,249,710,336 logical bytes).
- The exact Stage-3 Generator LoRA has 1,251,442,688 FP32 trainable elements over 409
  target modules. Mixed-dtype resident parameter storage is 31.375 GiB and CUDA
  residency is 31.493 GiB.
- Minimum no-grad forward succeeds in 671.57 ms at 31.502 GiB peak allocation.
- Default source-derived no-grad forward succeeds in 3.5556 s at 35.6614 GiB peak;
  reserved memory is 36.4473 GiB and driver-free memory after the call is 2.4398 GiB.
- Fake Score and Generator default peaks differ by only 0.0182 GiB; their cold times
  differ by 1.38 ms. Generator plus Score resident allocations sum to 62.9676 GiB.
- Both probes use clean source commit `44eb7f3233039c8714a031e0b06e05631539c6ec`.
  Tests: 38 passed.

### Interpretation

The hypothesis is verified. A real Distilled Generator does not require sharding for
an isolated no-grad call, but the measured 62.97 GiB two-backbone residency proves that
single-A100 colocation is impossible before any activation or optimizer state. The
dominant placement question is therefore not whether either inference call fits alone;
it is how to shard or separate the two roles while respecting returned gradients and
versions. The next gate is Generator training, not another proxy scheduling policy.

### Files Changed

- scripts/distillschedule/probe_model_load.py
- scripts/distillschedule/summarize_iteration17.py
- artifacts/manifests/iteration_017_real_generator_feasibility.json
- docs/00_hardware_feasibility.md
- docs/01_source_of_truth.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/07_experiment_plan.md
- docs/08_results.md
- docs/09_paper_story.md
- docs/10_open_questions.md
- README_AUTORESEARCH.md
- research_log.md

### Failure or Limitation

The calls are cold isolated no-grad forwards with synthetic contract-valid tensors and
local SDPA. They do not measure Generator gradients, AdamW state, EMA, DMD integration,
steady-state throughput, or output quality. The LoRA has the exact construction and
size but is freshly initialized rather than trained.

### Next Highest-Value Question

Does the exact default-shape Generator selected step require parameter sharding, and
can two-GPU ZeRO-3 complete its forward, returned-input-gradient backward, and AdamW
step with the same mixed BF16/FP32 constraints observed for Critic?

### Next Command

Generalize the protected DeepSpeed component probe to the Generator role, establish the
single-GPU training failure boundary, then compare minimum/default ZeRO-2 and ZeRO-3
forward/backward/step without adding EMA yet.


## Iteration 18

### Question

What is the exact Helios-Distilled Generator selected-step training boundary on
A100-40GB, and does the two-GPU ZeRO-3 configuration that made Critic feasible also
complete a Generator forward, returned-input-gradient backward, and AdamW step?

### Hypothesis

The minimum checkpointed Generator backward will fit on one A100, but the default
source-derived selected step will OOM with replicated parameters. Two-GPU ZeRO-3 with
persistent-parameter gathering disabled will complete default forward/backward/step,
with a boundary close to but slightly above Critic because Generator LoRA has 5.13M
more trainable elements.

### Importance

Iteration 17 proved that no-grad Generator and Score each fit alone but cannot colocate.
The Runtime cannot choose 8+8 groups, time sharing, or replay from inference residency:
it needs the trainable Generator peak and communication/materialization boundary.

### Evidence Needed

Clean-source single-GPU minimum/default checkpointed probes, exact Generator gradient
census and OOM site, and a two-GPU ZeRO-3 default forward/backward/step with per-rank
memory, time, optimizer result, and mixed-dtype configuration.

### Refutation Condition

The minimum single-GPU backward OOMs, the default replicated call unexpectedly
completes, or two-GPU ZeRO-3 cannot complete the default optimizer step.

### Actions

1. Extended the exact-model loader probe with checkpointed Generator training and a
   fixed synthetic returned-input-gradient loss.
2. Generalized the protected DeepSpeed Critic probe to Generator/Score roles without
   changing the model forward contract.
3. Ran minimum and default one-GPU Generator training probes with the default and
   expandable CUDA allocators.
4. Ran the default Generator on two A100-40GB ranks with ZeRO-3 and persistence
   disabled, then added a distributed ZeRO-partition checksum to distinguish a real
   optimizer update from a successful no-op.
5. Re-ran the accepted two-rank experiment from clean commit
   `96278add6f2b07184af5bc1aadd103c53213d5c6`, generated the Iteration-18
   manifest, and ran the full unit suite.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/probe_model_load.py \
  --model-root /share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled \
  --subfolder transformer_ode --role generator --adapter generator --rank 256 \
  --device cuda:0 --dtype bf16 --loader-mode low_cpu --loader-workers 4 \
  --forward-modes generator --latent-shape 1 2 2 --text-tokens 8 \
  --forward-requires-grad --backward --gradient-checkpointing --seed 49 \
  --output artifacts/profile/generator_backward_minimal_a100_001.json

# The default call used --latent-shape 9 48 80, --history-sizes 16 2 1,
# and --text-tokens 512; it was repeated with:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ...

CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/distillschedule/probe_deepspeed_critic.py \
  --role generator \
  --model-root /share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled \
  --subfolder transformer_ode --zero-stage 3 --rank 256 \
  --latent-shape 9 48 80 --history-sizes 16 2 1 --text-tokens 512 \
  --gradient-checkpointing --persistence-threshold 0 \
  --output-dir artifacts/profile/deepspeed_zero3_2gpu_generator_stage3_step_003_checksum

PYTHONPATH=. python scripts/distillschedule/summarize_iteration18.py
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
```

### Results

- **Measured:** one-GPU minimum checkpointed Generator forward/backward succeeds in
  0.662/0.610 s with 36.190 GiB maximum allocation.
- The minimum graph exposes 812 core Linear LoRA gradient tensors and
  1,246,314,496 FP32 gradient elements. It does not invoke the three history-patch
  LoRAs, whose 5,128,192 elements explain the difference from the full trainable set.
- **Measured:** the default replicated call OOMs in checkpointed forward, before loss
  or backward, on a 584 MiB FFN-LoRA allocation. Default and expandable allocators
  reach 37.793 and 38.410 GiB maximum allocation respectively; fragmentation is not
  the sole blocker.
- **Measured:** two-GPU ZeRO-3 completes the default forward/backward/AdamW step in
  3.967/8.459/0.652 s. Maximum allocated memory is 35.160 GiB and backward maximum
  reserved memory is 37.303 GiB.
- The distributed checksum covers all 1,251,442,688 sharded trainable elements,
  matches on both ranks, and changes from sum -162.296838 to -162.273077 after the
  step. Zero ordinary `.grad` tensors are therefore a ZeRO ownership artifact, not
  a no-op optimizer.
- Generator has 5,128,192 more trainable elements than Critic but only 0.0285 GiB
  more ZeRO-3 backward peak.
- **Tested:** 38 tests passed with two pre-existing SWIG deprecation warnings.

### Interpretation

The hypothesis is verified. Replicated default-shape selected-step training is not
feasible on one A100-40GB even with checkpointing and allocator mitigation, while
two-way ZeRO-3 crosses the exact Generator capacity boundary. The near-identical
Generator and Critic peaks strengthen parameter sharding as a physical-backend
requirement rather than a role-specific patch. The 37.303 GiB reserved peak also means
that two-way ZeRO-3 has too little margin to add a full EMA copy or colocated Score.

### Files Changed

- scripts/distillschedule/probe_model_load.py
- scripts/distillschedule/probe_deepspeed_critic.py
- scripts/distillschedule/summarize_iteration18.py
- artifacts/manifests/iteration_018_real_generator_training.json
- README_AUTORESEARCH.md
- docs/00_hardware_feasibility.md
- docs/01_source_of_truth.md
- docs/03_profile_report.md
- docs/04_problem_definition.md
- docs/07_experiment_plan.md
- docs/08_results.md
- docs/09_paper_story.md
- docs/10_open_questions.md
- research_log.md

### Failure or Limitation

The returned input gradient is fixed synthetic data, not a Score-computed DMD gradient.
The experiment is a cold isolated component capacity test, not steady-state throughput.
A Generator-specific ZeRO-2 run was not repeated; the replicated failure plus the
accepted Critic ZeRO-2 result establishes the mechanism, not a Generator ZeRO-2 timing.
EMA, cross-role state transfer, integrated loss ordering, optional attention kernels,
and multi-node execution remain absent.

### Next Highest-Value Question

Can upstream full-Generator EMA be constructed and updated under an
A100-compatible Generator ZeRO-3 group without recreating the two-backbone or
cold-start memory failure?

### Next Command

Audit `create_ema_zero3_lora.py` and its call order, then implement an exact isolated
EMA construction/update probe that records host RSS, per-rank CUDA peak, shard
ownership, update time, and numerical change before attempting integrated DMD.


## Iteration 19

### Question

Does Helios's upstream full-model ZeRO-3 EMA unnecessarily duplicate and rewrite the
frozen 14.3B Generator backbone, and can a trainable-state-only EMA preserve Stage-3
update semantics while crossing a smaller A100 sharding boundary?

### Hypothesis

The upstream path deep-copies the complete pre-cast FP32 Generator on every rank,
then creates a separate FP32 ZeRO-3 engine. Its `step` averages trainable parameters
but also copies every frozen shard on every synchronized update. Consequently, a live
two-rank Generator plus upstream EMA will exceed A100-40GB, whereas a sharded EMA over
only `requires_grad` state will fit beside the two-rank Generator, reproduce the
trainable EMA checksum, and reduce EMA storage and update bytes by more than 90%.

### Importance

EMA is enabled in the default Stage-3 configuration and is not an optional mock
component. If frozen base state can be represented as an immutable shared checkpoint,
this is a source-backed Physical State Sharing optimization rather than another name
for checkpointing or generic ZeRO. It can also turn the currently measured two-rank
Generator boundary into a complete Generator-plus-EMA residency plan.

### Evidence Needed

Exact static full/trainable parameter bytes; source evidence for copy, dtype, sharding,
frequency, and update behavior; clean two-rank full-EMA and trainable-only probes; a
four-rank upstream-equivalent success if two ranks OOM; per-rank host RSS, CUDA peaks,
initialization/update times, bytes touched, and distributed trainable checksums before
and after the same deterministic source mutation.

### Refutation Condition

The upstream EMA does not retain/copy frozen parameters, full Generator-plus-EMA fits
on two A100-40GB ranks, trainable-only EMA cannot reproduce the upstream trainable
state, or the measured storage/update reduction is below 50%.

### Actions

- Audited the upstream EMA construction, step loop, call predicate, and DeepSpeed
  configuration.
- Implemented `TrainableStateEMA`, retaining only mutable `requires_grad` state and
  reading ZeRO-3 `ds_tensor` partitions.
- Added vanilla/ZeRO, CPU-placement, and upstream-semantic unit tests.
- Implemented an exact 14B Generator+EMA probe with full/trainable-only modes, host/CUDA
  telemetry, checksums, and same-run CPU mirror verification.
- Ran two-GPU full/CUDA-sparse/CPU-sparse and three-GPU CUDA/CPU variants; preserved all
  failures and generated a hash-checked manifest.

### Commands

```bash
# Representative accepted runs; all used clean source commits.
/tmp/distillschedule-helios-probe-torch24/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/distillschedule/probe_deepspeed_ema.py \
  --ema-mode full --output-dir artifacts/profile/deepspeed_zero3_2gpu_generator_full_ema_step_002

/tmp/distillschedule-helios-probe-torch24/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/distillschedule/probe_deepspeed_ema.py \
  --ema-mode trainable_only --trainable-ema-device cpu \
  --output-dir artifacts/profile/deepspeed_zero3_2gpu_generator_trainable_ema_step_002_cpu

/tmp/distillschedule-helios-probe-torch24/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/distillschedule/probe_deepspeed_ema.py \
  --ema-mode trainable_only --trainable-ema-device cuda --verify-cpu-mirror \
  --output-dir artifacts/profile/deepspeed_zero3_3gpu_generator_trainable_ema_step_004_error

PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration19.py
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
```

### Results

- **Measured static state:** upstream full EMA is 15,563,870,272 FP32 elements,
  57.9799 GiB. Trainable-only state is 1,251,442,688 elements, 4.6620 GiB. The
  reduction is **91.9593%** (12.4367x); 53.3179 GiB of frozen state is avoided.
- **Measured upstream failure:** a full CPU deepcopy takes 56.09 s and each rank peaks
  at 147.036 GiB RSS. DeepSpeed OOMs during pre-partition `module.to(device)` at
  37.311 GiB allocated with a 100 MiB request. More ranks do not remove this per-rank
  transient in the current construction path.
- **Measured negative result:** two-GPU CUDA trainable EMA reaches a 3.982 s forward,
  then OOMs on the same 584 MiB FFN-LoRA allocation at 35.981 GiB peak.
- **Measured success, two GPUs:** CPU trainable EMA preserves the source-only 35.160
  GiB backward peak and completes forward/backward/AdamW/EMA in
  4.009/8.531/0.660/1.838 s. The expected checksum error is 5.33e-10.
- **Measured success, three GPUs:** CUDA trainable EMA completes at 29.940 GiB peak;
  forward/backward/AdamW/EMA are 4.207/8.587/0.666/0.0946 s.
- **Measured exactness:** the same three-GPU run's CPU mirror takes 1.237 s and is
  elementwise identical to CUDA over 1,251,443,430 physical partition elements:
  max/mean/L2 error are all zero. A prior norm-only discrepancy was reduction
  arithmetic, not a state difference.
- **Tests:** 42 passed, 5 dependency warnings.

### Interpretation

The hypothesis is **partially verified and partially refuted**. Upstream does duplicate
and rewrite frozen state, and trainable-only EMA is exact while reducing state by more
than 90%. However, sparse state alone does not fit beside the two-GPU Generator on CUDA.
EMA representation and placement are coupled planner decisions: CPU sparse state makes
2-GPU execution feasible; CUDA sparse state needs 3 GPUs for this contract. This adds a
measured Physical State Graph requirement—immutable alias versus mutable state—rather
than relabeling generic checkpointing.

### Files Changed

- `runtime/ema.py`
- `tests/test_ema.py`
- `scripts/distillschedule/probe_deepspeed_ema.py`
- `scripts/distillschedule/summarize_iteration19.py`
- `artifacts/manifests/iteration_019_real_ema_state.json`
- `README_AUTORESEARCH.md`, `docs/00_hardware_feasibility.md` through the relevant
  source/graph/profile/design/results/open-question documents

### Failure or Limitation

The loss uses a deterministic synthetic returned gradient rather than a real
Score-computed DMD gradient. Sparse EMA checkpoint save/resume is not integrated;
frozen state must be reconstructed from an immutable, versioned base checkpoint. CPU
EMA is synchronous/pageable. A complete Generator–Score–Critic–EMA iteration remains
Not Run.

### Next Highest-Value Question

Can the exact real Generator and Score execute the source-semantic DMD boundary—Fake
Score and Real Score under `no_grad`, `grad=pred_fake-pred_real`, detached surrogate
Generator backward—while preserving physical role peaks and parameter versions?

### Next Command

Audit the existing real Score and Generator probe helpers plus
`compute_distribution_matching_loss`, then implement the smallest role-split exact-shape
boundary probe. First validate the Score-produced `dL/dx_fake` artifact independently;
then consume it in the accepted ZeRO-3 Generator step before attempting concurrent
multi-group execution.

## Iteration 20

### Question

Can the exact real Generator and shared Score execute the default Stage-3 DMD gradient
boundary concurrently on one A100 node, without fabricating a synthetic returned
input gradient or changing the Generator parameter version?

### Hypothesis

A three-rank ZeRO-3 Generator group can retain the default selected-step graph while a
separate one-GPU Score worker sequentially evaluates each rank's `x_fake` in Fake,
Real-conditional, and Real-unconditional modes. Returning the normalized
`pred_fake - pred_real` gradient through the exact detached MSE surrogate will fit,
update all 1.251B Generator trainables, and expose a small tensor boundary relative to
model swapping. The Score adapter is architecture-real but randomly initialized, so
this tests execution semantics/capacity rather than learned loss quality.

### Importance

Iterations 17–19 establish isolated Generator, Score, and EMA boundaries, but no real
cross-component gradient has yet been produced or consumed. A successful 3+1 role split
would be the first Level-2 multi-component update and would validate that gradient-edge,
adapter-mode, and role-placement abstractions survive the 14B boundary. Failure would
identify whether the retained Generator graph, Score residency, file/transport edge, or
source surrogate is the next limiting mechanism.

### Evidence Needed

Source-exact shifted sigma/timestep/noise conversion, Fake/Real adapter order, CFG and
normalization; shape/checksum-bound exchange artifacts for every Generator rank;
concurrent per-role peak memory and phase time; returned-gradient size and wait time;
Generator pre/post ZeRO-shard checksums; derivative/unit tests for the detached
surrogate; a clean-commit run with raw logs and hashes.

### Refutation Condition

The 3+1 split OOMs, Score cannot execute all three source calls while Generator graphs
remain live, returned gradients do not match bundle identity/shape, the detached
surrogate derivative differs from the source gradient divided by element count, or the
Generator optimizer does not update all trainable shards.

### Actions

- Added source-semantic DMD schedule, noise, flow-to-x0, normalized-gradient, and
  detached-surrogate primitives plus derivative tests.
- Extended the protected ZeRO-3 Generator probe to publish digest-bound selected outputs
  and consume a Score response without releasing its graph.
- Extended the real Score probe to process Generator bundles sequentially through Fake,
  Real-conditional, and Real-unconditional modes and atomically return each gradient.
- Preserved a non-capacity JSON-serialization preflight failure, fixed it, and ran clean
  3G+1S and 2G+1S experiments.
- Verified raw tensor/file hashes, source commits, surrogate loss identities, distributed
  checksums, role peaks, boundary bytes, and collective wait behavior.

### Commands

```bash
# Generator role (2G accepted example)
CUDA_VISIBLE_DEVICES=0,1 /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/distillschedule/probe_deepspeed_critic.py \
  --role generator --zero-stage 3 --optimizer-step \
  --dmd-exchange-dir artifacts/profile/real_dmd_split_2g1s_step_003/exchange \
  --dmd-run-id iteration20-real-dmd-2g1s-step003 ...

# Concurrent Score role
CUDA_VISIBLE_DEVICES=2 /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/probe_model_load.py --role score --adapter critic \
  --dmd-world-size 2 \
  --dmd-exchange-dir artifacts/profile/real_dmd_split_2g1s_step_003/exchange \
  --dmd-run-id iteration20-real-dmd-2g1s-step003 ...

PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration20.py
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
```

### Results

- **Measured 2G+1S success:** 29.539 s component phase for two samples, 0.06771
  samples/s. Generator F/last-rank-B/step are 4.047/8.481/0.694 s; backward peaks at
  35.167 GiB. Score peaks at 35.650 GiB and spends 7.49–7.75 s/sample across three
  source calls.
- **Measured 3G+1S success:** 37.406 s for three samples, 0.08020 samples/s; 18.45%
  more phase throughput than 2G+1S while consuming one extra Generator GPU. Generator
  backward peak falls to 28.347 GiB; Score remains the 35.650-GiB role.
- **Measured boundary:** 11.399 MiB/sample forward tensors including prompts and history;
  1.055 MiB/sample returned gradient. Dynamic `x_fake + gradient` alone is 2.109 MiB.
- **Correctness:** all producer/consumer tensor digests match. Logged losses equal
  `0.5*mean(gradient^2)` exactly or within 6.94e-18. Global ZeRO checksums cover all
  1,251,442,688 logical trainables at world size 2 and change identically on both ranks.
- **Measured negative overlap:** at 2G, rank0 waits 7.92 s then reports 15.97 s backward;
  rank1 waits 15.76 s then reports 8.48 s. Both hit the same completion barrier because
  ZeRO backward collectives require both ranks.
- **Tests:** 47 passed, 5 dependency warnings before documentation-only changes.

### Interpretation

The main hypothesis is verified and strengthened: **2G+1S**, not only 3G+1S, is enough
for a real selected-step cross-component DMD update. The experiment validates the
Gradient Edge and shared adapter-mode abstractions at 14B scale. It also refutes a naive
rank-streaming insight. Tensor readiness is not operation progress when the consumer
contains group collectives; the Runtime IR/planner must attach collective participation
to operations and pipeline microbatches across the whole DP group instead.

### Files Changed

- `runtime/dmd_boundary.py`
- `tests/test_dmd_boundary.py`
- `scripts/distillschedule/probe_model_load.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration20.py`
- `artifacts/manifests/iteration_020_real_dmd_boundary.json`
- research and design documents/graphs

### Failure or Limitation

Step 001 failed before loading because a `Path` argument was not JSON serializable; it
is retained but provides no capacity evidence. Accepted runs use a random architecture-
real Critic LoRA, file rendezvous rather than NCCL, and local SDPA. They include one
selected Generator operation only—no rollout prefix, Critic update, EMA, convergence,
or quality evidence.

### Next Highest-Value Question

Can the successful real DMD role split preserve `G@v -> G@v+1 -> EMA@v+1` by attaching
the measured trainable-only EMA, and does CPU@2G versus CUDA@3G change the preferred
placement once Score serialization is included?

### Next Command

Extend the Generator DMD client with `TrainableStateEMA`, run 2G+1S with CPU EMA and
3G+1S with CUDA EMA from clean commits, then compare full component-phase throughput,
role peaks, update identity, and EMA placement cost before adding the real Critic step.

## Iteration 21

### Question

Once the real Score-produced DMD gradient advances `Generator@v` to `Generator@v+1`,
can trainable-only EMA observe that exact new version in the same role-split execution,
and which measured placement—2G+CPU EMA or 3G+CUDA EMA—is preferable under the Score
serialization bottleneck?

### Hypothesis

The 2G+1S plan will remain feasible when EMA shadows stay on CPU, with the Iteration-20
35.167-GiB Generator backward peak and roughly 1.8 s exposed EMA update. The 3G+1S plan
will remain feasible with CUDA EMA, add about 1.6 GiB/rank residency, and update in about
0.1 s. Both should match the exact Helios decay/checksum equation after the real DMD
optimizer step. Because Score costs roughly 7.5 s/sample, CUDA EMA should improve phase
throughput but CPU EMA may retain better GPU efficiency.

### Importance

This is the first chance to close an actual parameter-version chain across three logical
components and two physical roles: `Score gradient -> Generator optimizer -> EMA`.
Iterations 19 and 20 measured those boundaries separately. Their composition may expose
memory or synchronization interactions that isolated measurements miss, and it gives
the planner a concrete representation/placement/performance choice rather than a static
state-size heuristic.

### Evidence Needed

Clean concurrent 2G+1S CPU-EMA and 3G+1S CUDA-EMA runs; exact pre/post Generator and
EMA checksums; decay-equation error; EMA init/update time and memory; full component
phase and samples/s; Score phase/peak; GPU-count-normalized efficiency; retained
failures if the composed path crosses a capacity boundary.

### Refutation Condition

Either composed placement OOMs, EMA observes the pre-update Generator version, the
expected checksum equation fails, the real DMD surrogate changes relative to Iteration
20, or measured placement costs reverse the isolated EMA boundary enough that the
hypothesized tradeoff disappears.

### Actions

- Extended the real DMD Generator client to create trainable-only EMA after ZeRO-3
  initialization on CPU or CUDA.
- Recorded EMA state bytes, init/update phases, pre/post state checksums, the observed
  post-optimizer Generator checksum, and the exact Helios decay equation.
- Ran clean concurrent 2G+1S CPU-EMA and 3G+1S CUDA-EMA experiments.
- Matched every `x_fake`, returned-gradient digest, and loss against Iteration 20 and
  compared throughput, capacity, EMA cost, and samples/s/GPU.

### Commands

```bash
# 2G+1S CPU EMA
... probe_deepspeed_critic.py --role generator --zero-stage 3 --optimizer-step \
  --trainable-ema-device cpu --dmd-exchange-dir ...
... probe_model_load.py --role score --adapter critic --dmd-world-size 2 ...

# 3G+1S CUDA EMA
... probe_deepspeed_critic.py --role generator --zero-stage 3 --optimizer-step \
  --trainable-ema-device cuda --dmd-exchange-dir ...
... probe_model_load.py --role score --adapter critic --dmd-world-size 3 ...

PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration21.py
```

### Results

- **2G+1S CPU EMA:** success; 31.061 s/2 samples, 0.06439 samples/s, 35.163
  GiB Generator backward peak. EMA init/update are 1.544/1.870 s; checksum equation
  error is 5.32e-10.
- **3G+1S CUDA EMA:** success; 37.422 s/3 samples, 0.08017 samples/s, 29.944
  GiB Generator backward peak. EMA init/update are 0.107/0.0946 s; checksum equation
  error is 3.75e-10.
- CUDA@3G provides 24.51% more absolute throughput. CPU@2G provides 7.09% more
  samples/s/GPU and uses one fewer GPU.
- CPU EMA increases matched component phase by 5.16%. CUDA EMA changes it by 0.043%,
  below the resolution of one cold run; only the direct 19.8x update-time contrast is
  treated as robust.
- DMD `x_fake` and gradient SHA values and losses are identical to Iteration 20.
  Independent post-AdamW checksum sums differ by -2.92e-4 (2G) and 7.29e-3 (3G), so
  the distributed path is not cross-run bit deterministic.
- **Tests:** 47 passed before result-only documentation.

### Interpretation

The hypothesis is verified with a sharper planner conclusion. Both representation/
placement choices preserve `Score -> Generator@v+1 -> EMA@v+1`; neither universally
wins. CUDA EMA is the throughput plan, while CPU EMA is the capacity/GPU-efficiency
plan. Correctness must use within-run parameter-version equations and tensor identity,
not bitwise equality across independent DeepSpeed launches.

### Files Changed

- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration21.py`
- `artifacts/manifests/iteration_021_integrated_dmd_ema.json`
- research/design/results documents and graph validation metadata

### Failure or Limitation

Only one cold run per placement was executed; small phase differences are not robust
timing claims. The run still omits Generator rollout prefix, Critic rollout/update,
trained Critic state, sparse-EMA checkpoint reconstruction, and NCCL tensor transport.
Cross-run AdamW states are numerically close but not bit identical despite identical
inputs/loss.

### Next Highest-Value Question

Can the real post-Generator-update Critic path consume `G@v+1`, train the same physical
Score adapter from `C@v` to `C@v+1`, and complete one source-ordered iteration without
violating role capacity or duplicating Score state?

### Next Command

Audit `_critic_loss` inputs and optimizer ordering, then extend the role split with a
second post-step Generator output and a ZeRO-3 Critic update. Start with the smallest
source-semantic denoising surrogate and instrument the unresolved Generator-gradient
pollution hazard before adding full rollout prefixes.

## Iteration 22

### Question

Do the Iteration-20/21 capacity and placement conclusions survive after correcting the
Generator boundary from raw transformer flow output to the rollout's
`x_fake = x_t - sigma_t * flow` value?

### Hypothesis

Adding the exact higher-precision flow-to-x0 conversion at the selected model call will
change the `x_fake`, DMD gradient, loss, and Generator update, but add only a small tensor
lifetime. The 2G+1S CPU-EMA and 3G+1S CUDA-EMA placements should remain feasible with
similar peaks. The returned gradient should backpropagate through the conversion and
still update every Generator shard before EMA.

### Importance

The prior runs validate real-model capacity, adapter sequencing, gradient transport,
collective behavior, and version ordering, but their Generator output value is not the
value produced by `inference_with_trajectory_stage2`. Carrying that simplification into
the Critic path would overstate semantic fidelity. This correction separates a valid
systems boundary result from an invalid claim of selected-step numerical equivalence.

### Evidence Needed

Source line audit for selected-step conversion; a unit/autograd test through
flow-to-x0; clean 2G+1S CPU-EMA and 3G+1S CUDA-EMA reruns with an explicit sigma schedule;
flow and x0 digests; DMD loss identity; Generator/EMA checksums; peak/time deltas against
the raw-flow characterization; updated claim boundaries in all reports.

### Refutation Condition

Flow-to-x0 was not actually required by the source, the conversion disconnects the
Generator graph, either accepted placement OOMs, Generator/EMA checksums do not advance,
or the added lifetime materially changes the capacity/placement conclusion.

### Actions

- Audited `inference_with_trajectory_stage2` and confirmed that the selected transformer
  flow is converted to `pred_x0` before history mutation and return.
- Added `flow_prediction_to_x0`, explicit `--generator-output-kind flow|x0`, and
  `--generator-selected-sigma`; added a unit/autograd test for the conversion.
- Ran clean 2G+1S CPU-EMA and 3G+1S CUDA-EMA real-model role-split experiments at sigma
  0.5, then generated a hash-bound manifest comparing them to Iterations 20/21.
- Corrected all reports so Iterations 20/21 are raw-flow real-architecture
  characterization rather than selected-step numerical equivalence.

### Commands

```bash
grep -n "convert_flow_pred_to_x0\|pred_x0" helios/utils/utils_helios_post.py
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q tests/test_dmd_boundary.py tests/test_deepspeed_probe_helpers.py
bash /tmp/run_iter22_x0cpu2_step001.sh
bash /tmp/run_iter22_x0cuda3_step002.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python   scripts/distillschedule/summarize_iteration22.py
```

### Results

- **Measured Level 2:** 2G+1S CPU EMA succeeds: 30.762 s / 2 samples, 0.06501
  samples/s, 35.164 GiB Generator allocated peak, 35.650 GiB Score peak, 1.810 s EMA.
- **Measured Level 2:** 3G+1S CUDA EMA succeeds: 37.152 s / 3 samples, 0.08075
  samples/s, 29.945 GiB Generator peak, 35.650 GiB Score peak, 0.0930 s EMA.
- The underlying per-rank flow checksums exactly match the corresponding raw-flow runs.
  Every transported x0 SHA, returned DMD-gradient SHA, and loss differs, as required.
- All 48 repository tests pass. Every loss equals `0.5*mean(gradient^2)` within 1e-15. Generator global ZeRO
  checksums advance and EMA equations agree within 1.29e-9.
- Flow-to-x0 adds 1.56 MiB allocated peak on 2G and 0.19 MiB on 3G. Phase deltas
  (-0.96%/-0.72%) are one-cold-run noise and are not reported as speedups.

### Interpretation

The hypothesis is verified. The old runs remain valid for architecture capacity,
adapter sequence, tensor transport, ZeRO collective behavior, and version mechanics,
but are now correctly labeled raw-flow characterization. The corrected value changes
the optimization numerics without changing the feasible placements. Output semantics
and selected sigma must be explicit Runtime IR state; physical model-call identity is
not sufficient.

### Files Changed

- `runtime/dmd_boundary.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration22.py`
- `tests/test_dmd_boundary.py`
- `artifacts/manifests/iteration_022_generator_x0_correction.json`
- research/design/results documents

### Failure or Limitation

Only one shape-exact selected model call is executed. The full three-stage/two-step
rollout, actual sampled exit sigma, preceding no-grad/history mutations, Critic update,
trained Critic checkpoint, and NCCL tensor transport remain absent. Earlier profiles did
not persist a raw-flow tensor SHA separately, so exact underlying-flow equality is
checked by their recorded FP32 checksum; transported x0/gradient tensors are SHA-256
bound.

### Next Highest-Value Question

Can a two-rank ZeRO-3 Score group use one physical Score state for the three no-grad DMD
modes and the source Critic denoising backward, after consuming a post-Generator-step
converted x0, while preserving `C_i -> C_i+1` and fitting A100-40GB?

### Next Command

Extend the role protocol to publish a second post-Generator-step x0 bundle. Run 2G
Generator+CPU EMA on GPUs 0-1 and 2S Score/Critic ZeRO-3 on GPUs 2-3, then validate the
source denoising target, Score checksum update, Generator non-pollution, peaks, and
end-to-end version order.

## Iteration 23

### Question

Can one two-rank ZeRO-3 physical Score state first serve the three no-grad DMD modes at
`C_i`, then train the source default Critic denoising loss to `C_i+1` using a converted
post-Generator-step sample from `G_i+1`, with EMA delayed until the source control point
after Critic update?

### Hypothesis

A 2G+2S single-node split should fit A100-40GB when Generator uses CPU sparse EMA and
Score uses two-rank ZeRO-3. DMD evaluation must leave the Score checksum exactly at
`C_i`; Generator DMD must advance `G_i` to `G_i+1`; a second no-grad selected call must
publish x0 from `G_i+1`; the Critic MSE must advance the same Score adapter to `C_i+1`;
and only then may EMA read `G_i+1`. Because Score no-grad DMD and Critic training are
phase-separated, their peaks should not add.

### Importance

This is the first experiment to compose both trainable physical backbones and the shared
Real/Fake/Critic state in source control order. It tests whether Physical State Sharing
and Parameter Version edges are necessary executable Runtime constraints rather than
only source annotations. The source audit also corrects an ordering simplification in
Iteration 21/22: `train_helios.py:1890-1899` updates EMA after Critic, not immediately
after the Generator optimizer.

### Evidence Needed

A clean four-GPU run; SHA-bound initial x0/DMD gradient and post-step x0/Critic request;
exact DMD surrogate and source Critic denoising-loss identities; Score checksum equality
before/after no-grad DMD and inequality after Critic optimizer; Generator checksum
advance before the second sample and stability while Score trains; EMA checksum equation
after Critic acknowledgment; explicit operation timestamps, peaks, and source lines.

### Refutation Condition

Either role OOMs, the shared Score state changes during no-grad DMD, Critic backward
cannot follow adapter-mode switching on the same ZeRO-3 state, Generator changes while
waiting for Critic, Critic does not update its trainables, EMA executes before Critic
acknowledgment, or phase-local peaks combine and invalidate the proposed placement.

### Actions

- Added the exact default GAN-disabled Critic denoising objective and unit tests.
- Extended the protected ZeRO-3 probe with a two-group protocol: 2G Generator + CPU
  sparse EMA and 2S shared Score/Critic, using one Score engine for DMD and Critic.
- Published a second converted post-Generator-step x0, delayed EMA until Critic
  acknowledgment, and checksum-gated every parameter version.
- Preserved the first failed run, corrected an invalid exact scalar-reduction gate, and
  reran from a clean commit.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q tests/test_dmd_boundary.py
bash /tmp/run_iter23_2g2s_cpu_step001.sh
bash /tmp/run_iter23_2g2s_cpu_step002.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python scripts/distillschedule/summarize_iteration23.py
```

### Results

- All 50 repository tests pass. **Measured Level 2:** accepted clean 2G+2S run
  completes 2 samples in 41.121 s
  (0.04864 samples/s; 0.01216 samples/s/GPU).
- Generator initial F/B/step are 4.004/8.441/0.648 s max; backward peaks at 35.164
  GiB. The post-step selected call takes 3.661 s and peaks at 29.954 GiB.
- Shared Score DMD executes three modes concurrently across DP ranks in 8.883 s and
  peaks at 24.537 GiB. Its global checksum is exactly unchanged.
- Critic F/B/step are 3.497/8.387/0.688 s max; backward peaks at 35.139 GiB and the
  Score checksum sum changes by -0.0137734.
- Each post-step x0 SHA differs from its pre-step x0. Generator checksum advances before
  publishing it and remains exactly stable through Critic training.
- EMA begins only after Critic acknowledgment, takes 1.745 s, and satisfies its checksum
  equation within 1.59e-10.
- The first run completed Score/Critic but Generator rejected a valid GPU/CPU Critic-loss
  difference of 3.58e-7. SHA-identical operands plus a 1e-6 FP32 reduction tolerance
  accept the rerun; this is recorded as negative evidence, not hidden.

### Interpretation

The hypothesis is verified. Logical Real Score, Fake Score, and Critic are executable
modes of one physical trainable state: DMD reads C_i without mutation and Critic alone
advances C_i+1. Parameter version and control edges are separately necessary—EMA reads
G_i+1 but must be scheduled after Critic. Phase-local memory matters: adding the 24.537
GiB DMD peak and 35.139 GiB Critic peak would falsely reject a legal 40GB placement.

### Files Changed

- `runtime/dmd_boundary.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration23.py`
- `tests/test_dmd_boundary.py`
- `artifacts/manifests/iteration_023_shared_score_critic_chain.json`
- execution graphs and research/design/results documents

### Failure or Limitation

The two Generator emissions are isolated selected calls at explicit sigma 0.5, not the
source three-stage/two-step rollout. Continuous history mutation, sampled exit positions,
real data, trained Critic weights, GAN/Reward/VAE branches, NCCL role transport,
gradient accumulation, sparse EMA checkpoint reconstruction, and convergence are absent.
One cold run cannot support throughput superiority claims.

### Next Highest-Value Question

Can the real `inference_with_trajectory_stage2` nested rollout execute under the 2G
ZeRO-3 Generator group while retaining only its selected differentiable suffix, and can
its measured forward count/history lifetimes replace the synthetic selected-call inputs
without invalidating the 2G+2S placement?

### Next Command

Build a protected rollout harness around `inference_with_trajectory_stage2` using the
loaded scheduler/config and shape-reduced one-section default path. First trace every
stage/denoise grad mode and pred_x0/history shape without Score; then connect its selected
output to the accepted Iteration-23 protocol.

## Iteration 24

### Question

Can the unmodified source `run_generator -> inference_with_trajectory_stage2` path
execute a one-section, three-pyramid-stage, two-candidate-step Helios-Distilled 14B
rollout on A100-40GB, retain only sampled gradient boundaries, and complete the
Generator update under a practical ZeRO-3 placement?

### Hypothesis

The official structural settings produce 3–6 model calls with exactly one selected
autograd call per pyramid stage. Two-rank ZeRO-3 may fit; if not, a per-call trace should
locate the real nested lifetime hidden by the isolated-call probe. The final x0 should
backpropagate through all three selected stage outputs and exclude no-grad prefixes.

### Importance

Iterations 20–23 validated real selected-call, shared-state, and version mechanics, but
not the nested continuous state that distinguishes video distillation from a coarse
stage graph. This experiment decides whether Nested Axis changes actual placement and
whether the initial `N_forward >> N_backward` motivation holds for the default rollout.

### Evidence Needed

Clean source commits; exact scheduler/configuration; per-call stage, shape, timestep,
grad mode, output graph state, time, and memory; final-x0 identity; backward gradients
on every selected output; optimizer checksum; and preserved failures for import,
control divergence, OOM, or fragmentation.

### Refutation Condition

The source cannot run through ZeRO-3, selected calls do not form a multi-stage gradient
chain, or the two-rank placement OOMs. A strong forward-dominance claim is refuted if
default call counts are comparable to the three selected calls.

### Actions

- Added a non-invasive `NestedCallTracer` and four CPU tests; all 54 repository tests
  pass.
- Called the real source `run_generator` and `inference_with_trajectory_stage2` without
  editing Helios source, using shape-exact synthetic GT/history.
- Preserved a torch-2.4/diffusers-0.37 import failure and added fail-closed placeholders
  for three imported but uncalled training utilities.
- Preserved and terminated a confirmed ZeRO-3 rank-divergence stall caused by random
  history drop entering different amplify-first-chunk call sequences.
- Added a collective-control preflight, disabled synthetic random history drop only for
  protected placement measurement, and scanned 2G/3G plus allocator variants.
- Retained flow-output references and checked their gradients after final-x0 backward.
- Generated a deterministic manifest over 31 raw artifacts.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
/tmp/run_iter24_forward.sh       # import failure, preserved
/tmp/run_iter24_forward2.sh      # rank-divergent control, terminated/preserved
/tmp/run_iter24_forward3.sh      # 2G forward success
/tmp/run_iter24_backward4.sh     # 2G backward OOM
/tmp/run_iter24_backward5.sh     # 3G default-allocator OOM
/tmp/run_iter24_backward6.sh     # 3G expandable backward success
/tmp/run_iter24_step7.sh         # 3G expandable F/B/step success
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration24.py
```

### Results

- **Measured Level 2:** sampled exits are stage `1/0/0`, giving four model calls: one
  no-grad prefix and three selected calls at latent resolutions 12x20, 24x40, 48x80.
- Two-rank ZeRO-3 forward succeeds in 9.541 s at 34.408 GiB allocated, but backward
  OOMs during checkpoint recomputation. Failure reaches 36.112 GiB max allocated and
  leaves only 64–174 MiB driver-free before a 216-MiB request.
- Three ranks lower forward peak to 27.592 GiB, but the default allocator OOMs during
  backward with 5.08–5.40 GiB reserved but unallocated.
- With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, three ranks complete forward,
  backward, and optimizer in 10.116/15.346/0.307 s. Peak is 32.892 GiB allocated and
  36.195 GiB reserved. The component phase is 26.412 s for three samples
  (0.11358 samples/s).
- Final x0 gives nonzero retained-output gradient sums at all three selected stages
  (~0.0296, 0.288, 0.797); the prefix output has `requires_grad=false`.
- The global 1,251,443,430-element trainable checksum changes from sum
  -162.164627 to -162.113575 after optimizer step.
- Official random-history-drop evidence produces stage sequences `[0,0,0,0]` on one
  rank and `[0,0,1,2]` on another before stalling. Source lines 1241–1257 explain why:
  the all-zero amplify branch doubles steps and samples its exit unsynchronized.
- Default one-section `[2,2,2]` structurally has 3–6 calls and exactly 3 selected calls;
  the sampled 4/3 ratio refutes `N_forward >> N_backward` for this rollout alone.

### Interpretation

The original two-GPU feasibility hypothesis is refuted, while the broader nested-graph
hypothesis is verified. Nested Axis is not decorative: retaining three selected pyramid
boundaries changes the minimum measured Generator placement from 2G to 3G. Allocator
mode is also part of backend feasibility, because identical logical tensors fail or
succeed depending on fragmentation handling.

The strongest new systems insight is collective-safe conditional execution. Source
control valid for replicated ZeRO-2 is not automatically valid for ZeRO-3: rank-local
branches can reorder parameter all-gathers. A Runtime must reconcile/pad call sequences
or isolate groups. The paper should not use default-rollout forward dominance as a core
claim until the full Generator-plus-Critic iteration trace is measured.

### Files Changed

- `runtime/rollout_trace.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration24.py`
- `tests/test_rollout_trace.py`
- `artifacts/manifests/iteration_024_nested_generator_rollout.json`
- execution graphs and research/design/results documents

### Failure or Limitation

Synthetic shape-exact GT/history and returned input gradients are used; random history
drop is disabled only in the accepted ZeRO-3 placement. The compatibility import shim
is fail-closed but is not the official torch-2.10 stack. One exit pattern and one cold
run do not establish a distribution or convergence. The nested Generator is not yet
composed with real Score/DMD/Critic/EMA, and no general branch-reconciliation scheduler
exists.

### Next Highest-Value Question

Can the admitted 3G nested Generator replace both isolated Generator emissions in the
accepted shared-Score/Critic/EMA version chain on a 3G+2S placement, while transporting
the actual final x0/DMD gradient and preserving collective-safe control?

### Next Command

Extend the integrated exchange protocol to publish the nested final x0, consume the real
Score DMD gradient, complete the nested backward/step, then run the post-step Critic
rollout and EMA on a five-GPU single-node placement with expandable Generator allocation.


## Iteration 25

### Question

Can the real one-section, three-pyramid-stage Generator rollout replace both isolated
Generator emissions in the shared Score DMD/Critic/EMA protocol, first under a
symmetry-preserving 3G+3S ZeRO-3 placement, and what autograd state does the source
Critic-side `with torch.no_grad(): run_generator(...)` call actually retain?

### Hypothesis

With expandable CUDA allocator segments on both role groups and trainable-only EMA on
CPU, 3 Generator ranks plus 3 Score ranks should fit a complete composed control chain:
`nested G_i rollout -> shared Score DMD -> nested G backward/step -> post-step nested
G_i+1 rollout -> shared Score Critic backward/step -> EMA`. The first DMD gradient should
reach all three selected Generator stage outputs, DMD should leave Score at `C_i`, the
Critic optimizer should advance it to `C_i+1`, and EMA should read `G_i+1` only after the
Critic acknowledgment.

The post-step rollout is intended to be Generator-inference work, but the source nested
loop uses `torch.set_grad_enabled(should_compute_grad)` inside the outer no-grad block in
`_critic_loss`. Therefore the trace may refute the apparent no-grad boundary by showing
three selected calls with autograd enabled. The experiment must record that behavior
rather than silently correcting it.

### Importance

Iteration 24 proves that the nested Generator changes placement from 2G to 3G, while
Iteration 23 proves the selected-call multi-component version chain. Composing them is
the shortest path to a source-faithful real-14B iteration boundary and can expose a
new optimization opportunity: a graph that is retained during Critic data generation
although Generator parameters are not updated by the Critic optimizer.

A 3G+3S rank-symmetric file protocol is deliberately tested before asymmetric 3G+2S
mapping so any failure can be attributed to graph lifetime or component composition,
not fan-in/fan-out transport logic.

### Evidence Needed

Clean source commit; both nested call traces and final-x0 hashes; first-rollout DMD
gradient identity and gradients on all selected stage outputs; G/Score/EMA checksums and
version order; per-phase time and peak allocated/reserved memory; exact file identities;
post-step call grad modes and output `requires_grad`; and preserved OOM, deadlock, or
semantic mismatch failures.

### Refutation Condition

The 3G+3S chain OOMs or deadlocks, the real Score gradient fails to reach every selected
Generator stage, any parameter version changes out of order, the post-step output cannot
feed the Critic, or source behavior differs from the predicted inner-grad override.

### Actions

- Factored the nested Generator path into the existing two-role protocol and moved
  trainable-only EMA initialization before that path.
- Added SHA-bound first x0/DMD gradient and post-step x0/Critic request exchanges while
  keeping one physical Score/Critic ZeRO-3 state.
- Added a call-mode test proving inner `set_grad_enabled(True)` overrides outer no-grad.
- Ran a successful pilot, found that retained tracer outputs extended inference graph
  lifetimes, added a non-retaining tracer mode, and reran from a clean commit.
- Repeated the full chain without the allocator request; preserved the Generator OOM and
  manually terminated Score only after it had completed DMD and was waiting for an
  unreachable Critic input.
- Generated a deterministic manifest over 60 raw artifacts and corrected the earlier
  allocator mechanism wording after auditing runtime warnings.
- Ran all 56 repository tests.

### Commands

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
/tmp/run_iter25_3g3s_step001.sh   # successful pilot; post tracer retained outputs
/tmp/run_iter25_3g3s_step002.sh   # accepted non-retaining trace
/tmp/run_iter25_3g3s_step003.sh   # no allocator flag; Generator OOM, Score terminated after DMD
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration25.py
```

### Results

- **Measured Level 2:** the accepted clean 3G+3S run completes the composed chain in
  60.744 s for three samples: 0.04939 samples/s and 0.00823 samples/s/GPU.
- First nested Generator F/B/step maxima are 9.970/15.444/0.190 s. Forward peaks at
  27.577 GiB and backward at 32.896 GiB. The real Score-produced DMD gradient reaches
  all three selected pyramid outputs; detached-surrogate error is <=1.23e-9.
- Shared Score DMD takes 9.356 s and peaks at 17.720 GiB. Its checksum is exactly
  unchanged. Critic F/B/step takes 4.018/8.571/0.155 s, peaks at 28.313 GiB, and changes
  the Score checksum sum by -0.05751.
- The post-step source rollout takes 10.066 s and has five calls: two no-grad prefixes
  and three selected calls with `grad_enabled=true` and call outputs requiring grad.
  The final returned x0 nevertheless has `requires_grad=false`.
- Generator advances before the post-step rollout and remains checksum-identical during
  Critic training. EMA runs only after the acknowledgment and matches its equation
  within 8.99e-10. All three boundary tensors and Critic operands are SHA-bound.
- The successful pilot retained post-step call outputs and reported 30.693 GiB. The
  accepted non-retaining trace reports 28.820 GiB, a 1.873-GiB instrumentation correction;
  first x0 and DMD-gradient SHAs are unchanged.
- A no-allocator repeat OOMs on Generator rank 1 during backward at a 584-MiB request;
  4.73 GiB is reserved but unallocated. Score had completed DMD. This reproduces the
  earlier no-flag failure.
- Every requested-flag process warns `expandable_segments not supported on this
  platform`. Across Iterations 24-25 there are two no-flag failures and four
  requested-flag successes, so the configuration is empirically admitted but actual
  segment-expansion causality is **not verified**.

### Interpretation

The composition hypothesis is verified. The real nested graph and shared-state version
chain fit together; the minimum measured symmetric placement is 3G+3S, not the earlier
selected-call 2G+2S abstraction.

The predicted inner-grad override is also verified, but the feared persistent
Critic-to-Generator edge is refuted. Selected calls create transient autograd state;
outer-no-grad history operations detach it before the returned x0. This yields a sharper
Runtime abstraction—call-local activation lifetime and returned Gradient Edge are
separate—and a concrete next optimization: force only those selected inference calls to
no-grad, with exact-output gates.

The pilot exposes an instrumentation lesson: an observer that retains tensors changes
the lifetime it claims to measure. The allocator evidence is similarly narrowed to an
empirical admission result because the runtime warning prevents a clean mechanism claim.

### Files Changed

- `runtime/rollout_trace.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration24.py`
- `scripts/distillschedule/summarize_iteration25.py`
- `tests/test_rollout_trace.py`
- `artifacts/manifests/iteration_024_nested_generator_rollout.json`
- `artifacts/manifests/iteration_025_composed_nested_stage3_chain.json`
- execution graphs and research/design/results documents

### Failure or Limitation

Both rollouts reuse one synthetic shape-exact context; the trained Critic checkpoint and
real data are absent. File transport is not NCCL and explicitly realizes the detached
boundary. Random history drop remains disabled for collective safety. The run calls real
source rollout and losses but not the official training entrypoint, scheduler steps, or
checkpoint path. One cold accepted run is not a speedup or convergence result. The
3G+2S asymmetric placement remains Not Run.

### Next Highest-Value Question

Can a Runtime-enforced call-local no-grad policy remove the three transient selected-call
graphs in Critic data generation while preserving the final x0 bitwise and reducing
peak memory or phase time?

### Next Command

Add a protected non-retaining wrapper that forces only post-step Generator model calls
to no-grad, capture identical RNG/state for source and candidate rollouts, then compare
call/output SHAs, peak allocated memory, and end-to-end phase time on 3G.

## Iteration 26

### Question

Can the Runtime suppress call-local autograd in the post-Generator-step Critic data
rollout, while preserving the exact final x0 and full Generator/Score/Critic/EMA version
chain, and does doing so materially reduce the measured 28.820-GiB / 10.066-s boundary?

### Hypothesis

A wrapper that forces only the five physical Generator model calls in the second rollout
to execute under `torch.no_grad()` should preserve the source call sequence, timesteps,
and final x0 bitwise. The source path already detaches the returned x0 after every
selected call, no auxiliary consistency/GAN/Reward branch is enabled, and no Generator
optimizer consumes this rollout. Therefore its three transient selected-call graphs are
semantically dead for the measured default branch.

In a paired same-process replay with Python, CPU, and CUDA RNG state restored, the
candidate should produce the same SHA-256 output and zero call outputs requiring grad.
It should lower post-rollout peak allocated memory by at least 1 GiB or time by at least
5%. A candidate-only 3G+3S chain should preserve all DMD, optimizer, Critic, and EMA
version gates.

### Importance

This is the first profile-derived Runtime optimization on the real Helios nested path,
rather than an abstraction or placement feasibility result. It directly exercises the
distinction between call-local autograd lifetime and a returned Gradient Edge. If
bitwise correctness or useful savings failed, this mechanism would not be a contribution.

### Evidence Needed

A clean paired replay from identical model version, inputs, and complete RNG state;
call-by-call stage, timestep, shape, grad mode, and output graph state; final x0 SHA and
numeric equality; source/candidate phase peaks and times; then a clean candidate-only
composed chain with unchanged G/Score/EMA ownership and boundary identities.

### Refutation Condition

Any output SHA or call signature differs, a candidate output retains autograd, ZeRO-3
collectives diverge, peak saving is below 1 GiB and time saving below 5%, or the
candidate-only composed chain fails a parameter-version or loss gate.

### Actions

- Added a narrow `ForceNoGradCalls` wrapper and tested that it preserves values while
  suppressing only the wrapped physical call graph.
- Added `source`, `force_no_grad`, and paired post-rollout policies to the protected
  real-backend probe. Paired mode restores Python, CPU Torch, and CUDA RNG state before
  candidate replay and uses a non-retaining tracer.
- Ran a clean paired 3G+3S ZeRO-3 chain and a separate candidate-only full chain.
- Added a deterministic artifact summarizer and reran all tests.

### Commands

```bash
/tmp/run_iter26_3g3s_paired_step001.sh
/tmp/run_iter26_3g3s_candidate_step002.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration26.py
HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m pytest -q
```

### Results

- **Measured Level 2 paired ablation:** every rank executes the same five calls with
  identical stage/timestep/shape signatures. Source has three call outputs requiring
  grad; candidate has zero. Every source/candidate final x0 is bitwise equal, with the
  same SHA-256 and max absolute error 0.
- Max-rank post-rollout allocated peak falls from 28.8196 to 21.5795 GiB: **7.2401 GiB
  (25.12%)**. This passes the preregistered 1-GiB phase-memory threshold.
- Max-rank paired time falls from 10.0142 to 9.5509 s: **4.63%**, below the
  preregistered 5% time threshold. Timing always ran source first, so no speed claim is
  made.
- The candidate-only component chain completes in 60.0123 s for three samples
  (0.04999 samples/s, 0.00833 samples/s/GPU). It preserves all x0/gradient/Critic-input
  SHA gates, DMD Score immutability, both optimizer ownership edges, Generator stability
  during Critic, Critic loss identity, and EMA equation (max error 7.15e-10).
- The older source chain took 60.7444 s, a 1.21% single-run difference. Different
  commits and one cold run each make this context, not an end-to-end speedup estimate.
- The whole-chain maximum remains the first Generator backward at about 32.9 GiB, so
  this phase-local optimization does **not** reduce the current placement's global peak.
- `artifacts/manifests/iteration_026_call_local_nograd.json` hashes 66 raw artifacts and
  reruns deterministically at SHA-256
  `569f3f848088b937491153fa0dc05ca2c0e7effbef96f29e140ff1d81ac69021`.
- 57 tests pass with 5 dependency warnings.

### Interpretation

The hypothesis is partially verified. The correctness and useful phase-memory claims
pass decisively; the 5% timing threshold is refuted. The measured source constructs
three semantically dead physical-call graphs because a later no-grad state boundary
detaches x0. DistillSchedule can safely eliminate them only when both facts are proved:
the returned edge is detached and no enabled auxiliary branch consumes those activations.

This is a gradient-state-aware optimization, not generic `no_grad` propagation and not
a new global-capacity result for the existing 3G+3S plan. Its immediate systems value is
7.24 GiB of headroom during Critic data generation, which can become useful for
co-residency, prefetch, or dynamic role remapping.

### Files Changed

- `runtime/rollout_trace.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration26.py`
- `tests/test_rollout_trace.py`
- `artifacts/manifests/iteration_026_call_local_nograd.json`
- research/design/profile/results documents

### Failure or Limitation

The policy is validated only for the default GAN/Reward/VAE/consistency-disabled branch.
Paired timing has fixed source-first order. Peak is call-local allocated memory, not
driver/reserved/global peak. The allocator request still emits an unsupported-feature
warning. Synthetic context, random Critic adapter, file transport, and one iteration do
not establish quality or convergence.

### Next Highest-Value Question

Can the real composed chain use three Generator ranks but only two Score ranks by packing
three samples into an uneven 2+1 Score batch, while preserving collective call order,
returned-gradient identity, globally weighted Critic loss, and version semantics?

### Next Command

Preregister the 3G+2S uneven-batch mapping and add a CPU-only protocol test before
changing the real Score transport or loss scaling.

## Iteration 27

### Question

Can the real nested chain reduce the shared Score/Critic group from three A100s to two
without changing DMD gradients, Critic reduction semantics, collective order, or
parameter versions, and does doing so improve GPU efficiency?

### Hypothesis

Round-robin source ownership `[0,2]`, `[1]` should allow a 3G+2S plan. An uneven 2+1
batch was expected to preserve source-seeded DMD values and reconstruct the global
three-sample Critic mean with DP weights 4/3 and 2/3. The full plan should remove one
GPU with no more than 20% component-chain slowdown or improve samples/s/GPU.

### Importance

Iteration 26 creates only phase-local Generator headroom. This experiment tests whether
the Runtime's logical/physical graph can instead synthesize a smaller whole-plan role
group, and whether role asymmetry is an efficiency optimization or merely a capacity
fallback.

### Evidence Needed

Source-owned tensor digests; DMD gradients compared to the accepted 3G+3S run; identical
Score collective-call counts; exact global Critic loss reconstruction; gradient
accumulation/version checksums; full-chain success, phase memory, throughput, and
samples/s/GPU.

### Refutation Condition

Any collective divergence, OOM, changed DMD gradient, wrong Critic loss/version, more
than 20% slowdown without a capacity crossing, or lower GPU-normalized throughput.

### Actions

- Added a tested source-batch protocol with round-robin ownership, source-scoped RNG,
  uneven-DP loss weights, collective-aligned padded rounds, and explicit padding slots.
- Integrated both `uneven_batch` and `padded_rounds` into the real 14B shared-Score
  protected backend.
- Added `gradient_accumulation_steps=2`; real slots use a 4/3 loss scale and padding
  slots use a zero-loss graph so both Score ranks execute identical collectives.
- Ran the uneven protocol, exact padded protocol without split modulation, and padded
  protocol with Helios's pre-existing split-modulation lifetime patch.
- Added a deterministic manifest builder that validates raw gradients, OOM records,
  loss identity, update boundaries, every cross-role checksum, and artifact hashes.

### Commands

```text
/tmp/run_iter27_3g2s_asym_step001.sh
/tmp/run_iter27_3g2s_padded_step002.sh
/tmp/run_iter27_3g2s_padded_splitmod_step003.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration27.py
```

### Results

**Measured Level 2.** The one-call uneven 2+1 protocol completed DMD, but the packed
samples did not preserve batch-one numerics: source ranks 0 and 2 have relative-L2
gradient errors 19.23% and 12.86% (cosine 0.9813 and 0.9917); the singleton rank is
bitwise exact. The two-sample Score rank then OOMed in Critic forward after reaching
34.700 GiB allocated.

Collective-aligned rounds `[[0,1],[2,pad]]` make all three DMD gradients bitwise equal to
3G+3S. Without the pre-existing split-modulation patch, both Score ranks OOM in first
Critic backward after reaching 34.876 GiB. With that patch, the full 3G+2S chain
completes. DMD remains checksum-invariant; Critic loss reconstructs the three-sample
mean within 3.974e-8; GAS boundaries are `[false,true]`; the first `engine.step()` leaves
Score exactly unchanged and the second advances it. Generator, Critic, transport, and
EMA checks all pass.

The accepted five-GPU chain takes 83.532 s for three samples: 0.03591 samples/s and
0.007183 samples/s/GPU. The matched semantic 3G+3S reference takes 60.012 s:
0.04999 samples/s and 0.008332 samples/s/GPU. Thus the exact asymmetric plan removes
16.67% of GPUs but is 39.19% slower, loses 28.16% throughput, and loses 13.79%
GPU-normalized throughput. Accepted DMD and Critic-backward peaks are 23.318 and
34.088 GiB.

### Interpretation

The original efficiency hypothesis is refuted. Role-group shrinkage is not a transparent
placement rewrite: batch-dependent Score numerics make packed execution semantically
different, while exact padding duplicates physical calls. The exact 3G+2S protocol is a
valid capacity fallback, not an efficiency win. Its feasibility also relies on Helios's
existing split-modulation lifetime patch and is not a DistillSchedule contribution.

### Files Changed

- `runtime/source_batch.py`
- `tests/test_source_batch.py`
- `scripts/distillschedule/asymmetric_score_protocol.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration27.py`
- `artifacts/manifests/iteration_027_asymmetric_score_group.json`
- research and results documents

### Failure or Limitation

The timing comparison has one cold run per plan and different implementation commits;
the 39.19% duplicated-call cost is characterization, not a confidence interval. Context
is synthetic, the Critic adapter is random, transport is file-based, and one step does
not prove convergence. Packed-gradient drift is rejected by exact semantics; no claim is
made about its training-quality impact. The allocator request continues to warn that its
named feature is unsupported. The full repository suite passes: 66 tests, 5 dependency
warnings.

### Next Highest-Value Question

Can frequency awareness make the exact smaller Score group useful on iterations where
Critic training is skipped, or does duplicated batch-one DMD still make 3G+2S dominated
even on DMD-only steps?

### Next Command

Preregister a frequency-aware DMD-only 3G+3S versus exact padded 3G+2S comparison using
the already validated raw tensors and prohibit extrapolating one-step timing into an
unmeasured training-frequency claim.

## Iteration 28

### Question

The source default updates Generator/DMD only every fifth step but trains Critic every
step. Can the common Critic-only branch use a smaller 2G+2S ZeRO-3 placement than 3G+3S,
while preserving source-rank operands and GPU-normalized efficiency?

### Hypothesis

On four of each five post-EMA-start steps, Generator performs only the nested Critic-data
rollout: no Generator backward/step, DMD Score, or EMA update. Since two Generator ranks
fit nested forward and two Score ranks fit Critic backward, 2G+2S should complete, keep
Generator bitwise unchanged, update Score once, and achieve samples/s/GPU no more than
5% below 3G+3S.

This gate deliberately keeps one sample per rank: two samples/four GPUs versus three
samples/six GPUs. It tests role-group scaling, not fixed-global-batch equivalence.

### Importance

Source inspection refutes the Iteration-27 tentative "Critic-skipped/DMD-only" framing.
`train_helios.py:1428-1429,1478-1643,1717-1809` and
`stage_3_post.yaml:231` establish Generator/DMD every five steps and Critic every step.
If the common branch admits a smaller placement, frequency becomes a measured planner
input rather than a decorative field.

### Evidence Needed

One forced-no-grad nested Generator rollout; no DMD, Generator update, or post-start EMA;
one shared-Score Critic F/B/step; same clean commit for both placements; common-rank x0,
noise, timestep, prediction, and loss identity; role peaks, component time, samples/s,
samples/s/GPU, and parameter-version checksums.

### Refutation Condition

OOM, any digest/loss/version failure, common-rank operand drift, or 2G+2S samples/s/GPU
more than 5% below 3G+3S. A favorable result is still incomplete until fixed-three-sample
semantics are measured.

### Actions

- Added `runtime/frequency.py` and tests for the source ratio, Critic-every-step contract,
  and the pre/post-EMA-start distinction.
- Added a protected `critic_only` exchange that suppresses physical Generator call
  graphs, skips DMD/G-step/EMA, trains Score once, and validates both state owners.
- Preserved an implementation-debug run where the first trace assertion incorrectly
  derived exit calls only from grad-selected calls; fixed and committed before accepted
  measurements.
- Ran clean 2G+2S and 3G+3S real-architecture branches with the same source-scoped RNG.
- Added a deterministic manifest that verifies digests, checksums, raw artifacts, and
  the preregistered efficiency floor.

### Commands

```text
/tmp/run_iter28_2g2s_step002.sh
/tmp/run_iter28_3g3s_step003.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration28.py
```

### Results

**Measured Level 2.** Both placements succeed from clean commit `8fb5f88`. On common
ranks 0/1, Generator x0, Critic timestep, noise, prediction, and scalar loss are bitwise
or exactly identical. All physical Generator calls have no graph; Generator global
sharded checksums remain exactly unchanged. DMD is absent and Score advances once. Loss
identity error is at most 1.192e-7.

2G+2S completes two samples in 23.982 s: 0.08339 samples/s and 0.02085
samples/s/GPU. Generator forward and Score backward peaks are 25.247 and 34.088 GiB.
3G+3S completes three samples in 24.277 s: 0.12357 samples/s and 0.02060
samples/s/GPU, with 18.464 and 27.315-GiB peaks. The smaller gate removes 33.33% of GPUs,
loses 32.51% absolute throughput because it handles one fewer sample, and measures 1.23%
higher GPU-normalized throughput.

### Interpretation

The preregistered floor is verified, but the 1.23% difference is a single-run tie, not a
speedup. Frequency exposes a genuine capacity opportunity: the common branch fits four
GPUs with essentially proportional scaling. It does **not** yet yield a legal dynamic
plan at fixed global batch; a second round or accumulation is required to process the
third sample, and its cost may erase the benefit.

### Files Changed

- `runtime/frequency.py`
- `tests/test_frequency.py`
- `scripts/distillschedule/critic_only_protocol.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/summarize_iteration28.py`
- `artifacts/manifests/iteration_028_critic_only_frequency.json`
- source/profile/design/results documents

### Failure or Limitation

The optimizer global batch differs, so final Score checksums are not cross-placement
comparable. One cold run cannot distinguish the measured 1.23% difference from noise.
Synthetic context, random Critic adapter, file transport, and one step do not establish
quality or convergence. Both use Helios's existing split-modulation patch and the
unsupported-feature allocator request. Before `ema_start_step`, source EMA semantics
differ and are not represented by this run. The full repository suite passes: 70 tests,
5 dependency warnings.

### Next Highest-Value Question

Can 2G+2S process the same three-source Critic-only global batch through two
collective-aligned rounds with GAS=2, and does the fixed-batch capacity plan retain any
GPU-efficiency benefit?

### Next Command

Preregister exact three-source 2G+2S padding against the measured 3G+3S operands, reuse
the tested source-round primitive, and require global-loss and optimizer-boundary
identity before interpreting performance.

## Iteration 29

### Question

Before building a second 2G producer round, can two Score ranks consume the exact three
Critic inputs through collective-aligned batch-one rounds, reconstruct global update
semantics, and retain GPU efficiency versus three Score ranks?

### Hypothesis

Replaying three SHA-bound inputs as `[[0,1],[2,pad]]` with GAS=2, real-slot scale 4/3,
and zero-loss padding should preserve every per-source Score prediction/loss, leave Score
unchanged at the first `engine.step()`, and update once at the second. The fixed-batch 2S
phase should fit and remain within 5% of 3S samples/s/Score-GPU.

### Importance

Iteration 28 changes optimizer global batch. The Score side is the cheapest exact gate:
if duplicated Critic execution is already inefficient, implementing a second physical
Generator round cannot rescue the current sequential full plan.

### Evidence Needed

Tracked SHA-valid replay; clean-commit 2S and 3S runs over identical bundles; bitwise
per-source noise/prediction and exact loss; global mean-loss identity; GAS boundaries;
logical state checksums independent of ZeRO padding; phase time, peak, throughput, and
GPU-normalized throughput.

### Refutation Condition

OOM, any operand/loss/version mismatch, or 2S Score GPU efficiency more than 5% below
3S. Success would still be Score-only evidence rather than full 2G+2S execution.

### Actions

- Extended the padded protocol to skip all DMD work on the source-authentic Critic-only
  branch while retaining source identity, loss scaling, and version gates.
- Added `prepare_critic_replay.py`; it validates every source x0 and records hashes for
  x0, prompt, histories, source file, and rewritten destination.
- Ran a preliminary 2S replay. It exposed that the profiler counted 734 ZeRO tail-padding
  elements in the historical 3S checksum, making cross-world state comparison invalid.
- Added and tested logical partition slicing, committed it, then reran both 2S padded and
  3S symmetric Score phases from the same clean commit and source bundles.
- Added a deterministic manifest validating operands, global loss, optimizer boundaries,
  logical state identity, performance, and all raw artifact hashes.

### Commands

```text
/tmp/run_iter29_fixed3_score2_step002.sh
/tmp/run_iter29_fixed3_score3_step003.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration29.py
```

### Results

**Measured Level 2.** Both accepted Score runs succeed from clean commit `6ce95ee`. All
three x0, noise, timestep, prediction, and loss identities match exactly. The padded
mean matches the symmetric source-loss mean within 1e-6; its internal scaled-reduction
identity error is 7.947e-8. GAS boundaries are `[false,true]`; logical Score state is
unchanged after round one and advances after round two. DMD is not executed.

The 2S fixed-batch phase takes 27.685 s (7.978 forward, 19.099 backward, 0.608 step),
peaking at 28.970/34.088 GiB in forward/backward. The 3S phase takes 14.261 s
(4.277/9.811/0.174) at 21.525/27.315 GiB. For the same three samples, 2S saves one Score
GPU but is 94.12% slower, loses 48.49% throughput, and loses 22.73%
samples/s/Score-GPU. The preregistered efficiency hypothesis is refuted.

After excluding each `ds_tensor` tail beyond logical `ds_numel`, both world sizes report
1,246,314,496 elements and exactly the same initial sum. Post-update sums differ by
1.497e-4 from reduction/order effects; bitwise cross-sharding optimizer equality is not
claimed.

### Interpretation

Exact fixed-batch role shrinkage is capacity-feasible but not efficient in the current
sequential schedule. Padding duplicates a complete Critic forward/backward. Because the
Score-only gate already fails by 22.73%, adding an unmeasured second Generator round
cannot repair this sequential candidate. Only a separately measured overlap or different
batch-semantics design could reopen it.

### Files Changed

- `scripts/distillschedule/asymmetric_score_protocol.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `scripts/distillschedule/prepare_critic_replay.py`
- `runtime/zero_partition.py`
- `tests/test_zero_partition.py`
- `tests/test_frequency.py`
- `scripts/distillschedule/summarize_iteration29.py`
- `artifacts/manifests/iteration_029_fixed_batch_critic_only.json`
- research/design/profile/results documents

### Failure or Limitation

Generator inputs are replayed from the clean Iteration-28 run; the second 2G Generator
round and end-to-end overlap are Not Run. One cold run per placement is not a confidence
interval, though the 1.94x gap is far beyond the threshold. The Critic adapter is random,
context synthetic, and one update does not establish quality/convergence. Existing
split-modulation and the unsupported-feature allocator request remain prerequisites.
The full repository suite passes: 76 tests, 5 dependency warnings.

### Next Highest-Value Question

Does the measured call-local 7.24-GiB Critic-rollout headroom enable an actual overlap or
prefetch that reduces exposed Score time, or is there no independent work under ZeRO
collective/version constraints?

### Next Command

Use the completed phase trace to enumerate legal overlap windows before writing another
backend: reject any candidate that overlaps collectives sharing a group or changes the
three-source version barrier.

## Iteration 30

### Question

Can the three-rank Generator and three-rank Score ZeRO-3 engines share the same three
A100-40GB devices for a Critic-only step, and exactly which phase sets the remaining
co-residency gap?

### Hypothesis

Measured idle allocations are 13.567 GiB for Generator and 13.548 GiB for Score per
rank. Their 27.114-GiB sum should admit the forced-no-grad Generator rollout and Score
forward. Score backward was predicted to be the first OOM boundary near 40.88 GiB.
Releasing the Generator allocator cache before Score starts should separate true
live-state pressure from a cache artifact.

### Importance

If the chain succeeds, same-device frequency-aware co-residency is stronger than 2G+2S.
If it fails at a precisely identified phase, the next backend mechanism can target live
state instead of adding speculative prefetch.

### Evidence Needed

A clean same-device launch with two independent three-rank process groups; Score fully
ready before Generator launch; allocated/reserved and driver memory at engine-ready,
rollout, Score forward, and failure/success; clean source identity and retained logs.

### Refutation Condition

Dual engines cannot initialize, Generator rollout OOMs, or the full chain unexpectedly
succeeds. Each outcome refutes a different part of the additive steady-state model.

### Actions

- Added a cache-release boundary after the detached Critic input is published.
- The first launcher attempt was rejected because stale rank JSON triggered Generator
  launch before the new Score processes reached ready.
- Corrected the launcher to use a fresh output directory and reran from clean commit
  `60b2df8`, with Score reaching ready on all three ranks before Generator launch.
- Detected and manually cleaned an orphaned Score process tree after the expected peer
  failure; raw logs and the invalid attempt were retained.
- Added a deterministic manifest validating source identity, failure call site, memory
  stages, process exit codes, `nvidia-smi` samples, and raw artifact hashes.

### Commands

```text
bash /tmp/run_iter30_colocated3_step002.sh
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/distillschedule_pycache \
  /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration30.py
```

### Results

**Measured Level 2, negative capacity result.** Score reaches its protected ready state
at 13.548 GiB allocated and 15.133 GiB reserved per rank. Its own initialization had
already peaked at 31.942 GiB. With Score live, Generator fails before ZeRO-3 partitioning
inside `DeepSpeedEngine._configure_distributed_model -> module.to(device)`.

The failing Generator process has 20.392 GiB allocated / 20.400 GiB reserved and only
0.132 GiB driver-free memory. The simultaneous Score-ready plus Generator allocation is
33.940 GiB, while `nvidia-smi` observes up to 40,204 MiB used. The failed allocation is
136 MiB. No Generator model call executes.

### Interpretation

The steady-state shard sum is not the admission peak. DeepSpeed first materializes the
second full module on GPU before partitioning, while the first engine's live shards,
allocator, CUDA/NCCL context, and workspaces remain. `empty_cache()` cannot release any
of those live states. Same-device co-residency therefore requires either inactive-engine
parameter offload or partitioned construction; activation optimization cannot repair
this earlier initialization failure.

### Files Changed

- `scripts/distillschedule/critic_only_protocol.py`
- `scripts/distillschedule/summarize_iteration30.py`
- `artifacts/manifests/iteration_030_colocated_engine_gate.json`
- research/profile/design/results documents

### Failure or Limitation

This uses the pinned diagnostic Torch 2.4 environment without FlashAttention2 and two
separate process groups rather than a shared-process backend. Forward, backward, and
optimizer steps are Not Run. The launcher needs process-group-aware cleanup after peer
failure. It is a capacity admission result, not throughput evidence. The full repository
suite passes: 76 tests, 5 dependency warnings.

### Next Highest-Value Question

Can a standard parameter-offload or partitioned-construction backend remove the second
engine's pre-partition materialization peak while preserving the same Critic-only graph?

### Next Command

Add an explicit ZeRO-3 parameter-offload option to the probe, first verify an isolated
Score F/B/step, then retry same-device engine admission. In parallel, finish the official
Torch-2.10 environment gate required for H0.

## Iteration 31

### Question

Can standard ZeRO-3 CPU parameter offload preserve the real Score update while removing
enough live GPU state to admit a second full DeepSpeed engine on the same A100-40GB ranks?

### Hypothesis

`offload_param.device=cpu` will preserve isolated Score semantics, lower two-rank ready
allocation below 6 GiB, and allow Generator pre-partition admission while Score remains
live. Active execution may still be slower or fail later.

### Importance

Iteration 30 proves activation scheduling is downstream of engine admission. CPU
parameter offload is the standard fixed-policy mechanism closest to the requested H1
Manual-Offload baseline; it must be exhausted before claiming a new Runtime mechanism.

### Evidence Needed

A clean option and unit tests; fresh offload/no-offload Score F/B/step under identical
shape and seed; output, loss, update, GPU memory, host RSS, and time; then a fresh
Score-first same-device 3G+3S admission run with process-tree-safe cleanup.

### Refutation Condition

The isolated run fails or exceeds 1% update deviation, Score ready remains at least
6 GiB, or Generator still OOMs before `engine_ready`. Admission success alone would not
establish active-step feasibility or efficiency.

### Actions

- Added and tested a Runtime-owned DeepSpeed ZeRO config builder plus explicit
  `--zero-param-offload {none,cpu}` and pinned-memory validation.
- Ran clean two-rank no-offload and CPU-parameter-offload Score F/B/step probes from the
  same commit, shape, seed, buckets, and optimizer contract.
- Implemented a fresh-output, run-id-validating, process-group-safe same-device launcher.
- Reran the 3G+3S Critic-only admission with CPU parameter offload on both engines.
- In parallel, attempted the literal official requirements. It is currently
  unreproducible because unpinned Diffusers HEAD requires `huggingface-hub>=1.23.0`
  while Helios pins `1.4.1`; a date-pinned Diffusers environment is being built
  separately and is not counted as H0 evidence yet.
- Added a deterministic manifest validating semantics, capacity, time, failures, and
  raw artifact hashes.

### Commands

```text
torchrun --nproc_per_node=2 scripts/distillschedule/probe_deepspeed_critic.py \
  ... --zero-stage 3 --optimizer-step
torchrun --nproc_per_node=2 scripts/distillschedule/probe_deepspeed_critic.py \
  ... --zero-stage 3 --zero-param-offload cpu --optimizer-step
bash /tmp/run_iter31_colocated_paramoffload_step003.sh
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration31.py
```

### Results

**Measured Level 2.** Isolated CPU parameter offload succeeds and preserves the exact
Score output checksum and loss. Initial logical parameter sum is exact. Post-step update
checksum differences are 0.705% in sum and 0.00167% in norm delta, within the
preregistered 1% BF16/backend tolerance.

On two ranks, ready allocation falls from 20.321 to 4.643 GiB (15.678 GiB, 77.15%).
Forward peak falls from 29.560 to 13.882 GiB and backward peak from 35.083 to
19.405 GiB. This capacity costs 63.82% in measured F/B/step time: 13.480 to 22.082 s.

The same-device 3G+3S admission still fails. Three-rank Score is reduced to
3.095 GiB allocated / 3.107 GiB reserved, satisfying the 6-GiB threshold. Generator
nevertheless reaches 31.375 GiB while DeepSpeed broadcasts the full pre-partition
module; driver usage reaches 40,052 MiB and NCCL reports CUDA OOM. No Generator
`engine_ready` or model forward is reached.

### Interpretation

CPU parameter offload is a valid, strong baseline mechanism for isolated active Score:
it saves 15.68 GiB but slows the phase by 63.82%. It does not change DeepSpeed's
admission order. The remaining Score allocation is trainable optimizer/master state,
and the second model is still fully moved and broadcast before its own parameter offload
activates. The next minimal standard mechanism is CPU optimizer offload; otherwise the
backend needs partition-aware construction. Activation offload cannot fix this phase.

### Files Changed

- `runtime/deepspeed_backend.py`
- `runtime/__init__.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `tests/test_deepspeed_backend.py`
- `scripts/distillschedule/summarize_iteration31.py`
- `artifacts/manifests/iteration_031_zero_param_offload.json`
- research/profile/design/results documents

### Failure or Limitation

The probes use diagnostic Torch 2.4 without FlashAttention2. Full trainable deltas were
not checkpointed; correctness uses exact outputs/losses and aggregate update checksums.
Co-resident forward is Not Run. CPU offload is existing DeepSpeed functionality, not a
DistillSchedule contribution. The full repository suite passes: 80 tests, 5 dependency
warnings.

### Next Highest-Value Question

Does CPU optimizer offload remove the final 3.095-GiB Score-ready state and admit the
second engine, or is partition-aware construction mandatory?

### Next Command

Add a separately testable `offload_optimizer.device=cpu` option, validate the isolated
Score update, then retry same-device admission before implementing custom construction.

## Iteration 32

### Question

Does adding CPU optimizer offload to CPU parameter offload remove the remaining
three-rank Score-ready allocation and permit a second full Generator engine to pass
DeepSpeed admission and execute a real Critic-only step on the same A100-40GB devices?

### Hypothesis

`offload_optimizer.device=cpu` will preserve isolated Score output/loss and remain within
1% aggregate update tolerance, reduce two-rank ready allocation below 1 GiB and
three-rank Score ready below 0.5 GiB. The second Generator should reach `engine_ready`;
the active chain may still be too slow or OOM later.

### Importance

This is the final standard fixed-policy mechanism before custom partition-aware
construction. It determines the honest H1 Manual-Offload baseline and whether the
research problem is feasibility or exposed offload cost.

### Evidence Needed

A testable config, isolated two-rank Score F/B/step, exact output/loss and bounded update
differences, ready/phase memory, host RSS and time, followed by a fresh process-safe
same-device 3G+3S run and exact comparison to the seed-matched separate-groups chain.

### Refutation Condition

Isolated semantics fail, update deviation exceeds 1%, ready thresholds are missed,
Generator fails before `engine_ready`, or the active Critic-only chain fails.

### Actions

- Added explicit optimizer-offload backend state and validation.
- Preserved the client `torch.optim.AdamW` with an explicit DeepSpeed top-level override;
  otherwise DeepSpeed rejects it in favor of a different CPUAdam implementation.
- Ran the isolated two-rank real Score update with CPU parameter+optimizer offload.
- Ran a process-group-safe Score-first, same-device 3G+3S admission and active step.
- Compared it to the exact seed/shape separate 3G+3S Iteration-28 reference.
- Built and smoke-tested a node-local Torch 2.10 environment using a date-pinned Diffusers
  commit after proving literal unpinned requirements are dependency-inconsistent. This is
  environment evidence only; the official entrypoint is still Not Run.
- Added a hash-validating manifest and updated research/design/results documents.

### Commands

```text
torchrun --nproc_per_node=2 scripts/distillschedule/probe_deepspeed_critic.py \
  ... --zero-stage 3 --zero-param-offload cpu --zero-optimizer-offload cpu \
  --optimizer-step
bash /tmp/run_iter32_colocated_param_optimizer_offload.sh
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  scripts/distillschedule/summarize_iteration32.py
/tmp/distillschedule-helios-official-torch210/bin/python < official smoke script
```

### Results

**Measured Level 2.** Isolated output/loss and initial logical parameters are exact.
Update deltas differ by 0.150% in sum and 0.00146% in norm, below the preregistered 1%
gate. Ready allocation falls 4.643 GiB→approximately zero; forward/backward peaks fall
13.882→9.239 and 19.405→14.762 GiB; optimizer peak falls 13.954→0.469 GiB. F/B/step
rises 22.082→31.627 s (+43.21%).

Both three-rank engines then admit on the same three physical GPUs and the complete
source nested Generator→Critic F/B/AdamW update chain succeeds. Active allocated peaks
are 4.897 GiB for Generator and 13.767 GiB for Score backward, although construction
still reaches 38,096 MiB by `nvidia-smi`. Against the exact-seed separate 3G+3S run,
Generator x0 and Score noise/prediction/timestep/loss are exact; Critic update deltas
differ by 0.0846% in sum and 0.00083% in norm.

The co-resident chain takes 48.261 s versus 24.277 s (+98.79%) and halves physical GPUs
from six to three. Throughput falls 49.70%; samples/s/GPU changes +0.61%, a tie rather
than a speedup. The pinned official Torch 2.10+cu128 environment passes A100 BF16 CUDA
smoke on node 0, but no official training invocation has been counted.

### Interpretation

The central hypothesis is verified, including the active-step gate. Standard DeepSpeed
full CPU offload is already sufficient for this real-architecture Critic-only branch.
It is the strong baseline, not a contribution. The Runtime opportunity is to choose the
separate/co-resident Pareto point and hide or avoid the approximately 2x exposed offload
cost using branch frequency, safe release windows, CPU bandwidth, and network topology.
The full Generator-update/DMD/EMA branch remains unproved.

### Files Changed

- `runtime/deepspeed_backend.py`
- `runtime/__init__.py`
- `scripts/distillschedule/probe_deepspeed_critic.py`
- `tests/test_deepspeed_backend.py`
- `scripts/distillschedule/summarize_iteration32.py`
- `artifacts/manifests/iteration_032_zero_optimizer_offload.json`
- research/profile/design/results documents

### Failure or Limitation

The measured branch uses synthetic inputs in the diagnostic Torch 2.4 environment
without FlashAttention2. It is one real-loss Critic-only step, not official data loading,
checkpointing, convergence, or the Generator-update branch. CPU memory/bandwidth under
multiple simultaneous replicas and inter-node transport are not yet measured. Full CPU
offload is existing DeepSpeed functionality.

### Next Highest-Value Question

Can the pinned official Torch 2.10 environment be reproduced on both nodes, and what
measured NCCL/transport contract should govern the two-node H0 official-entry baseline?

### Next Command

Freeze the node-0 resolved environment, reproduce it node-locally on node 1, run paired
single-node/two-node NCCL and boundary-tensor benchmarks, then launch the smallest
official-entry cold-start gate with all 16 GPUs available.

## Iteration 33

### Question

Can the date-pinned official Helios Torch 2.10 environment be reproduced exactly on both
nodes, and does it provide a valid 16-rank NCCL contract for the official-entry H0 gate?

### Hypothesis

The resolved Helios requirements with Diffusers commit
`81c354d8796d7c30336f0628241b768df4e00476` will install on node 1 with the same key
package versions and direct source identity as node 0. Both nodes will pass A100 BF16
CUDA smoke. A fresh 16-rank Torch 2.10 NCCL all-reduce using the measured four-HCA path
will succeed, and 64/256-MiB algorithm bandwidth will remain within 20% of Iteration 0's
24.98/25.62 GB/s measurements.

### Importance

The user requires both eight-GPU nodes for the official strong baseline. A reproducible
software and communication contract must be established before an official-entry
failure can be classified as model capacity, code incompatibility, or network behavior.

### Evidence Needed

The resolved requirements and original conflict log; node-local package inventories,
Diffusers direct URLs, driver/CUDA/NCCL versions, BF16 CUDA smoke on both nodes; fresh
single-node and 16-rank all-reduce artifacts with identical source commit and HCA
configuration; raw logs, return codes, and hashes.

### Refutation Condition

Node 1 cannot reproduce the environment, either node fails CUDA smoke, 16-rank NCCL
fails/deadlocks, package identities differ, or either 64/256-MiB algorithm bandwidth is
more than 20% below the Iteration-0 reference. A communication slowdown does not block
all research, but it invalidates the assumed H0 cost contract.

### Actions

- Preregistered environment identity and communication thresholds before installing the
  second node or rerunning NCCL.
- Replayed the literal requirements and retained the dependency-resolution failure.
- Pinned Diffusers to `81c354d8796d7c30336f0628241b768df4e00476`.
- Built one wheel from that Git object on node 0 after node 1's direct GitHub access
  timed out, installed it node-locally, and verified the installed trees byte-for-byte.
- Ran BF16 CUDA smoke on both nodes, an eight-rank intra-node all-reduce, a sixteen-rank
  two-node all-reduce, and cross-node source-shaped BF16 point-to-point transfers.
- Added actual NCCL API version and source-state capture to the transport profilers.

### Commands

```bash
/tmp/distillschedule-helios-official-torch210/bin/pip install   -r configs/helios_official_requirements_20260312.txt
/tmp/distillschedule-helios-official-torch210/bin/python   scripts/distillschedule/bench_dist.py ...
/tmp/distillschedule-helios-official-torch210/bin/python   scripts/distillschedule/bench_p2p.py ...
PYTHONPATH=. /tmp/distillschedule-helios-official-torch210/bin/python   scripts/distillschedule/summarize_iteration33.py
```

Exact launch arguments, rank logs, return codes, package inventories, and hashes are
captured by `iteration_033_official_two_node_environment.json`.

### Results

**Measured environment/transport gate, not a model result.**

- The literal requirements are no longer resolvable: current Diffusers HEAD requires
  `huggingface-hub>=1.23.0`, while Helios pins `1.4.1`.
- Node 1's direct GitHub install timed out. The local-wheel fallback has SHA256
  `c90e20ddbdf626442d2d06f52560f7306453ba8608d67e67a3a192aca379ee21`.
- Both nodes expose 161 package names with identical non-Diffusers versions. Their
  installed Diffusers trees contain 858 files/22,028,576 bytes and share tree SHA256
  `ba6673a2bf7dfeaa5c69ae83a8f886639ea982ac5bf79caa886f1b38a4a44e34`.
- Both A100 nodes pass the same BF16 CUDA checksum under Torch 2.10.0+cu128.
- The clean sixteen-rank run uses all 16 GPUs and four HCAs. Algorithm bandwidth is
  24.814 GB/s at 64 MiB and 25.785 GB/s at 256 MiB: -0.658% and +0.648% versus
  Iteration 0, both far above the preregistered 80% floors.
- Cross-node 1.106/3.564/12.427-MB BF16 transfers reach 6.35--6.72,
  11.87--12.38, and 17.30--18.06 GiB/s by direction. Maximum latency regression
  versus Iteration 9 is 15.53%.
- The container's `NCCL_VERSION=2.18.5` environment variable is stale; the Torch API
  and NCCL logs report 2.27.5.

### Interpretation

The literal-install and node-1-direct-download portions of the hypothesis are refuted,
but the intended reproducibility contract is recovered with a content-addressed local
wheel. CUDA and sixteen-rank NCCL are verified. Network feasibility is not the blocker
for H0; subsequent failures should first be attributed to official-entry dependency,
asset, control-flow, or memory behavior. This does not prove Helios loads or trains.

### Files Changed

- `configs/helios_official_requirements_20260312.txt`
- `scripts/distillschedule/bench_dist.py`
- `scripts/distillschedule/bench_p2p.py`
- `scripts/distillschedule/summarize_iteration33.py`
- `artifacts/manifests/iteration_033_official_two_node_environment.json`
- environment and raw communication artifacts under ignored artifact directories
- project status documents

### Failure or Limitation

Direct internet installation is not reproducible on node 1, and the pinned environment
uses a newer resolved stack than an immutable upstream lockfile because Helios does not
provide one. This gate establishes environment and communication feasibility only; it
is not an official-entry or model-training result.

### Next Highest-Value Question

Can the unchanged official Stage-3 entrypoint reach its first model/data boundary on all
16 A100-40GB GPUs under the pinned environment, and where does the first failure occur?

### Next Command

Preregister Iteration 34, inventory the official launcher/config/checkpoint/data
contracts, then execute the smallest semantics-preserving sixteen-rank official-entry
cold start with ZeRO-3 CPU parameter and optimizer offload when the official ZeRO-2
placement cannot admit.


## Iteration 34

### Question

On the verified two-node stack, where does the unchanged official Stage-3 entrypoint
first fail when launched with all 16 A100-40GB GPUs?

### Hypothesis

The official Stage-3 ZeRO-2 configuration will fail independently on each rank before
dataset construction or `accelerator.prepare`: the source casts and moves the full
Generator to CUDA, then casts and moves the full Score backbone to the same CUDA device.
Because ZeRO-2 replicates model parameters and partitioning has not yet occurred, adding
ranks does not reduce this admission peak. The first capacity failure should be
`real_score_model.to(accelerator.device)` with the Generator already resident.

### Importance

This is the H0 strong-baseline boundary. If verified, “use all 16 GPUs” and low-VRAM
runtime calls cannot by themselves make the official path fit A100-40GB: the baseline
needs an official-entry-compatible construction/offload adaptation before activation
scheduling is relevant. If refuted, later Runtime work must target the actually measured
earlier or later boundary instead.

### Evidence Needed

A clean source commit; a derivative config changing only local asset/data/output paths
and finite-run controls; exact two-node/16-rank launch command; per-rank logs and return
codes; periodic `nvidia-smi`; first traceback and source line; proof of whether dataset
construction and either DeepSpeed `prepare` call were reached; host/GPU/process
topology; hashes of config and raw logs.

### Refutation Condition

The entrypoint reaches dataset construction or either DeepSpeed engine preparation
without this OOM, or it fails first because of a dependency, incompatible API, missing
asset, missing data, distributed initialization, or another source line. Any such result
must replace the predicted capacity boundary rather than be explained away.

### Actions

- Completed the pinned environment and sixteen-rank communication gate in Iteration 33.
- Audited the official launcher: it selects ZeRO-2 and creates two Accelerate DeepSpeed
  plugins, while model construction precedes either DeepSpeed `prepare`.
- Added a path-adapted one-step config, deterministic 16-sample source-shaped Stage-3
  dataset, two-node launcher, per-rank tee logs, and 0.5-second GPU-process monitoring.
- Preserved two invalid preflights: run 001 inherited inconsistent platform rendezvous
  variables; run 002 forced offline mode before the A100 Flash-Attention-2 binary was
  cached.
- Fetched the official `kernels-community/flash-attn2` A100 variant through the
  reachable Hugging Face mirror into a shared cache and verified it on both nodes.
- Ran the valid clean 16-rank H0 cold start and collected node-local return codes,
  rank-stream logs, model-load coverage, cgroup postmortems, and artifact hashes.

### Commands

```bash
/tmp/distillschedule-helios-official-torch210/bin/python   scripts/distillschedule/prepare_official_h0_dataset.py

# Run concurrently on node 0 and node 1:
scripts/distillschedule/launch_official_h0.sh 0 iteration34_official_h0_zero2_003
scripts/distillschedule/launch_official_h0.sh 1 iteration34_official_h0_zero2_003

PYTHONPATH=. /tmp/distillschedule-helios-official-torch210/bin/python   scripts/distillschedule/summarize_iteration34.py
```

### Results

**Measured H0 official-entry cold-start failure; no forward or training step.**

- All 16 rank logs report world size 16 under Torch 2.10.0+cu128/NCCL 2.27.5.
- Every rank loads the exact Helios-Distilled Generator and Helios-Base Score backbone:
  16/16 markers for each physical model.
- Both nodes have a 760-GiB host-memory cgroup limit and no swap. Each records one
  `oom_kill`; rank 3 on node 0 and rank 10 on node 1 are killed by SIGKILL.
- The process-level GPU monitor reaches only 12,202 MiB. The predicted
  `real_score_model.to(cuda)` OOM is therefore refuted: the job dies first from
  replicated host construction state.
- The last confirmed source stage is Score load/norm patch at
  `train_helios.py:345-351`. Dataset construction, either DeepSpeed
  `accelerator.prepare`, the low-VRAM manager, forward, backward, optimizer, and EMA
  update are not reached.
- Linux SIGKILL has no Python traceback, so the exact post-load allocation between the
  two-model load and dataset construction remains unresolved. The source candidates
  include rank-256 Generator adapter creation and full `copy.deepcopy` EMA state.

### Interpretation

The hypothesis is refuted in a more restrictive direction. Sixteen-way data parallelism
duplicates pre-sharding host state eight times per node, so it exhausts 760 GiB before
ZeRO-2 or Helios's hand-written low-VRAM policy can act. Merely adding GPUs, switching
the later engine to ZeRO-3, or offloading activations cannot fix this earlier boundary.
The strong A100 baseline must first reduce construction state: direct BF16 host load,
staged Generator/Score engine preparation with ZeRO-3 CPU parameter and optimizer
offload, release of unused VAE/text state, and sparse/reconstructable EMA.

### Files Changed

- `configs/experiments/iteration34_official_h0_zero2.yaml`
- `scripts/distillschedule/prepare_official_h0_dataset.py`
- `scripts/distillschedule/launch_official_h0.sh`
- `scripts/distillschedule/summarize_iteration34.py`
- `artifacts/manifests/iteration_034_official_h0_zero2.json`
- ignored raw logs/traces and the external deterministic dataset/cache
- project status documents

### Failure or Limitation

No model call or optimizer transition executed. The deterministic input preserves the
official loader and Stage-3 tensor shapes but is not an official training sample. The
precise CPU allocation that crossed the limit needs opt-in phase instrumentation before
the baseline adaptation is attributed to a single source statement.

### Next Highest-Value Question

Can a semantics-preserving official-entry adaptation cross host construction and GPU
admission, then execute one complete Generator-update/DMD/Critic/EMA step on all 16
A100-40GB GPUs?

### Next Command

Preregister Iteration 35; add opt-in CPU/GPU phase tracing; load frozen backbones directly
in BF16; release unused VAE/text state; replace full frozen EMA duplication with
trainable-only reconstructable EMA; and prepare Generator then Score with ZeRO-3 CPU
parameter plus optimizer offload before running the same one-step two-node entrypoint.


## Iteration 35

### Question

Can an explicitly labeled, semantics-preserving A100 adaptation of the official Stage-3
entrypoint cross the newly measured host-construction boundary and execute one complete
Generator/DMD/Critic/EMA iteration on all 16 GPUs?

### Hypothesis

Loading frozen backbones directly in BF16, releasing VAE/text state once their disabled
branches no longer consume it, preparing Generator and Score sequentially with ZeRO-3
CPU parameter plus optimizer offload, and storing EMA only for trainable/local shards
will keep each node below 650 GiB host memory and each GPU below 39 GiB. The run will
reach both DeepSpeed engines, build the real Stage-3 dataset, execute the global-step-0
Generator and Critic updates, perform one EMA transition, and exit without OOM.

### Importance

Iteration 34 proves the unadapted official strong baseline dies before existing low-VRAM
logic. This iteration determines whether standard backend mechanisms plus a
reconstructable sparse EMA are sufficient to create the honest H1 A100 baseline. If it
still fails, the measured next boundary—not a presumed Runtime innovation—must drive
the following change.

### Evidence Needed

Opt-in phase records with host cgroup/current memory, process RSS, CUDA
allocated/reserved, parameter dtype/count and phase timestamps; exact config and source
diff; BF16-load equivalence tests; sparse-EMA shard/update tests; 16-rank engine/dataset
markers; finite Generator/Critic losses; pre/post logical parameter checksums; EMA
optimization step and checksum; per-node host/GPU monitors; phase/end-to-end time; raw
logs, return codes, and hashes.

### Refutation Condition

Either node exceeds 650 GiB host memory, any process exceeds 39 GiB GPU memory, any rank
OOMs/deadlocks, construction changes trainable parameter identity, sparse EMA disagrees
with a dense reference beyond 1e-5 relative error, the official entrypoint does not
reach both optimizer updates and EMA, or any loss/update is non-finite. A successful
cold start without a complete step only partially verifies the hypothesis.

### Actions

- Preregistered the host/GPU bounds and complete-step contract before implementing the
  adaptation.

### Commands

Not Run.

### Results

Not Run.

### Interpretation

Not Run.

### Files Changed

- `research_log.md`

### Failure or Limitation

Not Run. This is a one-step systems/correctness baseline, not convergence or quality
evidence. Sparse EMA must be reconstructable from the immutable base checkpoint and
tracked trainable state before it can be treated as semantically equivalent.

### Next Highest-Value Question

Which exact post-load state must be removed or staged to make the 16-rank official
entrypoint executable?

### Next Command

Implement opt-in phase tracing and unit-test BF16 construction plus sparse EMA first;
then run a cold-start-only adapted 16-rank gate before enabling the full one-step path.


## Iteration 35 Result

### Question

Can ZeRO-3 CPU parameter and optimizer offload cross official Stage-3 construction
and complete one real 14B DMD step on 16 A100-40GB GPUs?

### Hypothesis

Direct BF16 construction, preprocessing release, sparse trainable-shard EMA, and
sequential ZeRO-3 prepare will stay below 650 GiB host and 39 GiB GPU memory and
complete Generator, DMD, Critic, and EMA.

### Importance

This tests whether a standard sharded backend alone is a sufficient A100 strong
baseline before attributing value to a cross-component Runtime.

### Evidence Needed

All-rank phase traces through both prepares and optimizer steps, host/GPU peaks,
finite losses/checksums, and exact failures.

### Refutation Condition

Either bound is crossed, a rank fails, or the complete step does not execute.

### Actions

Added opt-in phase tracing, direct BF16 construction, preprocessing release,
sparse reconstructable EMA, sequential two-engine ZeRO-3 prepare, and an exact
zero-sum Gaussian replacement for Torch 2.10's rejection of the official
singular block-noise Cholesky.

### Commands

Cold run: launch_official_h1.sh on both nodes with run
iteration35_official_h1_cold_004. Full run: the same launcher with
iteration35_official_h1_full_002.

### Results

Measured partial success and full-step refutation. Cold run 004 reaches both
ZeRO-3 prepares on all 16 ranks and exits cleanly. Phase span is 597.326 s,
per-node cgroup peaks are 709.112 and 709.099 GiB, refuting the 650-GiB bound,
and per-process GPU peak is 30,618 MiB. Full run 002 reaches the
Generator-to-Score DMD boundary, then nonzero ranks fail DeepSpeed's ZeRO-3
parameter coordinator with a collective-order disagreement. It is not an OOM.

### Interpretation

ZeRO-3 solves admission but is not a drop-in replacement for the nested
two-engine workload. Single-model sharding and the multi-component execution
graph cannot be treated as independent layers without engine-aware ordering.

### Files Changed

runtime/official_baseline.py, train_helios.py, zero3 CPU-offload config and
launcher, tests, and ignored raw traces.

### Failure or Limitation

No full ZeRO-3 step completed. A cold-start result is not training evidence.

### Next Highest-Value Question

Can official ZeRO-2 plus the existing hand-written model swaps execute a step?

### Next Command

Run a cold gate and one full step using official ZeRO-2 with CPU optimizer
offload and sequential construction.


## Iteration 36

### Question

Can official ZeRO-2 plus Helios's hand-written Generator/Score CPU swapping
complete a 384x640 one-step strong baseline?

### Hypothesis

Official ZeRO-2 avoids the ZeRO-3 collective mismatch and CPU swapping fits both
14B backbones.

### Importance

This is the strongest faithful baseline: official component order and loss,
official ZeRO-2, and the repository's low-VRAM algorithm.

### Evidence Needed

A successful cold gate followed by Generator, Critic, and EMA phase records.

### Refutation Condition

The full run OOMs or fails before both optimizer updates.

### Actions

Added ZeRO-2 CPU optimizer offload with conservative buckets; moved Generator
to CPU before Score prepare, then used OptimizedLowVRAMManager in-step.

### Commands

Run launch_official_h2.sh on both nodes for
iteration36_official_h2_cold_001 and iteration36_official_h2_full_001.

### Results

Measured cold success and full-resolution GPU OOM. Cold phase span is 611.140 s,
host peaks are 710.008/710.253 GiB, and GPU peak is 35,526 MiB. At 384x640 the
selected Generator forward OOMs at transformer_helios.py:809 while creating the
FP32 timestep modulation tensor. PyTorch has 38.05 GiB allocated, about
194-262 MiB free, and cannot allocate 1.26 GiB. Monitor peak is 40,232 MiB.

### Interpretation

The official schedule is reached, but checkpointing cannot remove this
selected-forward temporary. Reduce only input shape for Level-2 validation.

### Files Changed

ZeRO-2 CPU-offload configs, iteration36 config/launcher, and opt-in adaptation.

### Failure or Limitation

No 384x640 update completed; this is not quality evidence.

### Next Highest-Value Question

What shape first crosses the selected Generator forward?

### Next Command

Repeat at 256x448 with source-shaped deterministic inputs.


## Iteration 37

### Question

Does 256x448 make the full official hand-swap step executable?

### Hypothesis

The lower spatial-token count leaves enough room for the selected temporary.

### Importance

It identifies the Level-2 activation boundary without removing components.

### Evidence Needed

Full-step phases or the exact next allocation failure.

### Refutation Condition

Any OOM or missing optimizer phase.

### Actions

Created 16 BF16 samples with latent shape [3,16,9,32,56]; only resolution
changed.

### Commands

Run launch_official_h3.sh on both nodes for
iteration37_official_h3_full_001.

### Results

Measured GPU OOM at the same source line. The allocation requests 604 MiB with
564 MiB free; PyTorch has 37.77 GiB allocated and monitor peak is 40,236 MiB.

### Interpretation

The run misses by about 40 MiB, justifying a smaller spatial contract.

### Files Changed

Iteration37 config/launcher and external 256x448 dataset.

### Failure or Limitation

No complete step; inputs are synthetic features.

### Next Highest-Value Question

Will 192x320 cross GPU admission, and what is the next bottleneck?

### Next Command

Run the same graph at 192x320.


## Iteration 38

### Question

At 192x320, does the official schedule complete or reveal another boundary?

### Hypothesis

The smaller timestep temporary fits and permits DMD and Critic execution.

### Importance

It separates activation pressure from parameter residency/offload pressure.

### Evidence Needed

All-rank phases, cgroup events, GPU peaks, and residency pattern at failure.

### Refutation Condition

A GPU OOM or any different failure prevents the step.

### Actions

Created latent shape [3,16,9,24,40] and ran all 16 ranks with the same real
models, loss, ZeRO-2, and low-VRAM schedule.

### Commands

Run launch_official_h4.sh on both nodes for
iteration38_official_h4_full_001 and summarize_iteration38.py.

### Results

Measured GPU-boundary success followed by host OOM. All ranks complete both
prepares and sparse EMA. GPU monitor reaches 38,460 MiB without CUDA OOM. During
later full-model switching, each cgroup reaches 759.996 GiB against its 760-GiB
limit. Node 0 oom_kill rises from 1 to 3 and node 1 from 1 to 5. Some ranks are
still near 38.46-GiB GPU residency while others are near 8.08 GiB, locating the
failure inside staggered model movement. Resolution is no longer the blocker.

### Interpretation

The synchronous whole-model CPU/GPU policy combines replicated ZeRO-2 state,
transient dual-CPU residency, and rising host-memory troughs. Further resolution
reduction cannot fix this host boundary.

### Files Changed

Iteration38 config/launcher/summarizer, external 192x320 dataset, raw traces.

### Failure or Limitation

SIGKILL has no Python stack. The old run lacks per-move events.

### Next Highest-Value Question

Is the 760-GiB failure mostly reclaimable allocator/cache retention, or
unavoidable live dual-model CPU residency?

### Next Command

Instrument each move, cap glibc arenas, release unused CUDA host cache and call
malloc_trim after moves, then repeat 192x320.


## Iteration 39

### Question

Can host allocator reclamation make the official 192x320 baseline complete
without changing model, loss, or order?

### Hypothesis

MALLOC_ARENA_MAX=2, CUDA host-cache release, GC, and malloc_trim after every move
will keep the cgroup below 730 GiB and allow both optimizer updates plus EMA.
Per-move traces expose the exact boundary if refuted.

### Importance

If verified this becomes the runnable strong baseline. If refuted, live
replicated residency requires sharded or disk-backed materialization.

### Evidence Needed

Sixteen-rank model_move_start/end events, component/direction/duration,
cgroup/RSS/CUDA state, AnonPages/Shmem monitor data, return codes, finite losses,
checksums, EMA phase, and end-to-end time.

### Refutation Condition

The cgroup reaches 760 GiB, a rank fails, or completion phases are absent.

### Actions

Added move tracing with logical labels, CUDA host-cache release plus glibc trim,
MALLOC_ARENA_MAX=2, extended host monitor fields, and a unit test.

### Commands

PYTHONPATH=. with the Torch 2.4 probe environment ran
pytest -q tests/test_official_baseline.py: 9 passed.

### Results

Not Run.

### Interpretation

Not Run.

### Files Changed

runtime/official_baseline.py, utils_helios_post.py, train_helios.py,
launch_official_h4.sh, and tests/test_official_baseline.py.

### Failure or Limitation

Reclamation cannot remove genuinely live dual-model CPU residency.

### Next Highest-Value Question

If insufficient, can sharded inactive residency or frozen-backbone
rematerialization remove replication without the two-engine ZeRO-3 conflict?

### Next Command

Commit and launch iteration39_official_h4_trim_full_001 on both nodes.

## Iteration 39 Result

### Question

Can ordinary host allocator reclamation alone make the 192x320 official
hand-written baseline complete?

### Hypothesis

Capping glibc arenas and releasing CUDA host/glibc caches after each whole-model
move will keep each host below 730 GiB and complete Generator, Critic, and EMA.

### Importance

This separates reclaimable cache growth from genuinely live node-wide residency.

### Evidence Needed

Per-move traces, full update phases, resource peaks, and cgroup OOM deltas.

### Refutation Condition

Either cgroup reaches 760 GiB, any rank is killed, or all-rank EMA is missing.

### Actions

Executed iteration39_official_h4_trim_full_001 at the same 192x320 contract with
MALLOC_ARENA_MAX=2, CUDA host-cache release, malloc_trim, synchronized per-move
tracing, and extended AnonPages/Shmem monitoring.

### Commands

Run launch_official_h4.sh on both nodes with port 29669 and summarize with
summarize_iteration39.py.

### Results

Measured partial progress and final refutation.

- All 16 ranks complete Generator loss, backward, and optimizer update.
- All 16 ranks complete Critic loss, backward, and optimizer update.
- Only the eight node-0 ranks record EMA; node 1 OOM-kills two ranks during the
  final Score-D2H/Generator-H2D transition.
- Both node cgroups still reach about 759.996 GiB. Node 0 has no new OOM kill;
  node 1 oom_kill increases by two.
- The adaptation therefore crosses multiple swaps that Iteration 38 could not.
  Completed whole-model D2H medians are about 30.08 s for Generator and
  26.13 s for Score; H2D medians are about 11.91 and 15.35 s, including
  near-no-op calls in the aggregate.
- At a completed Score D2H boundary, rank-local RSS is about 95.7 GiB while
  CUDA allocation is about 4.7 GiB. Before that transition it is about
  51.8 GiB with 33.7 GiB CUDA allocation. This directly measures roughly
  44 GiB of per-rank transient host growth.
- The launcher later also exposes an independent post-step validation OOM when
  rank 0 reloads the text encoder beside a resident Generator. Validation is
  outside the one-step training contract and must be disabled explicitly for
  this feasibility run.

### Interpretation

The hypothesis is refuted: cache reclamation helps but cannot remove the live
two-model CPU overlap. More importantly, the failure is node-scoped. Sixteen
rank-local managers independently make legal moves, yet eight simultaneous
D2H copies exceed one node's shared host budget. The Runtime must schedule
residency transitions with node-level concurrency, not just per-GPU memory.

### Files Changed

Host trim and move tracing implementation, Iteration 39 summarizer and raw
traces.

### Failure or Limitation

The run proves both optimizer paths but not all-rank EMA or clean exit.

### Next Highest-Value Question

Can a node-budgeted rank-wave exchange bound transient host residency while
preserving the official component order?

### Next Command

Coalesce each adjacent outgoing-D2H/incoming-H2D pair, execute at most two
local ranks per wave with a global barrier, disable post-step validation, and
repeat the same 192x320 step.


## Iteration 40

### Question

Can node-budgeted wave swapping complete the official 14B
Generator/DMD/Critic/EMA step on all 16 A100-40GB GPUs?

### Hypothesis

Two local ranks per residency wave cap transient host growth near two times
44 GiB instead of eight times 44 GiB. With allocator reclamation retained,
both node cgroups will remain below 650 GiB, all ranks will record five
logical swap completions, both optimizer updates, EMA, and clean training exit.

### Importance

This is the first planner-like mechanism tested directly in the official
entrypoint. It converts the measured shared host budget into a legal
multi-rank materialization schedule rather than merely shrinking tensors.

### Evidence Needed

Clean source commit; all-rank model_swap_start/end and model_move pairs;
per-node cgroup/RSS/Shmem peaks; swap durations; finite losses/checksums;
Generator and Critic optimizer phases; EMA on 16 ranks; clean return codes;
and explicit proof that validation and final save alone were skipped.

### Refutation Condition

Any rank OOMs or deadlocks, cgroup exceeds 650 GiB, fewer than 16 ranks record
either optimizer or EMA, checksums/losses are non-finite, or return codes are
nonzero. If it completes but exceeds 650 GiB, execution is verified but the
memory-bound hypothesis is refuted.

### Actions

- Added a two-rank local wave partition with a global barrier after each wave.
- Coalesced the default-branch Generator/Score exchanges at Generator rollout,
  DMD, Critic rollout, and EMA boundaries.
- Left optional VAE/Reward paths on the original schedule.
- Added an explicit post-step validation skip for the one-step systems gate;
  no training component or loss is skipped.
- Added option and wave-partition unit tests.

### Commands

PYTHONPATH=. under the Torch 2.4 probe environment ran
pytest -q tests/test_official_baseline.py: 11 passed.

### Results

Not Run.

### Interpretation

Not Run.

### Files Changed

runtime/official_baseline.py, helios/utils/utils_helios_post.py,
train_helios.py, launch_official_h4.sh, tests/test_official_baseline.py.

### Failure or Limitation

The wave policy deliberately trades transfer concurrency for host headroom.
It is a feasibility policy until compared with other wave sizes.

### Next Highest-Value Question

If wave size two succeeds, what is the fastest safe wave size under the
measured 760-GiB node budget?

### Next Command

Commit and launch iteration40_official_h4_wave2_full_001 on both nodes.

## Iteration 40 Result

### Question

Can node-budgeted wave swapping complete the official 14B
Generator/DMD/Critic/EMA step on all 16 A100-40GB GPUs?

### Hypothesis

Two local ranks per residency wave bound the synchronized host transient and
complete the full one-step chain. The preregistered whole-run host bound was
650 GiB.

### Actions

- Committed the wave scheduler at source commit
  `25a36235e9cb60315c7a7e8d34a11f5bfd63f7f9` with a clean launch tree.
- Launched `iteration40_official_h4_wave2_full_001` on both eight-GPU nodes,
  with real 14B Generator and Score weights, official Stage-3 DMD/Critic loss,
  deterministic source-shaped 192x320 inputs, ZeRO-2, hand-written full-model
  CPU swapping, allocator reclamation, and two local ranks per wave.
- Skipped only post-step validation and final checkpoint save; no training
  component, loss, backward, optimizer update, or EMA update was skipped.

### Commands

```bash
DISTILL_MASTER_PORT=29675 \
  scripts/distillschedule/launch_official_h4.sh <node-rank> \
  iteration40_official_h4_wave2_full_001 full
python scripts/distillschedule/summarize_iteration40.py
```

### Results

**Measured.** Both launchers returned 0 with zero new cgroup OOM kills. All 16
ranks recorded Generator loss/backward/optimizer, Critic loss/backward/optimizer,
EMA, and `training_complete_skip_final_save`. The trace contains exactly 80/80
logical swap start/end events and 160/160 physical model-move start/end events,
with no incomplete transition.

- Whole-run cgroup peak: 709.612 GiB on node 0 and 710.303 GiB on node 1.
- Steady training-window cgroup peak: 503.513/502.883 GiB, versus the failed
  unbounded run's 760-GiB limit.
- Per-process GPU peak: 38,970 MiB on both nodes.
- Median one-step training interval: 769.532 s; complete trace span including
  construction: 1,361.339 s.
- Median logical swaps: 142.197 s for Score->Generator and 146.596 s for
  Generator->Score. The five exposed swaps sum to approximately 719.783 s,
  93.54% of the measured training interval.
- Generator loss range across ranks: 0.472708--0.473448; Critic loss range:
  1.545555--1.689627. All losses are finite. Generator and Critic post-update
  checksums are identical across all ranks.

### Interpretation

The execution hypothesis is verified: node-aware materialization concurrency
turns the official-path one-step baseline from host OOM into a clean all-rank
run without changing the training chain. The global 650-GiB hypothesis is
refuted because cold Score construction alone peaks near 710 GiB; the steady
training portion does satisfy it at about 503.5 GiB. This separates two runtime
states that need different plans: construction admission and repeated
residency exchange.

Wave size two is a feasibility policy, not a throughput policy. It removes
about 256.5 GiB (33.75%) from the failed steady-state host peak but serializes
four local waves per swap, making transfers dominate the step. The next
optimization must choose the largest safe wave from measured node budget,
not blindly minimize concurrency.

### Files Changed

`scripts/distillschedule/summarize_iteration40.py`,
`artifacts/manifests/iteration_040_official_h4_wave2.json`, raw run logs and
phase traces.

### Failure or Limitation

This is a real-weight, real-loss, one-step systems result with synthetic
source-shaped features. It is not convergence or generation-quality evidence.
The post-step validation and final save paths are outside this gate and were
explicitly disabled.

### Next Highest-Value Question

Can a profile-derived larger wave preserve the host safety proof while
recovering most of the transfer time?

### Next Command

Derive the maximum safe wave from the measured 760-GiB limit, 415.7-GiB
pre-swap steady state, approximately 44.6-GiB per-rank transient, and 64-GiB
reserve; then run the resulting wave-six policy on the same two-node step.


## Iteration 41

### Question

Can the profile-derived wave-six schedule materially outperform wave two
without reintroducing host OOM?

### Hypothesis

With a 760-GiB node limit, 64-GiB reserve, 415.7-GiB measured pre-swap state,
and 44.6-GiB transient per active rank, the safe concurrency is
`floor((760 - 64 - 415.7) / 44.6) = 6`. Wave six will keep steady training
below 710 GiB, complete all 16 Generator/Critic/EMA chains, preserve the exact
post-update checksums from Iteration 40, reduce median logical-swap time by at
least 25%, and reduce end-to-end training-step time by at least 20%.

### Importance

A successful result converts rank-wave swapping from a conservative capacity
patch into a profile-based scheduling decision with a measurable
memory-throughput tradeoff.

### Evidence Needed

Clean source commit; deterministic planner output; both return codes; OOM-kill
deltas; all-rank phase and checksum equality; node-wide cold and steady peaks;
logical swap durations; training-step duration; and direct comparison with
wave two.

### Refutation Condition

Any OOM/deadlock/nonzero return, missing optimizer or EMA phase, checksum drift,
steady peak above 710 GiB, logical-swap improvement below 25%, or training-step
improvement below 20%.

### Results

Not Run.

## Iteration 41 Result

### Question

Can the profile-derived wave-six schedule materially outperform wave two
without reintroducing host OOM?

### Hypothesis

The measured node budget admits six concurrent local ranks, preserving the
complete official one-step chain while cutting logical swap time by at least
25% and training-step time by at least 20%. The preregistration also required
bitwise-identical post-update checksums relative to wave two.

### Actions

- Added a pure host-budget planner and generated
  `artifacts/manifests/iteration_041_swap_wave_plan.json` from the measured
  760-GiB limit, 415.7-GiB steady base, 44.6-GiB per-rank transient, and
  64-GiB reserve. It selected wave size six and predicted 683.3 GiB.
- Launched `iteration41_official_h4_wave6_full_001` from clean commit
  `e50166adff58c930a8404dd415ab151b7e2193ae` on both eight-A100 nodes.
- Kept model, input shape, losses, optimizer paths, EMA, sharding, offload,
  validation skip, and final-save skip identical to Iteration 40.

### Commands

```bash
python scripts/distillschedule/plan_official_swap_wave.py \
  --local-world-size 8 --memory-limit-gib 760 \
  --steady-state-gib 415.7 --transient-per-rank-gib 44.6 \
  --reserve-gib 64 \
  --output artifacts/manifests/iteration_041_swap_wave_plan.json
DISTILLSCHEDULE_SWAP_WAVE_SIZE=6 DISTILL_MASTER_PORT=29681 \
  scripts/distillschedule/launch_official_h4.sh <node-rank> \
  iteration41_official_h4_wave6_full_001 full
python scripts/distillschedule/summarize_iteration41.py
```

### Results

**Measured.** Both nodes returned 0 with zero new OOM kills. All 16 ranks
recorded Generator loss/backward/update, Critic loss/backward/update, EMA, and
clean training completion. All 80 logical swaps and 160 physical moves have
matching end events.

| Metric | Wave 2 | Wave 6 | Change |
| --- | ---: | ---: | ---: |
| Score -> Generator median swap | 142.197 s | 85.043 s | -40.19% |
| Generator -> Score median swap | 146.596 s | 92.109 s | -37.17% |
| Training-step median | 769.532 s | 468.173 s | -39.16% / 1.644x |
| Full trace span including construction | 1,361.339 s | 1,108.377 s | -18.58% |
| Max steady node cgroup | 503.513 GiB | 679.600 GiB | +176.087 GiB |
| Max whole-run node cgroup | 710.303 GiB | 711.752 GiB | +1.449 GiB |
| Per-process GPU peak | 38,970 MiB | 38,970 MiB | unchanged |

The 683.3-GiB cost-model prediction is conservative by 3.70 GiB (0.54%)
against the 679.600-GiB measured maximum. Five median swaps still sum to
approximately 439.346 s, or 93.84% of the training interval.

Generator losses are bitwise equal to Iteration 40. Across policies, Generator
post-update checksum sum differs by 5.46e-8 relative and norm by 1.05e-12;
Critic loss differs by at most 1.16e-4, while Critic checksum sum/norm differ by
1.22e-7/2.57e-13 relative. Each run is internally identical across all ranks
and all values are finite.

### Interpretation

The capacity and performance hypotheses are verified. A measured node-level
budget selects a safe concurrency that is 1.644x faster per training step than
the conservative feasibility schedule, while the cold-start peak remains
unchanged because it belongs to model construction rather than repeated
swaps. The predicted steady peak is accurate to within 1%.

The literal bitwise cross-run checksum requirement is refuted by tiny
floating-point drift, despite exact Generator loss and relative update drift
below 1.3e-7. Correctness should use a preregistered numerical tolerance, not
bitwise equality, for distributed optimizer state. This does not hide the
negative result: the exact-checksum conjunct of Iteration 41 failed.

The strongest remaining performance fact is also negative: even the planned
wave spends about 94% of step time in exposed whole-model copies. Rank-wave
selection solves admission and recovers 39%, but it does not by itself make
CPU swapping efficient. Layer/chunk prefetch or fewer materializations is the
next optimization boundary.

### Files Changed

`runtime/official_baseline.py`,
`scripts/distillschedule/plan_official_swap_wave.py`,
`scripts/distillschedule/summarize_iteration41.py`, launch metadata,
`artifacts/manifests/iteration_041_swap_wave_plan.json`,
`artifacts/manifests/iteration_041_official_h4_wave6.json`, raw logs and traces.

### Failure or Limitation

The workload is one real-weight/real-loss step with source-shaped synthetic
features. Validation, checkpoint save, convergence, and generation quality
remain outside the result. The fixed 64-GiB reserve was profiled on these two
nodes and is not yet portable across hardware.

### Next Highest-Value Question

Do the accumulated source, Level-2, Level-3, and official-path results satisfy
the project success gate, and what narrow paper claim survives the negative
results?

### Next Command

Update the hardware/profile/results/problem/paper-story documents, run the
complete unit suite, audit all stop conditions, and record missing large-scale
experiments explicitly.


## Iteration 42

### Question

What is the narrowest evidence-supported DistillSchedule contribution after
the official two-node experiments?

### Hypothesis

The defensible contribution is node-budgeted, physical-state-aware
materialization scheduling for forward-dominant multi-component distillation,
not generic activation checkpointing or merely CPU offload. The evidence
should support an OOM-to-runnable capacity result and a profile-planned
throughput recovery, while keeping convergence and quality claims out.

### Results

Not Run.

## Iteration 42 Result

### Question

What is the narrowest evidence-supported DistillSchedule contribution after
the official two-node experiments?

### Hypothesis

The defensible contribution is node-budgeted, physical-state-aware
materialization scheduling for forward-dominant multi-component distillation,
not generic activation checkpointing or merely CPU offload.

### Actions

- Updated the hardware, profile, problem, Runtime design, experiment, results,
  paper-story, open-question, and top-level research summaries through
  Iteration 41.
- Audited all twelve success-stop conditions and serialized the evidence map in
  `artifacts/manifests/iteration_042_success_audit.json`.
- Re-ran the complete unit suite under the isolated Torch 2.4 probe environment
  and preserved stdout/stderr in
  `artifacts/logs/iteration42_success_audit/pytest.txt`.
- Validated the new manifests with `python -m json.tool` and the worktree patch
  with `git diff --check`.

### Commands

```bash
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m pytest -q tests
python -m json.tool artifacts/manifests/iteration_041_official_h4_wave6.json
python -m json.tool artifacts/manifests/iteration_042_success_audit.json
git diff --check
```

### Results

**Measured:** 95 tests passed with five dependency deprecation warnings in
58.32 s. All required files/directories and all twelve success-gate evidence
categories are present. The official-path evidence supports:

> Node-budgeted physical-state materialization scheduling converts the
> measured Helios one-step A100 baseline from host OOM to complete execution;
> a profile-derived wave is 1.644x faster than the conservative feasible wave
> and predicts steady host peak within 1%.

The evidence does not support convergence, generation quality, a second real
repository, validation/save, hardware portability, or an H100-scale result.

### Interpretation

The hypothesis is verified. The final insight comes from separating logical
component legality from physical node feasibility: eight legal rank-local
swaps can be an illegal shared-node schedule. Gradient/version/nested-axis IR
is needed to preserve the Stage-3 chain, but the official measured gain is
specifically attributable to node-budgeted physical materialization
concurrency.

The project satisfies its configured success stop condition. Further work is
paper-scale extension rather than evidence needed to establish the current
narrow systems result.

### Files Changed

All current report documents, Iteration 41 summarizer/manifest, Iteration 42
success-audit manifest, and the preserved unit-test log.

### Failure or Limitation

Synchronous whole-model copies still consume approximately 93.84% of the
accepted step. This is the largest remaining optimization opportunity, not a
hidden success.

### Next Highest-Value Question

Can chunked pinned prefetch or fewer physical materializations reduce exposed
copy time while preserving the 40-GB GPU and 760-GiB node budgets?

### Next Command

Not executed because the configured success stop condition is met. Resume with
a preregistered chunked-prefetch experiment if a new research cycle is desired.

## Iteration 43

### Track / Milestone

Track A / M0 reset and Gate G0 preregistration.

### Question

What exact 384x640, one-rank A100-40GB baseline sequence should replace the obsolete
reduced-shape project stop condition, and is the software/asset workspace still valid?

### Hypothesis

The repository and pinned environment remain reproducible, but the prior project-complete
claim is invalid under the new hard contract. A bounded B0a -> split-modulation ->
Generator/Critic activation-offload -> frozen-block-streaming -> joint-static sequence
will either yield complete `A-local`/`A-scalable` baselines or a traceable negative
capacity result.

### Importance

Without a 384x640 single-rank baseline, later sharding, placement, and pipeline results
would not have the required official-like A100 control.

### Evidence Needed

Clean HEAD and branch, full test result, both-node GPU/software/cgroup identity, exact
Generator/Score assets, normative execution document, M0 manifest, replayable input
trace, canonical checkpoint schema, and phase-attributed B0/B0a measurements.

### Refutation Condition

The repository, tests, pinned environment, or exact assets are unavailable; or the
new plan cannot be represented without changing the official 384x640 semantics.

### Allowed Code Changes

Documentation/status reset, M0 manifests/contracts/trace/checkpoint schema, independent
instrumentation, and only the bounded B0a-B4 handwritten mechanisms. M1-M4 and 8G+8S
work are not authorized before Gate G0.

### Commands

```bash
git rev-parse HEAD
git branch --show-current
git status --short
HF_HUB_OFFLINE=1 PYTHONPATH=. \
  /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
/tmp/distillschedule-helios-official-torch210/bin/python - <<'PYENV'
import torch, deepspeed
print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), deepspeed.__version__)
PYENV
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

### Measured Results

- Clean branch `distillschedule/autoresearch` at
  `787e37c5b05d8c8861c521c32074bcacafd19245` on both nodes.
- 95 tests passed with five dependency deprecation warnings in 92.26 seconds under the
  isolated Torch-2.4 probe environment.
- Both nodes expose eight idle NVIDIA A100-SXM4-40GB GPUs and the same shared worktree.
- Official runtime: Python 3.10.20, Torch 2.10.0+cu128, CUDA 12.8, NCCL 2.27.5,
  DeepSpeed 0.18.4.
- Each node has a 816043786240-byte (760-GiB) memory cgroup limit.
- The exact Distilled Generator asset is present under the shared asset root; the exact
  Base Score is present in the read-only Helios weight tree.
- `docs/11_master_execution_plan.md` now supersedes the Iteration-42 success stop.

### Correctness Result

Environment/test gate passed. No new 384x640 model execution has been claimed in this
iteration.

### Performance Result

Not Run. Historical 192x320 numbers remain reduced-shape evidence only.

### Interpretation

The workspace is valid and the prior completion interpretation is refuted by the new
authoritative contract. The research cycle is active at M0.

### Files Changed

- `docs/11_master_execution_plan.md`
- `README_AUTORESEARCH.md`
- `docs/10_open_questions.md`
- `research_log.md`

### Failure or Limitation

The official Torch-2.10 environment does not include pytest; the established isolated
Torch-2.4 probe environment remains the unit-test runner. B0/B0a performance and M0
contracts are not yet executed.

### Gate Status

M0: OPEN. Gate G0: BLOCKED on M0 evidence. Project completion: explicitly revoked.

### Next Highest-Value Question

Can a complete operation-keyed 384x640 SYN-FEATURE trace and canonical logical
checkpoint contract reproduce the exact source control path before B0/B0a profiling?

### Next Command

Create and validate the M0 manifest, 384x640 contract, operation-keyed trace generator/
replay validator, and canonical logical checkpoint schema; then run B0/B0a single-rank
phase profiling.

## Iteration 44

### Track / Milestone

Track A / M0 input-and-state contract.

### Question

Can the official-shape SYN-FEATURE workload be represented by an operation-keyed trace
whose replay is independent of ambient RNG consumption, before any B0/B0a performance
claim?

### Hypothesis

Explicit sampled tensors plus per-operation control records and exact representation
hashes can eliminate seed/order ambiguity while preserving official Stage-3 source,
history, pyramid, detached-DMD and Critic semantics.

### Importance

If false, policy comparisons can silently consume different history-drop, denoising-exit,
noise or timestep samples and invalidate both correctness and timing comparisons.

### Evidence Needed

A 384x640 machine-readable contract, bounded M0 manifest, explicit source/primary/
secondary/history/noise/timestep records, exact trace hash, replay under perturbed RNG,
canonical checkpoint schema, unit tests, and a fail-closed training integration.

### Refutation Condition

Replay changes after unrelated RNG consumption; tensors lose stride/layout identity;
operation records omit steps 0/5 Generator or any Critic update; or training cannot load
the verified bundle without changing default loss/version semantics.

### Allowed Code Changes

M0-only trace/schema/instrumentation and fail-closed hooks. No sharding, component split,
pipeline, mini-planner or Track-C optimization.

### Commands

```bash
PYTHONPATH=. /tmp/distillschedule-helios-official-torch210/bin/python \
  scripts/distillschedule/prepare_m0_operation_trace.py \
  --config configs/experiments/iteration36_official_h2_zero2_cpu_swap.yaml \
  --source /share/project/shuchenweng/DistillSchedule_assets/official_h0_dataset/h0_000_81_384_640.pt \
  --output /share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v1 \
  --steps 10
PYTHONPATH=. /tmp/distillschedule-helios-official-torch210/bin/python \
  scripts/distillschedule/validate_m0_operation_trace.py \
  /share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v1
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
```

### Measured Results

- Trace hash: `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- 129 explicit tensors and 35 operation records across steps 0--9.
- Generator update trace exactly `[0,5]`; Critic update count exactly 10.
- Replay after changing Python/Torch seeds and consuming unrelated random values: PASS.
- Complete repository tests: 99 passed, five dependency warnings, 101.86 seconds.
- Cached local Flash-Attention-2 variant is usable through `LOCAL_KERNELS`; importing
  through `kernels.get_kernel` without that binding attempted unavailable network access.

### Correctness Result

Trace bundle, tensor hashes, layout metadata, branch/update frequency and ambient-RNG
independence pass. The full 14B training path has compile/import coverage and explicit
trace hooks, but has not yet completed a model call; that remains the B0/B0a experiment.

### Performance Result

Not Run. No performance conclusion is drawn from trace generation or unit tests.

### Interpretation

The seed-only input contract is refuted and replaced by an explicit operation-keyed
contract. This is experimental control, not a DistillSchedule performance contribution.

### Files Changed

- `runtime/operation_trace.py`
- `scripts/distillschedule/prepare_m0_operation_trace.py`
- `scripts/distillschedule/validate_m0_operation_trace.py`
- `tests/test_operation_trace.py`
- `configs/m0/official_384x640_contract.json`
- `configs/m0/m0_manifest.json`
- `artifacts/schemas/operation_trace.schema.json`
- `artifacts/schemas/canonical_logical_checkpoint.schema.json`
- `docs/12_m0_trace_and_checkpoint_contract.md`
- M0 B0/B0a configs, launcher, phase/OOM instrumentation and trace hooks.

### Failure or Limitation

The trace is SYN-FEATURE, not convergence or quality evidence. The canonical checkpoint
serializer/restore is only a schema at this point. Full 14B trace consumption, backward,
Adam allocation and stable second update are not yet measured.

### Gate Status

M0: OPEN. Input trace sub-gate: PASS. B0/B0a execution: NOT RUN. Gate G0: BLOCKED.

### Next Highest-Value Question

Where does unmodified B0 first fail on one A100-40GB at 384x640, and how large is the
measured allocation gap before applying B0a?

### Next Command

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b0 iteration45_m0_b0_memory_001 memory 0
```

## Iteration 45

### Track / Milestone

Track A / M0 B0 upstream A100 baseline.

### Question

Where does the unmodified upstream-style Helios Stage-3 path first fail on one
A100-SXM4-40GB at the official 384x640 shape?

### Hypothesis

Before any training operation, construction will OOM while co-materializing the real
Score model after the Generator because upstream initialization does not yet apply the
B0a sequential-BF16/time-sharing substrate.

### Importance

This establishes the measured upstream failure boundary and prevents construction
repairs from being misreported as DistillSchedule lifecycle gains.

### Evidence Needed

An immutable run/config/commit/trace hash, OOM stack, failed allocation, allocator
state, driver-free memory, host peak, and the stage at which execution stops.

### Refutation Condition

B0 completes construction and reaches a real 384x640 training operation, or fails for a
non-memory reason before the measured model placement.

### Allowed Code Changes

Instrumentation and launch fixes only. No B0a substrate, split modulation, activation
offload, block streaming, sharding, component split, or reduced shapes.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b0 iteration45_m0_b0_memory_001 memory 0
```

### Measured Results

- Source commit: `11b141b60ae3604220103f008c40a73646129dea` (clean at launch).
- Input trace: `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- Device: one NVIDIA A100-SXM4-40GB; official 384x640 SYN-FEATURE contract.
- Result: CUDA OOM during `real_score_model.to(accelerator.device)` at
  `train_helios.py:601`; no Generator/Score forward or optimizer operation ran.
- Failed allocation: 52,428,800 bytes (50 MiB).
- At failure: 41,229,934,080 allocated/active bytes; 41,250,979,840 reserved bytes;
  29,753,344 driver-free bytes; zero inactive-split bytes.
- Required release for the failed allocation plus the preregistered 1-GiB safety
  margin: 1,096,417,280 bytes (1.021 GiB).
- Peak sampled node cgroup memory: 122,825,363,456 bytes (114.39 GiB); peak process RSS:
  113,798,004 KiB (108.53 GiB); VmPin sampled as zero.
- Wall interval sampled by the launcher: 463.78 seconds, dominated by cold model loading.
- Host cgroup OOM counters did not change; this was a GPU OOM.

### Correctness Result

Not Run. B0 did not cross construction, so it provides no training-semantic correctness
evidence.

### Performance Result

Measured cold-construction failure only. It is not a steady-state timing result.

### Interpretation

The hypothesis is verified. Upstream B0 cannot establish a one-A100 baseline because it
attempts to place the Score model while nearly all device memory is already allocated.
The next experiment must apply the common B0a feasibility substrate; this 1.021-GiB
number is a local construction gap, not an estimate of later 384x640 activation demand.

### Files Changed

- `artifacts/manifests/iteration_045_m0_b0_memory.json`
- `research_log.md`

### Failure or Limitation

No category-level tensor census was attempted in this timing-independent failure run.
The allocator snapshot reports an inactive block only, not a guaranteed driver
allocation. GPU phase timing and all post-construction phases remain unmeasured.

### Gate Status

B0: bounded negative result recorded. B0a: next. M0: OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Does the shared B0a feasibility substrate complete construction and reach the first
official-shape Generator selected call, and if not, where is its first measured OOM?

### Next Command

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b0a iteration46_m0_b0a_memory_001 memory 0
```

## Iteration 46

### Track / Milestone

Track A / M0 B0a common A100 feasibility substrate.

### Question

Does B0a complete sequential BF16 construction and reach the first official-shape
Generator selected call, and where is its first remaining memory failure?

### Hypothesis

B0a will remove the construction OOM, but the first gradient-enabled Generator block
will OOM on the full six-component FP32 modulation temporary, making B1 the next
preregistered mechanism.

### Importance

This separates mandatory A100 construction repairs from the first execution-lifetime
bottleneck and tests the exact ordering required by the M0 contract.

### Evidence Needed

Clean commit/trace/config identity, successful Generator and Score prepare, exact trace
input validation, first model-swap timings, OOM source line and allocation diagnostics.

### Refutation Condition

B0a fails before sequential construction/prepare for an unrelated reason, or crosses
Generator selected forward and fails in a different phase.

### Allowed Code Changes

Only the shared B0a substrate and instrumentation. No split modulation, activation
offload, frozen-block streaming, sharding, component placement, or reduced shapes.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b0a iteration46_m0_b0a_memory_001 memory 0
```

### Measured Results

- Clean source commit and trace hash match Iteration 45.
- Sequential BF16 Generator/Score prepare completed. Generator local parameter storage
  was 31,127,740,544 bytes and Score storage 31,117,484,160 bytes after prepare.
- Operation-keyed source tensors were hash-validated before training: PASS.
- The hand-written manager exposed 18.033 s for Score D2H followed by 11.974 s for
  Generator H2D in the cold first transition.
- The first gradient-enabled Generator call reached block execution and OOMed at
  `helios/modules/transformer_helios.py:809`, while materializing the full
  `(scale_shift_table.unsqueeze(0) + temb.float())` six-component tensor.
- Failed allocation: 562,036,736 bytes (536 MiB). At failure: 39,518,521,344 active/
  allocated bytes; 40,221,278,208 reserved bytes; 556,138,496 driver-free bytes.
- Peak allocator observations before failure: 40,592,492,544 allocated and
  40,766,537,728 reserved bytes. Required release for failed allocation plus 1-GiB
  safety: 1,079,640,064 bytes (1.006 GiB).
- Process peak RSS was 145,648,168,960 bytes (135.64 GiB). Largest phase-sampled current
  cgroup usage was 119,364,620,288 bytes; host OOM counters did not change.

### Correctness Result

Trace input identity passed. No complete Generator output/backward/update was produced,
so training correctness remains Not Run.

### Performance Result

Measured cold model moves only. No steady-state cycle timing is claimed.

### Interpretation

The hypothesis is verified. B0a is a necessary feasibility substrate and successfully
moves the failure boundary from construction into the first official-shape selected
Generator call. The next bounded mechanism is B1 split-modulation, not activation
offload or sharding.

### Files Changed

- `artifacts/manifests/iteration_046_m0_b0a_memory.json`
- `research_log.md`

### Failure or Limitation

The launcher's high-frequency resource sampler inherited `set -e` and exited when a
`/proc` status file disappeared. Phase/OOM traces are intact; the launcher is fixed
before B1 so later runs retain continuous sampling. This run did not complete any model
forward, backward, optimizer step, Critic update, or EMA update.

### Gate Status

B0a: bounded negative result with construction gate passed. B1: authorized next. M0:
OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Does per-component FP32 modulation materialization preserve block/input/LoRA gradients
and free enough memory for the first 384x640 selected Generator call?

### Next Command

```bash
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m pytest -q tests/test_modulation_lifetime.py
scripts/distillschedule/launch_m0_single_a100.sh \
  b1 iteration47_m0_b1_memory_001 memory 0
```

## Iteration 47

### Track / Milestone

Track A / M0 B1 split-modulation, implementation iteration 1 of 2.

### Question

Is splitting only each block's FP32 shift/scale/gate materialization sufficient to pass
the first 384x640 selected Generator call?

### Hypothesis

The six-way FP32 block temporary is the sole immediate allocation blocker, so phase-local
component materialization will let the first block execute.

### Importance

This tests the narrowest existing lifetime primitive before activation offload, exactly
following the bounded M0 order.

### Evidence Needed

Block output/input-gradient/LoRA-gradient tests, both physical-model flags in the phase
trace, and a full official-shape run with exact OOM attribution if it still fails.

### Refutation Condition

The run OOMs before block execution on another six-component eager temporary, or split
modulation changes tested outputs/gradients.

### Allowed Code Changes

B1 split-modulation only. No activation hooks, block streaming, sharding, placement or
shape changes.

### Commands

```bash
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m pytest -q tests/test_official_baseline.py \
  tests/test_modulation_lifetime.py tests/test_operation_trace.py
scripts/distillschedule/launch_m0_single_a100.sh \
  b1 iteration47_m0_b1_memory_001 memory 0
```

### Measured Results

- Unit/integration-contract tests: 20 passed. Exact toy-block outputs plus input,
  modulation and nonzero LoRA gradients matched for Generator-selected and Critic-train
  call labels.
- Both real physical backbones recorded 40 split-modulation blocks.
- B1 v1 moved the failure before `HeliosTransformerBlock.forward`, but did not complete a
  model call. It OOMed at `transformer_helios.py:1480` while concatenating history and
  main `timestep_proj` into an eager six-component BF16 tensor.
- Failed allocation: 679,477,248 bytes (648 MiB); allocated/active 40,177,590,784 bytes;
  reserved 40,284,192,768 bytes; driver-free 493,223,936 bytes.
- Required release for the allocation plus 1-GiB safety: 1,259,995,136 bytes.
- Continuous sampler: 1,464 samples over 607.16 s; peak cgroup 151,589,748,736 bytes;
  peak summed Python RSS 142,418,720 KiB; no host OOM or sampled VmPin.

### Correctness Result

Primitive-level tests PASS. Real training correctness remains Not Run because no full
Generator call completed.

### Performance Result

Not Run. The cold Score D2H and Generator H2D moves were 18.045 s and 18.977 s; they are
reported only as cold diagnostic values.

### Interpretation

The hypothesis is refuted. Splitting the block's FP32 conversion is necessary but not
sufficient: the producer eagerly materializes all six BF16 components first. B1 has one
preregistered implementation iteration remaining; it must carry a lazy segmented
timestep projection into the block and materialize one component at actual use.

### Files Changed

- `artifacts/manifests/iteration_047_m0_b1_split_fp32_memory.json`
- `research_log.md`

### Failure or Limitation

No block, backward, Adam, Critic or EMA boundary completed. This is not authorization to
skip to B2; the second and final B1 implementation is still within the bounded sequence.

### Gate Status

B1 implementation 1/2: negative. B1 implementation 2/2: authorized. M0: OPEN. Gate G0:
BLOCKED.

### Next Highest-Value Question

Can a lazy segmented timestep-projection carrier remove the eager 648-MiB six-component
concatenation while preserving exact component/block/gradient semantics?

### Next Command

Implement and test the final B1 lazy projection representation, then run
`iteration48_m0_b1_lazy_memory_001` on one A100 at 384x640.

## Iteration 48

### Track / Milestone

Track A / M0 B1 split-modulation, implementation iteration 2 of 2 (final).

### Question

Can a lazy segmented timestep-projection carrier eliminate the eager six-component
concat and complete the first 384x640 Generator forward?

### Hypothesis

After removing both the eager BF16 concat and full FP32 six-component conversion, one
component plus the block's ordinary computation will fit in the remaining device space.

### Importance

This is the final preregistered B1 attempt and determines whether the next legal action
is B2 activation offload.

### Evidence Needed

Exact component/block/gradient and checkpoint-recompute tests, real flags on both
backbones, and the next full-shape failure boundary.

### Refutation Condition

The first model call still OOMs on an ordinary one-component/block temporary, or the
lazy carrier changes output/gradient semantics.

### Allowed Code Changes

The second and final B1 implementation only: lazy repeat-only timestep segments and
one-component materialization. No activation offload, streaming or cross-GPU mechanism.

### Commands

```bash
PYTHONPATH=. /tmp/distillschedule-helios-probe-torch24/bin/python \
  -m pytest -q tests/test_official_baseline.py \
  tests/test_modulation_lifetime.py tests/test_operation_trace.py
scripts/distillschedule/launch_m0_single_a100.sh \
  b1 iteration48_m0_b1_lazy_memory_001 memory 0
```

### Measured Results

- 21 tests passed, including exact segmented-component values, checkpointed block output,
  input gradient and segment-base gradients.
- The eager 648-MiB six-component concat was removed: execution entered block 0.
- The failure shifted to the first no-grad rollout call (`utils_helios_post.py:1451`),
  at `transformer_helios.py:863` while allocating the ordinary full-sequence FP32
  normalization/modulation intermediate.
- Failed allocation: 226,492,416 bytes (216 MiB); allocated/active 39,943,135,744 bytes;
  reserved 40,556,822,528 bytes; driver-free 220,594,176 bytes.
- Required release for the allocation plus 1-GiB safety: 1,079,640,064 bytes.
- Continuous sampler: 1,395 samples over 589.85 s; peak cgroup 151,574,339,584 bytes;
  peak summed Python RSS 142,710,492 KiB; no host OOM or sampled VmPin.

### Correctness Result

B1 primitive and checkpoint-recompute tests PASS. Real end-to-end correctness is Not Run
because even the first no-grad model forward did not complete.

### Performance Result

Not Run. Cold Score D2H was 18.115 s and Generator H2D 7.803 s; these values are
non-steady diagnostics, not a speed comparison.

### Interpretation

The B1 hypothesis is refuted and the bounded two-implementation budget is exhausted.
B1 successfully removes the identified six-component temporaries but cannot overcome
the much tighter fact that a resident 31.1-GB backbone plus LoRA/DeepSpeed state leaves
insufficient space for a 216-MiB ordinary FP32 block temporary during a no-grad prefix.
B2 must still be implemented and measured in order, but selected/Critic saved-tensor
offload cannot affect this pre-selected no-grad failure boundary; this prediction is
explicitly falsifiable.

### Files Changed

- `artifacts/manifests/iteration_048_m0_b1_lazy_memory.json`
- `research_log.md`

### Failure or Limitation

No full no-grad forward, selected forward, backward, Adam, Critic or EMA boundary
completed. B1 is a bounded negative result, not a project stop.

### Gate Status

B1: bounded negative complete. B2: authorized next. M0: OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Does audited full activation offload ever activate before this no-grad OOM, and can it
change the measured boundary without violating the selected-call-only contract?

### Next Command

Implement/census B2 saved-tensor hooks, then run one immutable B2 384x640 memory trial.

## Iteration 49

### Track / Milestone

Track A / M0 B2 activation-offload primitive census.

### Question

Can the B2 hook enforce the saved-tensor safety contract and replay backward correctly
before it is enabled in the real 14B run?

### Hypothesis

A top-level grad-enabled-only hook can offload ORIGINAL_FORWARD activations, retain
parameter/buffer and zero-stride storages, deduplicate repeated saves, avoid recursive
checkpoint-recompute hooks, and reproduce gradients.

### Importance

Un-audited `save_on_cpu` can silently copy parameters, expanded views or duplicate
storages and can extend lifetimes; B2 must fail closed before the expensive real run.

### Evidence Needed

Ordered CUDA pack/unpack events, parameter/buffer exclusion, zero-stride exclusion,
duplicate-save deduplication, no live CPU handle after backward, and exact output/input/
parameter gradients against an unhooked reference.

### Refutation Condition

Any excluded storage is copied, duplicate saves create duplicate CPU copies, event order
is inconsistent, a handle leaks, recomputation opens a new top-level scope, or gradients
differ.

### Allowed Code Changes

B2 coarse saved-tensor hook and instrumentation only. No prefetch, selective policy,
block streaming or placement changes.

### Commands

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /tmp/distillschedule-helios-official-torch210/bin/python \
  scripts/distillschedule/probe_activation_offload_census.py \
  --output artifacts/profile/iteration49_b2_activation_census_committed
```

### Measured Results

- Source commit: `029ae936b4625e084832516e2d5aa9f24ac390a8`.
- MINI CUDA census correctness: PASS; event order 1--18 was strict.
- One ORIGINAL_FORWARD scope produced 8 packs and 8 unpacks.
- Three unique CPU copies served the eligible tensors; two duplicate packs reused an
  existing CPU handle.
- One parameter/buffer storage and two zero-stride views were explicitly kept on GPU.
- Peak live pinned saved-tensor storage was 4,096 bytes and returned to zero after
  backward.
- 23 relevant repository tests passed before this census.

### Correctness Result

PASS for the MINI primitive: output, input gradient and parameter gradient matched the
unhooked reference exactly.

### Performance Result

Not Run. This census is explicitly not a Helios performance result.

### Interpretation

The hypothesis is verified at primitive level. B2 is safe to enable for the one allowed
real 384x640 trial. Because B1 v2 fails in a no-grad prefix, the preregistered prediction
is that B2 will install but open zero activation scopes before the same OOM.

### Files Changed

- `runtime/activation_offload.py`
- `scripts/distillschedule/probe_activation_offload_census.py`
- `tests/test_activation_offload.py`
- `artifacts/manifests/iteration_049_m0_b2_activation_census.json`
- `research_log.md`

### Failure or Limitation

The workload is MINI. It does not establish real Helios pack sizes, pinned demand or
performance and is not included in the paper performance table.

### Gate Status

B2 primitive gate: PASS. B2 real run: authorized. M0: OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Will B2 open any saved-tensor scope before the known no-grad block OOM, or is the real
failure boundary provably unchanged?

### Next Command

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b2 iteration50_m0_b2_memory_001 memory 0
```

## Iteration 50

### Track / Milestone

Track A / M0 B2 real 384x640 standard activation offload.

### Question

Can the one preregistered standard ORIGINAL_FORWARD saved-tensor offload complete
Generator backward on one A100-40GB without changing the official checkpoint policy?

### Hypothesis

B2 will either expose enough selected-forward activation memory to cross backward, or
it will localize the next OOM to checkpoint recomputation, where B2 intentionally does
not recursively install another saved-tensor hook.

### Importance

This distinguishes saved activation pressure from immutable resident parameter pressure
and determines whether B3 frozen-block streaming is authorized.

### Evidence Needed

An immutable 384x640 run with the operation trace hash, ordered pack/unpack census,
phase trace through Generator loss/backward, exact CUDA OOM state, and host trace.

### Refutation Condition

The hook does not activate, copies excluded parameter/buffer storage, changes the input
trace, or the full Generator backward and optimizer step complete.

### Allowed Code Changes

None during the run. It used the already committed B2 standard coarse hook: selected
Generator/Critic ORIGINAL_FORWARD only, synchronous unpack, no prefetch and no
checkpoint-recompute offload.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b2 iteration50_m0_b2_memory_001 memory 0
```

### Measured Results

- Source commit `ce2882a568ff7c88d3aebdd1c4ba9fe02bb4bc7f`; trace hash
  `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- All three selected Generator stage calls completed and produced loss
  `0.45730964033703136`; the run entered Generator backward.
- Three ORIGINAL_FORWARD scopes packed 651 logical saves. 346 unique CPU copies
  represented 8,385,466,464 bytes; 242 duplicate saves reused existing handles and 63
  parameter/buffer saves stayed on GPU.
- Backward synchronously unpacked 14 handles, then the first checkpoint recomputation
  failed at `transformer_helios.py:964` on a 226,492,416-byte FP32 norm temporary.
- OOM state: allocated/active 39,993,078,272 bytes; reserved 40,428,896,256 bytes;
  driver free 224,722,944 bytes; failed allocation plus 1-GiB safety requires
  1,075,511,296 bytes of release.
- 1,565 resource samples over 656.89 s; peak cgroup 151,606,054,912 bytes; peak summed
  Python RSS 142,718,120 KiB; no host OOM and no sampled VmPin.
- Critic activation scope count was zero because Generator backward did not complete.

### Correctness Result

The committed B2 MINI CUDA primitive remains PASS. Official input trace validation and
Generator loss construction PASS. Real end-to-end update correctness is Not Run because
Generator backward OOMed before any optimizer step.

### Performance Result

Not Run. This was a memory instrumentation run and did not complete a training step.
Cold/model-swap times are retained in the manifest but are not steady-state claims.

### Interpretation

B2 materially moved the boundary: unlike B1, all selected stage forwards and loss
construction completed after 8.39 GB of saved tensors were offloaded. The remaining OOM
is inside non-reentrant checkpoint recomputation, where PyTorch's internal
`_recomputation_hook` owns saved-tensor bookkeeping; recursively replacing it with a
second generic `save_on_cpu` hook would be semantically unsafe rather than a valid B2
extension. The one preregistered standard B2 implementation is therefore a bounded
negative for completing backward. Because B1+B2 still cannot finish F/B/optimizer, B3
immutable frozen-block streaming is now authorized.

### Files Changed

- `artifacts/manifests/iteration_050_m0_b2_memory.json`
- `research_log.md`

### Failure or Limitation

No Generator optimizer, first Adam state initialization, Critic train call, EMA boundary,
or steady step completed. The result is not a baseline throughput number.

### Gate Status

B2 standard: bounded negative complete. B3 windows 1/2/4: authorized next. M0: OPEN.
Gate G0: BLOCKED.

### Next Highest-Value Question

Can streaming only immutable frozen block parameters lower residency enough for window 1
to complete checkpoint recomputation without moving LoRA or optimizer-owned state?

### Next Command

Implement and unit-test the B3 immutable frozen-block streamer and whole-model-manager
interlock, then launch the immutable window-1 384x640 memory run.

## Iteration 51

### Track / Milestone

Track A / M0 B3 window-1, initial non-reentrant checkpoint interlock.

### Question

Can a one-block immutable frozen-parameter window remain bounded during both original
forward and PyTorch non-reentrant checkpoint recomputation?

### Hypothesis

Materializing each 702,788,608-byte frozen block before use and releasing it after the
block's boundary-input gradients should keep streamed residency at one block through
forward and backward.

### Importance

B2 proved that the remaining OOM is in checkpoint recomputation. B3 is valid only if
frozen weights do not accumulate across the backward critical path.

### Evidence Needed

Per-block materialize/release order and resident bytes for Generator and Score, exact
backward failure state, and confirmation that only frozen, non-optimizer-owned block
parameters are captured.

### Refutation Condition

More than one block accumulates with window 1, trainable state is streamed, or full
Generator backward still OOMs.

### Allowed Code Changes

The first B3 implementation: immutable frozen block records, HOST_PRIVATE storage,
window 1, whole-model-manager interlock, and checkpoint-generation hooks. No selective
block subset, async prefetch, placement, sharding, or cross-GPU mechanism.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b3w1 iteration51_m0_b3w1_memory_001 memory 0
```

### Measured Results

- Source commit `0668e0bb99b7e304ca292e43c5281e163de8d492`; official trace hash
  `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- Each backbone exposed 1,080 eligible immutable parameters totaling 28,111,544,320
  bytes: exactly 40 blocks of 702,788,608 bytes. Trainable LoRA and optimizer-owned
  parameters were rejected by construction and remained in the whole-model state.
- Eager/no-grad and ORIGINAL_FORWARD streaming behaved as intended. Score executed 120
  block materializations and 120 releases with a 702,788,608-byte peak streamed
  residency. Generator completed all forward/loss construction with the identical B2
  loss `0.45730964033703136`.
- During non-reentrant checkpoint backward, recomputations for blocks 39 down through 11
  happened before their boundary-gradient hooks executed. 29 blocks accumulated to
  20,380,869,632 streamed resident bytes despite configured window 1.
- OOM at block 11 modulation materialization: failed 226,492,416 bytes; active/allocated
  39,809,081,856; reserved 40,783,314,944; driver free 40,239,104. Required release for
  allocation plus 1-GiB safety was 1,259,995,136 bytes.
- 1,513 host samples over 628.62 s; peak cgroup 151,900,352,512 bytes and peak summed
  Python RSS 142,826,440 KiB; no host OOM.

### Correctness Result

MINI output/input/LoRA-gradient tests for all host policies and windows remained PASS.
Official input identity and forward loss PASS. Real update correctness is Not Run because
backward did not finish.

### Performance Result

Not Run. This is a memory characterization failure, not a completed-step timing result.

### Interpretation

The hypothesis is refuted. The problem is not failure to offload a block after ordinary
forward; it is the execution semantics of non-reentrant checkpointing. PyTorch can
recompute a chain of checkpoint regions while resolving saved holders before the local
boundary gradients run, so a gradient hook is not a valid per-block release point.
The trace is direct evidence for a nested-axis/checkpoint interaction: a nominal
window-1 parameter plan became a 29-block runtime peak. The bounded corrective action is
a fixed reentrant per-block recompute region that reconstructs all tensor-bearing
arguments and inserts a dummy differentiable boundary when necessary, so each block's
recompute and local backward complete before the next block materializes.

### Files Changed

- `artifacts/manifests/iteration_051_m0_b3w1_nonreentrant.json`
- `research_log.md`

### Failure or Limitation

This run does not count as the final window-1 result because it exposed and localizes a
checkpoint/streaming correctness defect in the first B3 implementation. No optimizer,
Critic, or EMA boundary completed.

### Gate Status

B3 implementation correction required. M0: OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Does a tensor-flattened reentrant per-block checkpoint make recompute residency obey the
configured one-block window while preserving output and LoRA gradients?

### Next Command

Implement the bounded reentrant RecomputeRegion interlock, extend MINI tests to
non-differentiable and multi-tensor boundaries, then rerun immutable B3 window 1.

## Iteration 52

### Track / Milestone

Track A / M0 B3 window-1 corrected reentrant checkpoint region.

### Question

Does tensor-flattened reentrant per-block recomputation keep immutable streamed
residency at the configured one-block window while crossing backward, first Adam
initialization, the second Generator update, Critic updates, and EMA?

### Hypothesis

Reentrant local recompute/backward will consume each block before the next block is
materialized, eliminating the 29-block accumulation observed in Iteration 51.

### Importance

This is the feasibility boundary for an unsharded 384x640 A100 flow. If it fails, the
bounded B3 windows cannot make Track A executable.

### Evidence Needed

A clean immutable 10-step run; one-block peak streamed residency; two Generator
optimizer steps; ten Critic optimizer steps; ten EMA steps; finite official losses and
gradient norms; no GPU/host OOM.

### Refutation Condition

Streamed residency exceeds one block, any backward/optimizer boundary OOMs, a streamed
parameter becomes trainable/optimizer-owned, or the input trace changes.

### Allowed Code Changes

The bounded B3 checkpoint correction only: tensor-bearing argument flatten/rebuild,
reentrant per-block checkpoint, dummy differentiable scalar for Critic regions without
a differentiable data boundary, and all-boundary-gradient release hooks. No partial
block selection, async transfer, sharding, or placement.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b3w1 iteration52_m0_b3w1_reentrant_memory_001 memory 0
```

### Measured Results

- Source commit `5026548cfcd2e836e8cee900d43fde5a7a21b96b`; trace hash
  `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- The full official-shape 10-step run returned 0. It recorded two Generator losses and
  optimizer steps at logical steps 0 and 5, ten Critic losses/backwards/optimizer steps,
  and EMA optimization steps 1 through 10.
- Generator loss was `0.45730964033703136` at step 0 and
  `0.45189464652397326` at step 5. All reported losses and gradient norms were finite.
- Each physical backbone captured 1,080 immutable parameters / 28,111,544,320 bytes.
  Generator materialized 2,560 block uses (1,799,138,836,480 bytes total) and Score
  materialized 1,040 (730,900,152,320 bytes total).
- Peak streamed residency was exactly 702,788,608 bytes for both backbones: one block,
  versus 20,380,869,632 bytes in the rejected non-reentrant implementation.
- CUDA maxima were 33,727,460,864 allocated and 38,663,094,272 reserved bytes; no CUDA
  OOM occurred.
- The sampled cgroup peak reached 246,707,978,240 bytes and process high-water RSS
  reached 249,993,834,496 bytes. Host retention rose across the run even though live
  activation handles returned to zero; this run therefore does not yet satisfy the
  long-term A-local/A-scalable host-stability condition.

### Correctness Result

PASS for trace identity, phase/version order, finite loss/gradient execution, bounded
streamer state, and the full 10-step control path. Exact per-parameter update/moment
comparison and the explicit Critic-only `Generator.grad is None` assertion remain Not
Yet Verified and are required before A-local freeze.

### Performance Result

Not Run as a formal timing claim. The memory run took 669.6 s for the progress-bar
10-step region, but it emitted thousands of synchronous per-block JSON events and is
only a characterization measurement.

### Interpretation

The hypothesis is verified. The critical B3 requirement is not merely offloading frozen
weights; the parameter window must be coupled to checkpoint execution semantics.
Reentrant local recomputation restores the intended window bound and turns the prior
384x640 backward OOM into a complete 10-step run. However, full 40-block streaming
moves terabytes per run and retains too much host allocator/pinned state, so it is a
feasibility point rather than the frozen A-local baseline.

### Files Changed

- `runtime/frozen_block_streaming.py`
- `tests/test_frozen_block_streaming.py`
- `artifacts/manifests/iteration_052_m0_b3w1_reentrant_memory.json`
- `research_log.md`

### Failure or Limitation

This is a memory-instrumented SYN-FEATURE run, not formal timing, convergence, quality,
or raw-data throughput evidence. Host high-water memory is not stable across ten steps.

### Gate Status

B3 window 1: positive feasibility result. Windows 2/4 and B4 remain required. M0: OPEN.
Gate G0: BLOCKED.

### Next Highest-Value Question

Do the preregistered windows 2 and 4 also obey their bounds, and does either offer a
memory-run indication that invalidates window 1 as the static B4 starting point?

### Next Command

Run immutable one-step 384x640 B3 window-2 and window-4 memory trials.

## Iteration 53

### Track / Milestone

Track A / M0 B3 window-2 bounded variant.

### Question

Does the corrected reentrant mechanism obey a two-block residency window and complete
one official-shape step?

### Hypothesis

Peak streamed residency will be no greater than 2 x 702,788,608 bytes, with the same
trace and numerically consistent losses as window 1.

### Importance

All preregistered B3 windows must be measured before selecting the B4 static window.

### Evidence Needed

One complete step, bounded residency, phase completion, trace identity, and finite loss.

### Refutation Condition

Residency exceeds two blocks, the step fails, or losses exceed the correctness tolerance.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=1 scripts/distillschedule/launch_m0_single_a100.sh \
  b3w2 iteration53_m0_b3w2_reentrant_memory_001 memory 0
```

### Measured Results

- Source commit and trace hash matched Iteration 52.
- Return code 0; one Generator update, one Critic update, and EMA optimization step 1
  completed.
- Peak streamed residency was exactly 1,405,577,216 bytes for both backbones.
- Generator loss was exactly `0.45730964033703136`; Critic loss was
  `1.5418128967285156`, differing from window 1 by `2.0981e-5`.
- CUDA max allocated/reserved were again 33,727,460,864 / 38,663,094,272 bytes.
- Instrumented progress-bar step time was 139.05 s; this is not a formal timing result.

### Correctness Result

PASS for bounded residency, trace identity, complete phase order, and configured
loss tolerance. Exact update/moment comparison remains pending.

### Performance Result

Not Run formally; memory instrumentation only.

### Interpretation

The hypothesis is verified. Window 2 has no observed capacity advantage over window 1
and doubles the peak streamed residency. It remains a measured Pareto candidate but is
not selected as the B4 default.

### Files Changed

- `artifacts/manifests/iteration_053_m0_b3w2_reentrant_memory.json`
- `research_log.md`

### Failure or Limitation

Only one step was run; it does not test long-term host behavior.

### Gate Status

B3 window 2: complete. Window 4 and B4 remain. M0: OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Does window 4 remain bounded and offer enough synchronous-copy efficiency to justify its
larger GPU residency?

### Next Command

Run the immutable B3 window-4 one-step memory trial.

## Iteration 54

### Track / Milestone

Track A / M0 B3 window-4 bounded variant.

### Question

Does the corrected reentrant mechanism obey a four-block residency window and complete
one official-shape step?

### Hypothesis

Peak streamed residency will be no greater than 4 x 702,788,608 bytes, without changing
the operation trace or training semantics.

### Importance

This closes the preregistered B3 window set and enables a bounded B4 selection.

### Evidence Needed

One complete step, four-block residency bound, phase completion, and finite losses.

### Refutation Condition

Residency exceeds four blocks, execution fails, or loss tolerance is violated.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=1 scripts/distillschedule/launch_m0_single_a100.sh \
  b3w4 iteration54_m0_b3w4_reentrant_memory_001 memory 0
```

### Measured Results

- Source commit and trace hash matched Iterations 52--53.
- Return code 0; one Generator update, one Critic update, and EMA optimization step 1
  completed.
- Peak streamed residency was exactly 2,811,154,432 bytes for both backbones.
- Generator loss was exactly `0.45730964033703136`; Critic loss was
  `1.5418081283569336`, within `1.7e-5` of window 1.
- CUDA max allocated/reserved were 33,727,460,864 / 38,663,094,272 bytes.
- Instrumented progress-bar step time was 119.29 s. Per-block event and memory
  instrumentation make it non-formal.

### Correctness Result

PASS for the bounded window, trace, phase order, and configured loss tolerance. Exact
update/moment comparison remains pending.

### Performance Result

Not Run formally; memory instrumentation only.

### Interpretation

The hypothesis is verified. All windows 1/2/4 are feasible. Window 1 minimizes the
resident safety tax and had the lowest observed one-step memory-run time, although these
runs cannot establish a formal timing ordering. B4 therefore starts with window 1 and
reduces transfer volume by streaming only a profile-derived frozen tail subset.

### Files Changed

- `artifacts/manifests/iteration_054_m0_b3w4_reentrant_memory.json`
- `research_log.md`

### Failure or Limitation

Only one step was run, and memory instrumentation perturbs timing.

### Gate Status

B3 windows 1/2/4: bounded sequence complete. B4 authorized. M0: OPEN. Gate G0: BLOCKED.

### Next Highest-Value Question

Can streaming only four tail blocks plus an 8-MiB activation threshold preserve at
least the measured B2 safety gap while eliminating most full-streaming transfers?

### Next Command

Run the preregistered B4 partial-4 and partial-8 one-step 384x640 memory trials.

## Iteration 55

### Track / Milestone

Track A / M0 B4 partial frozen-tail streaming.

### Question

Can streaming only four frozen tail blocks plus the 8-MiB coarse activation policy close the measured B2 backward gap?

### Hypothesis

Four streamed blocks will release at least the failed allocation plus the required 1-GiB safety margin at the checkpoint-recompute peak.

### Importance

If true, B4 could avoid the transfer volume of full-backbone streaming.

### Evidence Needed

A complete official-shape Generator update, Critic update, EMA step, and an OOM-free backward.

### Refutation Condition

Any OOM before the complete optimizer/EMA sequence, or less than 1 GiB safety at the critical peak.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=1 scripts/distillschedule/launch_m0_single_a100.sh \
  b4p4 iteration55_m0_b4p4_memory_001 memory 0
```

### Measured Results

- Source `7df841f4316ea3beeb9e5d68ce75a4f6ef03439c`; trace hash `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- Generator loss forward completed (`0.45730964033703136`), then checkpoint recompute backward OOMed.
- Failed allocation: 226,492,416 bytes. Active/allocated at failure: 39,662,512,128 bytes; driver free: 191,234,048 bytes.
- Release required for the failed allocation plus 1-GiB margin: 1,109,000,192 bytes.
- Return code 1; no optimizer or EMA step completed.

### Correctness Result

FAIL: the complete update sequence did not execute.

### Performance Result

Not Run; failed memory-instrumentation trial.

### Interpretation

The hypothesis is refuted. The static B2 gap estimate did not predict the reentrant checkpoint-recompute peak. A four-block tail is insufficient.

### Files Changed

- `artifacts/manifests/iteration_055_m0_b4p4_memory.json`
- `research_log.md`

### Failure or Limitation

This is a precisely localized capacity failure, not a formal timing result.

### Gate Status

B4 remains OPEN; partial-8 is the next bounded candidate. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does partial-8 cross the recompute capacity boundary without full streaming?

### Next Command

Run B4 partial-8 plus 32-MiB activation offload for one official-shape step.

## Iteration 56

### Track / Milestone

Track A / M0 B4 partial-8 candidate.

### Question

Does partial-8 complete one update, and is its host activation state released after backward?

### Hypothesis

Eight streamed tail blocks will be GPU-feasible and all packed activation handles will become unreachable after the step.

### Importance

Both GPU feasibility and bounded long-lived host state are required before considering a scalable baseline.

### Evidence Needed

Return code 0, complete G/C/EMA sequence, finite losses, and zero final live activation bytes.

### Refutation Condition

OOM, incomplete phase order, or retained CPU activation storage after autograd completion.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=1 scripts/distillschedule/launch_m0_single_a100.sh \
  b4p8 iteration56_m0_b4p8_memory_001 memory 0
```

### Measured Results

- Return code 0; one Generator optimizer step, one Critic optimizer step, and EMA optimization step 1 completed.
- CUDA max allocated/reserved: 39,717,764,608 / 40,370,176,000 bytes.
- Generator/Critic losses: `0.45730964033703136` / `1.5418529510498047`.
- The final activation registry still retained 8,965,087,232 bytes even though backward had finished.
- Resource-sampled cgroup peak was 161,240,657,920 bytes. This includes cold construction and cannot be multiplied as steady rank-private state.

### Correctness Result

PASS for one-step training semantics; FAIL for the intended activation-handle lifetime invariant.

### Performance Result

Not Run formally; memory instrumentation only.

### Interpretation

GPU feasibility is verified, but the host retention was an instrumentation/runtime registry bug: alias-dedup entries held packed CPU tensors that autograd never unpacked. The run cannot establish steady host memory.

### Files Changed

- `artifacts/manifests/iteration_056_m0_b4p8_memory.json`
- `research_log.md`

### Failure or Limitation

The observed host retention was caused by our offload bookkeeping and not by Helios graph semantics.

### Gate Status

B4 OPEN pending a bounded alias-lifetime fix and rerun. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can alias dedup be scope-bounded without losing correctness or duplicating live CPU copies?

### Next Command

Clear alias lookup state at save-scope exit, retain packed handles only by true autograd ownership, add an unused-saved-tensor test, and rerun.

## Iteration 57

### Track / Milestone

Track A / M0 B4 full streaming plus coarse activation offload after alias-lifetime repair.

### Question

Does the repaired activation offloader release unused packed tensors while full frozen-block streaming completes the official-shape step?

### Hypothesis

Scope-bounded alias lookup plus weakref lifetime tracking will yield zero final live CPU activation bytes and preserve the complete update sequence.

### Importance

A false host leak would invalidate A-local/A-scalable admission accounting.

### Evidence Needed

Unit coverage for unused saved tensors, return code 0, G/C/EMA completion, and zero final live activation bytes.

### Refutation Condition

Duplicate live copies, use-after-free, incomplete backward, or nonzero final live activation bytes.

### Allowed Code Changes

Only the alias-registry lifetime repair and its focused tests.

### Commands

```bash
PYTHONPATH=. python -m pytest -q tests/test_activation_offload.py
DISTILL_M0_MAX_STEPS=1 scripts/distillschedule/launch_m0_single_a100.sh \
  b4full iteration57_m0_b4full_memory_001 memory 0
```

### Measured Results

- Focused lifetime tests passed.
- Return code 0; complete Generator/Critic/EMA sequence.
- Final live CPU activation bytes were zero for Generator and Critic.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes.
- Phase-sampled cgroup peak: 123,268,313,088 bytes; process high-water during construction: 145,640,157,184 bytes.
- Generator activation offload copied 16,581,918,720 logical bytes over six scopes, peak-live 8,290,959,360 bytes.
- Generator/Score frozen streaming moved 281,115,443,200 / 140,557,721,600 bytes in this one-step memory run.

### Correctness Result

PASS for unit invariants, phase order, finite losses, and released activation handles. Exact update/moment comparison remains pending.

### Performance Result

Not Run formally; full tracing and memory instrumentation were enabled.

### Interpretation

The hypothesis is verified. The prior retained 8.97 GB was bookkeeping-induced. Full streaming is a valid B4 capacity candidate, though transfer-heavy.

### Files Changed

- `runtime/activation_offload.py`
- `tests/test_activation_offload.py`
- `artifacts/manifests/iteration_057_m0_b4full_memory.json`
- `research_log.md`

### Failure or Limitation

One step does not establish steady host behavior or A-local performance.

### Gate Status

B4 capacity candidate positive; bounded candidates and multi-step validation remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can parameter-streaming-only partial-16 fit, avoiding activation traffic?

### Next Command

Run B4 partial-16 with no activation offload.

## Iteration 58

### Track / Milestone

Track A / M0 B4 parameter-streaming-only ablation.

### Question

Can partial-16 frozen-block streaming alone complete the official-shape step?

### Hypothesis

Streaming 16 tail blocks will create enough GPU headroom without activation offload.

### Importance

This isolates parameter residency from activation offload and preserves a required B4 ablation/Pareto point.

### Evidence Needed

Complete G/C/EMA sequence, finite losses, and measured GPU/host/transfer state.

### Refutation Condition

OOM or incomplete optimizer/version sequence.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=1 scripts/distillschedule/launch_m0_single_a100.sh \
  b4s16 iteration58_m0_b4s16_memory_001 memory 0
```

### Measured Results

- Return code 0; complete Generator/Critic/EMA sequence.
- CUDA max allocated/reserved: 37,032,203,264 / 38,663,094,272 bytes.
- Phase-sampled cgroup peak: 137,719,869,440 bytes; process construction high-water: 145,631,469,568 bytes.
- Generator/Score streamed 112,446,177,280 / 56,223,088,640 bytes.
- Generator/Critic losses: `0.45730964033703136` / `1.5418192148208618`.

### Correctness Result

PASS for one-step phase order and finite loss; exact update/moment comparison remains pending.

### Performance Result

Not Run formally; instrumented one-step progress time was 178.93 s.

### Interpretation

The hypothesis is verified. Parameter streaming alone can make one step feasible. Surprisingly, its instrumented run was slower than full streaming because the unstreamed portions still participate in repeated whole-model swaps; this is a hypothesis for formal timing, not yet a claim.

### Files Changed

- `artifacts/manifests/iteration_058_m0_b4s16_memory.json`
- `research_log.md`

### Failure or Limitation

The one-step memory run cannot freeze the fastest plan.

### Gate Status

B4 bounded capacity set has positive candidates. Multi-step leak/correctness and formal timing remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does the full-streaming 32-MiB/no-trim candidate stay host-stable over the mandated 10 correctness steps?

### Next Command

Run `b4fast` for steps 0--9 without changing source during execution.

## Iteration 59

### Track / Milestone

Track A / M0 B4 ten-step characterization.

### Question

Does `b4fast` cross two Generator updates and ten Critic/EMA updates without GPU or host-state growth?

### Hypothesis

After the alias-lifetime fix, host state will plateau and all ten logical steps will complete.

### Importance

This is the first official-shape observation spanning the required update-frequency boundary.

### Evidence Needed

Clean launcher exit, steps 0--9, G updates at 0/5, Critic and EMA every step, no live activation handles, and no monotonic long-term host growth.

### Refutation Condition

Any OOM, leak, incorrect phase count, or missing immutable launcher exit record.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration59_m0_b4fast_memory_10step_001 memory 0
```

### Measured Results

- The trainer emitted `training_complete_skip_final_save` at logical step 10.
- Phase counts show 2 Generator optimizer steps, 10 Critic optimizer steps, and 10 EMA steps.
- Final live CPU activation bytes were zero for both offload scopes.
- Phase-sampled cgroup memory plateaued near 134.9 GB after the first cycle; CUDA max allocated/reserved were 33,727,460,864 / 38,663,094,272 bytes.
- The shared launcher file was edited during this run. Although training completed, no `rc.txt` was recorded by the outer launcher.

### Correctness Result

INCOMPLETE. Trainer-internal phase evidence is positive, but immutable-launcher integrity and outer return code are missing.

### Performance Result

Not Run formally; memory instrumentation was enabled and launcher integrity was violated.

### Interpretation

The host-lifetime hypothesis is supported as characterization only. This run is deliberately not accepted as A-local correctness evidence and must be repeated cleanly.

### Files Changed

- `artifacts/manifests/iteration_059_m0_b4fast_10step_launcher_invalidated.json`
- `research_log.md`

### Failure or Limitation

Experiment provenance was invalidated by modifying a shared launcher during execution. The raw logs are retained rather than repaired or relabeled.

### Gate Status

B4 multi-step acceptance remains OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can the exact candidate reproduce the 10-step result from an immutable clean HEAD?

### Next Command

Revalidate the extended operation trace, then rerun `b4fast` steps 0--9 without workspace writes.

## Iteration 60

### Track / Milestone

Track A / M0 formal-cycle input contract.

### Question

Can the operation-keyed SYN-FEATURE trace cover multiple complete five-step cycles and remain independent of ambient RNG consumption?

### Hypothesis

A 20-step trace generated by the official sampler will validate exact tensor hashes, G-update positions 0/5/10/15, and replay identically under perturbed ambient seeds.

### Importance

Formal timing must not reuse correctness steps 0--9 and cannot rely on seed alone.

### Evidence Needed

Generation manifest, validation PASS, replay under distinct ambient seed, exact logical-step and update counts, and revalidation of the existing 10-step trace.

### Refutation Condition

Any hash mismatch, seed-dependent replay, missing operation, or invalid frequency pattern.

### Allowed Code Changes

Extend the trace validator from a hard-coded 10-step range to the contiguous range declared by each trace; add launcher trace-path override.

### Commands

```bash
python scripts/distillschedule/generate_m0_operation_trace.py \
  --output-root /share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20 \
  --steps 20
python scripts/distillschedule/validate_m0_operation_trace.py \
  /share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20
python scripts/distillschedule/validate_m0_operation_trace.py \
  /share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v1
```

### Measured Results

- 20-step trace: 249 tensors, 69 operations, logical steps 0--19.
- Generator updates at 0/5/10/15; 20 Critic updates.
- Trace hash: `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.
- Replay remained exact after ambient RNG perturbation.
- The original 10-step trace also revalidated with hash `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.

### Correctness Result

PASS for operation-keyed trace generation and replay independence.

### Performance Result

Not Run; trace-contract work only.

### Interpretation

The formal timing input contract now spans four complete update cycles and does not depend on random-number consumption order.

### Files Changed

- `scripts/distillschedule/validate_m0_operation_trace.py`
- `scripts/distillschedule/launch_m0_single_a100.sh`
- `artifacts/manifests/iteration_060_m0_trace10_revalidation.json`
- `artifacts/manifests/iteration_060_m0_trace20_generation.json`
- `artifacts/manifests/iteration_060_m0_trace20_validation.json`
- `research_log.md`

### Failure or Limitation

No canonical optimizer/EMA state has yet been serialized; this only freezes the input trace.

### Gate Status

Formal trace contract complete. B4 clean rerun and canonical checkpoint remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can shared immutable frozen-block host backing reduce per-rank long-lived state without changing one-step semantics?

### Next Command

Build the full Generator/Score frozen-block mmap cache with one rank and execute a one-step shared-host-policy validation.

## Iteration 61

### Track / Milestone

Track A / M0 A-scalable host-state substrate probe.

### Question

Can full frozen-block streaming use a content-addressed shared immutable mmap cache and still complete one official-shape update?

### Hypothesis

A single cache build will produce immutable backing for both 14B backbones and preserve the same complete G/C/EMA phase order.

### Importance

A-local private CPU state cannot simply be multiplied by eight; shared immutable backing is a candidate mechanism for A-scalable.

### Evidence Needed

Complete cache materialization, return code 0, one full update, finite losses, and cache identity recorded in the run.

### Refutation Condition

Incomplete cache, stale/mismatched tensor identity, OOM, or phase-order failure.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=1 \
DISTILL_M0_B3_HOST_POLICY=HOST_SHARED_IMMUTABLE \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4full iteration61_m0_b4full_shared_cache_build_001 memory 0
```

### Measured Results

- Source `56f432318b1e7211f1c8ffe24c9d7096c1291499`; return code 0.
- Cache contains 4,320 files and occupies 53 GiB under `DistillSchedule_assets/m0/frozen_block_cache`.
- One Generator optimizer step, one Critic optimizer step, and EMA optimization step 1 completed.
- Generator/Critic losses: `0.45730964033703136` / `1.5417805910110474`.
- Final live activation-offload bytes were zero.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes.
- Phase-sampled cgroup peak was 121,532,497,920 bytes. Whole-process cold-construction high-water was 145,630,265,344 bytes.

### Correctness Result

PASS for cache construction, one-step phase order, finite loss, and return code. Exact update/moment comparison and multi-rank sharing remain pending.

### Performance Result

Not Run formally; cache construction and detailed memory tracing were enabled.

### Interpretation

The shared immutable host-state primitive is feasible for one rank. It does not yet prove that eight ranks share physical pages under cgroup accounting or that the remaining private state satisfies the 760-GiB node contract.

### Files Changed

- `artifacts/manifests/iteration_061_m0_b4full_shared_cache_build.json`
- `research_log.md`

### Failure or Limitation

This run built/populated the cache and is not an 8-rank admission test.

### Gate Status

A-scalable substrate candidate positive; actual A-local/A-scalable acceptance and admission remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can `b4fast` complete the mandated ten-step correctness/leak run from a clean immutable HEAD?

### Next Command

Run a new clean 10-step `b4fast` memory trial with no workspace writes until launcher exit.

## Iteration 62

### Track / Milestone

Track A / M0 B4 clean ten-step correctness and leak acceptance.

### Question

Can the A-local candidate `b4fast` reproduce the prior trainer-internal result from a clean immutable HEAD and exit through the launcher?

### Hypothesis

Steps 0--9 will complete with G updates at 0/5, Critic and EMA every step, zero final live activation bytes, and a host-memory plateau after first-state initialization.

### Importance

The earlier ten-step run could not be accepted because its shared launcher changed during execution.

### Evidence Needed

Clean source status, return code 0, exact trace hash, complete phase counts, zero final activation handles, and bounded host/GPU state.

### Refutation Condition

Any OOM, missing update, nonzero final live activation storage, monotonic steady host growth, dirty source, or missing launcher return code.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration62_m0_b4fast_memory_10step_clean_001 memory 0
```

### Measured Results

- Source `cfc665055b165a79cdf24bc413448ae0e9dff286`; clean launcher metadata; trace hash `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- Return code 0 and `training_complete_skip_final_save` at logical step 10.
- Phase counts: Generator forward/backward/update 2/2/2; Critic forward/backward/update 10/10/10; EMA 10.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes; no allocator OOM/retry.
- Final live CPU activation bytes: zero for both Generator and Critic. Peak-live activation copies: 7,009,075,200 / 5,052,825,600 bytes.
- Phase-sampled steady cgroup peak: 134,961,541,120 bytes; the resource monitor's 151,638,953,984-byte peak was the cold-construction high-water, not steady rank-private state.
- Generator losses at steps 0/5: `0.45730964033703136` / `0.4518582879581576`.

### Correctness Result

PASS for the mandated 0--9 frequency, phase, finite-loss, immutable-provenance, and leak checks. Exact update/moment comparisons remain a separate pending correctness contract.

### Performance Result

Not Run formally; detailed memory/hot-path instrumentation was enabled. This run cannot freeze the timing winner.

### Interpretation

The clean run verifies that full frozen-block streaming plus 32-MiB coarse activation offload is a viable single-A100 official-shape baseline candidate across two full update cycles. Its steady per-rank host state is about 135 GB under HOST_PRIVATE, so direct eight-rank replication would exceed the 760-GiB node budget before swap transients and the required reserve.

### Files Changed

- `artifacts/manifests/iteration_062_m0_b4fast_10step_clean.json`
- `research_log.md`

### Failure or Limitation

No optimizer-moment tensor comparison or formal timing was performed.

### Gate Status

B4 ten-step capacity/leak acceptance positive. A-local formal timing and A-scalable admission remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does the shared immutable/trim candidate also remain stable for ten steps, and how much private host state remains?

### Next Command

Run the shared-cache B4 candidate for the same ten-step operation trace on the second node.

## Iteration 63

### Track / Milestone

Track A / M0 A-scalable per-rank candidate ten-step acceptance.

### Question

Does the shared immutable frozen-block cache remain correct and bounded across steps 0--9?

### Hypothesis

The cache-backed/trimmed candidate will complete all update frequencies, release activation handles, and reduce steady per-rank cgroup memory relative to HOST_PRIVATE.

### Importance

A-scalable requires a per-rank plan whose long-lived host state can plausibly be shared or replicated eight times under 760 GiB.

### Evidence Needed

Clean return code, identical trace, complete phase counts, zero final activation bytes, and measured host/GPU state.

### Refutation Condition

OOM, stale cache state, update mismatch, host leak, or no reduction in measured steady node state.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 \
DISTILL_M0_B3_HOST_POLICY=HOST_SHARED_IMMUTABLE \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4full iteration63_m0_b4full_shared_memory_10step_clean_001 memory 0
```

### Measured Results

- Same source and trace as Iteration 62; return code 0.
- Phase counts: 2 Generator updates, 10 Critic updates, 10 EMA updates; training completed at logical step 10.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes.
- Final live activation bytes were zero. Peak-live Generator/Critic activation copies were 8,290,959,360 / 5,064,130,560 bytes.
- Phase-sampled cgroup peak was 126,140,321,792 bytes versus 134,961,541,120 bytes for HOST_PRIVATE/no-trim. The cold-construction resource peak was 149,920,841,728 bytes.
- Generator losses at steps 0/5: `0.45730964033703136` / `0.4518967121748157`; the step-5 absolute delta to Iteration 62 was `3.8424e-5`, below the 2e-4 reference loss threshold.
- The run remained slower under memory instrumentation and host trimming; no formal performance claim is made.

### Correctness Result

PASS for trace, frequency, phase, finite-loss tolerance, and leak checks. Exact update/moment comparison and multi-rank physical-page sharing remain pending.

### Performance Result

Not Run formally; detailed tracing and trim were enabled.

### Interpretation

The shared-cache candidate is single-rank stable and lowers measured steady cgroup state by about 8.82 GB. That reduction alone does not prove 8-rank admission: shared page accounting, private optimizer/activation state, concurrent swap transients, and the 64-GiB reserve must be measured together.

### Files Changed

- `artifacts/manifests/iteration_063_m0_b4full_shared_10step_clean.json`
- `research_log.md`

### Failure or Limitation

A one-rank cgroup result cannot be multiplied or subtracted to infer physical page sharing reliably.

### Gate Status

A-scalable per-rank candidate positive; actual 8-rank/node admission remains. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can we freeze a canonical logical checkpoint and compare complete optimizer/EMA state across B4 candidates before formal timing?

### Next Command

Implement the M0-only canonical logical checkpoint save/load and correctness manifest, then validate it on focused CPU/unit fixtures before another 14B run.

## Iteration 64

### Track / Milestone

Track A / M0 canonical logical checkpoint substrate.

### Question

Can the B0a sparse CPU EMA participate in an actual save/resume path without reconstructing a third dense 14B model?

### Hypothesis

Bounded tensor shards plus immutable base identity can save and restore sparse EMA exactly, while Accelerate/DeepSpeed continues to own Generator/Critic optimizer and scheduler payloads.

### Importance

Formal steps 10--19 must start from one identical logical step-10 state. The previous B0a optimization intentionally disabled sparse-EMA resume, so seed-only restarts are not sufficient.

### Evidence Needed

Exact sparse-EMA unit round-trip, checksum rejection on corruption, full test-suite pass, and launcher support for explicit checkpoint/save and absolute resume paths.

### Refutation Condition

Shadow mismatch, silent corrupted-shard load, dense frozen-model materialization, or regression in existing tests.

### Allowed Code Changes

M0-only sparse EMA checkpoint I/O, resume integration, launcher overrides, and focused tests. No M1--M4 planner work.

### Commands

```bash
PYTHONPATH=. python -m pytest -q tests/test_official_baseline.py
PYTHONPATH=. python -m pytest -q
bash -n scripts/distillschedule/launch_m0_single_a100.sh
```

### Measured Results

- Focused tests: 16 passed.
- Full suite: 124 passed, 32 warnings.
- Sparse EMA now writes bounded CPU tensor shards, records per-shard SHA-256, and rejects corruption.
- Resume allocates placeholder shadows and loads only trainable EMA storage; it does not create a dense frozen-backbone EMA.
- Launcher can override `checkpointing_steps` and an absolute `resume_from_checkpoint`, and records both in immutable metadata.

### Correctness Result

PASS at unit level. Real 14B checkpoint save and resume are Not Run.

### Performance Result

Not Run.

### Interpretation

The missing state-path primitive is implemented narrowly enough for M0. It is not yet a canonical checkpoint until a real step-10 checkpoint is saved, hashed, resumed, and shown to preserve optimizer/EMA state.

### Files Changed

- `runtime/official_baseline.py`
- `train_helios.py`
- `scripts/distillschedule/launch_m0_single_a100.sh`
- `tests/test_official_baseline.py`
- `artifacts/manifests/iteration_064_m0_canonical_checkpoint_substrate.json`
- `research_log.md`

### Failure or Limitation

Accelerate/DeepSpeed checkpoint contents and save-time host peak remain unmeasured on the 14B workload.

### Gate Status

Canonical checkpoint substrate UNIT PASS; real save/resume OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can `b4fast` save a real canonical step-10 checkpoint without host/GPU OOM, and what exact payload does Accelerate/DeepSpeed produce?

### Next Command

Run `b4fast` steps 0--9 with `DISTILL_M0_CHECKPOINTING_STEPS=10`, then inspect and hash the resulting checkpoint before any resume.

## Iteration 65

### Track / Milestone

Track A / M0 real 14B canonical step-10 checkpoint construction.

### Question

Can `b4fast` save the complete Generator/Critic optimizer, scheduler, RNG, sparse EMA, and dataloader state at logical step 10 without reconstructing dense EMA or exhausting host memory?

### Hypothesis

The new sparse-EMA shards plus existing Accelerate/DeepSpeed state hooks will complete checkpoint-10 within the node budget.

### Importance

Formal timing at steps 10--19 must restart from one immutable state rather than regenerate step 0 from a seed.

### Evidence Needed

Complete ten-step phase sequence, checkpoint save, file hashes, sparse EMA optimization step 10, return code 0, and no host/GPU OOM.

### Refutation Condition

Any missing logical state, sparse-EMA dense materialization, OOM, corrupt/incomplete payload, or failed launcher exit.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 \
DISTILL_M0_CHECKPOINTING_STEPS=10 \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration65_m0_b4fast_canonical_checkpoint10_001 memory 0
```

### Measured Results

- Source `62435954803356d12fc65b74476eb07048b380eb`; return code 0; complete 2 Generator / 10 Critic / 10 EMA update sequence.
- Checkpoint root contains 22 files totaling 100,339,267,102 bytes (94 GiB); every file was SHA-256 hashed in 160.20 s.
- Payload includes Generator/Critic LoRA weights, two DeepSpeed model states, two CPU optimizer states, two schedulers, two RNG states, dataloader state, partial modules, and sparse EMA.
- Sparse EMA is 2,503,168,137 bytes across three bounded shards; manifest records optimization step 10 and decay 0.0.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes.
- Phase-sampled cgroup peak: 136,844,570,624 bytes; resource-sampled checkpoint/cold peak: 167,063,814,144 bytes. No OOM occurred.

### Correctness Result

PASS for save completeness, file hashing, phase order, and sparse-EMA step. Real resume is Not Run, so canonical acceptance remains pending.

### Performance Result

Checkpoint I/O is not part of steady timing. The checkpoint's backend payload is large because DeepSpeed also writes full model state; no compression claim is made.

### Interpretation

A full same-layout step-10 restart payload is feasible. It is backend-specific rather than a final cross-layout logical format, which remains M2 work; for M0 it can anchor identical 1-rank formal runs after a successful resume test.

### Files Changed

- `artifacts/manifests/iteration_065_m0_b4fast_canonical_checkpoint10.json`
- `artifacts/manifests/iteration_065_m0_checkpoint10_file_hashes.json`
- `research_log.md`

### Failure or Limitation

The 100.34-GB checkpoint is not yet proven loadable and contains redundant frozen/model state. It must not be called cross-layout canonical.

### Gate Status

Same-layout canonical payload SAVED, RESUME PENDING. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does the checkpoint restore exact global step, optimizer/scheduler/RNG, and sparse EMA before executing trace step 10?

### Next Command

Add low-overhead resume-state/timing markers, then perform a one-step resume probe from checkpoint-10 with the 20-step trace.

## Iteration 66

### Track / Milestone

Track A / M0 B4 partial-8 repaired ten-step candidate.

### Question

After the activation-registry fix, is partial-8 stable across two full update cycles, and is it competitive enough to retain for formal timing?

### Hypothesis

Partial-8 will remain capacity-feasible and bounded, but repeated whole-model swaps will make it materially slower than full streaming.

### Importance

The B4 action set must retain measured losing candidates rather than selecting the winner post hoc.

### Evidence Needed

Clean ten-step completion, GPU/host peak, activation lifetime, transfer count, and an instrumented screening time.

### Refutation Condition

OOM/leak, or evidence that partial-8 is plausibly the fastest B4 candidate.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 scripts/distillschedule/launch_m0_single_a100.sh \
  b4p8 iteration66_m0_b4p8_memory_10step_clean_001 memory 0
```

### Measured Results

- Return code 0; 2 Generator updates, 10 Critic updates, and 10 EMA steps completed.
- Final live activation bytes were zero.
- CUDA max allocated/reserved: 40,180,792,320 / 40,726,691,840 bytes, substantially closer to the 40-GB device limit than full streaming.
- Phase/resource cgroup peaks were 149,936,971,776 / 149,931,687,936 bytes; process construction high-water was 145,631,608,832 bytes.
- Generator/Score streamed 359,827,767,296 / 146,180,030,464 bytes, but the large unstreamed model portions still moved during component time-sharing.
- Instrumented training progress was 957.1 s for ten steps versus about 574 s for `b4fast`; both include memory tracing and are not formal timing.

### Correctness Result

PASS for phase/frequency/leak checks; exact update/moment comparison remains pending.

### Performance Result

Screening-only. Partial-8 was about 1.67x slower than `b4fast` in comparable memory runs and has less GPU safety margin. Formal ranking is not claimed.

### Interpretation

The hypothesis is supported. Partial-8 remains a valid capacity Pareto point but is pruned from the top formal-timing set unless low-overhead screening reverses the large observed gap.

### Files Changed

- `artifacts/manifests/iteration_066_m0_b4p8_10step_clean.json`
- `research_log.md`

### Failure or Limitation

Memory instrumentation perturbs timing, and the run occurred on the second node; the 1.67x ratio is only a pruning signal.

### Gate Status

B4 partial-8 bounded trial complete. Canonical resume and formal timing remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can checkpoint-10 be resumed exactly under the low-overhead timing contract?

### Next Command

Implement timing-only phase records and post-load state fingerprints, run tests, then resume one logical step from checkpoint-10.

## Iteration 67

### Track / Milestone

Track A / M0 resume verification and formal timing instrumentation.

### Question

Can formal timing retain operation-level phase boundaries without the memory queries, tensor checksums, EMA checksums, and 5-Hz `nvidia-smi` polling that contaminated prior memory runs?

### Hypothesis

A timing-only tracer with one CUDA synchronization at each iteration boundary can measure complete logical steps while removing known high-overhead instrumentation.

### Importance

A-local cannot be frozen from memory-profile progress bars. Generator/Critic/EMA checksum traversals alone touch billions of elements and materially perturb timing.

### Evidence Needed

Focused and full tests, explicit timing-only trace fields, no per-module synchronization, and checkpoint-resume fingerprints for model/optimizer/scheduler/EMA state.

### Refutation Condition

Missing step boundaries, timing mode still executing memory/checksum scans, or regression in memory-mode evidence.

### Allowed Code Changes

Instrumentation only: lightweight phase mode, per-step timing records, resume fingerprints, and timing-run monitor suppression.

### Commands

```bash
PYTHONPATH=. python -m pytest -q tests/test_official_baseline.py
PYTHONPATH=. python -m pytest -q
bash -n scripts/distillschedule/launch_m0_single_a100.sh
```

### Measured Results

- Focused tests: 18 passed.
- Full suite: 126 passed, 32 warnings.
- Timing mode omits CUDA allocator, RSS, cgroup, and memory-event queries from every phase record.
- Timing mode omits full Generator/Critic trainable checksums and the 2.5-GB sparse-EMA checksum from optimizer hot paths.
- Background 5-Hz `nvidia-smi`/RSS sampling is disabled in timing mode; memory runs retain it.
- Each step records `elapsed_seconds`, `data_wait_seconds`, update type, and logical step, with exactly two iteration-boundary CUDA synchronizations and no per-module synchronization.
- Resume mode now records post-load Generator/Critic, Adam-moment, scheduler, global-step, and sparse-EMA fingerprints before training.

### Correctness Result

PASS at unit/static level. Real checkpoint resume is Not Run.

### Performance Result

Not Run.

### Interpretation

The formal timing path is now separated from memory profiling. The next run is intentionally a resume/correctness probe, not yet one of the three timing repetitions.

### Files Changed

- `runtime/official_baseline.py`
- `train_helios.py`
- `scripts/distillschedule/launch_m0_single_a100.sh`
- `tests/test_official_baseline.py`
- `artifacts/manifests/iteration_067_m0_timing_resume_instrumentation.json`
- `research_log.md`

### Failure or Limitation

The generic optimizer fingerprint has not yet been exercised on the real DeepSpeed wrapper.

### Gate Status

Timing substrate UNIT PASS; real resume OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can the real DeepSpeed checkpoint and sparse EMA resume at logical step 10 and execute trace step 10 correctly?

### Next Command

Resume `b4fast` from Iteration 65 checkpoint-10 with the 20-step trace and stop after logical step 11.

## Iteration 68

### Track / Milestone

Track A / M0 same-layout canonical checkpoint resume.

### Question

Can the 100.34-GB step-10 payload be restored offline before executing trace step 10?

### Hypothesis

Accelerate/DeepSpeed plus the sparse-EMA hook will restore the saved Generator, Critic, optimizer, scheduler, RNG, and EMA state without network access.

### Importance

Formal timing must begin from one immutable logical state, not a seed-only reconstruction.

### Evidence Needed

A `checkpoint_resume_complete` event at global step 10 with optimizer/scheduler/EMA fingerprints, followed by a complete Generator/Critic/EMA step 10.

### Refutation Condition

Missing state, hash mismatch, OOM, or failure before the first resumed optimizer update.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=11 \
DISTILL_M0_TRACE_BUNDLE=/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20 \
DISTILL_M0_RESUME_FROM_CHECKPOINT=artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration68_m0_b4fast_resume10_step10_probe_001 memory 0
```

### Measured Results

- Source `da6c6a892007aea86fc095d70d9805127a5c4a91`; trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.
- Generator DeepSpeed payload loading began and completed its first state-loading phase.
- Resume then failed in the registered LoRA hook because offline `HeliosPipeline.lora_state_dict(input_dir)` did not specify the actual file name `pytorch_lora_weights.safetensors`.
- No training step or sparse-EMA restore was reached; return code 1.

### Correctness Result

FAIL (infrastructure): resume acceptance was not tested.

### Performance Result

Not Run. Construction/resume time is not a steady-state performance result.

### Interpretation

This is an offline-loading API defect, not a capacity or semantic negative result. The raw failure is retained and the resume hypothesis remains open.

### Files Changed

- `artifacts/manifests/iteration_068_m0_resume_offline_weight_name_failure.json`
- `research_log.md`

### Failure or Limitation

The run stopped before global-step, optimizer, scheduler, RNG, or sparse-EMA fingerprints were emitted.

### Gate Status

Canonical resume OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does specifying the exact LoRA weight file permit complete offline resume?

### Next Command

Patch the hook with the immutable saved filename, validate the source, and rerun the same resume probe.

## Iteration 69

### Track / Milestone

Track A / M0 B4 partial-16 bounded candidate.

### Question

Can partial-16 complete the required ten-step correctness/leak run?

### Hypothesis

Partial-16 should be capacity-feasible and provide a measured B4 Pareto point between partial-8 and full streaming.

### Importance

The bounded B4 candidate set must preserve every pre-registered streaming-block count rather than select only favorable outcomes.

### Evidence Needed

Steps 0--9, two Generator updates, ten Critic/EMA updates, memory peaks, and no monotonic state growth.

### Refutation Condition

A mechanism OOM, leak, or semantic mismatch after entering the training loop.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 scripts/distillschedule/launch_m0_single_a100.sh \
  b4s16 iteration69_m0_b4s16_memory_10step_clean_001 memory 0
```

### Measured Results

- Source `da6c6a892007aea86fc095d70d9805127a5c4a91`; official-shape trace hash `65872618a148785c884817e6d0166248ab10390e9c666c2d2806f5311261c2b4`.
- Construction, DeepSpeed prepare, split modulation, sparse EMA, and both block-streamer installations completed.
- The process failed before the first sample with `NameError: name 'time' is not defined` at the newly added iteration timing boundary; return code 1.

### Correctness Result

FAIL (instrumentation regression): no B4 mechanism conclusion.

### Performance Result

Not Run.

### Interpretation

The partial-16 hypothesis is neither verified nor refuted. This run is excluded from capacity/performance comparisons and must be repeated unchanged after fixing the instrumentation import.

### Files Changed

- `artifacts/manifests/iteration_069_m0_b4s16_instrumentation_import_failure.json`
- `research_log.md`

### Failure or Limitation

Failure occurred before the training loop due solely to a missing standard-library import.

### Gate Status

B4 partial-16 ten-step trial OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does the unchanged partial-16 mechanism complete after the timing import repair?

### Next Command

Import `time`, run static/full tests, and repeat the exact ten-step candidate.

## Iteration 70

### Track / Milestone

Track A / M0 resume and timing infrastructure repair.

### Question

Can the two Iteration 68/69 infrastructure failures be removed without changing Helios training semantics or any B4 policy?

### Hypothesis

Adding the missing standard-library import and naming the already-saved LoRA file are sufficient, narrowly scoped repairs.

### Importance

Neither failed experiment can be interpreted until the measurement and offline-resume substrate itself is valid.

### Evidence Needed

Python compilation, focused tests, full suite, launcher syntax validation, and a two-line semantic diff only.

### Refutation Condition

Any test regression, broader source change, or continued failure at either repaired line.

### Allowed Code Changes

Instrumentation/resume infrastructure only: import `time`; pass `weight_name="pytorch_lora_weights.safetensors"` to the existing load hook.

### Commands

```bash
python -m py_compile train_helios.py
PYTHONPATH=. python -m pytest -q tests/test_official_baseline.py
PYTHONPATH=. python -m pytest -q
bash -n scripts/distillschedule/launch_m0_single_a100.sh
```

### Measured Results

- Focused tests: 18 passed, 2 warnings.
- Full suite: 126 passed, 32 warnings.
- Python compilation and launcher syntax validation passed.
- No lifecycle primitive, model operation, loss, optimizer ordering, trace, or B4 policy was changed.

### Correctness Result

PASS at unit/static level; real resume and partial-16 reruns remain required.

### Performance Result

Not Run.

### Interpretation

Both observed failures were bounded infrastructure regressions. Their reruns can now provide actual evidence about checkpoint restoration and the partial-16 plan.

### Files Changed

- `train_helios.py`
- `artifacts/manifests/iteration_070_m0_resume_timing_repair.json`
- `research_log.md`

### Failure or Limitation

The offline load call and timing import have not yet been exercised by a real 14B rerun.

### Gate Status

Infrastructure FIXED at unit level. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can the same-layout step-10 checkpoint restore and execute the first complete resumed Generator/Critic/EMA iteration?

### Next Command

Rerun the Iteration 68 probe from checkpoint-10 with trace-v2; in parallel on the second node rerun partial-16 steps 0--9.

## Iteration 71

### Track / Milestone

Track A / M0 same-layout checkpoint resume, second infrastructure probe.

### Question

After naming the offline LoRA file explicitly, can Generator and Critic state restoration reach the logical step-10 boundary?

### Hypothesis

The prior LoRA filename fix is sufficient for both registered Accelerate load hooks.

### Importance

The formal timing contract remains blocked until the shared checkpoint can be restored rather than regenerated.

### Evidence Needed

Both DeepSpeed engines, both schedulers and RNG states, sparse EMA step 10, and a `checkpoint_resume_complete` event.

### Refutation Condition

Any failure before the post-load fingerprints or any missing state.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=11 \
DISTILL_M0_TRACE_BUNDLE=/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20 \
DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration71_m0_b4fast_resume10_step10_probe_001 memory 0
```

### Measured Results

- Source `652f65bfcd1041981c9c7f29251c4d0c50c2c2cb`; trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.
- Generator DeepSpeed model/optimizer, LoRA, partial modules, scheduler, sampler, and RNG load completed.
- Critic DeepSpeed model/optimizer load completed, then the generic hook redundantly attempted to reconstruct the Critic from the Generator asset root and failed at the nonexistent `Helios-Distilled/transformer/config.json`.
- Return code 1; sparse EMA and training were not reached.

### Correctness Result

FAIL (resume-hook infrastructure); the saved state itself was not rejected.

### Performance Result

Not Run.

### Interpretation

The upstream-style DeepSpeed load hook is performing a redundant second 14B model construction after DeepSpeed has already restored the full module. This is both an offline path bug and avoidable host/time overhead, not a lifecycle-plan negative result.

### Files Changed

- `artifacts/manifests/iteration_071_m0_resume_critic_source_failure.json`
- `research_log.md`

### Failure or Limitation

No `checkpoint_resume_complete` fingerprint exists, so same-layout canonical acceptance remains open.

### Gate Status

Canonical resume OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does the DeepSpeed payload already contain every LoRA and partial-module tensor needed to skip the redundant hook construction safely?

### Next Command

Inspect the saved Generator/Critic DeepSpeed module maps with mmap before changing the hook.

## Iteration 72

### Track / Milestone

Track A / M0 B4 partial-16 bounded candidate.

### Question

After repairing the timing import, does the pre-registered partial-16 streaming plan complete steps 0--9 at official shape?

### Hypothesis

Partial-16 is capacity-feasible but transfers substantially more frozen state than full streaming and is unlikely to beat `b4fast`.

### Importance

This closes the missing B4 block-count point and preserves a real memory/transfer Pareto candidate.

### Evidence Needed

Return code 0; G2/C10/EMA10; finite loss; bounded GPU/host state; no live activation leak; streamer counts and bytes.

### Refutation Condition

OOM, semantic/frequency error, leak, or evidence that partial-16 reverses the prior transfer/time trend.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 scripts/distillschedule/launch_m0_single_a100.sh \
  b4s16 iteration72_m0_b4s16_memory_10step_clean_001 memory 0
```

### Measured Results

- Source `652f65bfcd1041981c9c7f29251c4d0c50c2c2cb`; return code 0; G2/C10/EMA10 completed.
- Generator losses at steps 0/5: `0.45730964033703136` / `0.45188973426583906`; step-5 absolute delta from clean `b4fast` is about `3.14e-5`, within the `2e-4` screening threshold.
- CUDA max allocated/reserved: 37,032,203,264 / 38,663,094,272 bytes.
- Phase/resource cgroup peaks: 137,748,520,960 / 149,677,481,984 bytes.
- Generator: 1,024 materializations, 719,655,534,592 streamed bytes, 112.56 s cumulative H2D wall time.
- Score: 416 materializations, 292,360,060,928 streamed bytes, 74.31 s cumulative H2D wall time.
- Instrumented step 0/5 were 161.0/142.1 s; Critic-only steps were 58.2--80.3 s. These are memory-instrumented screening values, not formal timing.

### Correctness Result

PASS for phase frequency, finite loss, version order, and ten-step bounded execution. Exact optimizer-moment comparison remains pending formal checkpoint comparison.

### Performance Result

Screening-only. Partial-16 is retained as a measured capacity/transfer point, not declared a timing winner.

### Interpretation

The pre-registered partial-16 point is feasible, but it streams roughly 1.012 TB over ten steps and holds less GPU safety margin than full streaming. Formal low-overhead ranking is still required.

### Files Changed

- `artifacts/manifests/iteration_072_m0_b4s16_10step_clean.json`
- `research_log.md`

### Failure or Limitation

Memory instrumentation and model-move phase records perturb timing; no performance claim is made from this run.

### Gate Status

B4 partial-16 bounded trial COMPLETE. Canonical resume and formal timing OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can the common checkpoint be restored without a redundant 14B load hook?

### Next Command

Validate the DeepSpeed checkpoint module contents and bypass only the redundant DeepSpeed model reconstruction.

## Iteration 73

### Track / Milestone

Track A / M0 evidence-based DeepSpeed resume repair.

### Question

Can the redundant DeepSpeed load-hook construction be skipped without losing LoRA, partial-module, or dataloader state?

### Hypothesis

DeepSpeed already restores the complete module and optimizer before the registered hook, so the hook only needs the custom Generator dataloader DCP; non-DeepSpeed behavior must remain unchanged.

### Importance

This removes a false 14B reconstruction, fixes the Critic asset-path failure, and reduces resume-only host/time overhead without changing training semantics.

### Evidence Needed

Direct saved-module key inspection, narrow control-flow change, compilation, focused tests, and full tests.

### Refutation Condition

The DeepSpeed module lacks LoRA/partial keys, tests regress, or the change touches non-DeepSpeed load semantics.

### Allowed Code Changes

Checkpoint infrastructure only: early return after loading the custom Generator dataloader DCP when DeepSpeed has already restored the full module.

### Commands

```bash
/tmp/distillschedule-helios-official-torch210/bin/python - <<'PY'
# torch.load(..., mmap=True) on both mp_rank_00_model_states.pt payloads;
# count module, LoRA, and patch keys without materializing tensor storage.
PY
python -m py_compile train_helios.py
PYTHONPATH=. python -m pytest -q tests/test_official_baseline.py
PYTHONPATH=. python -m pytest -q
```

### Measured Results

- Generator DeepSpeed module: 1,919 keys, 818 LoRA keys, 12 patch keys.
- Critic DeepSpeed module: 1,913 keys, 812 LoRA keys, 6 patch keys.
- mmap inspection peak RSS was about 522 MiB and did not materialize the 31-GB storages.
- The hook now skips only redundant DeepSpeed model reconstruction, records the component/path, and preserves the custom Generator dataloader DCP load.
- Non-DeepSpeed LoRA/partial loading remains unchanged.
- Focused tests: 18 passed; full suite: 126 passed, 32 warnings; Python compilation passed.

### Correctness Result

PASS at static/payload/unit level; real 14B resume rerun required.

### Performance Result

Not Run. This is resume infrastructure, excluded from steady-state timing.

### Interpretation

The saved same-layout payload is self-contained for both physical models. Reconstructing a disposable 14B backbone in the load hook has no state-restoration role under DeepSpeed and can be removed safely from the M0 resume path.

### Files Changed

- `train_helios.py`
- `artifacts/manifests/iteration_073_m0_deepspeed_resume_hook_repair.json`
- `research_log.md`

### Failure or Limitation

The next real resume must still verify optimizer/scheduler/RNG/sparse-EMA fingerprints and execute step 10.

### Gate Status

Resume hook repaired at payload/unit level. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does the repaired hook restore global step 10 and execute a complete G/C/EMA step 10?

### Next Command

Repeat the checkpoint-10 resume probe with trace-v2 and stop after logical step 11.

## Iteration 74

### Track / Milestone

Track A / M0 same-layout canonical state restoration.

### Question

Does the repaired DeepSpeed hook restore every saved state and execute trace step 10?

### Hypothesis

Skipping the disposable backbone construction will expose the complete restored state and allow one G/C/EMA iteration.

### Importance

This is the last gate before all formal candidates can start from the same logical checkpoint.

### Evidence Needed

Post-load model/Adam/scheduler/EMA fingerprints at step 10 and a completed step 10.

### Refutation Condition

Missing/wrong state, OOM, or no resumed training operation.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=11 \
DISTILL_M0_TRACE_BUNDLE=/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20 \
DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration74_m0_b4fast_resume10_step10_probe_001 memory 0
```

### Measured Results

- Return code 0; both DeepSpeed module/optimizer loads and sparse-EMA load completed.
- `checkpoint_resume_complete` reported global step 10; Generator Adam step 2, Critic Adam step 10; scheduler last epochs 2/10; EMA optimization step 10.
- Critic checksum and EMA checksum match the final Iteration-65 records exactly at the summary level.
- Resumed Generator checksum differs from the last pre-save summary by only `1.07e-5` in sum and `1.37e-7` in norm; tensor-level update comparison remains pending.
- No `training_step_start` occurred. The saved StatefulDataLoader cursor had `_num_yielded=1` and `_sampler_iter_yielded=1` for a one-batch epoch, so the sole resumed epoch was consumed as an already-yielded batch.

### Correctness Result

PARTIAL PASS: all required state classes restored with correct counters, but step-10 execution was not reached.

### Performance Result

Not Run.

### Interpretation

The same-layout payload is loadable and state-complete. A checkpoint written after the last batch but before StatefulDataLoader observes `StopIteration` must normalize its cursor at an epoch boundary; blindly replaying that cursor skips the next logical operation.

### Files Changed

- `artifacts/manifests/iteration_074_m0_b4fast_resume10_step10_probe.json`
- `research_log.md`

### Failure or Limitation

Return code 0 alone is not acceptance: the required complete resumed G/C/EMA operation did not run.

### Gate Status

Canonical state restore PASS; resumed operation OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can epoch-boundary cursor normalization preserve mid-epoch restore while allowing trace step 10 to execute?

### Next Command

Add and test an explicit epoch-boundary dataloader restore policy, then repeat the probe.

## Iteration 75

### Track / Milestone

Track A / M0 operation-keyed resume cursor repair.

### Question

When should the custom StatefulDataLoader cursor be restored versus reset?

### Hypothesis

A mid-epoch checkpoint must restore the cursor, but a checkpoint with `checkpoint_step % updates_per_epoch == 0` must start a fresh epoch because the prior last batch is already committed.

### Importance

Seed-only restart is forbidden, but operation-keyed replay also must not drop the first operation after a canonical checkpoint.

### Evidence Needed

Raw DCP cursor values, a pure policy test covering boundary/mid-epoch cases, compilation, focused tests, and full tests.

### Refutation Condition

The saved cursor is not at the completed-epoch position, or the policy cannot distinguish boundary from mid-epoch restore.

### Allowed Code Changes

Resume infrastructure only: select DCP cursor restore/reset from checkpoint step and measured `num_update_steps_per_epoch`.

### Commands

```bash
# Inspect DCP .metadata and each BytesStorage chunk.
python -m py_compile train_helios.py runtime/official_baseline.py
PYTHONPATH=. python -m pytest -q tests/test_official_baseline.py
PYTHONPATH=. python -m pytest -q
```

### Measured Results

- Saved DCP cursor: `_sampler_iter_yielded=1`, `_num_yielded=1`, `_iterator_finished=False` for a one-batch epoch.
- The new pure policy resets checkpoints at update/epoch boundaries and restores mid-epoch cursors; invalid step/epoch counts raise errors.
- DeepSpeed Generator hook records `reset_completed_epoch_cursor` or `restore_mid_epoch_cursor`; Critic remains `not_applicable`.
- Focused tests: 19 passed; full suite: 127 passed, 32 warnings; compilation passed.

### Correctness Result

PASS at state-inspection/unit level; real resumed step remains required.

### Performance Result

Not Run.

### Interpretation

The failure was a precise checkpoint-boundary ambiguity, not RNG drift. The policy preserves operation-keyed replay and avoids reusing a finished batch cursor at the next epoch.

### Files Changed

- `runtime/official_baseline.py`
- `tests/test_official_baseline.py`
- `train_helios.py`
- `artifacts/manifests/iteration_075_m0_dataloader_boundary_resume_policy.json`
- `research_log.md`

### Failure or Limitation

The M0 one-sample contract exercises an epoch boundary; a mid-epoch real 14B resume is unit-covered but not dynamically measured here.

### Gate Status

Cursor policy UNIT PASS; resumed operation OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does trace step 10 now execute with the restored Generator/Critic/Adam/scheduler/EMA state?

### Next Command

Repeat the checkpoint-10 probe to max step 11 and require one complete Generator/Critic/EMA operation.

## Iteration 76

### Track / Milestone

Track A / M0 canonical checkpoint execution acceptance.

### Question

After epoch-boundary cursor normalization, does the restored step-10 state execute a complete official-shape G/C/EMA iteration?

### Hypothesis

The operation-keyed trace will now produce exactly one logical step 10, advancing Generator, Critic, schedulers, and EMA once.

### Importance

All formal timing candidates require a proven common starting state and the same next operation.

### Evidence Needed

One step-10 start/end, finite G/C losses, both optimizer updates, EMA 10→11, trace-v2 identity, no OOM/leak, return code 0.

### Refutation Condition

Skipped/duplicated operation, wrong update frequency/counter, non-finite loss, OOM, or nonzero return.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=11 \
DISTILL_M0_TRACE_BUNDLE=/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20 \
DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration76_m0_b4fast_resume10_step10_complete_001 memory 0
```

### Measured Results

- Source `96e63474d38317b2603c768102fbbf1d82df2d62`; trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`; return code 0.
- Restore fingerprint: global step 10, Generator Adam step 2, Critic Adam step 10, scheduler epochs 2/10, sparse EMA step 10.
- Exactly one logical step 10 executed: Generator loss `0.4372450740737085`, Critic loss `1.4989588260650635`; both backward and optimizer events completed; EMA advanced to 11.
- Memory-run step time was 97.93 s plus 0.073 s data wait. This is not formal timing.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes. Resource cgroup peak: 151,641,874,432 bytes; phase peak: 135,483,408,384 bytes.
- Generator/Critic activation offload ended with zero live CPU bytes. Generator/Score streamers ended with zero resident frozen bytes.

### Correctness Result

PASS for same-layout canonical restore and the complete next G/C/EMA operation. Repeated-strategy parameter-delta comparison remains part of formal correctness.

### Performance Result

Not formal; memory instrumentation was enabled.

### Interpretation

The 100.34-GB checkpoint is now an accepted M0 same-layout canonical start. It restores all required state classes and replays the next operation rather than merely loading successfully.

### Files Changed

- `artifacts/manifests/iteration_076_m0_b4fast_resume10_step10_complete.json`
- `research_log.md`

### Failure or Limitation

This checkpoint is backend/same-layout specific; cross-layout canonical conversion remains outside the currently authorized M0 scope.

### Gate Status

Canonical M0 start ACCEPTED. Formal timing and A-local/A-scalable admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Which feasible B3/B4 static plan is fastest under low-overhead steps 10--19 timing, and which plan can scale its long-lived host state to eight ranks per node?

### Next Command

Freeze the static action library, add a 32-MiB/shared/trim scalable candidate and a strict five-step-cycle analyzer, then run randomized timing candidates from checkpoint-10.

## Iteration 77

### Track / Milestone

Track A / M0 B4 action-library freeze and formal timing substrate.

### Question

Can formal comparison separate the fastest local plan from a host-scalable plan without changing mechanism families after seeing results?

### Hypothesis

`b4fast` (full frozen streaming, 32-MiB activation threshold, no host trim) is the A-local candidate, while a pre-registered `b4scale` variant with the same tensor policy plus host trim/shared immutable state is a plausible A-scalable candidate.

### Importance

A-local and A-scalable may differ; formal timing needs an immutable analyzer and pre-registered candidate policies.

### Evidence Needed

A frozen launcher variant, strict timing analyzer enforcing steps 10--19 and the 1:4 schedule, unit/full tests, and separate timing/memory runs.

### Refutation Condition

Analyzer accepts missing/wrong steps, the scalable variant changes training semantics, or tests regress.

### Allowed Code Changes

M0 expert-static action declaration and measurement tooling only; no automatic planner or new Track-C primitive.

### Commands

```bash
PYTHONPATH=. python -m pytest -q tests/test_m0_timing.py tests/test_official_baseline.py
PYTHONPATH=. python -m pytest -q
bash -n scripts/distillschedule/launch_m0_single_a100.sh
```

### Measured Results

- Added `b4scale`: the same full frozen-block streaming and 32-MiB activation threshold as `b4fast`, but with host allocator trimming; shared immutable mmap is selected explicitly at launch.
- Analyzer requires exactly ten `training_step_end` records at steps 10--19, Generator updates at 10/15, and Critic-only steps otherwise.
- End-to-end step time is `device-synchronized elapsed + data-wait`; it reports both five-step cycles, G/C means, official weighted mean, samples/s, and raw-file hashes.
- Focused tests: 21 passed; full suite: 129 passed, 32 warnings; compilation and launcher syntax passed.

### Correctness Result

PASS at unit/static level; `b4scale` real ten-step correctness and all formal timing runs remain pending.

### Performance Result

Not Run.

### Interpretation

The M0 expert action library is now frozen before timing: windows 1/2/4, partial block counts 4/8/16/full, activation thresholds full/8/32 MiB, host private/shared immutable, trim on/off, and synchronous fixed prefetch distance zero.

### Files Changed

- `scripts/distillschedule/launch_m0_single_a100.sh`
- `runtime/m0_timing.py`
- `scripts/distillschedule/analyze_m0_timing.py`
- `tests/test_m0_timing.py`
- `artifacts/manifests/iteration_077_m0_formal_timing_action_freeze.json`
- `research_log.md`

### Failure or Limitation

Formal timing still needs three independent starts for the selected A-local/A-scalable candidates and at least one measured screening run for retained alternatives.

### Gate Status

Formal timing substrate UNIT PASS. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does `b4scale` retain correctness and lower long-lived host state without losing enough throughput to change the Pareto frontier?

### Next Command

Run `b4scale` steps 0--9 in memory mode with shared immutable state while beginning randomized checkpoint-10 timing runs on the other node.

## Iteration 78

### Track / Milestone

Track A / M0 `b4scale` correctness and memory characterization.

### Question

Does the shared-immutable, host-trimmed 32-MiB selective plan preserve the ten-step semantics while providing a scalable-host candidate?

### Hypothesis

`b4scale` will match `b4fast` numerically, bound live offloaded state, and reduce steady host pressure through shared frozen storage and trimming.

### Importance

A-scalable must be chosen from an actually correct per-rank full pipeline, not inferred from the A-local winner.

### Evidence Needed

Steps 0--9, G2/C10/EMA10, finite/matching losses, zero live activation/frozen-block state, GPU/host peaks, shared-cache policy.

### Refutation Condition

OOM, semantic drift, leak, or long-lived state incompatible with later eight-rank admission.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=10 \
DISTILL_M0_B3_HOST_POLICY=HOST_SHARED_IMMUTABLE \
DISTILL_M0_B3_MMAP_CACHE=/share/project/shuchenweng/DistillSchedule_assets/m0/frozen_block_cache \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4scale iteration78_m0_b4scale_shared_memory_10step_001 memory 0
```

### Measured Results

- Source `b0f20b3950082d14eb5784cf4412b0467397ce54`; return code 0; G2/C10/EMA10.
- Generator losses steps 0/5: `0.45730964033703136` / `0.4518954727488033`; step-5 delta from clean `b4fast` is about `3.72e-5`.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes.
- Phase/resource cgroup peaks: 127,878,483,968 / 151,638,822,912 bytes; cold construction remains the resource high-water.
- Activation final live CPU bytes and streamer final resident bytes were zero.
- Ten-step frozen transfer: Generator 1,799,138,836,480 bytes; Score 730,900,152,320 bytes. These byte counts are logical repeated materializations, not unique host storage.

### Correctness Result

PASS for ten-step phase/frequency/leak and loss thresholds. Exact final moment/update comparison remains pending formal checkpoint outputs.

### Performance Result

Memory-instrumented screening only; no formal timing claim.

### Interpretation

`b4scale` is a valid A-scalable candidate. Its single-rank cold peak is not an eight-rank admission result; shared mappings and construction waves must be measured directly.

### Files Changed

- `artifacts/manifests/iteration_078_m0_b4scale_shared_10step.json`
- `research_log.md`

### Failure or Limitation

Single-process cgroup/RSS cannot be multiplied naively because mmap-backed frozen pages may be shared; actual node admission remains mandatory.

### Gate Status

`b4scale` correctness PASS. A-scalable admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

How much steady-state throughput does `b4scale` trade for its host policy relative to `b4fast`?

### Next Command

Run both candidates from checkpoint-10 in timing-only mode and complete three independent starts each.

## Iteration 79

### Track / Milestone

Track A / M0 A-local candidate formal timing repetition 1.

### Question

What is the low-overhead steady-state five-step-cycle time of `b4fast` from the accepted checkpoint?

### Hypothesis

The weighted mean will be materially lower than memory-instrumented screening and stable enough across two cycles to retain `b4fast` as A-local candidate.

### Importance

A-local must be frozen from end-to-end timing, not progress-bar or memory-profile estimates.

### Evidence Needed

Exactly steps 10--19, G updates 10/15, two cycles, timing-only instrumentation, identical checkpoint/trace, return code 0.

### Refutation Condition

Missing/wrong steps, nonzero exit, or performance instability large enough to invalidate one-run ranking.

### Allowed Code Changes

None during the immutable run.

### Commands

```bash
DISTILL_M0_MAX_STEPS=20 \
DISTILL_M0_TRACE_BUNDLE=/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20 \
DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration79_m0_b4fast_timing_rep1_steps10_19 timing 0
```

### Measured Results

- Return code 0; trace-v2 and checkpoint-10; exactly ten timed steps and two correct 1:4 cycles.
- Generator-update mean: 104.987 s; Critic-only mean: 47.949 s.
- Official weighted mean: 59.357 s/step; 0.01685 samples/s; 59.357 GPU-s/sample.
- Five-step cycles: 304.634 s and 288.934 s; mean 296.784 s.
- Ten-step synchronized device work: 587.541 s; total data-wait: 6.027 s; end-to-end: 593.568 s.

### Correctness Result

PASS for trace/checkpoint/update schedule. Cross-repetition loss/update consistency remains pending.

### Performance Result

Measured formal repetition 1; not yet a three-run aggregate.

### Interpretation

Timing-only instrumentation removes the prior checksum/memory census overhead, but per-step variation remains tied to the fixed operation trace. The comparison unit is therefore the complete two-cycle trace, not an arbitrary single step.

### Files Changed

- `artifacts/manifests/iteration_079_m0_b4fast_timing_rep1.json`
- `research_log.md`

### Failure or Limitation

One repetition is insufficient for the required independent-start aggregate.

### Gate Status

A-local timing 1/3. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does `b4fast` reproduce this timing and how does `b4scale` compare under the same trace?

### Next Command

Run `b4fast` repetition 2 and `b4scale` repetition 1 concurrently from checkpoint-10.

## Iteration 80

### Track / Milestone

Track A / M0 A-local candidate formal timing repetition 2.

### Question

Does an independent `b4fast` start reproduce repetition 1 under the same checkpoint and trace?

### Hypothesis

The complete two-cycle weighted mean will remain within normal run-to-run variation of repetition 1.

### Importance

A-local requires an independent-start aggregate rather than one favorable run.

### Evidence Needed

Steps 10--19, correct 1:4 schedule, two cycles, return code 0, and bounded timing variation.

### Refutation Condition

Semantic mismatch, failure, or large unexplained cycle-level instability.

### Allowed Code Changes

None.

### Commands

```bash
# Same checkpoint/trace/timing contract as Iteration 79.
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration80_m0_b4fast_timing_rep2_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 101.729 s; Critic-only mean 46.381 s.
- Official weighted mean 57.450 s/step.
- Five-step cycles: 287.165 s and 287.339 s.
- Relative weighted difference from repetition 1: about -3.21%.

### Correctness Result

PASS for checkpoint, trace, and schedule.

### Performance Result

Measured formal repetition 2/3.

### Interpretation

`b4fast` is reproducible at the full-cycle level; the second run is internally especially stable across its two cycles.

### Files Changed

- `artifacts/manifests/iteration_080_m0_b4fast_timing_rep2.json`
- `research_log.md`

### Failure or Limitation

One further independent start is required.

### Gate Status

A-local timing 2/3. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Does repetition 3 preserve this range?

### Next Command

Run `b4fast` repetition 3 in the next randomized wave.

## Iteration 81

### Track / Milestone

Track A / M0 A-scalable candidate formal timing repetition 1.

### Question

What throughput cost does shared-immutable `b4scale` pay relative to `b4fast`?

### Hypothesis

Host trimming will increase exposed allocation/copy overhead but may remain close enough to justify node scalability.

### Importance

A-scalable is a constrained winner, so its local cost must be measured before node admission.

### Evidence Needed

The same steps 10--19 and trace/checkpoint with shared mmap and host trim.

### Refutation Condition

Failure or a performance loss so large that another shared-host candidate must be evaluated.

### Allowed Code Changes

None.

### Commands

```bash
DISTILL_M0_B3_HOST_POLICY=HOST_SHARED_IMMUTABLE \
DISTILL_M0_B3_MMAP_CACHE=/share/project/shuchenweng/DistillSchedule_assets/m0/frozen_block_cache \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4scale iteration81_m0_b4scale_timing_rep1_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 131.323 s; Critic-only mean 54.509 s.
- Official weighted mean 69.872 s/step.
- Five-step cycles: 367.241 s and 331.474 s.
- Against `b4fast` repetitions 1/2, this first `b4scale` run is about 17.7--21.6% slower.

### Correctness Result

PASS for checkpoint, trace, and schedule.

### Performance Result

Measured formal repetition 1/3.

### Interpretation

Host trimming is not performance-neutral. `b4scale` can only win A-scalable if it enables eight-rank admission that faster policies cannot satisfy.

### Files Changed

- `artifacts/manifests/iteration_081_m0_b4scale_timing_rep1.json`
- `research_log.md`

### Failure or Limitation

The first cycle is slower than the second; two more independent starts are needed before aggregation.

### Gate Status

A-scalable-candidate timing 1/3. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Is the `b4scale` penalty repeatable, and can shared `b4fast` avoid it while satisfying host admission?

### Next Command

Run `b4scale` repetition 2 and `b4fast` repetition 3 concurrently.

## Iteration 82

### Track / Milestone

Track A / M0 `b4scale` formal timing repetition 2.

### Question

Is the `b4scale` overhead repeatable across an independent start?

### Hypothesis

The plan remains slower than `b4fast`, with some variance from synchronous host trimming and page/cache state.

### Importance

A constrained scalable winner still needs a defensible local performance distribution.

### Evidence Needed

Complete steps 10--19 and a second independent two-cycle measurement.

### Refutation Condition

Failure or performance matching `b4fast`, which would change the host-policy tradeoff.

### Allowed Code Changes

None.

### Commands

```bash
# Same b4scale checkpoint/trace/timing contract as Iteration 81.
scripts/distillschedule/launch_m0_single_a100.sh \
  b4scale iteration82_m0_b4scale_timing_rep2_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 152.499 s; Critic-only mean 60.364 s.
- Official weighted mean 78.791 s/step.
- Five-step cycles: 411.282 s and 376.627 s.

### Correctness Result

PASS for trace/checkpoint/1:4 schedule.

### Performance Result

Measured formal repetition 2/3.

### Interpretation

The synchronous trim/shared plan remains slower and exhibits greater run-to-run variance than `b4fast`; node admission, not local speed, must justify it.

### Files Changed

- `artifacts/manifests/iteration_082_m0_b4scale_timing_rep2.json`
- `research_log.md`

### Failure or Limitation

A third start is required; page-cache/allocator state is not isolated by the timing tracer.

### Gate Status

A-scalable-candidate timing 2/3. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

What is the third-run aggregate and can shared `b4fast` avoid trimming overhead?

### Next Command

Run `b4scale` repetition 3 and shared-immutable `b4fast` screening concurrently.

## Iteration 83

### Track / Milestone

Track A / M0 A-local formal timing repetition 3 and aggregate closure.

### Question

Does `b4fast` remain stable on its third independent start?

### Hypothesis

The weighted mean remains within roughly 5% of the first two repetitions and all losses satisfy the per-term threshold.

### Importance

This freezes the local performance distribution rather than a post-hoc best run.

### Evidence Needed

Third complete two-cycle measurement plus cross-run loss deltas.

### Refutation Condition

Timing outlier, schedule failure, or loss delta above `2e-4`.

### Allowed Code Changes

None.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration83_m0_b4fast_timing_rep3_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 99.170 s; Critic-only mean 46.915 s.
- Official weighted mean 57.366 s/step; cycles 290.963 / 282.695 s.
- Three-run weighted means: 59.357, 57.450, 57.366 s; arithmetic mean 58.058 s, sample standard deviation 1.126 s (1.94%).
- Step-10 Generator loss is exact across all three runs. Across all G/C loss terms, observed deltas remain below the `2e-4` screening threshold; EMA counters are identically 11--20.

### Correctness Result

PASS for schedule, per-term loss threshold, and EMA recurrence at timing-trace granularity.

### Performance Result

A-local candidate formal timing 3/3 COMPLETE: 58.058 ± 1.126 s weighted mean across independent starts.

### Interpretation

`b4fast` is a stable local winner candidate. Final A-local freeze still requires retained-alternative screening, but not further repetitions unless another plan is faster.

### Files Changed

- `artifacts/manifests/iteration_083_m0_b4fast_timing_rep3.json`
- `research_log.md`

### Failure or Limitation

Timing mode omits full parameter/Adam checksums by design; exact update-state comparison requires a separate correctness checkpoint run.

### Gate Status

A-local candidate timing COMPLETE; alternative screen OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can a shared-host no-trim plan approach `b4fast` speed while satisfying eight-rank host admission?

### Next Command

Run `b4scale` repetition 3 and shared `b4fast` screening.

## Iteration 84

### Track / Milestone

Track A / M0 `b4scale` formal timing repetition 3 and aggregate.

### Question

What is the three-start timing distribution of the host-trimmed scalable candidate?

### Hypothesis

`b4scale` remains slower and more variable than A-local, but completes correctly.

### Importance

A-scalable must carry an explicit performance cost rather than being called equivalent to A-local.

### Evidence Needed

Third complete two-cycle run and aggregate statistics.

### Refutation Condition

Failure, correctness drift, or performance converging to `b4fast`.

### Allowed Code Changes

None.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b4scale iteration84_m0_b4scale_timing_rep3_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 135.928 s; Critic-only mean 57.286 s; weighted 73.014 s/step.
- Five-step cycles: 399.889 / 330.256 s.
- Three-run `b4scale` weighted means: 69.872, 78.791, 73.014 s; mean 73.892 s, sample standard deviation 4.524 s (6.12%).
- This aggregate is 27.3% slower than the 58.058-s `b4fast` aggregate.

### Correctness Result

PASS for trace/checkpoint/update schedule.

### Performance Result

`b4scale` formal timing 3/3 COMPLETE: 73.892 ± 4.524 s.

### Interpretation

Synchronous host trimming has a material and variable exposed cost. It can be A-scalable only if admission rejects faster shared/no-trim candidates.

### Files Changed

- `artifacts/manifests/iteration_084_m0_b4scale_timing_rep3.json`
- `research_log.md`

### Failure or Limitation

The timing trace does not independently attribute allocator trim versus page-fault cost; M0 only needs their joint static-plan cost.

### Gate Status

`b4scale` timing COMPLETE. Node admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can shared immutable frozen state without trimming fit eight ranks and avoid most of the 27% penalty?

### Next Command

Measure shared/no-trim `b4fast` memory and node admission.

## Iteration 85

### Track / Milestone

Track A / M0 shared/no-trim scalable-candidate screen.

### Question

Does replacing private frozen host copies with shared mmap, while retaining `b4fast` no-trim behavior, preserve near-local throughput?

### Hypothesis

Shared backing alone is much cheaper than per-swap allocator trimming and should remain within 10% of A-local after the initial shared-page working set is touched.

### Importance

This candidate may dominate `b4scale` if it also satisfies eight-rank host admission.

### Evidence Needed

One complete formal screen under the same trace/checkpoint, followed by a memory run and actual admission.

### Refutation Condition

Failure or ≥10% repeatable local slowdown before admission.

### Allowed Code Changes

None; only select the pre-frozen `HOST_SHARED_IMMUTABLE` policy for `b4fast`.

### Commands

```bash
DISTILL_M0_B3_HOST_POLICY=HOST_SHARED_IMMUTABLE \
DISTILL_M0_B3_MMAP_CACHE=/share/project/shuchenweng/DistillSchedule_assets/m0/frozen_block_cache \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration85_m0_b4fast_shared_notrim_timing_screen_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 124.667 s; Critic-only mean 46.383 s; weighted 62.040 s/step.
- Cycles were 347.899 and 272.501 s; the first cycle paid a much larger first-touch cost.
- The one-run weighted screen is 6.86% slower than the private `b4fast` three-run mean and 16.0% faster than `b4scale` mean.

### Correctness Result

PASS for trace/checkpoint/update schedule.

### Performance Result

Measured screening run only; admission and repetitions pending.

### Interpretation

Shared/no-trim is the leading A-scalable candidate, but a timing screen cannot establish host feasibility. Its page working set and long-lived private state must be measured in memory mode and at eight-rank admission.

### Files Changed

- `artifacts/manifests/iteration_085_m0_b4fast_shared_notrim_timing_screen.json`
- `research_log.md`

### Failure or Limitation

Single-run first-touch/page-cache behavior is variable; do not freeze A-scalable from this timing alone.

### Gate Status

Shared/no-trim candidate retained. A-scalable admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

What is its per-rank host high-water and does shared backing materially reduce node-wide steady state?

### Next Command

Run shared/no-trim `b4fast` steps 0--9 in memory mode while screening B3 window-1 timing.

## Iteration 86

### Track / Milestone

Track A / M0 shared/no-trim host memory characterization.

### Question

What is the single-rank memory high-water and lifetime behavior of shared `b4fast`?

### Hypothesis

Single-rank cgroup peak will resemble private `b4fast`, while frozen-page sharing can only be established at multi-rank admission.

### Importance

Shared mappings are not a scalability result until node-wide accounting measures their amortization.

### Evidence Needed

Ten-step completion, phase/resource peaks, zero final live state, and shared host-policy metadata.

### Refutation Condition

OOM/leak or a single-rank host increase large enough to reject the candidate statically.

### Allowed Code Changes

None.

### Commands

```bash
DISTILL_M0_B3_HOST_POLICY=HOST_SHARED_IMMUTABLE \
DISTILL_M0_B3_MMAP_CACHE=/share/project/shuchenweng/DistillSchedule_assets/m0/frozen_block_cache \
scripts/distillschedule/launch_m0_single_a100.sh \
  b4fast iteration86_m0_b4fast_shared_notrim_memory_10step_001 memory 0
```

### Measured Results

- Return code 0; G2/C10/EMA10; final activation live bytes and streamer resident bytes zero.
- CUDA max allocated/reserved: 33,727,460,864 / 38,663,094,272 bytes.
- Phase/resource cgroup peaks: 138,663,514,112 / 151,709,167,616 bytes; process RSS high-water 145,638,592,512 bytes.
- These single-rank values are not multiplied to estimate eight-rank shared mappings.

### Correctness Result

PASS for ten-step semantics/leak checks; all losses remained within the existing screening tolerance.

### Performance Result

Memory-instrumented only.

### Interpretation

Shared backing does not reduce single-process cold high-water; its value, if any, is inter-process page sharing. The candidate survives static screening but requires actual eight-rank admission.

### Files Changed

- `artifacts/manifests/iteration_086_m0_b4fast_shared_notrim_memory.json`
- `research_log.md`

### Failure or Limitation

`VmRSS` double-counts shared pages across processes, so no node formula is inferred from this single process.

### Gate Status

Shared/no-trim per-rank characterization COMPLETE; node admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can eight ranks share enough immutable pages to fit under the 760-GiB cgroup limit?

### Next Command

After retained alternative screening, launch wave-controlled eight-rank admission with node-wide cgroup accounting.

## Iteration 87

### Track / Milestone

Track A / M0 B3 window-1 retained-alternative timing screen.

### Question

Does the coarse full-activation-offload B3 plan outperform selective B4?

### Hypothesis

Offloading every eligible saved tensor will expose more copy work and be slower than `b4fast`.

### Importance

A-local cannot be frozen without a measured strong existing-mechanism alternative.

### Evidence Needed

One complete checkpoint-10 two-cycle timing screen.

### Refutation Condition

B3 window-1 matches or beats the `b4fast` aggregate.

### Allowed Code Changes

None.

### Commands

```bash
scripts/distillschedule/launch_m0_single_a100.sh \
  b3w1 iteration87_m0_b3w1_timing_screen_steps10_19 timing 0
```

### Measured Results

- Generator-update mean 122.368 s; Critic-only mean 49.220 s.
- Official weighted mean 63.850 s/step; cycles 323.046 / 315.450 s.
- This is 9.98% slower than the 58.058-s `b4fast` three-run mean.

### Correctness Result

PASS for trace/checkpoint/update schedule.

### Performance Result

Measured one-run screen; B3 window-1 is pruned from A-local contention.

### Interpretation

Selective activation offload is worthwhile within the frozen action library, but this remains a hand-tuned B4 baseline result, not a DistillSchedule planner gain.

### Files Changed

- `artifacts/manifests/iteration_087_m0_b3w1_timing_screen.json`
- `research_log.md`

### Failure or Limitation

Windows 2/4 and other B4 alternatives still need retained screening or a documented pruning reason.

### Gate Status

B3 window-1 screen COMPLETE. A-local alternative screen OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question

Can partial-16 or the 8-MiB full-stream plan beat `b4fast`?

### Next Command

Run `b4s16` and `b4full` timing screens concurrently.

## Iteration 88

### Track / Milestone
Track A / M0 retained B4 partial-16 timing screen.

### Question
Can streaming only 16 frozen blocks outperform the current `b4fast` A-local candidate?

### Hypothesis
Reducing the streamed block count may cut materialization overhead while retaining enough GPU headroom.

### Importance
The preregistered B4 action library must retain and measure plausible partial-stream alternatives before freezing A-local.

### Evidence Needed
A checkpoint-10, operation-trace-bound, two-cycle timing run at 384x640.

### Refutation Condition
The weighted steady-state time is materially slower than the three-run `b4fast` mean.

### Allowed Code Changes
None; select the frozen `b4s16` policy only.

### Commands
`DISTILL_M0_MAX_STEPS=20 DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 scripts/distillschedule/launch_m0_single_a100.sh b4s16 iteration88_m0_b4s16_timing_screen_steps10_19 timing 0`

### Measured Results
Generator-update mean 152.428 s; Critic-only mean 69.383 s; official weighted mean 85.992 s/step. Five-step cycles were 437.286 and 422.636 s.

### Correctness Result
PASS for checkpoint boundary, trace identity, ten-step schedule, and finite losses.

### Performance Result
Measured screening result: 48.12% slower than the 58.058-s `b4fast` three-run mean.

### Interpretation
Partial-16 streaming is decisively pruned from A-local contention.

### Files Changed
`artifacts/manifests/iteration_088_m0_b4s16_timing_screen.json`; immutable raw logs/traces.

### Failure or Limitation
One formal screen is sufficient for pruning but not a three-repetition performance estimate.

### Gate Status
B4 partial-16 screen COMPLETE; A-local OPEN; M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does full streaming with the lower activation threshold beat `b4fast`?

### Next Command
Run the frozen `b4full` timing screen.

## Iteration 89

### Track / Milestone
Track A / M0 retained B4 full-stream timing screen.

### Question
Can the 8-MiB full-stream plan outperform the selective `b4fast` policy?

### Hypothesis
The lower activation threshold may lower stalls enough to compensate for its extra copy volume.

### Importance
This is the remaining full-stream B4 alternative in the frozen expert action library.

### Evidence Needed
A checkpoint-10, operation-trace-bound, two-cycle timing run at 384x640.

### Refutation Condition
The weighted steady-state time is materially slower than `b4fast`.

### Allowed Code Changes
None; select the frozen `b4full` policy only.

### Commands
`DISTILL_M0_MAX_STEPS=20 DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 scripts/distillschedule/launch_m0_single_a100.sh b4full iteration89_m0_b4full_timing_screen_steps10_19 timing 0`

### Measured Results
Generator-update mean 121.401 s; Critic-only mean 57.488 s; official weighted mean 70.271 s/step. Five-step cycles were 355.104 and 347.602 s.

### Correctness Result
PASS for checkpoint boundary, trace identity, ten-step schedule, and finite losses.

### Performance Result
Measured screening result: 21.04% slower than the 58.058-s `b4fast` three-run mean.

### Interpretation
The full-stream/lower-threshold alternative is pruned from A-local contention.

### Files Changed
`artifacts/manifests/iteration_089_m0_b4full_timing_screen.json`; immutable raw logs/traces.

### Failure or Limitation
One formal screen is a pruning measurement, not a repetition aggregate.

### Gate Status
B4 full-stream screen COMPLETE; B3 window alternatives remain. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can either preregistered B3 window 2 or 4 match `b4fast`?

### Next Command
Run B3 window-2 and window-4 screens concurrently on the two nodes.

## Iteration 90

### Track / Milestone
Track A / M0 B3 window-2 timing attempt.

### Question
Can B3 window-2 match the current A-local candidate?

### Hypothesis
A larger streaming window may reduce copy exposure relative to window 1.

### Importance
Window 2 is a preregistered B3 candidate and cannot be skipped.

### Evidence Needed
One complete checkpoint-10 two-cycle timing run.

### Refutation Condition
Completion materially slower than `b4fast`.

### Allowed Code Changes
None.

### Commands
The standard `b3w2` checkpoint-10 timing launcher, run in a non-persistent SSH command session.

### Measured Results
The launcher received SIGINT (signal 2), returned 1, and did not produce a complete timing sample.

### Correctness Result
INVALIDATED; incomplete execution.

### Performance Result
Not Run. No policy conclusion is drawn.

### Interpretation
This was an orchestration/session-lifetime failure, not evidence against B3 window-2.

### Files Changed
`artifacts/manifests/iteration_090_m0_b3w2_external_sigint.json`; failed raw logs retained.

### Failure or Limitation
The experiment was tied to a short-lived command channel.

### Gate Status
B3 window-2 OPEN; M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does the same run complete under a persistent SSH shell?

### Next Command
Rerun as Iteration 92 in the retained node-0 SSH session.

## Iteration 91

### Track / Milestone
Track A / M0 B3 window-4 timing attempt.

### Question
Can B3 window-4 match the current A-local candidate?

### Hypothesis
The largest preregistered window may hide enough block materialization to approach `b4fast`.

### Importance
Window 4 is the strongest plausible B3 alternative and requires a retained screen.

### Evidence Needed
One complete checkpoint-10 two-cycle timing run.

### Refutation Condition
Completion materially slower than `b4fast`.

### Allowed Code Changes
None.

### Commands
The standard `b3w4` checkpoint-10 timing launcher, run in a non-persistent SSH command session.

### Measured Results
The launcher received SIGINT (signal 2), returned 1, and did not produce a complete timing sample.

### Correctness Result
INVALIDATED; incomplete execution.

### Performance Result
Not Run. No policy conclusion is drawn.

### Interpretation
This is a command-session failure, not a B3 policy result.

### Files Changed
`artifacts/manifests/iteration_091_m0_b3w4_external_sigint.json`; failed raw logs retained.

### Failure or Limitation
The experiment was tied to a short-lived command channel.

### Gate Status
B3 window-4 OPEN; M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does the same run complete under a persistent SSH shell?

### Next Command
Rerun as Iteration 93 in the retained node-1 SSH session.

## Iteration 92

### Track / Milestone
Track A / M0 B3 window-2 persistent timing screen.

### Question
Can B3 window-2 match or beat the current `b4fast` candidate?

### Hypothesis
Window 2 will improve on window 1 but remain slower than selective B4.

### Importance
This closes a preregistered B3 mechanism point with valid execution evidence.

### Evidence Needed
One complete checkpoint-10, trace-bound, two-cycle timing run.

### Refutation Condition
Window 2 matches or beats the `b4fast` aggregate.

### Allowed Code Changes
None; execution moved to a persistent SSH shell only.

### Commands
`DISTILL_M0_MAX_STEPS=20 DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 scripts/distillschedule/launch_m0_single_a100.sh b3w2 iteration92_m0_b3w2_timing_screen_steps10_19 timing 0`

### Measured Results
Generator-update mean 107.858 s; Critic-only mean 50.000 s; official weighted mean 61.572 s/step. Five-step cycles were 319.752 and 295.965 s.

### Correctness Result
PASS for checkpoint boundary, trace identity, ten-step schedule, and finite losses.

### Performance Result
Measured screen: 6.05% slower than the 58.058-s `b4fast` mean.

### Interpretation
Window 2 improves over B3 window 1 but remains outside the leading local policy.

### Files Changed
`artifacts/manifests/iteration_092_m0_b3w2_timing_screen.json`; immutable raw logs/traces.

### Failure or Limitation
Single screen; it is pruned based on a gap larger than the close-contender threshold, not presented as a repetition aggregate.

### Gate Status
B3 window-2 COMPLETE; B3 window-4 close-contender decision OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Is B3 window-4 statistically distinguishable from `b4fast`?

### Next Command
Complete and, if close, repeat the B3 window-4 timing run.

## Iteration 93

### Track / Milestone
Track A / M0 B3 window-4 persistent timing screen.

### Question
Can B3 window-4 match or beat the current `b4fast` candidate?

### Hypothesis
Window 4 will be the fastest coarse B3 policy but remain behind selective B4.

### Importance
A close result would require repetitions before freezing A-local.

### Evidence Needed
One complete checkpoint-10, trace-bound, two-cycle timing run.

### Refutation Condition
Window 4 matches or beats the `b4fast` aggregate.

### Allowed Code Changes
None; execution moved to a persistent SSH shell only.

### Commands
`DISTILL_M0_MAX_STEPS=20 DISTILL_M0_RESUME_FROM_CHECKPOINT=/share/project/shuchenweng/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10 scripts/distillschedule/launch_m0_single_a100.sh b3w4 iteration93_m0_b3w4_timing_screen_steps10_19 timing 0`

### Measured Results
Generator-update mean 102.296 s; Critic-only mean 48.423 s; official weighted mean 59.198 s/step. Five-step cycles were 281.370 and 310.605 s.

### Correctness Result
PASS for checkpoint boundary, trace identity, ten-step schedule, and finite losses.

### Performance Result
Measured screen: only 1.96% slower than the 58.058-s `b4fast` three-run mean.

### Interpretation
B3 window-4 is a genuine close contender; one screen cannot freeze the A-local winner. Two additional independent repetitions are required under the existing three-run contract.

### Files Changed
`artifacts/manifests/iteration_093_m0_b3w4_timing_screen.json`; immutable raw logs/traces.

### Failure or Limitation
One run and visible cycle variability (281.370 vs 310.605 s).

### Gate Status
B3 window-4 retained for repetitions. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
What is the three-run B3 window-4 aggregate relative to `b4fast`?

### Next Command
Run two additional checkpoint-10 B3 window-4 repetitions concurrently on the two nodes.

### Workspace Relocation (administrative, no evidence claim)

After Iterations 92/93 completed and no training process remained, the mutable repository was atomically moved to `/share/project/shuchenweng/yongyan_liu/DistillSchedule`. The old path `/share/project/shuchenweng/DistillSchedule` is retained as a compatibility symlink so immutable historical configs, checkpoint metadata, and raw-evidence paths continue to resolve. The active M0 launcher now derives its canonical root from its own location rather than hard-coding the legacy path.

## Iteration 94

### Track / Milestone
Track A / M0 B3 window-4 repetition attempt.

### Question
Does B3 window-4 remain a close A-local contender on a second run?

### Hypothesis
The operation-keyed trace selected by the launcher covers the resumed logical range 10--19.

### Importance
A timing repetition is valid only if it replays the exact registered operations for every measured step.

### Evidence Needed
Trace coverage of steps 10--19 followed by a complete two-cycle run.

### Refutation Condition
Any required operation key is absent before the first measured step.

### Allowed Code Changes
None during the run.

### Commands
The `b3w4` checkpoint-10 timing launcher selected its default `syn_feature_trace_v1` bundle.

### Measured Results
After checkpoint restoration, execution failed before step 10 with `KeyError: step/10/generator/entry_noise`; v1 contains only steps 0--9.

### Correctness Result
INVALIDATED; the requested execution trace did not cover the measured range.

### Performance Result
Not Run.

### Interpretation
Environment-only trace selection is too easy to omit and must fail before 14B construction.

### Files Changed
`artifacts/manifests/iteration_094_m0_b3w4_trace_coverage_mismatch.json`; raw failure logs retained.

### Failure or Limitation
Approximately twelve minutes of cold construction/checkpoint loading were wasted before the missing operation key was discovered.

### Gate Status
No B3 policy conclusion. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can manifest-only required-step validation reject this configuration before construction?

### Next Command
Add a fail-closed trace-range preflight and rerun with `syn_feature_trace_v2_steps20`.

## Iteration 95

### Track / Milestone
Track A / M0 B3 window-4 repetition attempt on node 1.

### Question
Does the third B3 window-4 timing run complete with the selected trace?

### Hypothesis
The operation-keyed trace covers resumed steps 10--19.

### Importance
The independent repetition must satisfy the same trace contract.

### Evidence Needed
Trace coverage followed by a complete two-cycle run.

### Refutation Condition
Any required operation key is absent.

### Allowed Code Changes
None during the run.

### Commands
The node-1 `b3w4` checkpoint-10 timing launcher also selected default `syn_feature_trace_v1`.

### Measured Results
The same pre-step-10 `KeyError: step/10/generator/entry_noise` occurred independently.

### Correctness Result
INVALIDATED.

### Performance Result
Not Run.

### Interpretation
The duplicate failure confirms a deterministic contract error, not a node-specific model failure.

### Files Changed
`artifacts/manifests/iteration_095_m0_b3w4_trace_coverage_mismatch.json`; raw failure logs retained.

### Failure or Limitation
No timing sample was produced.

### Gate Status
No B3 policy conclusion. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does the corrected v2 trace produce two valid independent repetitions?

### Next Command
Run Iterations 96/97 with explicit `DISTILL_M0_TRACE_BUNDLE=.../syn_feature_trace_v2_steps20`.

### Trace-Coverage Preflight Remediation

A manifest-only preflight now derives the resumed logical start from `checkpoint-N` and rejects any trace that does not cover `[start, max_train_steps)` before creating an immutable run directory or constructing either 14B backbone. Unit tests pass for exact, superset, missing-step, and invalid-range cases. An integration check rejected v1 for steps 10--19 with no artifact path consumed; v2 passed with trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.

## Iteration 96

### Track / Milestone
Track A / M0 B3 window-4 repetition attempt.

### Question
Does B3 window-4 remain competitive on node 0 under the corrected v2 trace?

### Hypothesis
A launcher invoked from a clean commit remains immutable until its postlude records the return code.

### Importance
Formal repetitions require both a complete training trace and an immutable executable wrapper.

### Evidence Needed
Ten completed steps, training-complete marker, and launcher rc=0 from unchanged source.

### Refutation Condition
The launcher source changes while its shell is still reading it.

### Allowed Code Changes
None should have occurred during the run.

### Commands
`b3w4` checkpoint-10 timing with explicit `syn_feature_trace_v2_steps20`.

### Measured Results
The training process completed all ten steps and emitted the final marker. Supplemental weighted time was 63.526 s/step, but the wrapper then hit a shell parse error and returned 2 because the launcher file had been rewritten in place during execution.

### Correctness Result
Training trace complete; formal run INVALIDATED by mutable launcher source.

### Performance Result
Excluded from the A-local aggregate.

### Interpretation
Editing a long-running shell launcher is unsafe even when the Python child has already imported its modules; bash may continue reading the file after the child exits.

### Files Changed
`artifacts/manifests/iteration_096_m0_b3w4_mutable_launcher_invalidated.json`; raw trace retained.

### Failure or Limitation
No `rc.txt` was written. This is an orchestration failure, not a B3 result.

### Gate Status
B3 repetition still OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can the same command complete while the worktree remains byte-stable?

### Next Command
Commit all instrumentation first, then rerun without editing the repository.

## Iteration 97

### Track / Milestone
Track A / M0 B3 window-4 repetition attempt on node 1.

### Question
Does the independent node-1 repetition complete under the corrected trace?

### Hypothesis
The launcher remains immutable through its postlude.

### Importance
This is the second required independent timing repetition.

### Evidence Needed
Complete trace plus wrapper rc=0.

### Refutation Condition
Concurrent launcher mutation changes the executing shell stream.

### Allowed Code Changes
None should have occurred.

### Commands
The corresponding node-1 `b3w4` checkpoint-10 v2-trace timing command.

### Measured Results
Training completed and supplemental weighted time was 55.265 s/step, but the same post-training shell parse error returned 2 with no `rc.txt`.

### Correctness Result
Training trace complete; formal run INVALIDATED.

### Performance Result
Excluded from the formal aggregate.

### Interpretation
The replicated failure confirms that both wrappers shared the same mutating source file. The performance variability remains informative only for sizing repetitions.

### Files Changed
`artifacts/manifests/iteration_097_m0_b3w4_mutable_launcher_invalidated.json`; raw trace retained.

### Failure or Limitation
The run cannot satisfy immutable-command provenance despite a complete Python trace.

### Gate Status
B3 repetitions OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
What is the valid three-run aggregate under a frozen worktree?

### Next Command
Run Iterations 98/99 from clean commit with no writes until both return rc=0.

## Iteration 98

### Track / Milestone
Track A / M0 B3 window-4 formal repetition 2.

### Question
What is B3 window-4's second valid checkpoint-10 timing sample under a frozen worktree?

### Hypothesis
Its weighted time remains within 5% of `b4fast`.

### Importance
A close contender requires the same three-repetition contract as the leading policy.

### Evidence Needed
Trace preflight PASS, wrapper rc=0, complete steps 10--19, and two five-step cycles.

### Refutation Condition
Failure or a repeated gap above 5%.

### Allowed Code Changes
None until both concurrent repetitions completed.

### Commands
Explicit v2 trace, checkpoint-10, `b3w4`, timing mode on node 0.

### Measured Results
Generator-update mean 107.532 s; Critic-only mean 47.145 s; official weighted mean 59.223 s/step. Cycles were 301.625 and 290.602 s. Wrapper rc=0.

### Correctness Result
PASS for preflight, checkpoint boundary, trace identity, update schedule, and final completion.

### Performance Result
Measured formal repetition.

### Interpretation
B3 window-4 remains close but does not beat the 58.058-s `b4fast` aggregate in this repetition.

### Files Changed
`artifacts/manifests/iteration_098_m0_b3w4_timing_rep2.json`; immutable raw logs/traces.

### Failure or Limitation
One sample within the three-run aggregate; no standalone significance claim.

### Gate Status
B3 repetition 2 COMPLETE; repetition 3 COMPLETE concurrently. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Which complete policy has the fastest three-run mean?

### Next Command
Aggregate Iterations 93/98/99 against 79/80/83.

## Iteration 99

### Track / Milestone
Track A / M0 B3 window-4 formal repetition 3.

### Question
What is the independent node-1 B3 window-4 timing sample?

### Hypothesis
The aggregate remains close to but slower than `b4fast`.

### Importance
This closes the preregistered close-contender timing requirement.

### Evidence Needed
Trace preflight PASS, rc=0, complete steps 10--19 and two cycles.

### Refutation Condition
A three-run mean below `b4fast`.

### Allowed Code Changes
None during execution.

### Commands
Explicit v2 trace, checkpoint-10, `b3w4`, timing mode on node 1.

### Measured Results
Generator-update mean 105.288 s; Critic-only mean 48.885 s; official weighted mean 60.166 s/step. Cycles were 295.826 and 305.832 s. Wrapper rc=0.

### Correctness Result
PASS.

### Performance Result
Measured formal repetition.

### Interpretation
Together with Iterations 93 and 98, B3 window-4 averages 59.529 ± 0.552 s/step.

### Files Changed
`artifacts/manifests/iteration_099_m0_b3w4_timing_rep3.json`; immutable raw logs/traces.

### Failure or Limitation
The comparison is an empirical hardware mean; it is not presented as a statistically significant 2.53% difference.

### Gate Status
B3 close-contender timing COMPLETE. A-local freeze ready.

### Next Highest-Value Question
Freeze the fastest measured single-A100 handwritten policy.

### Next Command
Create the A-local decision manifest.

## Iteration 100

### Track / Milestone
Track A / M0 A-local freeze.

### Question
Which complete 384x640 handwritten policy is fastest on one A100-40GB?

### Hypothesis
Selective B4 `b4fast` remains faster than the strongest coarse B3 alternative after equal repetitions.

### Importance
A-local must be frozen before its direct eight-rank admission test and before A-scalable selection.

### Evidence Needed
Three independent valid timing runs for both close candidates plus retained screens for the bounded alternatives.

### Refutation Condition
B3 window-4 has a lower three-run mean.

### Allowed Code Changes
Aggregation only.

### Commands
Aggregate manifests 79/80/83 and 93/98/99; retain B3/B4 screens 87/92/88/89.

### Measured Results
`b4fast`: 58.058 ± 1.126 s/step (median 57.450). `b3w4`: 59.529 ± 0.552 s/step (median 59.223), 2.53% slower by mean. B3w1, B3w2, B4s16 and B4full screens were 63.850, 61.572, 85.992 and 70.271 s/step respectively.

### Correctness Result
PASS: all accepted timing runs replayed the same checkpoint-10 v2 trace and completed the official 1:4 update pattern.

### Performance Result
A-local FROZEN as `b4fast`: B0a + split modulation + selective activation offload (32 MiB threshold) + all frozen blocks streamed with window 1 + private host backing + no allocator trim.

### Interpretation
The fastest local baseline is a selective expert plan, but its single-rank cgroup peak is 151.639 GB and must not be assumed eight-way scalable.

### Files Changed
`artifacts/manifests/iteration_100_m0_a_local_freeze.json`; `research_log.md`.

### Failure or Limitation
The 2.53% difference is small; the freeze is an empirical fastest-policy decision, not a novelty or significance claim.

### Gate Status
A-local COMPLETE. A-local x8 admission and A-scalable remain OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can A-local's private long-lived host state admit eight ranks with the required 64-GiB reserve, and can shared immutable backing make a scalable policy admit?

### Next Command
Run A-local `b4fast/HOST_PRIVATE` wave-1 admission and `b4fast/HOST_SHARED_IMMUTABLE` wave-1 admission concurrently on the two nodes.

## Iteration 101

### Track / Milestone
Track B admission probe / M0 A-local x8.

### Question
Can frozen A-local (`b4fast`, `HOST_PRIVATE`) be copied to all eight ranks on one node with wave size 1?

### Hypothesis
Long-lived private host state will either exceed the 760-GiB cgroup or violate the required 64-GiB reserve.

### Importance
The master contract requires the fastest single-card plan's direct eight-way admission result even if it is not A-scalable.

### Evidence Needed
Eight rank-qualified traces through steps 0--9, node cgroup/RSS/PSS/private/shared peaks, memory-event deltas, and wrapper rc.

### Refutation Condition
All eight ranks finish with no OOM and node peak at or below 696 GiB (760 GiB minus the 64-GiB reserve).

### Allowed Code Changes
None during the run.

### Commands
`launch_m0_node_dp.sh b4fast iteration101_m0_alocal_private_x8_wave1_admission memory 1` with `HOST_PRIVATE` on node 0.

### Measured Results
Trace preflight passed for steps 0--9. All eight ranks constructed the real Generator and Score and reached ZeRO optimizer initialization. Node cgroup peak was 763,978,223,616 bytes; no new cgroup OOM kill occurred. The run then returned rc=1 because the one-file dataset yielded zero per-rank batches and `num_update_steps_per_epoch=0`.

### Correctness Result
INVALID / not reached: no logical training step executed.

### Performance Result
INVALID: construction/resource evidence is retained, but this is not an admission verdict.

### Interpretation
The attempted direct replication already crossed the 64-GiB reserve threshold, but the training admission question remains open because an input-cardinality bug terminated before steady state.

### Files Changed
Raw immutable logs/traces and `artifacts/manifests/iteration_101_m0_alocal_private_x8_wave1_admission.json`.

### Failure or Limitation
The single-rank SYN-FEATURE directory was incorrectly reused for eight manually sharded ranks.

### Gate Status
A-local x8 OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can the same plan execute after supplying one trace-compatible logical sample per rank?

### Next Command
Create a deterministic rank-8 hardlink dataset, fail fast on cardinality, and rerun.

## Iteration 102

### Track / Milestone
Track B admission probe / M0 A-scalable candidate.

### Question
Can `b4fast` with `HOST_SHARED_IMMUTABLE` admit eight ranks with wave size 1?

### Hypothesis
Shared immutable frozen-block backing reduces long-lived private PSS enough to make the per-rank policy scalable.

### Importance
Wave scheduling cannot repair long-lived state; shared backing is the first admissible A-scalable mechanism from the frozen action library.

### Evidence Needed
Eight completed rank traces, node PSS/private/shared breakdown, cgroup peak plus 64-GiB reserve, and no leaks.

### Refutation Condition
The complete run OOMs, violates reserve, or fails correctness.

### Allowed Code Changes
None during the run.

### Commands
`launch_m0_node_dp.sh b4fast iteration102_m0_b4fast_shared_x8_wave1_admission memory 1` with `HOST_SHARED_IMMUTABLE` on node 1.

### Measured Results
Trace preflight passed. All eight ranks constructed both 14B backbones and reached optimizer initialization. Node cgroup peak was 761,585,369,088 bytes with zero new OOM kills. The run returned rc=1 at the same zero-batch division as Iteration 101.

### Correctness Result
INVALID / not reached.

### Performance Result
INVALID: retained only as construction and host-accounting evidence.

### Interpretation
During preparation, shared backing reduced sampled eight-rank PSS/private state substantially versus private backing, but no A-scalable conclusion is allowed until full training executes.

### Files Changed
Raw immutable logs/traces and `artifacts/manifests/iteration_102_m0_b4fast_shared_x8_wave1_admission.json`.

### Failure or Limitation
The same one-file dataset cardinality bug invalidated the admission run.

### Gate Status
A-scalable OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can the fixed rank-8 data contract carry both candidates through steps 0--9?

### Next Command
Rerun private and shared candidates after Iteration 103.

## Iteration 103

### Track / Milestone
Track B substrate / M0 input-contract repair.

### Question
Why did both eight-rank runs reach model preparation but produce zero training batches?

### Hypothesis
`BucketedSampler` manually shards the single feature over eight ranks with `drop_last=True`, so each rank receives zero samples.

### Importance
Without a cardinality preflight, each bad launch wastes several minutes and hundreds of GiB before failing for a non-scheduler reason.

### Evidence Needed
Sampler source, dataset cardinality, a deterministic eight-sample asset, launcher fail-fast validation, and tests.

### Refutation Condition
Eight trace-compatible logical files still yield zero batches before training.

### Allowed Code Changes
Dataset preparation utility and node launcher input resolution only; no training semantics or memory policy changes.

### Commands
Create eight hardlinks to the immutable source-shaped feature; point only the node-DP resolved config to this root; run the selected M0 tests.

### Measured Results
Created eight logical `.pt` samples backed by one inode and the exact source SHA-256 `cb6ba67f4a5b06727c94cfc8fcc4d6130c4d8c570292fdb880b9c194cfc4ea60`. The launcher now fails before 14B construction if fewer than eight feature files exist. Selected regression suite: 44 passed, 0 failed.

### Correctness Result
PASS for deterministic physical data identity and cardinality preflight.

### Performance Result
Not Run; this iteration changes only input admission.

### Interpretation
The previous failure was refuted as a scheduler failure: it was an input cardinality bug. Eight logical samples can share immutable bytes without inflating host storage.

### Files Changed
`scripts/distillschedule/prepare_m0_rank_dataset.py`; `scripts/distillschedule/launch_m0_node_dp.sh`; `tests/test_m0_rank_dataset.py`; `artifacts/manifests/iteration_103_m0_rank8_dataset_contract.json`; `research_log.md`.

### Failure or Limitation
The eight logical samples intentionally contain identical SYN-FEATURE bytes for resource admission. This is not a diversity or convergence experiment.

### Gate Status
Input substrate repaired. A-local x8 and A-scalable remain OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Do private and shared host policies complete the official update schedule under the repaired contract?

### Next Command
Run Iterations 104 and 105 concurrently on the two eight-A100 nodes.

## Iteration 104

### Track / Milestone
Track B admission probe / M0 A-local x8 retry.

### Question
Does the repaired eight-sample cardinality contract carry private A-local into training?

### Hypothesis
Each rank will validate the common source feature and begin step 0.

### Importance
This separates dataset cardinality from host admission.

### Evidence Needed
Eight input validations followed by complete steps 0--9.

### Refutation Condition
Any rank rejects its trace before the first model operation.

### Allowed Code Changes
None during execution.

### Commands
Private `b4fast`, wave 1, memory mode, eight ranks on node 0.

### Measured Results
The dataset reported eight examples and one batch per rank. Construction and both optimizer preparations completed. Node cgroup peak was 757,902,880,768 bytes with no new OOM kill. Six ranks explicitly rejected the primary tensor against the single rank-0 trace; remaining ranks were terminated by the elastic launcher.

### Correctness Result
INVALID: operation trace was not sample-qualified.

### Performance Result
INVALID; retained as a failed preflight and resource trace only.

### Interpretation
Hardlink byte identity is insufficient because `prepare_stage1_latent` uses the dataset index to select a video section.

### Files Changed
Raw evidence and `artifacts/manifests/iteration_104_m0_alocal_private_x8_wave1_admission.json`.

### Failure or Limitation
The trace contract still represented only dataset index 0.

### Gate Status
A-local x8 OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can operation replay be made rank/sample-qualified without weakening exact input validation?

### Next Command
Build and validate an eight-entry trace bundle.

## Iteration 105

### Track / Milestone
Track B admission probe / M0 shared-host retry.

### Question
Does the repaired cardinality contract carry the shared-host candidate into training?

### Hypothesis
The shared plan will begin step 0 once every rank receives one logical sample.

### Importance
A-scalable cannot be judged from construction-only evidence.

### Evidence Needed
Eight exact input validations and complete training steps.

### Refutation Condition
Rank-specific input drift or resource failure.

### Allowed Code Changes
None during execution.

### Commands
Shared-immutable `b4fast`, wave 1, memory mode, eight ranks on node 1.

### Measured Results
Eight examples and one batch per rank were admitted; construction and optimizer setup completed. Cgroup peak was 764,611,792,896 bytes with zero new OOM kills. The same rank-indexed target mismatch terminated the run before model execution.

### Correctness Result
INVALID: trace/data identity failed by design rather than numerical drift.

### Performance Result
INVALID.

### Interpretation
The failure replicated across host policies, proving it is independent of residency policy.

### Files Changed
Raw evidence and `artifacts/manifests/iteration_105_m0_b4fast_shared_x8_wave1_admission.json`.

### Failure or Limitation
Single-sample operation trace.

### Gate Status
A-scalable OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can each rank replay its exact sampler output and operation sequence?

### Next Command
Iteration 106 rank-qualified trace bundle.

## Iteration 106

### Track / Milestone
Track B substrate / M0 operation-trace correction.

### Question
How should eight DP ranks replay exact inputs when Helios sampling depends on dataset index?

### Hypothesis
A bundle containing one immutable operation trace per rank/sample will pass exact primary/secondary validation while preserving independent operation-keyed randomness.

### Importance
This directly enforces the master contract and prevents scheduling from changing RNG consumption or sample selection.

### Evidence Needed
Contiguous rank manifest, per-rank trace hashes, deterministic file order, exact target comparisons, coverage of steps 0--9, and tests.

### Refutation Condition
Any sorted rank input differs from its resolved trace or any trace misses a logical step.

### Allowed Code Changes
Trace resolution/generation/preflight and deterministic dataset enumeration only; no loss, model, optimizer, or memory-policy changes.

### Commands
Generate eight trace roots from official sampler indices 0--7; build a hashed bundle manifest; run coverage and input checks before 14B construction.

### Measured Results
Generated an eight-rank 983-MiB bundle with bundle hash `48498f4da9ad5836bf0a17e336e21c101b3194492d9b2808486e0e0e6ec68a18`. All eight traces cover steps 0--9, and each rank's primary/secondary target is SHA-exact with the sorted hardlink dataset. Targeted tests: 12 passed; broader selected suite before the final preflight change: 47 passed.

### Correctness Result
PASS for rank resolution, bundle tamper checks, deterministic sample ordering, step coverage, and exact input replay.

### Performance Result
Not Run; substrate-only change.

### Interpretation
The execution contract is now genuinely sample-qualified rather than seed-only or rank-0-only.

### Files Changed
`runtime/operation_trace.py`; `train_helios.py`; `helios/dataset/dataloader_dmd.py`; trace preparation/check scripts; launcher; tests; `artifacts/manifests/iteration_106_m0_rank_trace_bundle.json`.

### Failure or Limitation
The eight source-shaped samples share immutable feature bytes; they differ through official index-dependent section selection and operation traces, not source-video diversity. The first two launcher preflight invocations (reserved run IDs 107/108) exited before creating run artifacts because the checker lacked a self-contained repo import path; the checker was fixed and then passed without `PYTHONPATH`.

### Gate Status
Trace substrate PASS. A-local x8 and A-scalable OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Do the two host policies now pass exact input validation and complete the official update schedule?

### Next Command
Run Iterations 107 and 108 concurrently with the frozen rank trace bundle.

## Iteration 109

### Track / Milestone
Track B / M0 direct A-local x8 admission.

### Question
Does frozen A-local run unchanged on eight ranks after the rank-qualified trace repair?

### Hypothesis
The private host plan may fail node reserve or backward even if all inputs validate.

### Importance
The contract requires the direct replication result, not an inferred eight-times estimate.

### Evidence Needed
Eight exact input validations, backward outcome, cgroup peak, and OOM events.

### Refutation Condition
All ranks complete steps 0--9 within the reserve.

### Allowed Code Changes
None during execution.

### Commands
A-local `b4fast`, private host, ZeRO-2 config unchanged, wave 1, memory mode.

### Measured Results
All 8/8 rank-specific input traces validated and Generator backward began. Ranks then failed inside ZeRO-2 with `Gradient computed twice for this partition`; cgroup peak was 757,807,587,328 bytes, reserve failed, and no new host/GPU OOM was recorded.

### Correctness Result
FAIL at first Generator backward; no optimizer step completed.

### Performance Result
Direct A-local x8 admission is negative.

### Interpretation
The single-rank ZeRO-2-shaped substrate changes behavior when world size becomes eight: stage-2 per-parameter reduction is incompatible with Helios reentrant checkpoint/parameter reuse. Direct replication also misses the 64-GiB node reserve.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_109_m0_alocal_private_x8_wave1_admission.json`.

### Failure or Limitation
This is a backend-semantics failure before a steady host-state measurement.

### Gate Status
Required A-local x8 admission COMPLETE as a negative result. A-scalable OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can a replicated-parameter backend without ZeRO-2 gradient partition hooks complete backward?

### Next Command
Test ZeRO-1 CPU optimizer.

## Iteration 110

### Track / Milestone
Track B / M0 shared-host ZeRO-2 candidate.

### Question
Does shared immutable backing repair the direct plan once exact traces pass?

### Hypothesis
It repairs long-lived host state but not a ZeRO-2 gradient-hook incompatibility.

### Importance
Host residency and distributed-backend validity must be separated.

### Evidence Needed
Same trace and backward evidence under only the host-policy change.

### Refutation Condition
Shared policy completes backward while private fails.

### Allowed Code Changes
None during execution.

### Commands
`b4fast`, shared immutable host, ZeRO-2, wave 1, memory mode.

### Measured Results
All 8/8 input traces validated. Sampled training-state cgroup usage dropped to roughly 125--193 GB versus roughly 558--655 GB for private backing, but Generator backward hit the identical ZeRO-2 double-reduction assertion. Construction cgroup peak was 764,232,949,760 bytes; reserve failed; no new OOM kill.

### Correctness Result
FAIL at the same backend hook.

### Performance Result
Invalid for throughput; useful host-state evidence only.

### Interpretation
Shared immutable backing is effective for long-lived host memory, but cannot fix gradient semantics. Construction concurrency remains a separate reserve issue.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_110_m0_b4fast_shared_x8_wave1_admission.json`.

### Failure or Limitation
No completed step.

### Gate Status
Shared host policy retained; ZeRO-2 rejected for A-scalable. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can ZeRO-1 preserve replicated parameters and CPU optimizer while tolerating repeated gradient production?

### Next Command
Create a frozen ZeRO-1 CPU-optimizer candidate and run private/shared admission.

## Iteration 111

### Track / Milestone
Track B substrate / M0 A-scalable backend candidate.

### Question
What is the minimal backend change that removes ZeRO-2's repeated-gradient reduction hook without parameter sharding?

### Hypothesis
ZeRO-1 keeps full parameters and gradients replicated, partitions only optimizer state, and therefore tolerates Helios checkpoint recomputation while retaining CPU optimizer offload.

### Importance
A-scalable must stay in the full-parameter DP family; jumping to ZeRO-3/FSDP would violate the current M0 scope.

### Evidence Needed
A stage-1 config with CPU optimizer/no parameter offload, immutable resolved config provenance, and full backward execution.

### Refutation Condition
The same assertion recurs or GPU/host memory becomes infeasible.

### Allowed Code Changes
Add the ZeRO-1 config and node-launcher selection only; no model/loss/checkpoint changes.

### Commands
Set `DISTILL_M0_DP_ZERO_STAGE=1`; run selected tests before real launch.

### Measured Results
The new config has ZeRO stage 1, CPU optimizer offload, and no parameter offload. Selected tests: 33 passed, 0 failed.

### Correctness Result
Static contract PASS; real execution pending.

### Performance Result
Not Run.

### Interpretation
This is a baseline scale-out compatibility candidate, not a DistillSchedule optimization and not parameter sharding.

### Files Changed
`scripts/accelerate_configs/zero1_cpu_optimizer_offload.json`; node launcher; test; research log.

### Failure or Limitation
ZeRO-1 increases replicated gradient memory relative to ZeRO-2 and may OOM A100-40GB.

### Gate Status
A-scalable backend candidate READY. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does ZeRO-1 complete Generator backward and optimizer initialization on eight A100s?

### Next Command
Run private and shared ZeRO-1 candidates concurrently.

## Iteration 112

### Track / Milestone
Track B / M0 direct A-local x8 under replicated ZeRO-1.

### Question
Does removing ZeRO-2's per-parameter reduction hook let the unchanged private-host A-local family complete the full official update schedule?

### Hypothesis
ZeRO-1 will complete backward and both first/steady optimizer updates, but eight private host copies will violate the mandatory 64-GiB node reserve.

### Importance
The required A-local x8 admission result must distinguish backend semantics from long-lived host feasibility.

### Evidence Needed
Eight exact trace validations, steps 0--9, Generator updates at 0/5, ten Critic updates, final zero activation/streamer liveness, cgroup peak, and OOM delta.

### Refutation Condition
A backend assertion/OOM occurs, or the run both completes and passes the reserve.

### Allowed Code Changes
None during execution.

### Commands
`b4fast`, `HOST_PRIVATE`, replicated ZeRO-1 plus CPU optimizer, eight ranks, swap wave 1, memory mode.

### Measured Results
PASS for execution: all eight ranks completed steps 0--9, Generator updates at 0/5, ten Critic backward/optimizer steps, and ten EMA steps. Peak CUDA allocated/reserved on rank 0 was 33,727,460,864 / 36,482,056,192 bytes. Node cgroup peak was 764,940,455,936 bytes, above the 760-GiB-minus-64-GiB admission limit; reserve failed with zero new OOM kills. Training-state tail median was 720,673,159,168 bytes. Memory-mode mean was 146.21 s/logical step (273.02 s G-update; 114.50 s Critic-only) and is not formal timing evidence.

### Correctness Result
PASS for the step/update/liveness contract. Distinct trace hash count was eight; activation and streamer final live bytes were zero on every rank.

### Performance Result
NEGATIVE admission: direct A-local x8 completes but does not preserve the required node reserve.

### Interpretation
The ZeRO-2 failure was a reduction-hook incompatibility, not a Helios semantic failure. Replicated ZeRO-1 fixes it, while private immutable backbone copies remain unscalable.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_112_m0_b4fast_zero1_private_x8_wave1.json`.

### Failure or Limitation
Memory-mode timings include instrumentation and cannot freeze Track-B throughput.

### Gate Status
Required A-local x8 admission COMPLETE as a reserve-negative result. A-scalable OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does shared immutable host backing satisfy long-lived state and full correctness under the same ZeRO-1 backend?

### Next Command
Summarize Iteration 113.

## Iteration 113

### Track / Milestone
Track B / M0 A-scalable shared-host ZeRO-1 candidate.

### Question
Can shared immutable frozen-block backing complete steps 0--9 while keeping the eight-rank steady state below the node budget?

### Hypothesis
The run will complete with a much lower long-lived cgroup footprint, but simultaneous cold model construction may still violate the reserve.

### Importance
A-scalable needs both complete training semantics and separate accounting of steady versus transient host state.

### Evidence Needed
The same eight-rank correctness evidence as Iteration 112, steady cgroup footprint, cold peak, and OOM delta.

### Refutation Condition
Training fails, steady state exceeds the admission limit, or sharing has no measurable host effect.

### Allowed Code Changes
None during execution.

### Commands
`b4fast`, `HOST_SHARED_IMMUTABLE`, replicated ZeRO-1 plus CPU optimizer, eight ranks, swap wave 1, memory mode.

### Measured Results
PASS for execution: all eight ranks completed steps 0--9 with the required Generator/Critic/EMA counts and eight distinct exact input traces. Training-state tail median was 346,491,676,672 bytes, versus 720,673,159,168 bytes for private backing (374,181,482,496 bytes lower). Rank-0 CUDA peak matched the private run at 33,727,460,864 allocated and 36,482,056,192 reserved bytes. Simultaneous cold construction still reached 762,136,793,088 cgroup bytes, so the reserve failed despite zero new OOM kills. Memory-mode mean was 148.61 s/logical step (303.33 s G-update; 109.94 s Critic-only), not formal timing evidence.

### Correctness Result
PASS for steps 0--9, two Generator updates, ten Critic updates, ten EMA steps, exact trace identity, and zero final activation/streamer liveness.

### Performance Result
PARTIAL admission: long-lived state passes comfortably, but cold construction concurrency violates the 64-GiB reserve.

### Interpretation
Wave scheduling cannot repair long-lived private copies, but shared immutable backing does. The remaining A-scalable failure is a bounded cold-construction transient, so a separate static construction wave is justified under Track B.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_113_m0_b4fast_zero1_shared_x8_wave1.json`.

### Failure or Limitation
The peak occurs before shared block streaming is installed; a model-swap wave does not control this phase.

### Gate Status
A-scalable semantics and steady state PASS; cold admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can a node-local construction wave reduce the cold peak below 696 GiB without changing any per-rank training policy?

### Next Command
Implement and test a static backbone-construction wave, then rerun the shared candidate.

## Iteration 114

### Track / Milestone
Track B substrate / M0 cold-construction admission.

### Question
What is the minimal Track-B mechanism that bounds simultaneous 14B `from_pretrained` transients without changing per-rank state policy?

### Hypothesis
Serializing Generator and Score construction by node-local rank slots will preserve identical full replicas and reduce the cold peak; matching local-rank slots can execute concurrently on two nodes.

### Importance
Iteration 113 already proves the long-lived shared plan fits. A-scalable cannot be frozen until the cold peak also preserves 64 GiB.

### Evidence Needed
Fail-closed option parsing, exactly one constructor invocation per local rank, barriers around every slot, phase provenance, unit tests, and a real eight-rank memory run.

### Refutation Condition
The wave deadlocks, changes the model contract, or peak remains above the admission limit.

### Allowed Code Changes
Static construction admission only; no activation, streaming, loss, optimizer, trace, or swap-policy changes.

### Commands
Add `DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE`, unit-test local rank slot execution, then launch the Iteration-113 plan with construction wave 1.

### Measured Results
Static implementation complete. The helper traverses common slots with pre/post barriers and invokes the factory exactly once on the owning rank. Selected tests: 37 passed, 0 failed. Real execution pending.

### Correctness Result
Static/unit contract PASS.

### Performance Result
Not Run.

### Interpretation
Construction admission is now explicitly separate from steady model-swap admission, matching the node-memory contract.

### Files Changed
`runtime/official_baseline.py`; `train_helios.py`; `scripts/distillschedule/launch_m0_node_dp.sh`; `tests/test_official_baseline.py`; research log; Iteration 112/113 manifests.

### Failure or Limitation
A barrier-based wave assumes matching local-world sizes across nodes; the launcher records the chosen size and real run must prove progress.

### Gate Status
A-scalable candidate READY for cold-admission rerun. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does construction wave 1 bring the real shared eight-rank peak below the reserve while retaining steps 0--9?

### Next Command
Run Iteration 115: shared immutable, ZeRO-1, construction wave 1, swap wave 1, memory mode.

## Iteration 115

### Track / Milestone
Track B / M0 A-scalable cold admission, construction wave 1.

### Question
Does serial per-node backbone construction eliminate the cold reserve violation while preserving the exact eight-rank training contract?

### Hypothesis
Wave 1 will pass the reserve and steps 0--9, at a substantial cold-start cost.

### Importance
This is the safest static admission point and validates the construction-wave primitive before increasing concurrency.

### Evidence Needed
Peak cgroup memory, construction span, zero OOM delta, eight exact traces, complete update counts, and final zero liveness.

### Refutation Condition
Deadlock, reserve failure, or semantic divergence.

### Allowed Code Changes
None during execution.

### Commands
Shared-immutable `b4fast`, ZeRO-1 CPU optimizer, construction wave 1, swap wave 1, eight ranks, memory mode.

### Measured Results
PASS. All eight ranks completed steps 0--9. Cgroup peak was 611,466,653,696 bytes against a 747,324,309,504-byte admission limit; zero new OOM kills. Construction span was 3,363.33 s and total monitored span 4,924.78 s. Training-state tail median was 345,911,146,496 bytes. Memory-mode logical-step mean was 137.95 s and is not formal timing.

### Correctness Result
PASS: eight trace hashes; Generator updates 0/5; ten Critic and EMA steps; activation/streamer final live bytes zero on every rank.

### Performance Result
PASS for capacity; cold construction is slow.

### Interpretation
The remaining Iteration-113 failure was transient concurrency, not steady-state capacity. The safe wave can be widened.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_115_m0_b4fast_zero1_shared_constructwave1_x8.json`.

### Failure or Limitation
Memory mode and wave-1 cold time are not a final throughput result.

### Gate Status
A-scalable has one admitted point; fastest safe wave still OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can construction wave 2 preserve reserve while cutting cold time?

### Next Command
Iteration 116, same plan with construction wave 2.

## Iteration 116

### Track / Milestone
Track B / M0 A-scalable cold admission, construction wave 2.

### Question
Does doubling construction concurrency remain below the reserve and reduce cold start?

### Hypothesis
Wave 2 will remain admitted and roughly halve construction time relative to wave 1.

### Importance
Track B requires the best static node-level admission rather than the most conservative one.

### Evidence Needed
Same correctness/capacity evidence as Iteration 115 plus construction span.

### Refutation Condition
Peak exceeds the limit or construction does not improve.

### Allowed Code Changes
None during execution.

### Commands
Shared-immutable `b4fast`, ZeRO-1 CPU optimizer, construction wave 2, swap wave 1, eight ranks, memory mode.

### Measured Results
PASS. Cgroup peak was 594,250,063,872 bytes, reserve passed with zero OOM kills. Construction span was 1,704.48 s, 49.3% shorter than wave 1; total monitored span was 3,463.40 s. Tail median was 341,838,530,560 bytes. All 80 rank-steps completed. Memory-mode step mean was 157.96 s and is not formal timing.

### Correctness Result
PASS for all rank traces, update counts, and final liveness.

### Performance Result
Capacity PASS; cold construction faster than wave 1.

### Interpretation
Measured peaks are not monotonic enough for a purely additive prediction, so wave 4 must be measured rather than inferred.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_116_m0_b4fast_zero1_shared_constructwave2_x8.json`.

### Failure or Limitation
No formal steady timing claim.

### Gate Status
A-scalable admitted; fastest safe wave OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Is wave 4 the largest safe candidate, given that wave 8 already failed reserve in Iteration 113?

### Next Command
Iteration 117, construction wave 4.

## Iteration 117

### Track / Milestone
Track B / M0 A-scalable cold admission, construction wave 4.

### Question
Does wave 4 remain below the reserve and become the fastest safe construction point among 1/2/4/8?

### Hypothesis
Wave 4 will pass; wave 8 is already a measured reserve failure, so wave 4 will close the bounded search.

### Importance
This freezes the node-level cold admission policy without unbounded tuning.

### Evidence Needed
Full steps 0--9, exact traces, peak below 747,324,309,504 bytes, construction span, and comparison to waves 1/2/8.

### Refutation Condition
Reserve failure or semantic divergence.

### Allowed Code Changes
None during execution.

### Commands
Shared-immutable `b4fast`, ZeRO-1 CPU optimizer, construction wave 4, swap wave 1, eight ranks, memory mode.

### Measured Results
PASS. Cgroup peak was 646,712,553,472 bytes, leaving 100,611,756,032 bytes below the admission limit; zero OOM kills. Construction span was 882.89 s, 48.2% shorter than wave 2 and 73.7% shorter than wave 1. Total monitored span was 2,467.75 s. Tail median was 344,847,491,072 bytes. Wave 8/no construction admission previously peaked at 762,136,793,088 bytes and failed reserve. Memory-mode step mean was 143.02 s and is not formal timing.

### Correctness Result
PASS for eight exact traces, steps 0--9, Generator updates 0/5, ten Critic/EMA steps, and zero final activation/streamer liveness.

### Performance Result
Wave 4 is the largest measured safe point and fastest cold start in the preregistered 1/2/4/8 set.

### Interpretation
A-scalable candidate is now: shared immutable frozen blocks + replicated ZeRO-1 gradients/parameters + CPU optimizer + construction wave 4 + model-swap wave 1. This is baseline scale-out engineering, not DistillSchedule planner gain.

### Files Changed
Raw logs/traces and `artifacts/manifests/iteration_117_m0_b4fast_zero1_shared_constructwave4_x8.json`.

### Failure or Limitation
Formal timing, canonical checkpoint, EMA 748--752, and the 16-rank two-node run remain open.

### Gate Status
A-scalable admission candidate FROZEN; formal freeze OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
What cross-node transport is actually available, and can the frozen per-node plan run as 16-rank pure DP without changing per-rank policy?

### Next Command
Measure Ethernet/RDMA NCCL bandwidth, then launch a 16-rank connectivity probe and full Track-B run.

## Iteration 118

### Track / Milestone
Track B / M0 two-node transport gate.

### Question
What transport do the two jobs actually use, and what bandwidth is available for 16-rank pure-DP collectives?

### Hypothesis
The jobs have active InfiniBand/RDMA despite eth0 exposing only a 10-Gb/s link; NCCL should select GPUDirect RDMA and materially exceed socket transport.

### Importance
The 16-rank baseline must not assume InfiniBand or infer network cost from link names.

### Evidence Needed
Two-node NCCL logs naming the selected transport/HCA/GDR mode and measured all-reduce latency across payload sizes, plus a forced-socket control.

### Refutation Condition
NCCL selects sockets only, RDMA fails, or measured RDMA bandwidth is not materially above Ethernet.

### Allowed Code Changes
A standalone non-repo probe and immutable raw evidence only; no Helios/runtime changes.

### Commands
One GPU/rank per node, PyTorch 2.10.0+cu128 NCCL all-reduce for 1 MiB, 64 MiB, 256 MiB, and 1 GiB payloads; three measured repeats. Run auto RDMA and forced `NCCL_NET=Socket` over eth0.

### Measured Results
Verified GPUDirect RDMA through NCCL RDMA Plugin v6 over active `mlx5_2`, `mlx5_5`, `mlx5_8`, and `mlx5_9`; eth0 is OOB. At 1 GiB the median payload bandwidth was 9.754 GB/s (78.03 Gb/s), with 110.08 ms all-reduce latency. Forced socket over eth0 measured 0.1148 GB/s (0.918 Gb/s), 9.354 s latency. RDMA was 84.97x faster for the largest payload. The 64/256-MiB RDMA points were 76.76/77.42 Gb/s. Setting only `NCCL_IB_DISABLE=1` did not disable the external plugin and remained on RDMA; `NCCL_NET=Socket` was required.

### Correctness Result
PASS: both ranks completed every collective and gathered mutually consistent timing samples.

### Performance Result
Measured transport characterization only; not a 16-rank application throughput result.

### Interpretation
Cross-node communication is not limited to the 10-Gb/s eth0 path. The real 16-rank baseline should retain NCCL's RDMA plugin and must record transport logs; forcing sockets would be an invalidly weak baseline.

### Files Changed
`artifacts/logs/iteration118_network/*`; `artifacts/manifests/iteration_118_m0_network_probe.json`; research log.

### Failure or Limitation
Two ranks and one GPU per node; full 16-rank collective behavior remains unmeasured.

### Gate Status
Two-node transport gate PASS. A-scalable formal/16-rank validation OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can 16 distinct logical samples and operation traces be replayed under one 16-rank collective-safe full-pipeline launch?

### Next Command
Build a 16-sample hardlink dataset and 16-rank trace bundle, validate exact replay, then run a 16-rank connectivity probe.

## Iteration 119

### Track / Milestone
Track B substrate / M0 16-sample execution contract.

### Question
Can the two-node run use 16 distinct operation-keyed logical samples rather than repeating the eight-rank bundle?

### Hypothesis
The existing rank-qualified builders generalize to 16 global ranks and preserve exact official index-dependent primary/secondary selections.

### Importance
Track B is defined as 16 independent full-pipeline logical samples; rank 8--15 may not alias rank 0--7 trace identity.

### Evidence Needed
Sixteen hardlink dataset entries, a contiguous 16-rank trace manifest, sixteen distinct trace hashes, steps 0--9 coverage, exact primary/secondary replay, and tests.

### Refutation Condition
Any global rank lacks a trace, any input hash differs, or coverage is incomplete.

### Allowed Code Changes
No runtime changes; generate immutable external assets and a summary manifest only.

### Commands
Build `syn_feature_rank16` and `syn_feature_trace_rank16_v1` with official sampler indices 0--15; validate coverage and exact inputs under perturbed ambient state.

### Measured Results
PASS. The dataset contains 16 hardlinks to source SHA-256 `cb6ba67f...ea60`. The 2.0-GiB trace bundle has bundle hash `a79d68d36086d97797fdb0c28ee034e558a0b28a6bb92d3bce519d531156d496`, 16 distinct trace hashes, and complete steps 0--9 coverage. All 16 primary/secondary targets are SHA-exact. Targeted tests passed.

### Correctness Result
PASS for global ranks 0--15 and exact replay independent of seed-only assumptions.

### Performance Result
Not Run; execution-contract substrate only.

### Interpretation
The 16-rank launch can now bind one distinct source-shaped logical sample and operation sequence to every global rank.

### Files Changed
External assets under `DistillSchedule_assets/m0/{datasets,traces}`; `artifacts/manifests/iteration_119_m0_rank16_trace_bundle.json`; research log.

### Failure or Limitation
All samples hardlink the same immutable SYN-FEATURE source and differ through official index-dependent sampling/trace; this is runtime profiling, not dataset-diversity or convergence evidence.

### Gate Status
16-rank input contract PASS. Full two-node launch OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can 16 ranks initialize one collective group and resolve their global-rank traces before expensive construction?

### Next Command
Implement a two-node launcher with per-node immutable artifacts and run a cheap 16-rank pre-construction collective/trace probe.

## Iteration 120

### Track / Milestone
Track B / M0 16-rank pre-construction contract.

### Question
Can all 16 ranks initialize one cross-node collective group, bind distinct global-rank operation traces, and use the measured RDMA path before any 14B construction cost?

### Hypothesis
The rank-16 bundle and two-node NCCL rendezvous will produce one exact trace mapping per global rank and a correct CUDA collective over GPUDirect RDMA.

### Importance
A full 14B two-node launch is expensive. This probe separates trace/rendezvous errors from model construction, host admission, and training failures.

### Evidence Needed
Sixteen verified tensor payloads, expected global-rank-to-host mapping, sixteen distinct expected trace hashes, an exact all-reduce, zero worker failures, and NCCL transport evidence.

### Refutation Condition
Missing or aliased trace, wrong rank mapping, collective mismatch/deadlock, socket-only transport, or any nonzero worker exit.

### Allowed Code Changes
A standalone M0 preflight and immutable two-node launcher; no Helios loss, optimizer, model placement, or lifecycle changes.

### Commands
Two nodes, eight A100 ranks per node, PyTorch 2.10.0+cu128; each rank fully loads/verifies its operation trace after intentionally different RNG perturbations, all-gathers descriptors, then executes a CUDA all-reduce.

### Measured Results
PASS. All 16 ranks exited zero. Global ranks 0--7 mapped to node 0 and 8--15 to node 1, the observed trace hash vector exactly matched bundle `a79d68d...d496`, all 16 hashes were unique, and the rank-sum all-reduce returned exactly 136. NCCL logs show `NCCL RDMA Plugin v6` and GDRDMA across nodes. Rank-0 measured 2.278 s from initialized process group through full payload verification and collectives.

### Correctness Result
PASS for exact rank-qualified trace identity and collective result despite deliberately distinct per-rank RNG states.

### Performance Result
Not an application performance result; no Helios model was constructed.

### Interpretation
The next 16-rank failure, if any, can be localized to construction/admission, DeepSpeed preparation, or full training rather than input identity or cross-node rendezvous. Operation-keyed replay demonstrably does not rely on all ranks sharing a seed.

### Files Changed
`scripts/distillschedule/probe_m0_multinode_contract.py`; `scripts/distillschedule/launch_m0_multinode_contract_probe.sh`; raw logs; `artifacts/manifests/iteration_120_m0_rank16_multinode_contract.json`; research log.

### Failure or Limitation
The probe does not establish 14B host/GPU admission, backward correctness, optimizer initialization, EMA behavior, or application throughput.

### Gate Status
16-rank trace/NCCL preflight PASS. Full A-scalable x16 correctness/admission OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does the frozen A-scalable per-rank mechanism complete steps 0--9 on both nodes simultaneously while each node remains below the 760-GiB-minus-64-GiB reserve?

### Next Command
Launch two-node 16-rank shared-immutable `b4fast`, replicated ZeRO-1 parameters/gradients, CPU optimizer, construction wave 4, swap wave 1 in memory mode.

## Iteration 121

### Track / Milestone
Track B / M0 A-scalable x16 correctness and per-node admission.

### Question
Does the frozen A-scalable per-rank mechanism complete the full steps 0--9 contract concurrently on two eight-A100 nodes while preserving the 64-GiB/node reserve?

### Hypothesis
Shared immutable frozen-block backing, replicated ZeRO-1 parameters/gradients with CPU optimizer-state partitioning, construction wave 4, and model-swap wave 1 will preserve the eight-rank admission result independently on both nodes and remain collective-safe across 16 ranks.

### Importance
This is the first complete official-shape Track-B execution: 16 independent full Helios pipelines, not a selected forward, one-component probe, reduced resolution, or G/S split.

### Evidence Needed
Sixteen exact operation traces; steps 0--9 on every rank; Generator backward/optimizer at 0 and 5; ten Critic backward/optimizer and EMA steps; zero final activation/streamer liveness; zero new OOM kills; each node peak below 747,324,309,504 bytes; and RDMA transport evidence.

### Refutation Condition
Any worker/trace/update mismatch, incomplete backward/Adam state initialization, memory leak, node reserve failure, OOM kill, or socket-only cross-node transport.

### Allowed Code Changes
A fixed two-node launcher and evidence summarizer only. The frozen per-rank activation, streaming, checkpoint, pinned-pool, and host-state policies were not changed.

### Commands
Two nodes x eight A100-40GB, 16 global ranks, 384x640 SYN-FEATURE trace bundle `a79d68d...d496`, BF16, `b4fast`, `HOST_SHARED_IMMUTABLE`, ZeRO-1 CPU optimizer, construction wave 4, model-swap wave 1, memory instrumentation, logical steps 0--9.

### Measured Results
PASS and admitted. Node 0 peaked at 645,368,602,624 bytes and node 1 at 645,750,177,792 bytes, respectively 101,955,706,880 and 101,574,131,712 bytes below the reserve-qualified limit. Both had zero new OOM kills. Tail medians were 321,363,412,992 and 317,620,815,872 bytes. All 16 ranks and 160 rank-steps completed. NCCL logs on both nodes show `NCCL RDMA Plugin v6` with GDRDMA. GPU max allocation remained approximately 33.73 GB/rank.

Memory-mode timing (not formal): global construction wall span 917.72 s; training wall span 1,588.58 s. Across rank-step records, Generator-update median was 310.29 s and Critic-only median 116.61 s. Step 0 includes first Adam-state initialization and measured 349.83 s median; the second Generator-update step 5 measured 271.40 s.

### Correctness Result
PASS. Each rank recorded exact input validation, Generator backward/optimizer count 2, Critic backward/optimizer count 10, EMA count 10, Generator-update steps [0,5], complete step IDs [0..9], and final activation/streamer live bytes [0,0]. Losses remained finite and all processes exited zero.

### Performance Result
Capacity and correctness result only. Timing came from the memory-instrumented run and is explicitly excluded from the formal throughput table.

### Interpretation
A-scalable x16 is feasible without parameter sharding, G/S placement, or component pipeline. The static node policy solves both long-lived host replication (shared immutable blocks) and cold transient admission (construction wave 4); wave alone would not have repaired the failed private/direct A-local replication. ZeRO-1 partitions optimizer state only and does not reduce replicated parameter/gradient residency.

### Files Changed
`scripts/distillschedule/launch_m0_multinode_dp.sh`; `scripts/distillschedule/summarize_m0_multinode_run.py`; raw node logs/traces; `artifacts/manifests/iteration_121_m0_b4fast_zero1_shared_cw4_x16.json`; research log.

### Failure or Limitation
This is SYN-FEATURE runtime evidence, not preprocessing, convergence, quality, or formal timing. Critic-only Generator `.grad is None`, frozen Score no-grad, and version checks inherit the already-tested implementation path but were not newly serialized as per-step assertions in this run. EMA bootstrap boundary 748--752, canonical logical checkpoint materialization, and formal A-scalable timing repetitions remain open.

### Gate Status
A-scalable x16 correctness/admission PASS. A-scalable formal freeze OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can the remaining M0 semantic assertions and EMA 748--752 bootstrap/steady recurrence be made directly traceable without another ambiguous seed-only or partial-component run?

### Next Command
Add non-retaining per-step correctness assertions/version records, construct an operation-keyed 748--752 trace, and run the bounded EMA boundary validation before formal A-scalable timing.

## Iteration 122

### Track / Milestone
Track A / M0 direct semantic assertions and EMA 748--752 boundary.

### Question
Can the frozen 384x640 `b4fast` full-pipeline baseline directly prove Generator/Critic update ownership, Critic-only Generator isolation, and the official EMA bootstrap/steady predicate without relying on a seed-only replay or a 748-step training history?

### Hypothesis
An operation-keyed logical-offset trace for steps 748--752, non-retaining parameter gradient hooks, exact trainable-state hashes, and a synthetic EMA optimization-step offset will exercise: Critic-only steps 748/749, the Generator update at 750, EMA updates at 748--750, and EMA skips at 751/752 while preserving official model/loss/update semantics.

### Importance
M0 cannot be frozen from update counts alone. The contract explicitly requires Generator `.grad is None` after Critic-only work, frozen Score no-grad, nonzero trainable updates, parameter-version order, and EMA recurrence at the 750 boundary.

### Evidence Needed
A source-shaped 384x640 trace independent of ambient RNG; steps 748--752 exactly once; Generator update only at 750; Critic update every step; pre-clear gradient delivery to all trainable LoRA parameters; exact before/after parameter SHA changes; zero retained Generator gradient on a Critic-only step; frozen Score gradient count zero; EMA optimization-step recurrence 748->751; exact zero-decay source equality; and a zero exit with no extra step.

### Refutation Condition
Any missing/extra logical step, absent gradient delivery, unchanged trainable state, leaked/frozen gradient, wrong version, wrong EMA predicate/recurrence, non-exact zero-decay source, OOM, or nonzero exit.

### Allowed Code Changes
Non-retaining correctness instrumentation, operation-trace start-step support, and a fail-closed synthetic boundary-loop guard. No loss, optimizer, model placement, activation policy, streaming window, or formal timing change.

### Commands
Generated and seed-perturbation replay-validated `/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_ema748_752_v1` (trace hash `8128284f...0650d`); ran targeted tests (40 passed) and the full suite (146 passed); launched `b4fast` on one A100-40GB with `DISTILL_M0_LOGICAL_START_STEP=748`, `DISTILL_M0_SYNTHETIC_EMA_OPTIMIZATION_STEP=748`, semantic assertions enabled, and `max_train_steps=753`. The accepted immutable run is `iteration122c_m0_b4fast_ema748_752_semantics`.

### Measured Results
PASS. The accepted run exited zero and executed exactly [748,749,750,751,752]. Generator gradient hooks fired for all 818 trainable tensors / 1,251,442,688 elements at step 750; Critic hooks fired for all 812 trainable tensors / 1,246,314,496 elements at step 748. Generator SHA changed `b06ad4...20b54 -> 0437f6...cfcd`; Critic SHA changed `253ad4...bb5f -> f2abac...165d`. The Critic-only step retained the exact Generator SHA and reported zero trainable/frozen Generator gradients. Frozen Score gradient count was zero before Critic backward. Final versions were Generator 151 and Critic 753.

EMA stepped on 748, 749, and 750 with optimization-step transitions [748,749,750] -> [749,750,751], decay 0.0, and exact equality to the current Generator source on every update. It skipped 751 and 752 with optimization step fixed at 751. Peak GPU allocated/reserved was 33,727,460,864 / 38,663,094,272 bytes; sampled cgroup peak was 151,800,127,488 bytes; new OOM-kill delta was zero.

### Correctness Result
PASS. `semantic_contract_complete` recorded all of `generator_update_verified`, `critic_update_verified`, and `critic_only_verified` true with no missing checks. Trace replay was operation-keyed and did not depend on the ambient RNG sequence.

### Performance Result
Not a formal timing result. Exact SHA scans, per-boundary diagnostics, and memory instrumentation intentionally perturb runtime. This run only supplies semantic and capacity evidence.

### Interpretation
The official Stage-3 update boundary is now directly observed rather than inferred: Score/Critic trainable state updates independently; a Critic-only step neither mutates nor leaves gradients on Generator; step 750 advances Generator before the Critic and EMA observations; EMA's bootstrap predicate is driven by the logical step before 750 and thereafter by Generator-update frequency.

### Files Changed
`runtime/official_baseline.py`; `train_helios.py`; trace builders/launch mapping/tests from commits `c5439aa`, `5373e08`, and `4d5332c`; external EMA trace asset; `artifacts/manifests/iteration_122_ema_trace*.json`; `artifacts/manifests/iteration_122_m0_ema_semantic_boundary.json`; raw logs/traces; research log.

### Failure or Limitation
The first attempt was deliberately interrupted during construction after source inspection showed that DeepSpeed clears gradients inside `Accelerator.backward`, making post-return `.grad` an invalid production test. A non-retaining pre-clear parameter-hook observer replaced it. The second attempt completed all five target steps but exposed an extra-batch bug in the synthetic logical-offset loop and failed on missing step-753 trace input; the outer/inner max-step guard fixed it. Both failed runs are preserved. The accepted probe synthetically offsets logical/EMA counters; it is not a model trained for 748 steps and is not convergence or quality evidence.

### Gate Status
EMA/update semantic gate PASS. A-local remains frozen. A-scalable x16 correctness/admission is PASS; its same-layout checkpoint and three formal timing repetitions remain OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can the A-scalable world16 layout save and restore a same-layout canonical step-10 checkpoint without a distributed sparse-EMA writer race, then execute steps 10--19 under formal timing mode?

### Next Command
Add bounded world16 checkpoint/resume launcher support and a single-writer sparse-EMA checkpoint barrier, generate/validate a 20-step rank16 trace, then create and restore checkpoint-10 before formal timing.

## Iteration 123

### Track / Milestone
Track B / M0 world16 canonical-checkpoint and formal-timing substrate.

### Question
Can the already-admitted A-scalable world16 layout be given an operation-keyed 20-step trace plus fail-closed same-layout checkpoint/resume support without changing its per-rank memory policy?

### Hypothesis
A trace covering 0--19, resume-derived coverage bounds, DeepSpeed all-rank model/optimizer checkpointing, and a single global writer for replicated sparse CPU EMA will support checkpoint-10 construction and steps 10--19 timing without rank races or seed-only replay.

### Importance
Formal Track-B timing must start independently from the same canonical logical state. Reusing the world1 checkpoint is invalid because ZeRO-1 optimizer partitions are world-size/layout specific.

### Evidence Needed
Sixteen distinct source-shaped traces with exact 0--19 coverage; exact rank-qualified input replay; launcher rejection of malformed checkpoint paths; resolved checkpointing/resume fields; a single sparse-EMA writer bracketed by all-rank barriers; resumed-step-aware result summarization; tests and prior-run regression.

### Refutation Condition
Missing/rank-aliased trace, malformed resume accepted, trace coverage restarting at zero after resume, sparse-EMA multi-writer race, prior x16 result no longer summarizes, or tests fail.

### Allowed Code Changes
Track-B checkpoint/resume launcher plumbing, replicated sparse-EMA save synchronization, and evidence summarization. No activation set, frozen-block window, checkpoint/recompute policy, host-state policy, loss, optimizer order, placement, or pipeline change.

### Commands
Generated `/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_rank16_v2_steps20`; validated ranges 0--9 and 10--19 plus all sixteen source inputs; ran `bash -n`, Python compile, malformed-resume fail-closed probe, a regression resummary of Iteration 121, and 39 targeted tests.

### Measured Results
Substrate PASS. Bundle hash `bd97ddc9...f12f` contains 16 distinct trace hashes, 249 tensors and 69 operation records per rank, with complete steps 0--19. Both half-range coverage checks and all sixteen exact primary/secondary input checks pass. The patched summarizer reproduces Iteration 121's admitted result and 1,588.578 s ten-step memory-run window exactly. A malformed resume path exits 2 before launch.

### Correctness Result
PASS for trace identity, coverage, fail-closed path parsing, source compilation, and prior-run regression. The sparse-EMA distributed save/restore path still requires the real world16 execution in the next iteration.

### Performance Result
Not Run. This iteration adds measurement substrate only and makes no throughput claim.

### Interpretation
M0 can now create a world16 same-layout checkpoint and measure a logically contiguous replay without conflating world1 and world16 optimizer layouts. The sparse EMA remains replicated; single-writer persistence is an I/O correctness fix, not model-state sharding or a DistillSchedule optimization.

### Files Changed
`scripts/distillschedule/launch_m0_multinode_dp.sh`; `scripts/distillschedule/summarize_m0_multinode_run.py`; `train_helios.py`; external rank16 trace v2; `artifacts/manifests/iteration_123_m0_rank16_trace_v2_steps20.json`; research log.

### Failure or Limitation
DeepSpeed model/optimizer checkpointing still requires every rank; only the replicated sparse EMA is single-writer. The 20-step trace is SYN-FEATURE runtime input, not training convergence or quality evidence. Real checkpoint I/O can introduce node/page-cache transients not predicted by the earlier training-only admission run.

### Gate Status
Checkpoint/timing substrate PASS. Real world16 checkpoint construction, resume acceptance, and three formal timing repetitions OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can the full world16 A-scalable execution save checkpoint-10, including both DeepSpeed engines and sparse EMA, while preserving node reserve and then restore step 10 on all ranks?

### Next Command
Launch `b4fast`, shared immutable host state, ZeRO-1 CPU optimizer, construction wave 4, swap wave 1, steps 0--9 in memory mode with `checkpointing_steps=10`; hash and resume the resulting checkpoint.

## Iteration 124

### Track / Milestone
Track B / M0 world16 same-layout checkpoint-10 construction and inventory.

### Question
Can the admitted A-scalable world16 layout complete steps 0--9 and persist a complete, race-free same-layout checkpoint containing both DeepSpeed engines, rank-local optimizer/RNG state, scheduler state, sparse EMA, and the logical step needed for formal replay?

### Hypothesis
All-rank DeepSpeed checkpointing plus a barrier-bracketed single sparse-EMA writer will create one resumable world16 checkpoint without changing the frozen per-rank policy or violating either node's 64-GiB reserve.

### Importance
Formal A-scalable timing must begin from the same world16 optimizer/EMA state. A world1 checkpoint, a LoRA-only export, or a directory that merely exists cannot establish this contract.

### Evidence Needed
Zero exit on both nodes; all 16 ranks complete exact steps 0--9; reserve pass and no new OOM kill on each node; 16 optimizer shards and 16 RNG states for both Generator and Critic; distributed logical-state files for all ranks; sparse EMA with optimization step 10; exactly one global EMA writer bracketed by 16-rank barriers; and a content-hashed inventory.

### Refutation Condition
Any missing rank/shard/state, multi-writer EMA race, wrong logical/EMA step, trace mismatch, worker failure, OOM/reserve failure, or inventory inconsistency.

### Allowed Code Changes
A fail-closed same-layout checkpoint inventory validator and unit test only. No loss, optimizer, activation, checkpoint/recompute, streaming, placement, host-state, or wave policy change.

### Commands
Ran two nodes x eight A100-40GB, 384x640 SYN-FEATURE rank16 trace bundle `bd97ddc9...f12f`, BF16 `b4fast`, shared immutable host blocks, ZeRO-1 CPU optimizer, construction wave 4, swap wave 1, memory instrumentation, steps 0--9, and `checkpointing_steps=10`. Then validated and SHA-256 hashed all checkpoint files with `validate_m0_same_layout_checkpoint.py`. The full test suite passed (147 tests).

### Measured Results
PASS. Both node launchers exited zero; all 16 ranks completed the expected two Generator and ten Critic/EMA updates. Node cgroup peaks were 650,537,316,352 and 645,585,743,872 bytes, leaving 96,786,993,152 and 101,738,565,632 bytes below the reserve-qualified 747,324,309,504-byte limit; new OOM-kill deltas were zero. The checkpoint contains 97 files and 99,826,243,222 payload bytes. Each engine has one model-state file, 16 ZeRO-1 optimizer shards, 16 rank RNG files, scheduler state, and LoRA export. Sparse EMA has 818 shadow tensors in three files, `optimization_step=10`, and exactly one writer across 16 start/end barrier events. The full 99.83-GB file inventory was SHA-256 hashed and passed.

Memory-mode timing remains non-formal: construction wall span 927.21 s; ten-step training window 1,501.96 s; Generator-update median 298.88 s; Critic-only median 110.81 s. The sparse-EMA writer barrier lasted 6.09 s.

### Correctness Result
PASS for complete steps 0--9, exact rank-qualified traces, file/rank inventory, EMA step/writer semantics, and reserve-qualified execution. Checkpoint value restoration is not yet claimed until the next measured resume.

### Performance Result
Capacity/checkpoint result only. The run used memory instrumentation and checkpoint I/O; its 150.20-s weighted step is excluded from the formal throughput table.

### Interpretation
A world16 canonical timing start state can be materialized in the already-frozen A-scalable layout. The artifact is explicitly a same-layout runtime checkpoint, not the later layout-independent logical serializer. Sparse EMA single-writer persistence removes an I/O race; it is correctness plumbing, not a scheduling speedup.

### Files Changed
`scripts/distillschedule/validate_m0_same_layout_checkpoint.py`; `tests/test_m0_checkpoint.py`; raw run logs/traces/checkpoint; `artifacts/manifests/iteration_124_m0_world16_checkpoint10_run.json`; `artifacts/manifests/iteration_124_m0_world16_checkpoint10_inventory.json`; research log.

### Failure or Limitation
During construction, an external diagnostic in the node-0 monitor deserialized an unrelated historical optimizer checkpoint. It may have transiently increased page cache/cgroup usage; therefore Iteration 121 remains the clean A-scalable admission authority. Iteration 124 still passed reserve and is used for checkpoint correctness. Inventory completeness does not prove restored optimizer values; the next run must load and execute step 10. This is SYN-FEATURE runtime evidence, not preprocessing, convergence, quality, or formal timing.

### Gate Status
World16 checkpoint construction/inventory PASS. Same-layout resume acceptance and three formal timing repetitions OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Can all 16 ranks restore Generator/Critic optimizer and scheduler state, sparse EMA optimization step 10, logical step 10, and execute the operation-keyed Generator+Critic step 10 without state drift or layout mismatch?

### Next Command
Resume `iteration124.../checkpoint-10` in the identical world16 layout, run exactly logical step 10 in memory mode, and summarize per-rank restore checksums/events before launching formal steps 10--19.

## Iteration 125

### Track / Milestone
Track B / M0 world16 checkpoint resume acceptance (failed input-authority probe).

### Question
Can checkpoint-10 restore all world16 state and immediately execute operation-keyed logical step 10 with the existing trace-input validation?

### Hypothesis
The saved ZeRO-1 partitions, schedulers, RNG, dataloader, sparse EMA, and operation trace will restore at step 10 and the first resumed batch will remain identical to each rank's trace source.

### Importance
A directory-complete checkpoint is insufficient: formal timing needs restored optimizer/EMA state and exact next input identity, not merely the same random seed.

### Evidence Needed
Sixteen `checkpoint_resume_complete` events with Generator optimizer step 2, Critic step 10, sparse EMA step 10, exact rank traces, then one successful Generator+Critic update at logical step 10.

### Refutation Condition
Any state/layout load failure, wrong restored counter/checksum, different source tensor, missing update, worker failure, or OOM.

### Allowed Code Changes
No code change before the immutable probe. After failure, only operation-trace source-authority plumbing and tests; no loss, model, optimizer, placement, lifecycle, or wave-policy changes.

### Commands
Resumed the Iteration-124 checkpoint in the identical two-node world16 `b4fast`/ZeRO-1/shared-immutable/construction-wave-4/swap-wave-1 memory layout with trace coverage restricted to logical step 10. After failure, added RNG-free authoritative source materialization and ran the full test suite.

### Measured Results
Partially verified, then correctly failed closed. All 16 ranks loaded Generator optimizer step 2, Critic optimizer step 10, both scheduler states, sparse EMA, RNG and logical global step 10. Rank-0 Generator checkpoint `exp_avg` independently measured sum/norm `-0.0500975422 / 0.02316612375`; the restored event reported `-0.0500975437 / 0.02316612351`, with the small difference explained by diagnostic reduction precision. Both nodes passed reserve (647,603,556,352 and 646,407,200,768 bytes) with zero new OOM kills.

Execution then failed before model compute: only 3/16 ranks happened to receive a dataloader sample equal to their fixed trace source; the others raised `ValueError: primary dataset tensor does not match operation trace`. No rank completed step 10. The StatefulDataLoader correctly restored epoch/sampler position, but that made its next shuffled sample differ from the operation-keyed source.

### Correctness Result
REFUTED for end-to-end resume; VERIFIED for checkpoint state restoration. The failed run exposed that the trainer treated the dataloader payload, rather than the trace payload, as authoritative at the resume boundary. This violated the explicit operation-keyed input contract even though seed and trace coverage were correct.

### Performance Result
Rejected. No training step completed and the run is excluded from all timing claims.

### Interpretation
Checkpoint restore and input replay are separate contracts. Restoring a stateful sampler is correct for ordinary training, but formal strategy comparison must replay the trace's source tensors regardless of sampler position. A rank-qualified trace cannot merely validate the first fresh-run batch and then defer source identity to shuffle order.

### Files Changed
Failure manifest `artifacts/manifests/iteration_125_m0_world16_resume10_failed_input_authority.json`; preserved raw logs/traces; `runtime/operation_trace.py`; `train_helios.py`; `tests/test_operation_trace.py`; research log.

### Failure or Limitation
The fix makes the trace authoritative for primary/secondary x0/history/target and text embeddings while still advancing the official StatefulDataLoader control flow. This is correctness substrate, not an optimization. The existing trace already controls rollout histories/noise/timesteps/branches; deterministic packing indices depend only on shape and are unchanged. A real world16 retry is still required.

### Gate Status
Checkpoint state restore PASS; resumed step execution FAIL due identified input-authority bug. Fix unit/replay tests PASS (148 full-suite tests). Retry OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
Does authoritative operation-trace source materialization make all 16 ranks consume exact rank-qualified inputs and complete logical step 10 from the restored optimizer/EMA state?

### Next Command
Commit the fail-closed trace-authority fix, then rerun the identical world16 step-10 resume under a new immutable run ID.

## Iteration 126

### Track / Milestone
Track B / M0 world16 checkpoint resume acceptance with authoritative trace source.

### Question
Does RNG-free operation-trace source materialization restore exact rank-qualified inputs and allow all 16 ranks to complete logical step 10 from checkpoint-10?

### Hypothesis
Replacing only the resumed dataloader payload with the already-verified trace's x0/history/target/text tensors will preserve official control flow and make the checkpointed world16 update execute exactly once.

### Importance
This closes the separation exposed in Iteration 125 between checkpoint-state restoration and input identity, which is required before any formal timing can be accepted.

### Evidence Needed
Sixteen exact source-authority events with seven materialized tensor hashes each; Generator/Critic optimizer restore steps 2/10; initial versions 2/10; one complete step 10; versions 3/11; EMA optimization step 10->11; reserve pass and zero OOM kill.

### Refutation Condition
Any trace-source hash mismatch, missing rank, state/version drift, failed update, nonfinite loss, OOM, or nonzero worker exit.

### Allowed Code Changes
The committed Iteration-125 trace-authority fix and stricter evidence summarization only. No model/loss/optimizer/lifecycle/placement/wave-policy change.

### Commands
Resumed the same 99.83-GB Iteration-124 checkpoint in the identical two-node world16 memory layout; required trace coverage [10,11); ran one Generator+Critic logical step in memory mode; resummarized with resume source/version gates.

### Measured Results
PASS. Both nodes exited zero and all 16 ranks recorded `source_authority=operation_trace` with seven exact materialized source hashes. Every rank restored Generator optimizer step 2, Critic step 10, initial parameter versions G2/C10, then completed Generator and Critic updates at logical step 10. Rank 0 advanced G2->G3 and C10->C11; its losses were finite (Generator 0.4396124, Critic 1.5121899). Sparse EMA advanced optimization step 10->11 with the post-update G3 source version. Node peaks were 652,040,196,096 and 646,884,024,320 bytes, respectively 95,284,113,408 and 100,440,285,184 bytes below the reserve-qualified limit, with zero new OOM kills and RDMA observed.

### Correctness Result
PASS for same-layout checkpoint restoration, exact next-input identity, optimizer/version ordering, one complete G+Critic update, EMA recurrence, and no final activation/streamer liveness.

### Performance Result
Not formal. This memory-instrumented single update measured a 330.53-s global wall span and is excluded from throughput claims.

### Interpretation
The same checkpoint now provides a valid canonical timing start for three independent world16 launches. A stateful dataloader may advance normally, but its shuffled payload cannot override the operation-keyed feature source in a controlled strategy comparison.

### Files Changed
`artifacts/manifests/iteration_126_m0_world16_resume10_trace_authority.json`; stricter source/resume fields in M0 summarizers; raw logs/traces; research log.

### Failure or Limitation
This proves same-layout restore only, not cross-layout serialization. It uses SYN-FEATURE and does not claim preprocessing, convergence, quality, or formal throughput. Formal timing still requires three independent ten-step runs.

### Gate Status
World16 same-layout checkpoint/resume gate PASS. Three formal A-scalable timing repetitions OPEN. M0 OPEN; G0 BLOCKED.

### Next Highest-Value Question
What is the repeatable formal 10-step steady-state performance of frozen A-scalable under timing-only instrumentation from the identical checkpoint?

### Next Command
Launch three independent timing-mode world16 runs of logical steps 10--19 from checkpoint-10, summarize each with strict resume/input/version gates, then freeze A-scalable.

## Iteration 127

### Track / Milestone

Track B / M0 formal A-scalable timing.

### Question

Does the frozen per-rank A-scalable policy produce repeatable formal 384x640
performance over three independent world16 launches from one canonical checkpoint?

### Hypothesis

The `b4fast + HOST_SHARED_IMMUTABLE + ZeRO-1 CPU optimizer + construction-wave-4 +
swap-wave-1` policy will complete logical steps 10--19 on all 16 ranks in each launch,
preserve trace/state/version contracts, and pass each node's 64-GiB reserve.

### Importance

A memory-mode correctness run cannot freeze Track B performance. M0 requires three
independent timing-mode repetitions before A-scalable and Gate G0 are valid.

### Evidence Needed

Two zero return codes per repetition; 16 rank traces; exact steps 10--19; operation-trace
source authority; restored Generator/Critic optimizer steps 2/10 and initial versions
G2/C10; Generator updates 10/15; Critic updates every step; final versions G4/C20;
EMA recurrence; finite losses; formal timing flag; no new OOM kill; per-node reserve pass.

### Refutation Condition

Any worker failure, missing step/rank, trace-source mismatch, optimizer/version drift,
nonfinite loss, OOM, reserve violation, or fewer than three accepted repetitions.

### Allowed Code Changes

None to the workload, policy, loss, optimizer, trace, or launcher during timing. Only
post-run immutable manifests were added.

### Commands

On each node, launched `scripts/distillschedule/launch_m0_multinode_dp.sh b4fast` in
`timing` mode with rank-specific machine rank, world size 16, trace bundle
`syn_feature_trace_rank16_v2_steps20`, `DISTILL_M0_MAX_STEPS=20`, resume from Iteration
124 `checkpoint-10`, ZeRO stage 1, construction wave 4, shared immutable frozen blocks,
swap wave 1, and ports 29757/29758/29759. Runs were detached server-side with
`nohup setsid`; monitoring used one-shot IPv4 SSH.

### Measured Results

Three accepted weighted results were 145.170340, 156.526335, and 150.689128 s/logical
step. Mean was 150.795268 s, median 150.689128 s, sample stdev 5.678741 s, CV 3.766%.
Mean Generator-update and Critic-only median-derived times were 297.156348 and
113.838993 s. Mean construction wall span was 873.100737 s. First and second five-step
cycles averaged 801.621850 and 706.041184 s. Global throughput at the primary full-window
mean was 0.106104 samples/s, or 0.006632 samples/s/GPU.

Timing-mode node peaks across the three runs were 653.982--654.844 GB on node 0 and
651.756--652.141 GB on node 1. Every repetition passed the reserve, observed RDMA on
both nodes, and recorded zero new OOM kills.

### Correctness Result

PASS. All 16 ranks in all three launches restored the canonical state, used operation
trace source authority, completed steps 10--19, updated Generator at 10/15 and Critic at
every step, advanced versions G2/C10 to G4/C20, and completed EMA through optimization
step 20 with finite losses.

### Performance Result

Measured formal Track-B performance: 150.795268 s/logical step mean and 0.106104
samples/s across 16 GPUs. The first cycle is consistently slower; both cycles remain in
the preregistered ten-step primary statistic.

### Interpretation

The scalable policy is reproducible but pays severe shared-node contention: its
per-rank weighted time is 2.60x A-local and pure-DP scaling efficiency is 38.50% of the
ideal A-local-derived value. This is a baseline result, not an M1-M4 speedup.

### Files Changed

`artifacts/manifests/iteration_127a_m0_a_scalable_x16_formal_timing_failed_ssh_transport.json`;
`iteration_127b_m0_a_scalable_x16_formal_timing_rep1.json`;
`iteration_127c_m0_a_scalable_x16_formal_timing_rep2.json`;
`iteration_127d_m0_a_scalable_x16_formal_timing_rep3.json`; raw logs/traces.

### Failure or Limitation

Iteration 127a used foreground SSH; the transport expired after checkpoint restore and
before any logical step. It is preserved but excluded from timing because it is not a
Helios/OOM failure. The accepted launches use SYN-FEATURE and same-layout resume; they
do not measure preprocessing, convergence, quality, or cross-layout restore.

### Gate Status

Formal A-scalable timing PASS. A-scalable freeze ready; M0 otherwise complete.

### Next Highest-Value Question

Which accepted scalable policy should be frozen, and does the complete M0 evidence route
Gate G0 to M1 or M2?

### Next Command

Aggregate Iterations 127b--127d, freeze A-scalable, then emit Gate G0.

## Iteration 128

### Track / Milestone

Track A/B / M0 A-scalable freeze.

### Question

Can the fastest actually admissible complete per-rank policy be frozen without
conflating it with A-local or allowing Track B to retune per-rank lifecycle actions?

### Hypothesis

The world16 `b4fast/shared/ZeRO-1/cw4/sw1` policy is A-scalable because it is the
fastest measured policy that preserves a complete independent flow per rank and passes
actual 8-rank-per-node long-lived-state admission.

### Importance

Track B must use A-scalable x16 plus node admission. It may not silently replace the
per-rank action set after seeing node pressure.

### Evidence Needed

Three formal repetitions, clean memory-mode x16 admission, direct A-local x8 admission,
fixed per-rank action policy, exact checkpoint/trace identity, and complete correctness.

### Refutation Condition

A faster admissible candidate, reserve failure, per-rank parameter sharding, missing
full-flow semantics, or any formal repetition failure.

### Allowed Code Changes

Manifest aggregation and documentation only.

### Commands

Aggregated Iterations 127b--127d with Python `statistics`, bound them to Iteration 121
memory admission, Iteration 109 A-local direct-copy rejection, Iteration 124 checkpoint
inventory, and the rank16-v2 trace bundle.

### Measured Results

A-scalable frozen mean is 150.795268 s/step with 0.106104 samples/s world16 throughput.
The clean admission authority peaks at 645,368,602,624 and 645,750,177,792 bytes,
leaving a minimum 101,574,131,712-byte reserve-qualified margin. Direct A-local x8
peaks at 757,807,587,328 bytes, exceeding the 747,324,309,504-byte qualified limit.

### Correctness Result

PASS under the operation trace, canonical checkpoint, optimizer/version, gradient
ownership, and EMA contracts.

### Performance Result

A-scalable is frozen as the Track-B strong baseline. It is not the single-card winner:
A-local remains frozen separately at 58.057628 s/step.

### Interpretation

A-local and A-scalable are necessarily different. Sharing immutable host state and
sharding CPU optimizer state make eight replicas admissible, while node contention
reduces per-GPU efficiency. A wave controls transient concurrency but does not make the
host-private A-local state admissible.

### Files Changed

`artifacts/manifests/iteration_128_m0_a_scalable_freeze.json`.

### Failure or Limitation

The scalable baseline uses ZeRO-1 optimizer sharding, not parameter or gradient
sharding. Timing-mode peaks support but do not replace Iteration 121's memory-mode
admission authority.

### Gate Status

A-local FROZEN; A-scalable FROZEN; A-local x8 admission REJECTED WITH RESERVE; Track B
world16 PASS. M0 complete; Gate G0 ready.

### Next Highest-Value Question

Does measured lifecycle pressure or placement infeasibility have higher priority after
M0?

### Next Command

Compute Gate G0 only; do not implement M1--M4.

## Iteration 129

### Track / Milestone

Gate G0.

### Question

Should the next authorized phase prioritize M1 lifecycle management or M2
sharding/placement?

### Hypothesis

A clear measured memory/performance Pareto remains even if the exact exposed PCIe
fraction cannot be proven above 50%; therefore Gate G0 should prioritize M1.

### Importance

The master contract forbids starting M1--M4 before this evidence-based decision and
requires the current loop to stop after G0.

### Evidence Needed

Frozen A-local/A-scalable, A-local x8 admission, bounded B0a--B4 outcomes, measured
parameter/activation movement, formal performance, and explicit uncertainty.

### Refutation Condition

No material lifecycle Pareto, single-card bounded failure, or a clearly dominant
measured multi-GPU placement result. The latter was not authorized or run in this cycle.

### Allowed Code Changes

Gate manifest and documentation only. No lifecycle, placement, sharding, or pipeline
primitive.

### Commands

Combined Iterations 45--58, 62, 100, 109, 121--128 and computed movement and scaling
ratios. Parameter movement was kept separate from untimed activation pack/unpack.

### Measured Results

A-local streams 2,530,038,988,800 bytes of frozen blocks in ten steps. Frozen-block H2D
and residual whole-model movement consume 264.357375 and 23.792146 s. Their 288.149521-s
sum equals 49.6316% of the A-local three-run mean ten-step window. The same memory run
offloads 134,364,364,800 bytes of saved-tensor storage, but its copy time is not isolated.
A-scalable is 2.597x slower per rank than A-local; world16 pure-DP scaling efficiency is
38.501% of ideal. Host-private x8 fails reserve while shared/ZeRO-1 x16 passes.

### Correctness Result

PASS for all evidence required by M0 and G0.

### Performance Result

No DistillSchedule M1--M4 performance claim. This iteration selects the next question;
it does not answer it.

### Interpretation

The strict “exposed PCIe >50%” predicate is not claimed: measured parameter movement is
49.63% and activation-copy time is unisolated. The alternative gate predicate is
satisfied directly by the local/scalable memory-performance Pareto. M1 real-14B joint
activation/parameter lifecycle management is therefore the next priority. M2 remains
deferred, not refuted.

### Files Changed

`artifacts/manifests/iteration_129_gate_g0.json`; `docs/13_gate_g0.md`;
`README_AUTORESEARCH.md`; `docs/00_hardware_feasibility.md`;
`docs/03_profile_report.md`; `docs/08_results.md`; `docs/10_open_questions.md`;
`research_log.md`.

### Failure or Limitation

SYN-FEATURE excludes source preprocessing, convergence, and quality. Exact activation
copy exposure, held-out lifecycle policies, and M2 placement families remain unmeasured.
The current authorization ends at G0.

### Gate Status

**G0 COMPLETE. Next priority: M1. Current auto-research loop STOPPED at authorized
boundary.**

### Next Highest-Value Question

After authorization, can an automatic joint lifecycle plan approach the frozen expert
Pareto while reducing exposed copies or host state on held-out traces/bandwidths?

### Next Command

Not run: await authorization before starting M1.

## Iteration 130

### Track / Milestone

M1 planning and preregistration after Gate G0. No M1 runtime implementation or performance experiment.

### Question

What is the smallest bounded next-stage program that can distinguish a real joint lifecycle-runtime contribution from copy/accounting noise, a fixed expert policy, or a node-contention problem that should instead route to M2?

### Hypothesis

The highest-value next experiment is not an async planner. It is a causal, low-overhead lifecycle profile of the frozen 384x640 A-local/A-scalable policies. If that profile exposes at least a 10% legal time ceiling or a material memory/admission opportunity, a versioned storage-lifetime IR and fixed-placement planner are justified; otherwise M1 should close negatively and route to M2.

### Importance

A-scalable is 2.597x slower per rank than A-local, but the delta mixes PCIe copies, allocation/staging, admission queueing, CPU memory contention, ZeRO-1/NCCL, and stragglers. Attributing the whole delta to offload would create the wrong paper problem and could lead to a large planner whose theoretical end-to-end ceiling is already below 10%.

### Evidence Needed

A clean live worktree and test suite; current runtime integration boundaries; frozen M0/G0 manifests; a same-primitive baseline set; calibration and operation-structure/value/contention held-outs; separate logical-byte, issued-copy, observed-copy, copy-engine-union, exposed-wait, host-overhead, and host-state accounting; preregistered positive/partial/negative Gate-G1 outcomes.

### Refutation Condition

The plan is invalid if M0/G0 evidence is stale, the current code already synthesizes storage-level lifecycle actions rather than selecting scalar profiles, fixed-placement lifecycle has a conservative legal ceiling below 10% with no memory Pareto opportunity, or held-out correctness cannot be bound to operation-keyed traces and canonical logical checkpoints.

### Allowed Code Changes

Planning artifacts only: one normative M1 execution plan, one machine-readable contract, README/open-question synchronization, and this append-only log entry. No model, optimizer, trace, checkpoint, lifecycle primitive, or launcher behavior changed.

### Commands

- inspected live HEAD/status and latest commits in `/share/project/shuchenweng/yongyan_liu/DistillSchedule`;
- verified `/share/project/shuchenweng/DistillSchedule_assets` (114G) and `/home/shuchenweng/xty/Helios-main` remain available;
- audited `runtime/ir.py`, `runtime/planner.py`, `runtime/residency.py`, `runtime/frozen_block_streaming.py`, `runtime/activation_offload.py`, `train_helios.py`, and M0 launchers;
- accidentally invoked the unit suite under the bootstrap interpreter backed by system Torch 2.1, which failed collection because current Diffusers references `torch.xpu`;
- reran the authoritative suite with:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 \
  /tmp/distillschedule-helios-probe-torch24/bin/python -m pytest -q
```

### Measured Results

Live input HEAD was `493b55fb44e89a820effb665cb95202a9f5145f7` on clean branch `distillschedule/autoresearch`. The authoritative Torch-2.4 probe environment completed **148 tests** twice: 46.30 s before edits and 47.23 s in the post-edit rerun, each with 32 warnings. The wrong bootstrap/system-Torch invocation produced three collection errors (`torch.xpu` missing); this is an environment-selection failure, not an accepted regression.

Code audit confirms that whole-module residency, frozen-block streaming, and saved-tensor offload are separate synchronous mechanisms. Frozen-block `_materialize_block` and activation pack/unpack use `non_blocking=False`; activation offload allocates a CPU tensor per packed storage; current `BudgetPlanner` selects among premeasured scalar `StrategyProfile` objects. These facts justify a separate storage-lifetime IR/profile/backend rather than relabeling the existing selector as M1.

### Correctness Result

Planning inputs PASS: clean authoritative worktree, source/assets present, and 148/148 tests pass in the documented probe environment. No new training correctness result was produced.

### Performance Result

Not Run. Existing M0 measurements remain the only real-performance evidence. This iteration makes no DistillSchedule speedup claim.

### Interpretation

M1 should proceed through four gates: causal ceiling, shadow-IR admission accuracy, synchronous planner regret, then bounded async overlap. The fair denominator is the strongest admitted expert using the same primitives, not B0/B2 OOM or full-offload alone. Held-out validation must vary operation structure, feature values, GPU budget, and rank contention—not only a seed.

### Files Changed

- `docs/14_m1_execution_plan.md`;
- `configs/m1_lifecycle_contract.json`;
- `README_AUTORESEARCH.md`;
- `docs/10_open_questions.md`;
- `research_log.md`;
- `artifacts/manifests/iteration_130_m1_planning.json` (generated after document validation).

### Failure or Limitation

No M1 causal profile exists yet. Activation-copy exposure remains unknown, and the 49.63% parameter-movement wall fraction is not equivalent to exposed PCIe time. The 34/36/38-GiB planner budgets are preregistered policy budgets but still require runtime/driver admission validation. M2 remains deferred, not disproved.

### Gate Status

M0/G0 remain complete. M1 plan is frozen and implementation is Not Run. The first M1 gate is G1a; large async/planner work is forbidden until causal profiling establishes sufficient ceiling.

### Next Highest-Value Question

Can a low-overhead profiler reconcile runtime-owned copy bytes and separate copy-engine work, host allocation, actual consumer wait, wave/NCCL wait, and node contention without changing Helios semantics?

### Next Command

Implement `runtime/lifecycle_profile.py` and `tests/test_lifecycle_profile.py`, rerun the complete suite, then run one 384x640 frozen-A-local memory/profile execution followed by a separate paired timing execution to measure tracer overhead.

## Iteration 131

### Milestone / Gate

M1.0 contract substrate and M1.1 profiler implementation; G1.0 open.

### Question

Can the frozen official-shape execution be bound to operation-keyed calibration/held-out traces and a layout-independent logical checkpoint while a low-overhead profiler separates issued bytes, copy-associated intervals, and exposed waits?

### Hypothesis

The M0 movement counters conflate copy, allocation, enqueue, synchronization, and contention; explicit runtime-owned byte counters plus deferred CUDA events can preserve semantics and expose the causal categories needed by Gate G1a.

### Importance

A planner or prefetch runtime is not justified until checkpoint/trace comparability and causal accounting are real. In particular, `.to()` wall time is not PCIe time.

### Evidence Needed

Exact CAL-A/HOLD-S/HOLD-V replay independent of ambient RNG; physical GPU-derived budgets; world16-to-logical export and native world1/world16 import/verify; profiler tests and full regression suite; no retained tensor references or per-copy synchronization.

### Refutation Condition

Trace/checkpoint state cannot be made layout-independent in two repairs, profiler overhead exceeds 3%, or runtime-owned byte accounting cannot be reconciled.

### Allowed Changes

Lifecycle instrumentation, checkpoint/trace tooling, M1 launchers/summarizers/tests, manifests, and documentation only. No lifecycle policy, prefetch, loss, optimizer, or frozen baseline change.

### Commands

Rechecked live branch/assets/GPU processes; generated and replayed the three official 384x640 trace classes; exported/imported/verified logical checkpoint v3; added `runtime/lifecycle_profile.py`, profile launchers, summaries, and focused tests; ran the full authoritative test suite under `/tmp/distillschedule-helios-probe-torch24`.

### Measured Results

- Live baseline anchor: `493b55fb44e89a820effb665cb95202a9f5145f7`; implementation advanced through commit `de3001a96b8b3dbf22c5d6845661ea5ae44349ad`.
- GPU physical memory: 42,298,834,944 bytes. Frozen aligned budgets are 37/35/33 GiB for relaxed/medium/tight after the physical two-GiB reserve rule.
- CAL-A hash: `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.
- HOLD-S hash: `19c4f8ed21429faa13c22e0993eff080cb6109d4e3f7902f7b3673d9bddf3fb3`.
- HOLD-V hash: `8407a08a620e7cdc978c8617558061b090e463f649acacf745d18604b7803dc4`.
- Logical step-10 manifest hash: `68eed293970be008764ee6acadb2aac6ce54de2ce1f55a17fca2246f0d04e4aa`.
- Full regression: 154 passed, 32 warnings, 41.21 seconds.

### Correctness

Trace replay, logical tensor import/verify, optimizer moments/steps, scheduler metadata, sparse EMA, and RNG inventories pass at the checkpoint boundary. Cross-layout post-update equivalence remained open at the end of this iteration.

### Performance

Not Run. The profiler is instrumentation only; no M1 performance claim is made.

### Interpretation

The implementation now distinguishes logical unique storage, issued transfer bytes, CUDA-event copy-associated interval union, explicit dependency waits, measurement-boundary synchronization, barrier time, and host-call residual. `observed_memcpy_bytes` deliberately remains null until CUDA activity/Nsight evidence exists.

### Files Changed

`runtime/lifecycle_profile.py`; `tests/test_lifecycle_profile.py`; `scripts/distillschedule/summarize_iteration131.py`; `scripts/distillschedule/summarize_m1_profile.py`; M1 launchers; trace/calibration preparation tools; checkpoint and budget manifests.

### Failure or Limitation

Logical unique bytes do not cover every non-streamed whole-state alias, and CUDA events are not CUPTI memcpy observations. The held-out traces remain sealed for tuning. Cross-layout one-step numerical equivalence is not implied by exact import.

### Gate Status

G1.0 OPEN; G1a BLOCKED.

### Next Highest-Value Question

Does one official G-update from the same logical step-10 state remain within the preregistered update tolerance in native world1 ZeRO-2 and world16 ZeRO-1 layouts?

### Next Command

Run native single/world16 step 10 from imported logical state, export both step-11 states, and compare logical parameter/optimizer/EMA updates.

## Iteration 132

### Milestone / Gate

M1.0 cross-layout one-step equivalence, repair 1 of at most 2.

### Question

Do native A-local world1 ZeRO-2 and A-scalable world16 ZeRO-1 produce equivalent one-step logical updates from the exact same imported state and CAL-A operation trace?

### Hypothesis

Exact input tensors, exact imported logical state, FP32 collective communication, and identical optimizer metadata are sufficient for the native layouts to meet cosine >=0.999 and update relative L2 <=2e-3.

### Importance

M1 comparisons cannot be called cross-layout canonical if the native layouts silently begin from different logical state or diverge beyond the correctness contract before any lifecycle policy changes.

### Evidence Needed

Successful world1/world16 step-10 execution, exact loss/trace/version records, exported logical step-11 packages, and chunked update comparison for BF16 module, FP32 master, Adam m/v, scheduler, and EMA.

### Refutation Condition

Any parameter update has cosine <0.999 or relative L2 >2e-3, scheduler/optimizer versions differ logically, or EMA fails its recurrence/equality check.

### Allowed Changes

Checkpoint/trace launch and comparison tooling only. No lifecycle, model, loss, optimizer, or M0 baseline policy change.

### Commands

Ran a single-rank imported checkpoint step and a two-node 16-rank imported checkpoint step at 384x640; exported both post-step logical packages; re-exported the common before-state with the CAL-A trace identity; ran chunked logical update comparison.

### Measured Results

Both runs returned zero. Generator loss was exactly `0.43723111103534534` in both layouts. Critic loss was `1.498900294303894` (world1) versus `1.4989155530929565` (world16). Generator master update cosine/relative-L2 were `0.9999966667 / 0.00258218`; Generator BF16 module was `0.9999885007 / 0.00479583`. Critic master was `0.9994933556 / 0.03188889`; Critic first moment was `0.9988771502 / 0.04778886`. EMA hashes differed because decay was zero and therefore inherited the Generator BF16 difference. Comparison wall time was 825.69 seconds.

### Correctness

FAIL under the preregistered delta-relative tolerance. Exact checkpoint-boundary import, input hashes, optimizer steps, scheduler logical state, versions, and phase order pass. Post-update state is numerically close in direct-state norm, but update-relative error exceeds the contract and may not be relabeled as PASS.

### Performance

Not Run. The 278.96-second world16 diagnostic step and 90.71-second single diagnostic step include checkpoint-gate conditions and are not formal timing results.

### Interpretation

The failed hypothesis is not a lifecycle failure. The Generator loss is bit-identical and the direct Generator first-moment difference is small, while adaptive master/module deltas amplify it and the post-Generator Critic update amplifies it further. A final bounded repair will test DeepSpeed's mathematically equivalent pre-divide reduction path. It is gate-only and cannot be counted as M1 performance or silently replace the frozen A-scalable baseline.

### Files Changed

Logical checkpoint export/comparison manifests; immutable raw logs/traces; `research_log.md`; gate-only launcher guard for an optional gradient pre-divide diagnostic.

### Failure or Limitation

The first compare attempt also exposed a stale pre-CAL trace hash in the before-state manifest; it was repaired by a non-destructive CAL-A re-export with identical payload SHA values. Native numerical divergence remains. The CLI returned process code zero while recording JSON status FAIL; downstream gates must inspect manifest status, not process code alone.

### Gate Status

G1.0 OPEN for one final repair; G1a BLOCKED. If the second repair fails, the M1 contract requires a blocked/negative report and no planner or async implementation.

### Next Highest-Value Question

Does world-size pre-division remove collective-order rounding sufficiently to satisfy native cross-layout update tolerances, or is the remaining difference inherent to ZeRO optimizer partition layout?

### Next Command

Run one world16 gate-only diagnostic from the same native checkpoint with `gradient_predivide_factor=16`, export step 11, and compare against the unchanged single-rank reference.

## Iteration 133

### Milestone / Gate

M1.0 final bounded cross-layout repair and Gate G1 closure.

### Question

Can a gate-only DeepSpeed `gradient_predivide_factor=16` diagnostic make the native world1 ZeRO-2 and world16 ZeRO-1 one-step updates satisfy the frozen cross-layout correctness contract?

### Hypothesis

If collective post-scaling and reduction rounding are the dominant source of the native cross-layout update divergence, world-size pre-division will reduce the Generator and Critic update relative L2 errors to at most `2e-3` without changing the logical training state or trace.

### Importance

This is the second and final repair allowed by G1.0. Failure requires stopping M1 before causal profiling, planner construction, or async runtime work rather than silently weakening the canonical-checkpoint contract.

### Evidence Needed

A successful official 384x640 world16 step from the same CAL-A logical step-10 package, exact trace/version/phase records, an exported logical step-11 package, and chunked comparisons against the unchanged world1 reference for module/master parameters, Adam m/v, scheduler, and EMA.

### Refutation Condition

Any required update has cosine below `0.999` or relative L2 above `2e-3`, or the logical scheduler/optimizer/EMA/version state differs.

### Allowed Changes

A hard-guarded checkpoint-gate-only DeepSpeed gradient pre-divide option, checkpoint comparison exit-code repair, manifests, tests, and documentation. No frozen M0 policy, lifecycle policy, prefetch, loss, optimizer semantics, trace, or model change.

### Commands

Ran the two-node world16 official 384x640 CAL-A diagnostic with `DISTILL_M1_CHECKPOINT_GATE_MODE=1` and `gradient_predivide_factor=16`, exported `/share/project/shuchenweng/DistillSchedule_assets/m1/checkpoints/logical_world16_predivide16_step11_v3`, compared it with the unchanged world1 logical step-11 reference, reran focused tests, and reran the full authoritative suite in `/tmp/distillschedule-helios-probe-torch24`.

### Measured Results

Both node launchers returned zero. Generator master cosine/relative-L2 were `0.9999966499 / 0.0025886646`; Generator BF16 module was `0.9999886848 / 0.0047572696`; Generator Adam m was `0.9999999024 / 0.0004419272`; Generator Adam v was `0.9999999997 / 0.0000235932`. Critic master was `0.9995180231 / 0.0310447556`; Critic BF16 module was `0.9996254106 / 0.0273698969`; Critic Adam m was `0.9997496307 / 0.0223771978`; Critic Adam v was `0.9999960166 / 0.0028227760`. EMA equality was false. Logical comparison took 817.77 seconds. The final full suite passed: `159 passed, 32 warnings in 42.43s`.

### Correctness

FAIL. Exact logical checkpoint serialization/import, payload inventories, CAL-A trace identity, and native execution pass, but both the default and pre-divide native updates violate the frozen `relative L2 <= 2e-3` contract. The pre-divide repair does not make the Generator master update pass and leaves much larger Critic divergence.

### Performance

Not Run. M1.1 causal profiling, strong static Pareto expansion, planner, and async runtime were not executed. The checkpoint-gate diagnostic times are not formal performance measurements.

### Interpretation

The narrow hypothesis that collective post-scaling is the dominant cause is refuted. The remaining divergence is consistent with, but does not prove, ZeRO partition/update arithmetic plus sequential amplification into the post-Generator Critic phase. This is a G1.0 contract blocker, not evidence that lifecycle optimization itself is ineffective.

### Files Changed

`scripts/distillschedule/summarize_iteration131.py`; Iteration-133 checkpoint/compare/gate manifests; `docs/15_m1_profile_report.md`; `docs/16_m1_gate_report.md`; `README_AUTORESEARCH.md`; `docs/10_open_questions.md`; lifecycle artifact placeholders/tables; `research_log.md`.

### Failure or Limitation

The two allowed G1.0 repairs are exhausted. No causal lifecycle profile was run, so no claim is made about removable copy wait, achievable overlap, a strongest M1 expert, or an automatic plan. The checkpoint comparison CLI was corrected to return nonzero when its JSON status is FAIL.

### Gate Status

**Gate G1: NEGATIVE / contract-blocked at G1.0.** M1 stops here under the frozen execution contract. M2 was not run.

### Next Highest-Value Question

Should future work define a layout-invariant logical optimizer reference, bind correctness comparisons to the same ZeRO layout while reporting cross-layout drift separately, or re-freeze a numerically invariant backend before lifecycle experiments?

### Next Command

Not Run. Await explicit authorization and a correctness-contract decision; do not enter M2 automatically.

## Iteration 134

### Milestone / Gate

Post-G1, explicitly authorized M1.1 causal transfer audit; cross-layout Gate G1.0 verdict unchanged.

### Question

Is the frozen A-local block streamer limited by the machine's PCIe/NVLink-host path, or by pageable storage, per-tensor allocation/launch, NUMA placement, and Python/runtime overhead?

### Hypothesis

A paired pageable/pinned transfer sweep plus an exact 27-tensor Helios-block control can establish a measured bandwidth ceiling and identify the dominant primitive without calling `.to()` wall time PCIe time.

### Importance

A-local issued 2.530 TB of immutable materialization in ten steps and spent 264.357 s in the materialization wall interval. If its effective rate is far below a legal pinned ceiling, lifecycle work has a causal performance target rather than only an estimated motivation.

### Evidence Needed

Both nodes; official torch/CUDA environment; 16 MiB–1 GiB; H2D/D2H; pageable/pinned; one/four streams; local and deliberately remote NUMA; exact 27-tensor/670.231-MiB block distribution; independent wall and CUDA-event intervals.

### Refutation Condition

Pinned bandwidth is not materially above pageable bandwidth, current streamer is already near the pinned ceiling, or local-vs-remote NUMA explains the whole gap.

### Allowed Changes

Transfer benchmark, summarizer, manifests, docs, and a synchronous pinned-host primitive with tests. No loss, optimizer, operation trace, frozen baseline, prefetch, or placement change.

### Commands

Ran `bench_transfer_audit.py` concurrently on both nodes under torch 2.10.0+cu128; bound GPU0 controls to NUMA 3 and NUMA 7; extracted the real block-0 shape inventory from the official safetensors index; ran `bench_streamer_path.py` on both nodes; added private pinned immutable-host support; ran focused tests and the full suite.

### Measured Results

Large-transfer medians across the two nodes: pageable H2D 11.702 GiB/s, pinned H2D 24.027 GiB/s (one stream) and 24.252 GiB/s (four streams), pageable D2H 9.028 GiB/s, pinned D2H 24.411 GiB/s. Wrong-NUMA large H2D changed pageable by 0.57% and pinned by -0.05%. The exact 27-tensor block achieved 11.409 GiB/s on the current pageable allocation path, 24.015 GiB/s when the same tensors were pinned, and 24.394 GiB/s when flat+pinned. Frozen A-local's wall-derived rate is 8.913 GiB/s, 36.75% of the pinned ceiling. Full regression: 161 passed, 34 warnings in 52.77 s.

### Correctness

The transfer microbenchmark performs exact deterministic copy checks. The new pinned-host streamer test preserved output, input gradient, LoRA gradient, shape, stride, and idle lifetime invariants exactly.

### Performance

Measured causal primitive: pinning gives 2.10x over the current per-tensor pageable block microbenchmark. The 41.34–44.27 s/logical-step projections are explicitly upper-bound estimates, not Helios measurements.

### Interpretation

Pinned ownership is the dominant measured lever. Flattening/preallocation is secondary for this shape. NUMA distance does not explain the standalone large-transfer gap on these nodes. The production streamer remains below even the pageable control, leaving allocation/Python/runtime residual in addition to pageable DMA.

### Files Changed

`runtime/frozen_block_streaming.py`; `runtime/official_baseline.py`; `train_helios.py`; transfer/block benchmark and summary scripts; tests; Iteration-134 manifest and raw artifacts.

### Failure or Limitation

CUDA events are not CUPTI-observed memcpy bytes, and no end-to-end pinned 14B result exists in this iteration. The prior cross-layout contract remains failed.

### Gate Status

Causal ceiling positive. Planner still not authorized by this result alone.

### Next Highest-Value Question

Can persistent or bounded pinned state realize the primitive gain in official-shape training without violating host admission?

### Next Command

Run synchronous bounded staging control, then official 384x640 persistent-pinning probes from the same native checkpoint and CAL-A trace.

## Iteration 135

### Milestone / Gate

M1.1 pinned-host feasibility probes; environment blocked before an unconfounded end-to-end result.

### Question

Can a bounded pinned arena or persistent pinned immutable state improve the actual A-local streamer while respecting host capacity?

### Hypothesis

A bounded synchronous arena will retain most pinned H2D benefit, or persistent pinning of at least the Generator will complete and expose a measurable official-shape speedup.

### Importance

The primitive ceiling is not a system result. A real path must include host ownership, CPU staging, model swaps, checkpoint recomputation, and node admission.

### Evidence Needed

Exact-block bounded-staging microbenchmark; official 384x640 step-10 resume; successful Generator and Score streamer installation; one G-update and one Critic-only step; host/GPU traces and OOM counters.

### Refutation Condition

Synchronous staging is slower than pageable direct copy, persistent pinning is rejected, or host/driver failure prevents reaching a training boundary.

### Allowed Changes

Bounded staging benchmark and a component selector for the same pinned-host primitive. No async copy, planner, loss, optimizer, trace, or model change.

### Commands

Measured a single reusable 135-MiB pinned staging arena on the exact block shape. Ran full-Generator+Score pinning on node0 and Generator-only pinning on node1 from `checkpoint-10`, CAL-A hash `ebce071...`, official 384x640, with CPU and memory bound to NUMA node 3. Added a component selector and its option tests. Attempted in-container GPU reset after both nodes entered CUDA error 802.

### Measured Results

Synchronous bounded staging reached 7.706 GiB/s (7.740 GiB/s with preallocated GPU destinations), 32.37% below the 11.395-GiB/s current pageable control. Full pinning installed 28,111,544,320 pinned Generator bytes, then failed during Score capture with CUDA unspecified launch failure. Generator-only pinning installed the same Generator bytes and a pageable Score streamer, then was SIGKILLed during the first Score model move. Its sampled cgroup peak was 128.42 GiB. NUMA node 3 provides 129,003 MiB (125.98 GiB), and node1 `oom_kill` increased 7->8. The full-pinning probe sampled 130.80 GiB and also increased OOM-kill counters.

### Correctness

No training boundary was reached, so loss/update correctness is Not Run. Option tests passed 29/29 after clearing inherited launcher environment. The earlier full suite and pinned-streamer CUDA test passed before the component-selector-only change.

### Performance

Synchronous bounded staging is a measured negative result. Persistent-pinning end-to-end performance is Not Run/inconclusive because both official runs were launch-confounded.

### Interpretation

The key new systems finding is not “pinning fails”; it is that binding the entire 100+-GiB CPU-offloaded process to the GPU-local 126-GiB NUMA node is invalid. Node-wide free memory remained ample, but the local node exhausted. The transfer audit already showed no large-transfer penalty for remote-node pages, so whole-process membind had cost without benefit. Async staging remains untested.

### Files Changed

Component-scoped pin option; staging microbenchmark; `artifacts/manifests/iteration_135_m1_pinned_host_probe.json`; profile/gate report addenda; failure logs and traces.

### Failure or Limitation

After the NUMA-local failures both nodes return CUDA error 802 (`system not yet initialized`). `nvidia-smi --gpu-reset` fails because Fabric Manager is unavailable and NVSwitch GPU reset is unsupported. This environment cannot run another GPU experiment until the platform restarts the job/nodes.

### Gate Status

Cross-layout G1.0 unchanged FAIL. Causal transfer ceiling positive; sync staging negative; persistent pinning inconclusive; planner/async Not Run.

### Next Highest-Value Question

Does persistent all-state or Generator-only pinning complete without whole-process `--membind`, and what end-to-end rate does it realize?

### Next Command

After node restart, rerun the same two official-shape probes without whole-process memory binding (record NUMA placement rather than forcing all host state into one node), then perform paired timing only if the memory run reaches both training phases.

## Iteration 136

### Milestone / Gate
M1.1 environment recovery and host-admission safety.

### Question
Was the earlier failure intrinsic pinned-memory infeasibility, or an invalid launch constraint?

### Hypothesis
Binding the complete 100+-GiB CPU-offloaded process to one 125.98-GiB NUMA node caused
the cgroup/NUMA OOM; removing process-wide membind will allow the same policy to run.

### Importance
Without this distinction, a launcher bug would be misreported as a lifecycle negative result.

### Evidence Needed
Healthy CUDA/IB after restart, no running GPU jobs, successful unbound official-shape probe,
and unchanged cgroup OOM counters.

### Refutation Condition
The same unbound policy fails at comparable host usage or reaches a genuine GPU OOM.

### Allowed Changes
Host admission guard, launcher safety, tests, and documentation only.

### Commands
Revalidated both 8xA100 nodes, CUDA, IB, exact environment, HEAD, and assets; added a
pre-allocation rejection for process-wide NUMA binding; ran the complete test suite.

### Measured Results
Both nodes returned healthy CUDA/IB. The complete suite passed 166 tests. No GPU process
was active. The subsequent unbound probes completed with no cgroup OOM event.

### Correctness
Launcher behavior and rejection tests pass; model semantics unchanged.

### Performance
Not measured in this iteration.

### Interpretation
The 128–131-GiB crash was a CPU/NUMA placement failure, not a pinned-state or GPU-memory
capacity result.

### Files Changed
`runtime/host_admission.py`, `tests/test_host_admission.py`, M1 launcher, documentation.

### Failure or Limitation
No per-allocation NUMA policy was added; only the unsafe whole-process policy is rejected.

### Gate Status
Environment gate open.

### Next Highest-Value Question
Does unbound persistent pinning deliver an official-shape end-to-end benefit?

### Next Command
Run Generator-only and Generator+Score memory probes, then independent formal timing.

## Iteration 151

### Milestone / Gate
M1.1 causal profile; Gate G1a.

### Question
Can persistent pinned ownership of immutable frozen blocks convert the standalone H2D
ceiling into a real-14B, same-layout end-to-end Pareto point?

### Hypothesis
Pinning Generator alone or both backbones will reduce lifecycle time by at least 10%
without changing the official 384x640 execution contract.

### Importance
This is the decision gate for building the bounded M1 static Pareto rather than closing
lifecycle optimization as a negative result.

### Evidence Needed
Successful memory probes, three independent formal runs per policy, matched trace and
checkpoint, host/GPU admission, version/loss sanity, and paired tracer overhead.

### Refutation Condition
No admitted policy reduces end-to-end time by 10%, or required state is below the 10%
footprint threshold.

### Allowed Changes
Persistent pinned host ownership and low-overhead counters only; no prefetch, planner,
loss, optimizer, trace, placement, or model change.

### Commands
Ran unbound Generator-only and all-pinned memory probes; ran three independent steps
10–19 timing launches for pageable, Generator-pinned, and all-pinned policies; ran a
paired counters-overhead launch and formal summarizer.

### Measured Results
Pageable mean 61.3314 s/logical step. Generator-pinned mean 48.1654 s (1.273x,
-21.47%) with 28.112 GB pinned state. All-pinned mean 43.6389 s (1.405x, -28.85%) with
56.223 GB pinned state. The all-pinned memory probe peaked at 172.48 GiB cgroup and
38,222 MiB GPU. Paired median profiler overhead was -4.10%, interpreted as no measurable
positive overhead under variance.

### Correctness
Partial pass: exact trace/checkpoint identity, finite loss/gradient norms, correct G/C
version advancement, and exact block-level output/input-gradient/LoRA-gradient tests.
Full policy boundary SHA/update equivalence and steps 0–9 leak run remain pending.

### Performance
Measured positive causal result. All-pinned reduces end-to-end time 28.85% versus the
same-layout pageable control.

### Interpretation
Pinned immutable ownership is a strong primitive and creates a real memory-performance
tradeoff. It is not yet an automatic-planner result and private pinning is not assumed
8x/node scalable.

### Files Changed
Iteration-151 manifest, M1 profile/gate report, README, open questions, and research log.

### Failure or Limitation
Cross-layout strict update contract remains separately failed. Expert-Minimum-Share,
1/2/4-rank calibration, 8-rank admission, shadow IR, planner, and async are Not Run.

### Gate Status
G1a PASS via condition B. Proceed to M1.2 only.

### Next Highest-Value Question
What is the smallest mixed private/shared immutable block set that admits eight ranks
per node while retaining most private/pinned performance?

### Next Command
Compute exact per-block host savings, implement mixed host ownership with tests, then run
static admission before an actual eight-rank/node experiment.

## Iteration 152

### Milestone / Gate
M1.2 Strong Static Pareto — Expert-Minimum-Share planning and primitive.

### Question
What is the smallest mixed private/shared immutable block set projected to admit direct
A-local x8 under the mandatory 64-GiB reserve?

### Hypothesis
Because the measured deficit is 10.483 GB and each shared 702.789-MB block saves seven
private copies, three shared blocks are sufficient; choosing lower-frequency Score blocks
minimizes expected slowdown.

### Importance
This tests the missing middle between fast all-private A-local and slow all-shared
A-scalable instead of comparing only two extremes.

### Evidence Needed
Exact per-block bytes/materialization counts, conservative admission arithmetic,
mixed-ownership correctness tests, then actual one-rank and eight-rank measurements.

### Refutation Condition
The static model needs more than the bounded candidate set, mixed ownership changes
outputs/gradients, or actual cgroup peak violates the reserve-qualified limit.

### Allowed Changes
Per-block immutable host ownership, static admission helper, expert launcher, tests, and
artifacts. No planner, async transfer, loss, optimizer, trace, or placement change.

### Commands
Parsed real streamer events from Iteration 137; computed minimum set from the measured
A-local x8 peak/cgroup cap; implemented component-specific mixed ownership and a frozen
minimum-share launcher; ran the full test suite.

### Measured Results
Each of 80 physical blocks is 702,788,608 bytes. Generator and Score materialization
counts are 16 and 7 per block in the two-step CAL-A probe. Three Score blocks project
14,758,560,768 bytes node saving, peak 743,049,026,560 bytes, and 4,275,282,944 bytes
reserve-qualified headroom.

### Correctness
Unit/CUDA tests pass 172/172, including exact block output, input gradient, and LoRA
gradient under mixed ownership. Real-14B mixed-policy correctness is Not Run.

### Performance
Not Run. Admission figures are Estimated, not Measured.

### Interpretation
Three Score blocks are the preregistered minimum-share candidate. The model is optimistic
about physical page sharing, so actual cgroup/PSS evidence remains authoritative.

### Files Changed
Mixed ownership runtime/options/train seam; minimum-share admission helper/planner script;
node launcher; tests; Iteration-152 manifest; profile report and research log.

### Failure or Limitation
Projected headroom is only 3.98 GiB. Residual/runtime variance may require the bounded
four-block fallback, but no post-hoc larger set may be called the three-block minimum.

### Gate Status
M1.2 primitive and static plan ready; actual admission pending.

### Next Highest-Value Question
Does the three-Score-block policy preserve official-shape semantics and pass actual
eight-rank/node reserve admission?

### Next Command
Commit the clean primitive, run a one-rank mixed-policy memory probe, then launch the
matched CAL-A ZeRO-1 all-private control, then the actual eight-rank/node mixed-policy
admission with wave 1.

## Iteration 153

### Milestone / Gate
M1.2 static Pareto setup.

### Question
Can the preregistered three-Score-block single-rank run resume native checkpoint-10?

### Hypothesis
The relative resume path names the frozen checkpoint and the run will start at logical step 10.

### Importance
A policy run started from the wrong logical state is invalid even if its tensors and shape are otherwise correct.

### Evidence Needed
`checkpoint_resume_complete`, step IDs 10--11, trace hash, and successful return code.

### Refutation Condition
The launcher reports that the checkpoint does not exist or begins at step 0.

### Allowed Changes
None during the run.

### Commands
Launched `iteration153_m1_minshare3_single_step10_12_memory`; inspected console and launcher metadata.

### Measured Results
The relative path was interpreted after changing working directory. Console reported that the checkpoint did not exist and began at step 0. The run was manually stopped and returned 1.

### Correctness
FAIL: wrong logical start state.

### Performance
Excluded.

### Interpretation
This is a setup failure, not a policy result. The launcher must canonicalize and validate resume paths before model construction.

### Files Changed
Subsequent launcher fail-fast fix; diagnostic manifest retained.

### Failure or Limitation
Three unintended steps were executed before manual termination; they are not evidence.

### Gate Status
M1.2 unchanged.

### Next Highest-Value Question
Does an absolute checkpoint path expose any real mixed-ownership semantic error?

### Next Command
Rerun with a verified absolute resume path.

## Iteration 154

### Milestone / Gate
M1.2 matched eight-rank host-admission control.

### Question
What is the actual absolute cgroup peak of the all-private ZeRO-1 policy with construction wave disabled?

### Hypothesis
The historical deficit inferred from an increment peak is accurate enough to select three shared blocks.

### Importance
Expert-Minimum-Share cannot be evaluated against a ZeRO-2 failure or an increment-from-initial value.

### Evidence Needed
Official-shape one-step completion, eight phase traces, cgroup absolute peak, reserve admission, and OOM counters.

### Refutation Condition
The matched absolute deficit differs materially or its maximum occurs before ownership activation.

### Allowed Changes
No policy change; matched all-private control only.

### Commands
Ran `iteration154_m1_alocal_private_zero1_cal_a_x8_control` with CAL-A, ZeRO-1, construction wave 0, swap wave 1.

### Measured Results
All eight ranks completed one G+C step in 253.39 s console time. Absolute peak was 769,603,149,824 B (716.749 GiB), above the 747,324,309,504-B admission limit. No OOM event occurred. The maximum preceded rank-0 `score_backbone_loaded` by about 29.6 s.

### Correctness
PASS at phase/version/loss sanity level; no complete formal update-delta comparison in this memory run.

### Performance
One-step memory-mode diagnostic only.

### Interpretation
The prior 757.808-GB number was an increment above initial cgroup usage. More importantly, the blocking peak is a cold phase that later host ownership cannot affect.

### Files Changed
Iteration-154 manifest and raw traces.

### Failure or Limitation
Resource sampling is periodic; later summaries conservatively reconcile phase-boundary samples.

### Gate Status
Old scalar Expert-Minimum-Share model refuted.

### Next Highest-Value Question
How much memory does partial sharing actually save after its ownership policy is active?

### Next Command
Run a matched mixed-ownership candidate without changing construction admission.

## Iteration 155

### Milestone / Gate
M1.2 partial immutable ownership.

### Question
Can five shared Score blocks alone close the matched reserve deficit?

### Hypothesis
The corrected absolute-byte model will admit the node once five lower-frequency Score blocks are shared.

### Importance
This tests the missing middle between all-private and all-shared host state.

### Evidence Needed
Eight-rank completion, per-rank ownership counters, phase-aware cgroup deltas, reserve result, and loss/version sanity.

### Refutation Condition
The cold peak remains above the admission limit even though ownership-active memory falls.

### Allowed Changes
Five Score blocks use `HOST_SHARED_IMMUTABLE`; all other per-rank policies remain fixed.

### Commands
Ran `iteration155_m1_minshare5_zero1_cal_a_x8_admission`; compared it with Iteration 154 using phase timestamps.

### Measured Results
Every rank installed five shared Score blocks (3,513,943,040 B/rank). The run completed, but peak remained 764,006,031,360 B (711.536 GiB), so reserve admission failed. Ownership-active peak fell by 16.051 GB, while all-time cold peak fell only 5.597 GB.

### Correctness
Generator loss matched exactly at `0.45730964033703136`; Critic loss delta was below `2e-4`; versions advanced correctly.

### Performance
Memory-mode one-step only; no formal speed claim.

### Interpretation
Partial sharing is a real steady-state memory primitive but cannot be applied retroactively to construction. A phase-unaware planner would false-admit this plan.

### Files Changed
Phase-aware admission fields, host-ownership summarizer, tests, manifest, and tables.

### Failure or Limitation
The 8-rank result is validation-only and is excluded from planner coefficient fitting.

### Gate Status
Sharing-only admission hypothesis REFUTED; memory-Pareto hypothesis VERIFIED.

### Next Highest-Value Question
Can a construction-specific wave remove the unaffected cold peak without changing per-rank state policy?

### Next Command
Run matched all-private with construction wave four.

## Iteration 156

### Milestone / Gate
M1.2 mixed-ownership correctness diagnostic.

### Question
Does the imported single-rank checkpoint complete the strict same-step semantic assertions?

### Hypothesis
Mixed ownership preserves Generator/Critic/EMA semantics.

### Importance
A memory policy cannot enter the expert set without at least one complete G+C and Critic-only semantic run.

### Evidence Needed
Steps 10--11, semantic contract completion, exact zero-decay EMA, and zero live state.

### Refutation Condition
A semantic assertion fails.

### Allowed Changes
No loss, optimizer, trace, or policy changes.

### Commands
Ran `iteration156_m1_minshare3_single_step10_12_memory` from the imported checkpoint.

### Measured Results
The run reached step 10 but failed `zero-decay EMA must exactly match the Generator source`.

### Correctness
FAIL at EMA assertion.

### Performance
Excluded.

### Interpretation
The failure could still be cross-layout checkpoint arithmetic; a native same-layout control is required before blaming mixed ownership.

### Files Changed
Failure manifest only.

### Failure or Limitation
Cross-layout and policy effects were confounded.

### Gate Status
Correctness unresolved.

### Next Highest-Value Question
Does the exact same failure reproduce from the native single-rank M0 checkpoint?

### Next Command
Run the same policy from native checkpoint-10.

## Iteration 157

### Milestone / Gate
M1.2 EMA root-cause control.

### Question
Is the zero-decay EMA failure caused by cross-layout import?

### Hypothesis
The native same-layout checkpoint will pass if import arithmetic is the cause.

### Importance
The repair must target the actual semantic defect rather than relax an assertion.

### Evidence Needed
Same assertion under the native checkpoint with otherwise identical policy.

### Refutation Condition
The native same-layout run fails identically.

### Allowed Changes
Checkpoint source only.

### Commands
Ran `iteration157_m1_minshare3_single_same_layout_step10_12_memory`.

### Measured Results
The native checkpoint reproduced the exact zero-decay EMA assertion.

### Correctness
FAIL, but cross-layout import is refuted as the cause.

### Performance
Excluded.

### Interpretation
BF16 algebra `shadow - (shadow - source)` rounds twice when decay is zero. The EMA equation requires an exact assignment at bootstrap.

### Files Changed
`SparseShardEMA.step` exact decay-zero branch and BF16 unit test.

### Failure or Limitation
This changes M1 EMA bootstrap state relative to historical M0; it is treated as a shared correctness substrate fix and receives no performance credit.

### Gate Status
Bounded repair identified.

### Next Highest-Value Question
Does exact copy make the complete native same-layout semantic run pass?

### Next Command
Rerun steps 10--11 with the corrected EMA branch.

## Iteration 158

### Milestone / Gate
M1.2 phase-specific node admission.

### Question
Can construction wave four alone admit the all-private eight-rank policy?

### Hypothesis
The rejected maximum is a simultaneous construction transient; limiting construction concurrency to four removes it.

### Importance
This distinguishes a transient admission problem from long-lived host ownership replication.

### Evidence Needed
Eight-rank completion, no OOM, conservative peak below 696 GiB, and phase of new maximum.

### Refutation Condition
Peak remains above the reserve-qualified limit or steady state becomes the same blocker.

### Allowed Changes
Construction wave only; host state remains all private.

### Commands
Ran `iteration158_m1_alocal_private_zero1_cal_a_x8_cw4`.

### Measured Results
The run completed with no OOM. Conservative observed peak was 730,855,907,328 B (680.663 GiB), 15.337 GiB below the admission limit. The maximum moved to Critic backward.

### Correctness
PASS at phase/version/loss sanity level.

### Performance
Median rank step was 241.582 s in memory mode; not formal timing.

### Interpretation
Construction wave four solves the cold transient. Under this admission policy, zero shared blocks are sufficient; `Expert-Minimum-Share` becomes the zero-share point.

### Files Changed
Iteration-158 manifest and phase-aware reports.

### Failure or Limitation
Cold construction becomes longer; formal steady timing is unaffected in principle but not claimed here.

### Gate Status
Phase-specific admission hypothesis VERIFIED.

### Next Highest-Value Question
Does partial sharing add a second Pareto improvement after construction is admitted?

### Next Command
Combine construction wave four with five shared Score blocks.

## Iteration 159

### Milestone / Gate
M1.2 same-layout real-14B semantic gate.

### Question
Does the corrected EMA substrate complete a mixed-ownership G+C plus Critic-only sequence?

### Hypothesis
The only failure in Iterations 156--157 was BF16 zero-decay EMA arithmetic.

### Importance
The static primitive requires end-to-end semantic evidence, not just block unit tests.

### Evidence Needed
Native checkpoint resume, steps 10--11, version/gradient assertions, exact EMA, and zero live offload/streamer state.

### Refutation Condition
Any semantic assertion or leak remains.

### Allowed Changes
Exact decay-zero EMA copy and resume-path fail-fast only.

### Commands
Ran `iteration159_m1_minshare3_single_same_layout_step10_12_ema_fixed`; ran the clean full test suite.

### Measured Results
Return code 0. One Generator update, two Critic updates, two EMA steps, and `semantic_contract_complete` were observed. Final activation live bytes and streamer resident bytes were zero. GPU max allocated/reserved were 33.727/38.663 GB; sampled cgroup peak was 157.613 GB. The full suite passed 175 tests at this checkpoint and 176 after adding the final peak-reconciliation test.

### Correctness
PASS for the specified same-layout semantic sequence.

### Performance
Not a formal timing run.

### Interpretation
Mixed ownership is not the source of the EMA failure. The M1 substrate now implements the intended exact bootstrap equation.

### Files Changed
EMA fix, tests, launcher path validation, Iteration-159 manifest.

### Failure or Limitation
Full update-vector comparison against an all-private run remains pending; the cross-layout numerical contract remains historically failed.

### Gate Status
Same-layout primitive correctness gate PASS.

### Next Highest-Value Question
What is the measured joint node Pareto with both phase-specific levers active?

### Next Command
Complete and summarize Iteration 160.

## Iteration 160

### Milestone / Gate
M1.2 joint expert static admission point.

### Question
Does five-block partial sharing improve the admitted construction-wave-four policy?

### Hypothesis
Once construction is controlled, partial ownership will reduce the Critic-backward peak without changing model semantics.

### Importance
This is the first measured joint lifecycle point and tests whether decisions compose only in their valid phases.

### Evidence Needed
Matched 8-rank run, conservative peak reconciliation, reserve headroom, phase deltas, loss/version sanity, and no OOM/leak.

### Refutation Condition
No peak reduction, reserve failure, or semantic divergence beyond the reference threshold.

### Allowed Changes
Five Score blocks shared; construction wave remains four; all other actions match Iteration 158.

### Commands
Ran `iteration160_m1_minshare5_zero1_cal_a_x8_cw4_joint`; summarized against Iteration 158.

### Measured Results
The run completed with no OOM. Conservative peak was 714,650,247,168 B (665.570 GiB), 15.093 GiB below the matched control and 30.430 GiB below the reserve-qualified limit. Generator loss was exact; Critic loss delta was `5.5313e-5`. Median rank step was 219.390 versus 241.582 s (9.19% lower) in one memory-mode run.

### Correctness
PASS at phase/version/loss and zero-resident-state level; full formal update-vector comparison remains pending.

### Performance
Not Yet Verified. The diagnostic reduction is below 10% and lacks randomized repeated timing.

### Interpretation
Construction concurrency and steady ownership are orthogonal, phase-scoped actions. The joint point is a memory Pareto improvement; automatic selection is still Not Run.

### Files Changed
Observed-peak reconciliation, joint manifest/table, profile/gate reports, README, open questions, and research log.

### Failure or Limitation
Eight-rank validation cannot be used to fit planner parameters. No formal multiple-cycle timing or bounded measured oracle yet.

### Gate Status
M1.2 causal admission point measured; final Gate G1 remains OPEN.

### Next Highest-Value Question
Does this partial-share point remain Pareto under same-checkpoint repeated formal timing, and can the frozen action set be represented by a leak-free shadow lifetime graph?

### Next Command
Freeze the bounded static action library, generate same-layout timing checkpoints, then implement the shadow executor without changing execution.

## Iteration 161

### Milestone / Gate
M1.2 / bounded same-primitive static Pareto.

### Question
Can the static oracle action space be frozen before further timing without leaking HOLD-S/HOLD-V or silently giving automatic planning extra primitives?

### Hypothesis
A causal-pruned set below 20 named plans is sufficient to cover the measured local pinning frontier, phase-aware scalable ownership, streaming windows, activation thresholds, and two legal recompute regions without a full Cartesian product.

### Importance
The measured oracle is only meaningful if its candidate set is executable, preregistered, shared by expert/heuristic/automatic methods, and fixed before held-out profiling.

### Evidence Needed
A versioned JSON action library, unique plan identifiers, exact runtime-setting validation, explicit implementation-required states, CAL-A-only provenance, and a formal run order.

### Refutation Condition
Duplicate or ambiguous plans, an action label that can disagree with its launched environment, use of held-out profiles, or an unbounded Cartesian search.

### Allowed Changes
Action-library/config validation, fixed-plan launcher seam, tests, manifests, and documentation. No model, loss, optimizer, trace, placement, or prefetch change.

### Commands
Created `configs/m1_static_action_library.json`; added `runtime/m1_action_library.py`, `validate_m1_action.py`, and exact launcher/action validation; began retarget/import of the CAL-A logical state into a native world8 ZeRO-1 checkpoint.

### Measured Results
The frozen library contains 14 named plans. Iterations 151, 158, and 160 are the only measured anchors. HOLD-S/HOLD-V fine profiles were not read for tuning. The first formal pair is preregistered as all-private wave4 versus Score-blocks-0--4 shared wave4 from the same world8 checkpoint.

### Correctness
Action labels now fail closed when host ownership, shared indices, streaming window, activation threshold, pin policy, or construction wave differs from the frozen plan. The two recompute-region plans are explicitly non-launchable until their executor parity exists.

### Performance
Not Run in this iteration. One-step Iteration 158/160 durations remain diagnostic only.

### Interpretation
The bounded oracle is now a measured-plan concept rather than a cost-model prediction. Under construction wave4, zero shared blocks are sufficient for admission, so the five-block point is `Expert-Partial-Share-5`, not `Expert-Minimum-Share`.

### Files Changed
`configs/m1_static_action_library.json`, action resolver/validator, node profile launcher, unit tests, Iteration-161 manifest, reports, and research log.

### Failure or Limitation
Recompute-region execution and formal world8 repetitions remain pending. The current action library freezes primitives and run order; it does not establish an oracle winner.

### Gate Status
G1a PASS; M1.2 action library FROZEN; G1b/G1c Not Run; final G1 OPEN.

### Next Highest-Value Question
Does partial Score sharing retain its memory Pareto without degrading formal same-checkpoint world8 steady-state timing?

### Next Command
Verify the imported world8 logical checkpoint, then run randomized independent `S-PRIVATE-CW4-W1-A32` and `S-SCORE5-CW4-W1-A32` timing repetitions over complete five-step cycles.

## Iteration 162

### Milestone / Gate
M1.2 / world8 same-layout correctness precheck.

### Question
Can the frozen five-Score-block partial-share action resume the native world8 logical state and complete one Generator+Critic step followed by one Critic-only step?

### Hypothesis
`S-SCORE5-CW4-W1-A32` changes only immutable host ownership, so trace, loss path, parameter-version ordering, EMA, and leak invariants should remain valid.

### Importance
Formal timing is invalid until the exact world8 action can complete from the same checkpoint state used by its control.

### Evidence Needed
All eight ranks complete steps 10--11 at official 384x640; exact trace hash; G version 2->3; C version 10->11->12; finite losses; exact zero-decay EMA; no Generator update on the Critic-only step; zero final activation/streamer live state; reserve admission.

### Refutation Condition
Any rank failure, OOM, trace mismatch, semantic assertion failure, EMA mismatch, version violation, or leaked live state.

### Allowed Changes
No model/loss/optimizer/trace/action change. Only run execution and evidence summarization.

### Commands
Launched `iteration162_m1_world8_score5_step10_12_correctness` on node 0 with action `S-SCORE5-CW4-W1-A32`, world8 ZeRO-1, events/memory mode, and native checkpoint 10.

### Measured Results
Return code 0; eight ranks completed steps 10 and 11. Conservative node peak was 713,789,034,496 B (664.788 GiB), leaving 33,535,275,008 B under the 696-GiB reserve-qualified limit. Rank-0 G loss was 0.43723111103534534; Critic losses were 1.49895441532135 and 1.5109851360321045.

### Correctness
PASS for trace/version/semantic/EMA/leak precheck on all ranks. Exact update-vector and Adam-state equivalence remain Not Run.

### Performance
Not a formal timing run. Memory/event instrumentation was enabled and node identity is not controlled.

### Interpretation
The partial-share action is executable from the native world8 checkpoint and is safe to admit into the preregistered static comparison.

### Files Changed
Iteration-162 manifest and raw logs/traces.

### Failure or Limitation
Only two logical steps; no vector-level update comparison and no throughput claim.

### Gate Status
M1.2 candidate correctness precheck PASS; final Gate G1 OPEN.

### Next Highest-Value Question
Does the all-private control pass the identical native-world8 contract?

### Next Command
Run `S-PRIVATE-CW4-W1-A32` from the same checkpoint and trace.

## Iteration 163

### Milestone / Gate
M1.2 / world8 same-layout correctness control.

### Question
Can the frozen all-private action complete the identical native-world8 sequence used by Iteration 162?

### Hypothesis
`S-PRIVATE-CW4-W1-A32` should be a reserve-admitted semantic control under construction wave four.

### Importance
A causal policy comparison requires the same world size, checkpoint, trace, logical steps, and semantic assertions.

### Evidence Needed
The same evidence contract as Iteration 162.

### Refutation Condition
Any rank failure, OOM, trace mismatch, semantic assertion failure, version violation, EMA mismatch, or leak.

### Allowed Changes
No model/loss/optimizer/trace/action change. Only all-private host ownership.

### Commands
Launched `iteration163_m1_world8_private_step10_12_correctness` on node 1 concurrently with Iteration 162.

### Measured Results
Return code 0; all eight ranks completed. Conservative peak was 735,143,759,872 B (684.656 GiB), leaving 12,180,549,632 B under the reserve-qualified limit. Rank-0 G loss was 0.43723111103534534; Critic losses were 1.4989327192306519 and 1.5109798908233643.

### Correctness
PASS for trace/version/semantic/EMA/leak precheck on all ranks. Exact update-vector and Adam-state equivalence remain Not Run.

### Performance
Not a formal timing run.

### Interpretation
Both frozen policies are executable and reserve admitted from the same native world8 state.

### Files Changed
Iteration-163 manifest and raw logs/traces.

### Failure or Limitation
The control ran on a different node from the candidate; timing and cross-node residual memory are confounded.

### Gate Status
M1.2 control correctness precheck PASS; final Gate G1 OPEN.

### Next Highest-Value Question
Do corresponding ranks satisfy the loss/semantic contract, and what memory difference is directly observed?

### Next Command
Generate a rank-paired correctness manifest with explicit guardrails.

## Iteration 164

### Milestone / Gate
M1.2 / paired world8 correctness and admission summary.

### Question
Does partial Score sharing preserve the observable training contract relative to all-private execution from the same logical checkpoint?

### Hypothesis
Corresponding rank/step losses remain within the frozen `2e-4` absolute threshold, all semantic assertions pass, and partial sharing increases reserve headroom.

### Importance
This is the minimum evidence required before spending long GPU time on repeated steady-state timing.

### Evidence Needed
Exact trace identity, all-rank semantic checks, paired per-term losses, EMA exactness, version sequence, no leaks, and conservative node peaks.

### Refutation Condition
Any semantic failure, trace mismatch, max loss delta above `2e-4`, OOM, or reserve rejection.

### Allowed Changes
Evidence summarization only. No runtime-policy change.

### Commands
Reconciled Iterations 162/163 into `iteration_164_m1_world8_private_score5_correctness_pair.json`; fixed the node-run summarizer to derive expected steps from `trace_start_step`; ran the full test suite.

### Measured Results
Trace hash was exact. Maximum paired loss absolute delta across 24 corresponding rank/phase/step records was `2.1696090698242188e-05`. Partial sharing reduced the observed cross-node peak by 21,354,725,376 B (2.905%) and increased reserve headroom from 12,180,549,632 B to 33,535,275,008 B. The full suite passed 182 tests.

### Correctness
PASS for same-layout trace, loss, semantic, EMA, version, and leak precheck. Full update-vector cosine/relative-L2 and Adam m/v vector comparison remain Not Run.

### Performance
Not Run formally. The memory-mode diagnostic median was 176.304 s for Score5 versus 157.015 s for private, a 12.28% slowdown, but the policies ran concurrently on different nodes. This sign reverses the earlier Iteration-158/160 diagnostic and is not a performance claim.

### Interpretation
Partial sharing is a real capacity/headroom lever, but its speed effect is unresolved and may be dominated by shared page-cache/DRAM contention. A randomized node-crossover timing design is required.

### Files Changed
Resume-aware summarizer, regression test, Iteration-164 manifest, reports, and research log.

### Failure or Limitation
Cross-node residual affected even ownership-unaffected peak; Iteration 158/160 remains the cleaner same-node causal memory pair. Vector-level correctness is still pending.

### Gate Status
M1.2 world8 semantic precheck PASS; static timing/oracle remains OPEN; G1b/G1c Not Run.

### Next Highest-Value Question
Does Score5 remain Pareto under repeated same-checkpoint, complete-cycle, randomized node-crossover timing?

### Next Command
Run control-mode steps 10--19 with private on node0 and Score5 on node1, then reverse assignments in subsequent rounds.

## Iteration 165

### Milestone / Gate
M1.2 / scalable static formal timing round 1 control.

### Question
What is the control-mode steady timing of the frozen all-private world8 action from checkpoint 10?

### Hypothesis
Two complete five-step cycles execute without profiler overhead or reserve failure and establish the first node-crossover control point.

### Importance
Memory-mode one-step durations cannot choose the strongest scalable expert.

### Evidence Needed
All ranks complete steps 10--19, two G-update steps, eight Critic-only steps, exact trace, rc=0, and reserve admission.

### Refutation Condition
Incomplete cycle, OOM, trace mismatch, or nonzero return code.

### Allowed Changes
No runtime change; control-mode execution only.

### Commands
Ran `iteration165_m1_world8_private_formal_round1_node0`, action `S-PRIVATE-CW4-W1-A32`, on node0.

### Measured Results
G-update mean 201.102 s; Critic-only mean 86.562 s; official weighted step 109.470 s; node throughput 0.073080 samples/s. Sampled cgroup peak was 734,150,598,656 B. All ten steps completed with rc=0.

### Correctness
Trace and complete-cycle execution PASS; detailed semantic gate was already passed in Iteration 163 and was disabled for formal timing.

### Performance
Measured formal per-run timing. Policy comparison remains pending node crossover.

### Interpretation
This is one independent all-private formal launch, not yet the final expert estimate.

### Files Changed
Iteration-165 manifest and timing CSV.

### Failure or Limitation
Node0-only; one launch.

### Gate Status
Static formal timing 1/3 for private; OPEN.

### Next Highest-Value Question
What is Score5 timing under the paired first node assignment?

### Next Command
Summarize Iteration 166.

## Iteration 166

### Milestone / Gate
M1.2 / scalable static formal timing round 1 candidate.

### Question
What is the control-mode steady timing of five-block partial sharing from the same logical state?

### Hypothesis
The capacity benefit will not impose more than 5% slowdown relative to all-private after profiler overhead is removed.

### Importance
A >=10% memory reduction with <=5% slowdown would be a positive Pareto claim; otherwise sharing may be capacity-only.

### Evidence Needed
Same formal contract as Iteration 165.

### Refutation Condition
OOM/incomplete run or slowdown above 5% after crossover.

### Allowed Changes
Only Score blocks 0--4 use shared immutable backing.

### Commands
Ran `iteration166_m1_world8_score5_formal_round1_node1`, action `S-SCORE5-CW4-W1-A32`, on node1.

### Measured Results
G-update mean 214.644 s; Critic-only mean 93.745 s; official weighted step 117.925 s; node throughput 0.067840 samples/s. Sampled cgroup peak was 717,674,156,032 B. All ten steps completed with rc=0.

### Correctness
Trace and complete-cycle execution PASS; detailed semantic precheck passed in Iteration 162.

### Performance
Measured formal per-run timing; causal comparison remains pending crossover.

### Interpretation
On this node assignment Score5 is slower, consistent with added shared-backing/page-cache contention.

### Files Changed
Iteration-166 manifest and timing CSV.

### Failure or Limitation
Node1-only; one launch.

### Gate Status
Static formal timing 1/3 for Score5; OPEN.

### Next Highest-Value Question
How large is the round-1 tradeoff and does it survive reversing nodes?

### Next Command
Pair Iterations 165/166, then launch reversed assignment.

## Iteration 167

### Milestone / Gate
M1.2 / node-crossover timing round 1 comparison.

### Question
Does Score5 dominate all-private in the first formal assignment?

### Hypothesis
Partial sharing creates a memory-performance Pareto rather than a pure capacity/slowdown tradeoff.

### Importance
This decides whether the frozen action library should spend further measurements on partial ownership.

### Evidence Needed
Official weighted time, G/C phase means, node peak, throughput, and explicit node-confounding guardrail.

### Refutation Condition
Score5 is slower without enough memory benefit, or node reversal changes the conclusion.

### Allowed Changes
Summary only.

### Commands
Paired the Iteration-165/166 formal manifests.

### Measured Results
Score5 was 7.7235% slower in official weighted step (117.925 vs 109.470 s), 6.734% slower for G-update and 8.298% slower for Critic-only. It reduced sampled peak by 16,476,442,624 B (2.244%) and throughput by 7.170%.

### Correctness
Both source runs completed the same trace and cycles; full update-vector equivalence remains a separate pending gate.

### Performance
One formal node assignment measured. No final claim until reversed assignment and three launches per policy.

### Interpretation
Round 1 does not meet either positive Pareto threshold: memory reduction is below 10%, and slowdown exceeds 5%. It suggests shared backing is an admission/headroom tool, not a speed optimization, but node identity is still confounded.

### Files Changed
Iteration-167 pair manifest, reports, and research log.

### Failure or Limitation
Policy and node identity change together in this round.

### Gate Status
Static timing round 1 complete; M1.2 OPEN.

### Next Highest-Value Question
Does the 7.7% slowdown persist after swapping policies across nodes?

### Next Command
Launch Score5 on node0 and private on node1 from the same checkpoint and trace.

## Iteration 168

### Milestone / Gate
M1.2 formal timing round 2, Score5 on node0.

### Question
Does Score5's slowdown persist when moved from node1 to node0?

### Hypothesis
If shared backing itself is costly, Score5 remains slower after node reversal.

### Importance
Round 1 confounded policy with node identity.

### Evidence Needed
Two complete cycles, same checkpoint/trace, rc=0, official weighted timing.

### Refutation Condition
Score5 becomes faster than the paired private run after reversal.

### Allowed Changes
Node assignment only.

### Commands
Ran `iteration168_m1_world8_score5_formal_round2_node0`.

### Measured Results
G-update 227.520 s; Critic-only 99.219 s; weighted step 124.879 s; throughput 0.064062 sample/s; sampled peak 717,705,519,104 B; rc=0.

### Correctness
Complete trace/cycles PASS; detailed semantics inherited from Iteration 162.

### Performance
Second independent formal Score5 launch measured.

### Interpretation
Score5 remains slower in the reversed assignment.

### Files Changed
Iteration-168 manifest/CSV.

### Failure or Limitation
Round 2 is globally slower; raw cross-round differences need a blocked model.

### Gate Status
Score5 formal launches 2/3; OPEN.

### Next Highest-Value Question
What is private timing on node1 in the same round?

### Next Command
Summarize Iteration 169.

## Iteration 169

### Milestone / Gate
M1.2 formal timing round 2, private on node1.

### Question
Does the all-private control complete the reversed pairing?

### Hypothesis
Private remains faster than Score5 within round 2.

### Importance
Completes the 2x2 node-crossover design.

### Evidence Needed
Same formal contract as Iteration 168.

### Refutation Condition
Incomplete/OOM or private is slower than Score5.

### Allowed Changes
Node assignment only.

### Commands
Ran `iteration169_m1_world8_private_formal_round2_node1`.

### Measured Results
G-update 218.191 s; Critic-only 95.139 s; weighted step 119.749 s; throughput 0.066806 sample/s; sampled peak 736,070,295,552 B; rc=0.

### Correctness
Complete trace/cycles PASS; detailed semantics inherited from Iteration 163.

### Performance
Second independent formal private launch measured. Within round 2, Score5 is 4.284% slower.

### Interpretation
The slowdown direction survives node reversal, although magnitude changes.

### Files Changed
Iteration-169 manifest/CSV.

### Failure or Limitation
Only two launches per policy.

### Gate Status
Private formal launches 2/3; OPEN.

### Next Highest-Value Question
What policy effect remains after separating node and round offsets?

### Next Command
Fit the preregistered two-round additive crossover summary.

## Iteration 170

### Milestone / Gate
M1.2 two-round node-crossover analysis.

### Question
After separating additive node and round effects, is Score5 a positive Pareto point?

### Hypothesis
Partial sharing saves enough memory with no more than 5% adjusted slowdown.

### Importance
Determines whether partial sharing is a performance Pareto action or only an admission action.

### Evidence Needed
Four formal runs, balanced node assignment, policy means, sampled peaks, and explicit model assumptions.

### Refutation Condition
Memory saving <10% and adjusted slowdown >5%.

### Allowed Changes
Analysis only.

### Commands
Combined Iterations 165, 166, 168, and 169 in a policy+node+round additive fit.

### Measured Results
Raw policy means were 114.610 s private and 121.402 s Score5. The balanced additive estimate assigns +6.793 s (+5.927%) to Score5, +1.662 s to node1, and +8.617 s to round 2. Mean sampled peak falls from 735.110 GB to 717.690 GB: 17.421 GB or 2.370%.

### Correctness
All four runs completed identical checkpoint/trace cycles; vector-level equivalence remains pending.

### Performance
The positive Pareto hypothesis is REFUTED after two rounds: slowdown exceeds 5% and memory saving is below 10%. A third launch remains required by the formal contract before freezing the winner.

### Interpretation
Five-block sharing is currently an admission/headroom mechanism, not a throughput optimization. Large round drift confirms why one-shot timing was misleading.

### Files Changed
Iteration-170 crossover manifest, reports, and research log.

### Failure or Limitation
The additive fit cannot estimate policy-by-node or policy-by-round interactions with one 2x2 crossover.

### Gate Status
Crossover direction stable; static winner not frozen until 3/3 launches.

### Next Highest-Value Question
Does a third randomized assignment keep private faster?

### Next Command
With fixed seed 20260727, run private on node0 and Score5 on node1 for round 3.

## Iteration 171

### Milestone / Gate
M1.2 formal timing round 3, private on node0.

### Question
Does a third independent all-private launch preserve the apparent timing advantage?

### Hypothesis
Private remains faster than Score5 after another complete-cycle launch.

### Importance
The performance contract requires at least three launches per policy and earlier rounds showed large drift.

### Evidence Needed
Official checkpoint-10 steps 10--19, rc=0, weighted/G/C timing and node peak.

### Refutation Condition
The paired Score5 run is faster or the private run fails admission.

### Allowed Changes
Node assignment only; frozen action, checkpoint, trace, loss and optimizer semantics unchanged.

### Commands
Ran `iteration171_m1_world8_private_formal_round3_node0` in control mode.

### Measured Results
G-update 231.322 s; Critic-only 102.935 s; weighted step 128.612 s; throughput 0.062202 sample/s; sampled peak 734,017,085,440 B; rc=0.

### Correctness
Complete trace and two five-step cycles PASS. Detailed same-layout semantic precheck is Iteration 163; vector-level Adam/update parity remains pending.

### Performance
Third independent private launch measured.

### Interpretation
This run is materially slower than the first two private launches, confirming substantial round drift.

### Files Changed
Iteration-171 manifest and CSV.

### Failure or Limitation
Control-mode cgroup sampling is not PSS/VmPin decomposition.

### Gate Status
Private formal launches 3/3.

### Next Highest-Value Question
What is the paired Score5 timing in round 3?

### Next Command
Summarize Iteration 172.

## Iteration 172

### Milestone / Gate
M1.2 formal timing round 3, Score5 on node1.

### Question
Does Score5 remain slower in the third paired assignment?

### Hypothesis
Shared immutable backing imposes a consistent time cost.

### Importance
Rounds 1/2 suggested slowdown, but the formal contract requires a third launch.

### Evidence Needed
Same checkpoint/trace and complete-cycle timing as Iteration 171.

### Refutation Condition
Score5 is faster than private within round 3.

### Allowed Changes
Ownership action only.

### Commands
Ran `iteration172_m1_world8_score5_formal_round3_node1` in control mode.

### Measured Results
G-update 216.039 s; Critic-only 93.997 s; weighted step 118.405 s; throughput 0.067565 sample/s; sampled peak 713,364,803,584 B; rc=0.

### Correctness
Complete trace and cycles PASS; detailed semantics inherited from Iteration 162.

### Performance
Score5 is 7.94% faster within round 3, reversing rounds 1 and 2.

### Interpretation
The stable-slowdown hypothesis is refuted. Static policy conclusions require all three launches and explicit round blocking.

### Files Changed
Iteration-172 manifest and CSV.

### Failure or Limitation
Round 3 repeats the round-1 node assignment; interaction estimates remain weak.

### Gate Status
Score5 formal launches 3/3.

### Next Highest-Value Question
What policy effect and memory tradeoff remain after combining all three rounds?

### Next Command
Run the registered three-round crossover summarizer.

## Iteration 173

### Milestone / Gate
M1.2 three-round static crossover freeze.

### Question
Is Score5 a positive memory-performance Pareto point after three independent launches per policy?

### Hypothesis
Score5 saves at least 10% node memory with no more than 5% adjusted slowdown, or improves time by at least 10%.

### Importance
This decides whether five-block sharing is a strong performance baseline or only an admission action.

### Evidence Needed
Three private and three Score5 complete-cycle manifests, identical contract/hash, per-policy distributions, paired rounds and a blocked descriptive fit.

### Refutation Condition
Memory saving is below 10% without a 10% speedup.

### Allowed Changes
Analysis, tests and documentation only.

### Commands
Added `summarize_m1_crossover.py`, ran the three-round summary, and ran the full test suite.

### Measured Results
Private mean 119.277 s; Score5 mean 120.403 s. The blocked estimate is +2.127 s or +1.783% for Score5. Mean sampled peak falls from 734.746 GB to 716.248 GB: 18.498 GB or 2.518%. Paired Score5 effects are +7.72%, +4.28%, and -7.94%.

### Correctness
All six runs completed the same checkpoint/trace contract; summarizer validation passes. Full suite: 183 passed, 21 warnings.

### Performance
Positive Pareto hypothesis REFUTED. Score5 remains a capacity/headroom point only.

### Interpretation
One-shot timing would have supported opposite conclusions depending on the round. The relevant systems result is a noisy capacity/performance tradeoff, not a reliable speedup.

### Files Changed
Crossover summarizer/test, Iteration-171/172/173 manifests and CSVs, reports and research log.

### Failure or Limitation
The blocked model has one residual degree of freedom and is descriptive, not an uncertainty interval. Full vector-level update/Adam parity remains pending.

### Gate Status
Private/Score5 pair frozen; M1.2 remains OPEN.

### Next Highest-Value Question
Do preregistered Score40 and all-shared ownership points create a >=10% host-memory Pareto or only add contention?

### Next Command
Run official-shape world8 correctness/admission probes for `S-SCORE40-CW4-W1-A32` and `S-ALLSHARED-CW4-W1-A32`.

## Iteration 174

### Milestone / Gate
M1.2 Score40 correctness/admission precheck.

### Question
Can sharing all 40 frozen Score blocks create a >=10% reserve-safe capacity point without changing training semantics?

### Hypothesis
`S-SCORE40-CW4-W1-A32` lowers node peak by at least 10% versus private and preserves the same-layout contract.

### Importance
Score5 saved only 2.518%; a larger ownership point is needed for a meaningful memory Pareto.

### Evidence Needed
Official steps 10--11, all-rank semantic/loss/EMA/leak checks, conservative node peak and reserve admission.

### Refutation Condition
OOM/rejection, semantic failure, loss delta above 2e-4, or memory saving below 10%.

### Allowed Changes
Frozen host-ownership action only.

### Commands
Ran `iteration174_m1_world8_score40_step10_12_correctness` on node0 in events mode.

### Measured Results
Peak 655,496,982,528 B, saving 79,646,777,344 B (10.834%) versus private; headroom 91,827,326,976 B; maximum loss delta 2.3723e-5; rc=0.

### Correctness
PASS on all eight ranks for trace, versions, gradients, Critic-only immutability, EMA and leaks.

### Performance
Two-step diagnostic median 176.988 s; not formal throughput.

### Interpretation
Score40 survives capacity pruning and requires formal timing.

### Files Changed
Iteration-174 manifest/raw evidence.

### Failure or Limitation
Cross-node control and memory instrumentation prevent a speed claim.

### Gate Status
Score40 correctness/admission PASS; M1.2 OPEN.

### Next Highest-Value Question
Does all-shared add meaningful capacity beyond Score40?

### Next Command
Summarize Iteration 175.

## Iteration 175

### Milestone / Gate
M1.2 all-shared correctness/admission precheck.

### Question
Does sharing both frozen Generator and Score state materially improve capacity over Score40 while preserving semantics?

### Hypothesis
All-shared produces the lowest node peak and remains correct.

### Importance
It is the historical A-scalable ownership extreme and a required bounded-oracle point.

### Evidence Needed
Same contract as Iteration 174.

### Refutation Condition
Failure, semantic drift, or no capacity improvement.

### Allowed Changes
Frozen all-shared ownership action only.

### Commands
Ran `iteration175_m1_world8_allshared_step10_12_correctness` on node1 in events mode.

### Measured Results
Peak 652,581,457,920 B, saving 82,562,301,952 B (11.231%) versus private and only 2.916 GB beyond Score40; headroom 94,742,851,584 B; maximum loss delta 3.1590e-5; rc=0.

### Correctness
PASS on all eight ranks.

### Performance
Two-step diagnostic median 184.526 s; not formal throughput.

### Interpretation
All-shared is admitted, but its small incremental capacity benefit means formal timing can easily make Score40 dominate it.

### Files Changed
Iteration-175 manifest/raw evidence.

### Failure or Limitation
Different node and profiler mode.

### Gate Status
All-shared correctness/admission PASS; M1.2 OPEN.

### Next Highest-Value Question
Do Score40 and all-shared pass the paired loss/semantic capacity gate against private?

### Next Command
Build Iteration-176 correctness set.

## Iteration 176

### Milestone / Gate
M1.2 larger-ownership correctness/capacity set.

### Question
Should Score40 and all-shared be admitted to formal timing?

### Hypothesis
Both reduce node peak by >=10%, remain reserve-safe and pass the frozen observable correctness contract.

### Importance
Only admitted, correct, preregistered plans may enter the measured oracle.

### Evidence Needed
Private-referenced peak deltas, per-loss deltas and all-rank semantic assertions.

### Refutation Condition
Any candidate misses capacity or correctness thresholds.

### Allowed Changes
Analysis and documentation only.

### Commands
Compared Iterations 174/175 with the frozen Iteration-163 private correctness control.

### Measured Results
Score40 saves 10.834%; all-shared saves 11.231%. Both stay below the 696-GiB reserve-qualified limit. Maximum loss deltas are 2.3723e-5 and 3.1590e-5.

### Correctness
PASS. An initial one-off summary incorrectly expected two `semantic_critic_post_step` events; step 11 is represented by `semantic_contract_complete`. The analysis was corrected before commit.

### Performance
Not Run formally.

### Interpretation
Both points survive capacity pruning. All-shared adds only 0.397 percentage points over Score40.

### Files Changed
Iteration-176 correctness-set manifest and reports.

### Failure or Limitation
Full update-vector and Adam m/v equivalence remain Not Run.

### Gate Status
Ownership capacity precheck PASS; formal timing required.

### Next Highest-Value Question
Does either >=10%-memory action stay within the allowed 5% slowdown?

### Next Command
Run complete-cycle control-mode Score40 and all-shared timing with node crossover.

## Iteration 177

### Milestone / Gate
M1.2 Score40 formal timing round 1.

### Question
Does Score40 remain a viable complete-cycle ownership point under formal timing?

### Hypothesis
Score40 completes two official five-step cycles from the frozen checkpoint without changing semantics.

### Importance
A capacity-admitted action cannot enter the bounded measured oracle without formal steady timing.

### Evidence Needed
World8 control-mode rc=0, exact CAL-A trace, steps 10--19, separate G/C timing and sampled node resources.

### Refutation Condition
Failure, reduced shape, incomplete cycles or trace mismatch.

### Allowed Changes
No policy changes; frozen `S-SCORE40-CW4-W1-A32` only.

### Commands
Ran `iteration177_m1_world8_score40_formal_round1_node0` on node0 from native-world8 checkpoint 10.

### Measured Results
G-update 204.229 s; Critic-only 80.707 s; weighted 105.411 s; throughput 0.075893 sample/s; sampled cgroup peak 657,583,845,376 B; rc=0.

### Correctness
The previously admitted action/trace contract is preserved; semantic assertions were intentionally disabled in this timing-only run.

### Performance
Measured formal round 1; not yet a causal policy estimate.

### Interpretation
Score40 remains executable and enters the repeated formal timing set.

### Files Changed
Iteration-177 manifest/raw evidence.

### Failure or Limitation
Single node/launch; sampled cgroup peak includes cold construction.

### Gate Status
M1.2 timing in progress.

### Next Highest-Value Question
How does all-shared behave under the same formal contract?

### Next Command
Run Iteration 178.

## Iteration 178

### Milestone / Gate
M1.2 all-shared formal timing round 1.

### Question
Does all-shared remain viable under complete-cycle formal timing?

### Hypothesis
All-shared completes the same two cycles and may reduce memory beyond Score40.

### Importance
It is the maximum-sharing endpoint of the bounded ownership action space.

### Evidence Needed
Same contract as Iteration 177.

### Refutation Condition
Failure, trace mismatch or invalid cycle/version sequence.

### Allowed Changes
No policy changes; frozen `S-ALLSHARED-CW4-W1-A32` only.

### Commands
Ran `iteration178_m1_world8_allshared_formal_round1_node1` on node1.

### Measured Results
G-update 232.669 s; Critic-only 91.590 s; weighted 119.806 s; throughput 0.066775 sample/s; sampled cgroup peak 658,380,328,960 B; rc=0.

### Correctness
The previously admitted action/trace contract is preserved; semantic assertions were disabled for timing.

### Performance
Measured formal round 1; not yet causal because policy and node are confounded.

### Interpretation
All-shared completes, but the first assignment does not reproduce its small two-step memory advantage.

### Files Changed
Iteration-178 manifest/raw evidence.

### Failure or Limitation
Single node/launch and construction-inclusive peak.

### Gate Status
M1.2 timing in progress.

### Next Highest-Value Question
Is the 13.655% time gap a policy effect or node/round drift?

### Next Command
Reverse node assignment.

## Iteration 181

### Milestone / Gate
M1.2 Score40/all-shared round-1 comparison and crossover launch.

### Question
Can round 1 alone classify either ownership point as Pareto-dominated?

### Hypothesis
No: previous crossover data imply node/round blocking is required.

### Importance
Premature pruning would bias the bounded measured oracle.

### Evidence Needed
Matched contract manifests plus reversed assignment.

### Refutation Condition
A node-independent repeated estimate already exists.

### Allowed Changes
Analysis, documentation and preregistered reversed launches only.

### Commands
Built the Iteration-181 comparison manifest; launched all-shared/node0 as Iteration 179 and Score40/node1 as Iteration 180.

### Measured Results
Round 1 places all-shared 13.655% slower, with 0.796 GB higher sampled peak, but both quantities are node-confounded.

### Correctness
Both source runs return zero and use trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.

### Performance
Descriptive only pending reversed assignment.

### Interpretation
No candidate is pruned. Formal crossover is now running.

### Files Changed
Iteration-181 manifest, M1 profile report and research log.

### Failure or Limitation
Round 2 is in progress; full update-vector/Adam parity remains Not Run.

### Gate Status
M1.2 OPEN; G1b/G1c not started.

### Next Highest-Value Question
Does the reversed assignment preserve the timing sign?

### Next Command
Wait for Iterations 179/180, summarize, then schedule the third independent pair.

## Iteration 179

### Milestone / Gate
M1.2 all-shared formal timing round 2, node-crossover backfill.

### Question
Does the all-shared slowdown from round 1 persist when all-shared moves from node1 to node0?

### Hypothesis
If the slowdown is caused by ownership contention rather than node identity, all-shared remains slower after node reversal.

### Importance
The bounded measured oracle cannot prune all-shared from a single node-confounded launch.

### Evidence Needed
Return code zero, exact CAL-A trace, checkpoint-10 steps 10--19, two complete five-step cycles, G-update/Critic-only timing, throughput, and node peak.

### Refutation Condition
Failure, trace mismatch, incomplete cycles, or all-shared becoming faster after reversal.

### Allowed Changes
No model, loss, optimizer, trace, checkpoint, activation, streaming, or ownership changes; frozen `S-ALLSHARED-CW4-W1-A32` only.

### Commands
Ran `iteration179_m1_world8_allshared_formal_round2_node0` in control mode on node0 and summarized it with `summarize_m1_profile.py`.

### Measured Results
Return code 0. G-update mean 245.575210 s; Critic-only mean 96.597246 s; official weighted step 126.392839 s; global throughput 0.0632947 samples/s; sampled cgroup peak 656,948,707,328 B. All ten logical steps 10--19 completed.

### Correctness
Exact frozen trace/checkpoint contract and two complete cycles pass. Detailed semantic assertions were intentionally disabled in the formal timing run; the matched all-shared semantic precheck is Iteration 175.

### Performance
Measured formal launch 2 for all-shared. Compared with the round-2 Score40 control, all-shared is 6.420% slower.

### Interpretation
The slowdown direction persists after node reversal. Two rounds still provide no residual degree of freedom for uncertainty, so a third independent launch remains required before freezing the policy effect.

### Files Changed
`artifacts/manifests/iteration_179_m1_world8_allshared_formal_round2_node0.json` and raw run evidence.

### Failure or Limitation
Construction-inclusive sampled cgroup peak is not PSS/VmPin and cannot isolate ownership-active steady state. Full update-vector and Adam m/v parity remain Not Run.

### Gate Status
All-shared formal launches 2/3; M1.2 OPEN; G1b/G1c Not Run.

### Next Highest-Value Question
Does Score40 complete the reversed round on node1, and what is the blocked two-round direction?

### Next Command
Summarize Iteration 180, then schedule the third independent pair.

## Iteration 180

### Milestone / Gate
M1.2 Score40 formal timing round 2, node-crossover backfill.

### Question
Does Score40 remain faster than all-shared after moving from node0 to node1?

### Hypothesis
Score40 preserves most of the capacity benefit while avoiding some Generator shared-backing contention.

### Importance
This determines whether all-shared deserves further oracle budget or is likely dominated by Score40.

### Evidence Needed
The same formal contract as Iteration 179.

### Refutation Condition
Failure, trace mismatch, incomplete cycles, or Score40 becoming slower than all-shared in the reversed round.

### Allowed Changes
No policy changes; frozen `S-SCORE40-CW4-W1-A32` only.

### Commands
Ran `iteration180_m1_world8_score40_formal_round2_node1` in control mode on node1 and summarized it with `summarize_m1_profile.py`.

### Measured Results
Return code 0. G-update mean 226.206102 s; Critic-only mean 91.908007 s; official weighted step 118.767626 s; global throughput 0.0673584 samples/s; sampled cgroup peak 659,027,697,664 B. All ten logical steps 10--19 completed.

### Correctness
Exact frozen trace/checkpoint contract and two complete cycles pass. Detailed same-layout semantic evidence is Iteration 174.

### Performance
Measured formal launch 2 for Score40. It is 6.420% faster than all-shared in round 2.

### Interpretation
Both crossover rounds favor Score40. Across two launches, raw weighted means are 112.089526 s for Score40 and 123.099173 s for all-shared, a descriptive all-shared slowdown of 9.822%. This is not yet the frozen three-launch estimate.

### Files Changed
`artifacts/manifests/iteration_180_m1_world8_score40_formal_round2_node1.json` and raw run evidence.

### Failure or Limitation
The two-round additive policy/node/round model is saturated and cannot provide an uncertainty estimate. Formal node peaks average 658.306 GB for Score40 and 657.665 GB for all-shared, only a 0.097% difference; this does not replace the separate memory-mode ownership-active capacity evidence.

### Gate Status
Score40 formal launches 2/3; M1.2 OPEN; G1b/G1c Not Run.

### Next Highest-Value Question
Does a third independent pair preserve the Score40 advantage and allow the larger-ownership branch to be frozen?

### Next Command
Run the preregistered third independent Score40/all-shared pair, then stop ownership repetitions and continue the bounded static action sequence.


## Iteration 107

### Milestone / Gate
M0 trace-substrate launcher preflight (reserved run ID; backfilled from Iteration 106 source evidence).

### Question
Can the rank-qualified operation-trace checker run as a self-contained launcher preflight without an inherited `PYTHONPATH`?

### Hypothesis
The new checker will validate the eight-rank trace bundle before any 14B construction.

### Importance
A launcher preflight must fail closed on trace identity without depending on the caller's shell import path.

### Evidence Needed
A preflight result and, on success, a run artifact.

### Refutation Condition
The checker exits before model construction because it cannot import the repository modules.

### Allowed Changes
Trace-checker import/bootstrap only; no model, optimizer, loss, trace content, or memory-policy changes.

### Commands
The exact shell command was not preserved. Iteration 106 records this as one of the first two reserved launcher preflight invocations (`107/108`).

### Measured Results
The invocation exited before creating a run artifact because the checker lacked a self-contained repository import path. No GPU model construction, training step, memory peak, or timing metric was produced.

### Correctness
FAIL at launcher-preflight infrastructure. The trace itself was not shown incorrect.

### Performance
Not Run.

### Interpretation
This was an import-path failure, not a Helios or lifecycle result. Iteration 106 subsequently fixed the checker and verified it without `PYTHONPATH`.

### Files Changed
No run artifact exists for reserved ID 107; authoritative backfill source is the Iteration 106 record.

### Failure or Limitation
The exact stderr and command line were not retained, so no additional detail is inferred.

### Gate Status
Superseded by the repaired Iteration 106 trace preflight.

### Next Highest-Value Question
Can the repaired rank-qualified trace bundle reach actual eight-rank construction and training?

### Next Command
Proceed to Iteration 109 direct A-local x8 admission.

## Iteration 108

### Milestone / Gate
M0 trace-substrate launcher preflight (reserved run ID; backfilled from Iteration 106 source evidence).

### Question
Can the rank-qualified operation-trace checker run as a self-contained launcher preflight without an inherited `PYTHONPATH`?

### Hypothesis
The new checker will validate the eight-rank trace bundle before any 14B construction.

### Importance
A launcher preflight must fail closed on trace identity without depending on the caller's shell import path.

### Evidence Needed
A preflight result and, on success, a run artifact.

### Refutation Condition
The checker exits before model construction because it cannot import the repository modules.

### Allowed Changes
Trace-checker import/bootstrap only; no model, optimizer, loss, trace content, or memory-policy changes.

### Commands
The exact shell command was not preserved. Iteration 106 records this as one of the first two reserved launcher preflight invocations (`107/108`).

### Measured Results
The invocation exited before creating a run artifact because the checker lacked a self-contained repository import path. No GPU model construction, training step, memory peak, or timing metric was produced.

### Correctness
FAIL at launcher-preflight infrastructure. The trace itself was not shown incorrect.

### Performance
Not Run.

### Interpretation
This was an import-path failure, not a Helios or lifecycle result. Iteration 106 subsequently fixed the checker and verified it without `PYTHONPATH`.

### Files Changed
No run artifact exists for reserved ID 108; authoritative backfill source is the Iteration 106 record.

### Failure or Limitation
The exact stderr and command line were not retained, so no additional detail is inferred.

### Gate Status
Superseded by the repaired Iteration 106 trace preflight.

### Next Highest-Value Question
Can the repaired rank-qualified trace bundle reach actual eight-rank construction and training?

### Next Command
Proceed to Iteration 109 direct A-local x8 admission.

## Iteration 137

### Milestone / Gate
M1.1 real-14B unbound persistent-pinning memory probe (backfilled from raw trace).

### Question
Does `Generator pinned; Score pageable` complete one official-shape G-update and one Critic-only step after removing invalid whole-process NUMA binding?

### Hypothesis
Persistent pinned immutable owners will be admitted by aggregate node memory and complete both phase types without changing Helios semantics.

### Importance
This distinguishes intrinsic pinned-state feasibility from the earlier NUMA-node OOM.

### Evidence Needed
Official 384x640 trace/checkpoint identity, steps 10 and 11, GPU/host peaks, pinned ownership bytes, materialization counts, and finite training outputs.

### Refutation Condition
Host/GPU OOM, failed phase boundary, trace drift, or non-finite loss/gradient.

### Allowed Changes
Persistent pinned ownership selector only; no async prefetch, planner, loss, optimizer, trace, or placement change.

### Commands
Run ID `iteration137_m1_pinned_generator_unbound_step10_12_memory` from checkpoint-10 with CAL-A trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`, world1, A100-40GB, BF16, official 384x640.

### Measured Results
- Return code: 0; step 10 G+C: 95.258274 s; step 11 Critic-only: 42.994935 s.
- Diagnostic `(G+4C)/5` extrapolation: 53.447602 s; this is **not** a formal multi-cycle timing result because only two steps ran.
- Runtime-owned pinned immutable state: 28,111,544,320 B.
- Max trace cgroup current: 159,656,337,408 B; max process RSS peak: 157,459,968,000 B.
- CUDA allocator max allocated/reserved: 33,727,460,864 / 38,663,094,272 B; no allocator OOM.
- Generator streamer: 640 materializations, 449,784,709,120 B materialized, 24.625313 s H2D wall intervals.
- Score streamer: 280 materializations, 196,780,810,240 B materialized, 21.978667 s H2D wall intervals.
- Generator/Critic activation offload logical bytes: 20,980,531,200 / 9,879,552,000 B; duplicate packs: Generator 1; Critic 2.

### Correctness
Partial PASS: exact trace hash, successful checkpoint resume, finite loss/gradient norms, steps 10/11 completed, and expected G/C version advancement. Full policy boundary SHA and update-delta equivalence were Not Run in this memory probe.

### Performance
Diagnostic only. Formal timing requires steps 10--19 and independent repeats.

### Interpretation
The unbound persistent-pinning policy is feasible for one rank. Streamer H2D wall intervals still mix Python, allocation/enqueue, and synchronization and are not labeled observed PCIe memcpy time.

### Files Changed
Raw log `artifacts/logs/iteration137_m1_pinned_generator_unbound_step10_12_memory/`; trace `artifacts/traces/iteration137_m1_pinned_generator_unbound_step10_12_memory/`.

### Failure or Limitation
`VmPin` is not reliable for CUDA host allocations on this platform; runtime-owned pinned bytes are authoritative. Two steps do not establish steady-state performance.

### Gate Status
Memory feasibility PASS; formal timing OPEN.

### Next Highest-Value Question
Does this policy retain its benefit over multiple complete five-step cycles?

### Next Command
Run independent steps 10--19 timing launches.

## Iteration 138

### Milestone / Gate
M1.1 real-14B unbound persistent-pinning memory probe (backfilled from raw trace).

### Question
Does `Generator pinned; Score pinned` complete one official-shape G-update and one Critic-only step after removing invalid whole-process NUMA binding?

### Hypothesis
Persistent pinned immutable owners will be admitted by aggregate node memory and complete both phase types without changing Helios semantics.

### Importance
This distinguishes intrinsic pinned-state feasibility from the earlier NUMA-node OOM.

### Evidence Needed
Official 384x640 trace/checkpoint identity, steps 10 and 11, GPU/host peaks, pinned ownership bytes, materialization counts, and finite training outputs.

### Refutation Condition
Host/GPU OOM, failed phase boundary, trace drift, or non-finite loss/gradient.

### Allowed Changes
Persistent pinned ownership selector only; no async prefetch, planner, loss, optimizer, trace, or placement change.

### Commands
Run ID `iteration138_m1_pinned_all_unbound_step10_12_memory` from checkpoint-10 with CAL-A trace hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`, world1, A100-40GB, BF16, official 384x640.

### Measured Results
- Return code: 0; step 10 G+C: 115.388877 s; step 11 Critic-only: 41.596515 s.
- Diagnostic `(G+4C)/5` extrapolation: 56.354988 s; this is **not** a formal multi-cycle timing result because only two steps ran.
- Runtime-owned pinned immutable state: 56,223,088,640 B.
- Max trace cgroup current: 175,322,173,440 B; max process RSS peak: 173,016,772,608 B.
- CUDA allocator max allocated/reserved: 33,727,460,864 / 38,663,094,272 B; no allocator OOM.
- Generator streamer: 640 materializations, 449,784,709,120 B materialized, 23.819004 s H2D wall intervals.
- Score streamer: 280 materializations, 196,780,810,240 B materialized, 13.595506 s H2D wall intervals.
- Generator/Critic activation offload logical bytes: 21,027,225,600 / 10,105,651,200 B; duplicate packs: Generator 0; Critic 0.

### Correctness
Partial PASS: exact trace hash, successful checkpoint resume, finite loss/gradient norms, steps 10/11 completed, and expected G/C version advancement. Full policy boundary SHA and update-delta equivalence were Not Run in this memory probe.

### Performance
Diagnostic only. Formal timing requires steps 10--19 and independent repeats.

### Interpretation
The unbound persistent-pinning policy is feasible for one rank. Streamer H2D wall intervals still mix Python, allocation/enqueue, and synchronization and are not labeled observed PCIe memcpy time.

### Files Changed
Raw log `artifacts/logs/iteration138_m1_pinned_all_unbound_step10_12_memory/`; trace `artifacts/traces/iteration138_m1_pinned_all_unbound_step10_12_memory/`.

### Failure or Limitation
`VmPin` is not reliable for CUDA host allocations on this platform; runtime-owned pinned bytes are authoritative. Two steps do not establish steady-state performance.

### Gate Status
Memory feasibility PASS; formal timing OPEN.

### Next Highest-Value Question
Does this policy retain its benefit over multiple complete five-step cycles?

### Next Command
Run independent steps 10--19 timing launches.

## Iteration 139

### Milestone / Gate
M1.1 timing-protocol correction (backfilled from aborted run artifact).

### Question
Is an unpinned steps 10--11 control sufficient for formal performance?

### Hypothesis
A two-step run would provide a quick matched control.

### Importance
Formal results must average complete five-step cycles rather than extrapolate from one G and one C step.

### Evidence Needed
Multiple complete five-step cycles from the canonical checkpoint.

### Refutation Condition
`max_steps=12` covers only steps 10 and 11.

### Allowed Changes
Run control only; no code or policy changes.

### Commands
Run ID `iteration139_m1_unpinned_control_node0_step10_12`, official 384x640, CAL-A trace preflight.

### Measured Results
Trace preflight PASS with hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`. The researcher terminated the launch during construction with SIGTERM; return code 143. The sparse resource sampler observed at most 18,208,915,456 B cgroup current before termination.

### Correctness
Not Run; no training boundary completed.

### Performance
INVALID / Not Run.

### Interpretation
The protocol was corrected before collecting a misleading two-step formal result. The run was superseded by steps 10--19 launches.

### Files Changed
`artifacts/logs/iteration139_m1_unpinned_control_node0_step10_12/`, wrapper log, and trace preflight artifact.

### Failure or Limitation
The resource sample is construction-only and is not a training memory peak.

### Gate Status
Superseded; formal timing remains OPEN.

### Next Highest-Value Question
What is the policy's mean over two complete five-step cycles?

### Next Command
Restart with `max_steps=20`.

## Iteration 140

### Milestone / Gate
M1.1 timing-protocol correction (backfilled from aborted run artifact).

### Question
Is an unpinned steps 10--11 control sufficient for formal performance?

### Hypothesis
A two-step run would provide a quick matched control.

### Importance
Formal results must average complete five-step cycles rather than extrapolate from one G and one C step.

### Evidence Needed
Multiple complete five-step cycles from the canonical checkpoint.

### Refutation Condition
`max_steps=12` covers only steps 10 and 11.

### Allowed Changes
Run control only; no code or policy changes.

### Commands
Run ID `iteration140_m1_unpinned_control_node1_step10_12`, official 384x640, CAL-A trace preflight.

### Measured Results
Trace preflight PASS with hash `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`. The researcher terminated the launch during construction with SIGTERM; return code 143. The sparse resource sampler observed at most 28,071,485,440 B cgroup current before termination.

### Correctness
Not Run; no training boundary completed.

### Performance
INVALID / Not Run.

### Interpretation
The protocol was corrected before collecting a misleading two-step formal result. The run was superseded by steps 10--19 launches.

### Files Changed
`artifacts/logs/iteration140_m1_unpinned_control_node1_step10_12/`, wrapper log, and trace preflight artifact.

### Failure or Limitation
The resource sample is construction-only and is not a training memory peak.

### Gate Status
Superseded; formal timing remains OPEN.

### Next Highest-Value Question
What is the policy's mean over two complete five-step cycles?

### Next Command
Restart with `max_steps=20`.

## Iteration 141

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Pageable immutable control, repeat 1 / node0` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration141_m1_unpinned_control_node0_step10_20_rep1`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 109.313364 s.
- Critic-only mean: 52.496851 s.
- Official weighted `(G+4C)/5`: 63.860153 s.
- Throughput: 0.015659217 samples/s; GPU-seconds/sample: 63.860153.
- Per-step makespan seconds: 10=108.272138 (G), 11=52.646239 (C), 12=53.862496 (C), 13=38.130538 (C), 14=75.768873 (C), 15=110.354590 (G), 16=43.021002 (C), 17=55.520264 (C), 18=51.326486 (C), 19=49.698906 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration141_m1_unpinned_control_node0_step10_20_rep1.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 142

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Pageable immutable control, independent repeat / node1` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration142_m1_unpinned_control_node1_step10_20_rep1`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 104.962940 s.
- Critic-only mean: 48.865857 s.
- Official weighted `(G+4C)/5`: 60.085273 s.
- Throughput: 0.016643013 samples/s; GPU-seconds/sample: 60.085273.
- Per-step makespan seconds: 10=105.526770 (G), 11=49.708995 (C), 12=55.577283 (C), 13=38.654782 (C), 14=70.427977 (C), 15=104.399110 (G), 16=41.759028 (C), 17=51.205783 (C), 18=41.825133 (C), 19=41.767873 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration142_m1_unpinned_control_node1_step10_20_rep1.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 143

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Persistent pin Generator, repeat 1 / node0` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration143_m1_pinned_generator_control_node0_step10_20_rep1`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 79.938591 s.
- Critic-only mean: 36.372440 s.
- Official weighted `(G+4C)/5`: 45.085670 s.
- Throughput: 0.022179997 samples/s; GPU-seconds/sample: 45.085670.
- Per-step makespan seconds: 10=76.008167 (G), 11=32.625316 (C), 12=38.873599 (C), 13=29.925435 (C), 14=44.611539 (C), 15=83.869014 (G), 16=34.130919 (C), 17=39.759447 (C), 18=36.418564 (C), 19=34.634699 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration143_m1_pinned_generator_control_node0_step10_20_rep1.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 144

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Persistent pin Generator+Score, repeat 1 / node1` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration144_m1_pinned_all_control_node1_step10_20_rep1`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 75.549218 s.
- Critic-only mean: 35.303073 s.
- Official weighted `(G+4C)/5`: 43.352302 s.
- Throughput: 0.023066826 samples/s; GPU-seconds/sample: 43.352302.
- Per-step makespan seconds: 10=72.241738 (G), 11=29.909433 (C), 12=37.272803 (C), 13=27.796633 (C), 14=44.515384 (C), 15=78.856697 (G), 16=33.557632 (C), 17=39.742686 (C), 18=35.446818 (C), 19=34.183192 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration144_m1_pinned_all_control_node1_step10_20_rep1.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 145

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Persistent pin Generator+Score, repeat 2 / node0` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration145_m1_pinned_all_control_node0_step10_20_rep2`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 75.815653 s.
- Critic-only mean: 34.982051 s.
- Official weighted `(G+4C)/5`: 43.148772 s.
- Throughput: 0.023175631 samples/s; GPU-seconds/sample: 43.148772.
- Per-step makespan seconds: 10=71.090772 (G), 11=32.840157 (C), 12=37.919927 (C), 13=29.904309 (C), 14=43.172079 (C), 15=80.540534 (G), 16=31.815449 (C), 17=36.830579 (C), 18=33.764541 (C), 19=33.609368 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration145_m1_pinned_all_control_node0_step10_20_rep2.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 146

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Persistent pin Generator, repeat 2 / node1` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration146_m1_pinned_generator_control_node1_step10_20_rep2`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 81.867597 s.
- Critic-only mean: 39.127763 s.
- Official weighted `(G+4C)/5`: 47.675730 s.
- Throughput: 0.020975033 samples/s; GPU-seconds/sample: 47.675730.
- Per-step makespan seconds: 10=76.301628 (G), 11=34.204552 (C), 12=39.439145 (C), 13=30.559512 (C), 14=45.087092 (C), 15=87.433566 (G), 16=40.897938 (C), 17=40.634879 (C), 18=42.310932 (C), 19=39.888053 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration146_m1_pinned_generator_control_node1_step10_20_rep2.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 147

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Persistent pin Generator, repeat 3 / node0` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration147_m1_pinned_generator_control_node0_step10_20_rep3`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 90.110602 s.
- Critic-only mean: 42.140666 s.
- Official weighted `(G+4C)/5`: 51.734653 s.
- Throughput: 0.019329404 samples/s; GPU-seconds/sample: 51.734653.
- Per-step makespan seconds: 10=85.858888 (G), 11=40.376082 (C), 12=46.120197 (C), 13=37.650536 (C), 14=47.212635 (C), 15=94.362316 (G), 16=39.286471 (C), 17=49.257516 (C), 18=41.948877 (C), 19=35.273014 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration147_m1_pinned_generator_control_node0_step10_20_rep3.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 148

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Persistent pin Generator+Score, repeat 3 / node1` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration148_m1_pinned_all_control_node1_step10_20_rep3`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 77.819240 s.
- Critic-only mean: 36.064734 s.
- Official weighted `(G+4C)/5`: 44.415635 s.
- Throughput: 0.022514594 samples/s; GPU-seconds/sample: 44.415635.
- Per-step makespan seconds: 10=75.101165 (G), 11=30.624072 (C), 12=39.541446 (C), 13=31.250497 (C), 14=45.163720 (C), 15=80.537315 (G), 16=33.734442 (C), 17=39.358846 (C), 18=35.364807 (C), 19=33.480045 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration148_m1_pinned_all_control_node1_step10_20_rep3.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 149

### Milestone / Gate
M1.1 official-shape persistent-pinning formal timing (backfilled from machine-readable profile).

### Question
What is the steady-state timing of `Pageable immutable control, repeat 3 / node0` on the frozen same-layout real-14B contract?

### Hypothesis
Persistent pinned immutable state reduces pageable streamer overhead without changing the operation trace or optimizer semantics.

### Importance
This run contributes one independent sample to the causal G1a comparison.

### Evidence Needed
Steps 10--19 from the same logical checkpoint/trace, separated timing mode, successful completion, per-step makespans, G/C means, and weighted step time.

### Refutation Condition
Nonzero return code, trace mismatch, non-finite training value, or no end-to-end improvement.

### Allowed Changes
Frozen host-ownership policy only; no profiler counters, planner, prefetch, loss, optimizer, trace, or placement changes.

### Commands
Run ID `iteration149_m1_unpinned_control_node0_step10_20_rep3`; world1; official 384x640; checkpoint-10; CAL-A; BF16; timing mode.

### Measured Results
- Return code 0; Generator-update mean: 102.226881 s.
- Critic-only mean: 49.504385 s.
- Official weighted `(G+4C)/5`: 60.048884 s.
- Throughput: 0.016653099 samples/s; GPU-seconds/sample: 60.048884.
- Per-step makespan seconds: 10=97.448148 (G), 11=45.836507 (C), 12=55.164567 (C), 13=39.253698 (C), 14=62.987783 (C), 15=107.005613 (G), 16=45.845257 (C), 17=49.799063 (C), 18=54.400239 (C), 19=42.747969 (C).
- Formal timing mode did not collect an authoritative training memory peak; its sparse resource file is retained but is not used as a memory result.

### Correctness
Partial PASS: exact contract/trace, successful canonical resume, all ten steps completed, finite logged loss/gradient norms, and expected version progression. Full cross-policy boundary SHA/update-delta equivalence was Not Run in this timing launch.

### Performance
Measured. This is one independent launch, not the aggregate policy claim.

### Interpretation
The result is retained verbatim for the three-run aggregate; no single run is selected post hoc as the winner.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration149_m1_unpinned_control_node0_step10_20_rep3.json`, corresponding CSV, raw trace, rc file, config, and wrapper log.

### Failure or Limitation
Timing variance remains; policy claims require the preregistered three independent launches.

### Gate Status
Contributes to G1a causal profile; aggregate decision deferred to Iteration 151.

### Next Highest-Value Question
Does the three-run policy aggregate exceed the 10% causal ceiling?

### Next Command
Complete the remaining randomized policy repeats.

## Iteration 150

### Milestone / Gate
M1.1 low-overhead lifecycle-counter paired run (backfilled from machine-readable profile).

### Question
Can runtime-owned byte/wait counters reconcile lifecycle activity without adding more than 3% positive timing overhead?

### Hypothesis
The lightweight event/interval tracer isolates issued bytes and dependency waits while remaining performance-neutral.

### Importance
G1a needs causal counters; a profiler that materially slows Helios would invalidate its own timing evidence.

### Evidence Needed
Same checkpoint/trace, ten-step instrumented timing, issued/unique bytes, waits, per-key counters, and paired control comparison.

### Refutation Condition
Positive paired median overhead exceeds 3%, counter leak, or incomplete reconciliation.

### Allowed Changes
Counters only; no host policy, prefetch, planner, loss, optimizer, trace, or placement change.

### Commands
Run ID `iteration150_m1_unpinned_counters_node1_step10_20_overhead`; world1 pageable control; official 384x640; checkpoint-10; CAL-A; steps 10--19.

### Measured Results
- G-update mean 103.917125 s; Critic-only mean 45.941456 s; weighted 57.536590 s; throughput 0.017380245 samples/s.
- Per-step makespan seconds: 10=96.938624 (G), 11=38.240700 (C), 12=56.316264 (C), 13=35.550809 (C), 14=59.430650 (C), 15=110.895626 (G), 16=42.754405 (C), 17=46.651248 (C), 18=41.750759 (C), 19=46.836811 (C).
- Runtime-issued transfer bytes: 3,269,218,644,480; logical unique storage bytes: 82,622,627,840.
- Exposed dependency wait: 0.004963845 s; measurement-boundary wait: 0.084757696 s; distributed barrier: 0.008330475 s.
- Generator frozen H2D: 2,192,700,456,960 B, 3120 calls, 247.374051 s host-call wall.
- Score frozen H2D: 730,900,152,320 B, 1040 calls, 81.866820 s host-call wall.
- Generator saved-tensor D2H/H2D: 83,902,464,000 / 10,105,651,200 B.
- Critic saved-tensor D2H/H2D: 50,528,256,000 / 50,528,256,000 B.
- Paired median relative overhead reported in the aggregate analysis: -4.10%; interpreted as no measurable positive overhead under variance, not as a profiler speedup.
- Observed memcpy bytes remain `Not Run`; host-call walls are not relabeled as PCIe memcpy time.

### Correctness
PASS for trace/checkpoint identity, run completion, counter cleanup, and finite training values. Full update-delta parity remains outside this timing/counter run.

### Performance
Profiler overhead gate PASS: no measured positive overhead above 3%.

### Interpretation
The counters separate issued data and dependency waits, but CUPTI/Nsight observed memcpy remains a residual. The evidence supports continuing G1a analysis without claiming `.to()` wall time is raw PCIe time.

### Files Changed
`artifacts/profile/m1/official_pinned/iteration150_m1_unpinned_counters_node1_step10_20_overhead.json`, corresponding CSV, raw lifecycle events, rc/config/wrapper logs.

### Failure or Limitation
`observed_memcpy_bytes` is Not Run, so reconciliation applies to runtime-owned issued bytes, not all device activity.

### Gate Status
Low-overhead tracer PASS; causal G1a decision recorded in Iteration 151.

### Next Highest-Value Question
Does persistent pinning create at least a 10% real end-to-end or memory Pareto point?

### Next Command
Aggregate all three policy repeats and issue Gate G1a.

## Iteration 182

### Milestone / Gate
Critic-only Pool / target-world4 C11 checkpoint gate.

### Question
Can the real world16 ZeRO-1 step-11 logical state be imported once into a
world4 ZeRO-1 target layout and frozen as the common Sequential/Pipeline
starting point without claiming cross-world update equivalence?

### Hypothesis
The v3 logical package can reconstruct a native world4 C11 checkpoint with
exact logical Generator/Critic module, FP32 master, Adam m/v, optimizer,
scheduler, EMA and RNG state.

### Importance
Runtime conversion during a formal run would confound scheduling with layout
conversion. Both policies need the same immutable target-layout state.

### Evidence Needed
Native world4 import, immediate native verification, re-export, exact payload
SHA inventory, G3/C11 versions, optimizer/scheduler steps, and explicit EMA and
claim boundaries.

### Refutation Condition
Any logical payload, optimizer moment, scheduler, EMA, RNG, version or target
world-size mismatch.

### Allowed Changes
Offline checkpoint retarget/import/verify/export and derived contract only; no
model execution, optimizer update or baseline policy change.

### Commands
`checkpoint-template-retarget --world-size 4`, `checkpoint-import`,
`checkpoint-verify`, `checkpoint-export`, then
`validate_critic_pool_target_package.py`.

### Measured Results
- Template retarget PASS in 39.708 s.
- Import PASS in 142.519 s.
- Native verification PASS in 57.798 s.
- Generator: 1,251,442,688 logical elements, G3, optimizer step 3,
  3,754,328,064 optimizer elements compared.
- Critic: 1,246,314,496 logical elements, C11, optimizer step 11,
  3,738,943,488 optimizer elements compared.
- Generator/Critic BF16 module, FP32 master, Adam m/v SHA inventories are exact
  between source logical package and target re-export.
- Source manifest:
  `d114d98dc222280f83c15b5161a819a2bc141225074c4fe54c5aa74a98a703f8`.
- Target manifest:
  `c474735da1c0764eb8da6002010bd606bad583bed9213ae55acb6f7583cfae7e`.
- Bound target contract:
  `ba3d0b48b0937032f7eb0484ebb9c89253ddc45ddfbd95abf9ef92238cc8c2a3`.

### Correctness
PASS for target-layout import/export state. This does not assert that a future
world4 update is numerically identical to the source world16 update.

### Performance
Not Run. These are offline gate durations, not training performance.

### Interpretation
Formal Sequential/Pipeline runs can now start from the same native world4 C11
state. The Generator optimizer remains a logical fingerprint and will not be
materialized on Generator ranks.

### Files Changed
`scripts/distillschedule/validate_critic_pool_target_package.py`;
`artifacts/logs/iteration182_critic_pool_target_world4_c11/`; asset package
`target_world4_C11_package_v1`.

### Failure or Limitation
The package starts from real G3/C11 state but the experiment intentionally
skips early bootstrap EMA, so it is an isolated Critic-only subgraph rather
than complete official step 11--14 continuation.

### Gate Status
Target-world4 C11 gate PASS.

### Next Highest-Value Question
Can a 16-entry operation-keyed trace bind four waves and four producer ranks
without seed-order dependence?

### Next Command
Build and verify the Critic pool trace v2.

## Iteration 183

### Milestone / Gate
Critic-only Pool / 4-wave × 4-rank operation-trace gate.

### Question
Can the official-shape Critic-only inputs, including all history index tensors,
be frozen as 16 unique logical entries independently of ambient RNG order?

### Hypothesis
Four rank-qualified traces covering logical steps 11--14 can provide exact
prompt/history/noise/timestep/control identity while allowing SYN-FEATURE
physical files to be reused.

### Importance
Sequential and Pipeline reorder work and therefore cannot rely on a shared
process seed.

### Evidence Needed
Sixteen unique trace keys, slots, RNG streams and identities; four rank traces;
official shapes; all four history index tensors; independent state-provenance
and execution hashes.

### Refutation Condition
Missing/duplicate entry, missing history index, reduced shape, ambient RNG
dependence, or conflated checkpoint/execution trace identity.

### Allowed Changes
Trace preparation and validation only.

### Commands
`prepare_critic_pool_trace.py` with official config, four official
SYN-FEATURE files, start step 11 and four steps.

### Measured Results
- Initial launch failed before writing tensors because the script lacked the
  repository root on `sys.path`; the empty failed output is preserved.
- Retry PASS: 16 entries, slots 0--15, 16 unique RNG streams.
- Rank trace hashes:
  `91333a5a...`, `efcbbf47...`, `4362ea5a...`, `7f0adf10...`.
- Execution pool trace:
  `ba5859dd39d516d459b3d37636f1e7bd55f18ab7255f51e9d2b207a5b57ba9fc`.
- State provenance trace remains:
  `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.
- Each rank trace has 13 operations and 65 tensor records.

### Correctness
PASS. Unit tests also prove replay is invariant to intervening Python/Torch RNG
consumption and fail closed on missing history-index tensors.

### Performance
Not Run. Trace generation is not a training benchmark.

### Interpretation
The experiment now has 16 logical sample identities without falsely claiming
16 distinct real videos.

### Files Changed
`runtime/critic_pool_trace.py`,
`scripts/distillschedule/prepare_m0_operation_trace.py`,
`scripts/distillschedule/prepare_critic_pool_trace.py`,
`tests/test_critic_pool_trace.py`, and
`artifacts/logs/iteration183_critic_pool_trace_v2/`.

### Failure or Limitation
Input classification remains SYN-FEATURE with precomputed source-shaped
features.

### Gate Status
Trace gate PASS.

### Next Highest-Value Question
Can DeepSpeed ZeRO-1 be scoped strictly to Student ranks 4--7 while Generator
ranks avoid DeepSpeed entirely?

### Next Command
Run the world8 Student subgroup gate.

## Iteration 184

### Milestone / Gate
Critic-only Pool / world8 ZeRO-1 Student subgroup and P2P gate.

### Question
Does the minimal `mpu=` adapter keep all Student optimizer collectives inside
ranks 4--7 while preserving exact local/global batch semantics?

### Hypothesis
Student-only DeepSpeed ZeRO-1 can coexist with four non-DeepSpeed Generator
ranks in one world8 job, and paired NCCL x0/ACK transport can complete without
WORLD collective leakage or deadlock.

### Importance
This is the control-plane prerequisite for the real 14B 4G+4S layout.

### Evidence Needed
All eight ranks complete; only ranks 4--7 construct DeepSpeed; resolved DP
world = 4, micro batch = 1, GAS = 1, train batch = 4; one Student update and
paired ACK per rank.

### Refutation Condition
DeepSpeed resolves WORLD=8, a Generator rank enters a Student collective,
batch fields drift, P2P identity fails, or the job deadlocks.

### Allowed Changes
Tiny control-plane model, explicit process groups, ZeRO-1 subgroup adapter and
the same official-shape x0 wire payload; no real-model policy or timing claim.

### Commands
Single-node `torchrun --nproc_per_node=8
probe_critic_pool_zero1_subgroup.py`.

### Measured Results
- 8/8 ranks succeeded.
- Generator ranks 0--3 did not initialize DeepSpeed.
- Student ranks 4--7 resolved DP ranks 0--3 and all reported:
  `dp_world_size=4`, `train_micro_batch_size_per_gpu=1`,
  `gradient_accumulation_steps=1`, `train_batch_size=4`.
- Paired header/x0/update/commit-ACK completed for every rank.
- Makespan was approximately 8.014 s for this control-plane probe; it is not a
  real-14B performance result.

### Correctness
PASS for group isolation, batch contract, x0 transport identity, one update and
ACK identity.

### Performance
Not Run for Helios. The tiny-model wall time is not reported as a Pipeline
speedup.

### Interpretation
The bounded repair succeeded: no two-job fallback is needed. The real Student
can now use the same adapter and fail-closed assertions.

### Files Changed
`runtime/critic_pool.py`, `runtime/critic_pool_transport.py`,
`runtime/deepspeed_backend.py`,
`scripts/distillschedule/probe_critic_pool_zero1_subgroup.py`,
and corresponding tests/logs.

### Failure or Limitation
The gate does not prove 14B Student memory admission or resident Generator
nested rollout admission.

### Gate Status
World8 Student subgroup gate PASS.

### Next Highest-Value Question
Do the real 14B Generator resident rollout and real 14B Student ZeRO-1
backward/update each pass their isolated memory gates?

### Next Command
Run the two real-model admission gates before integrating four waves.

## Iteration 185

### Milestone / Gate
Critic-only Pool / real-14B role admission gates.

### Question
Can four G ranks keep a complete mixed-precision Generator resident for the
full official 384×640 nested no-grad rollout, and can four S ranks independently
complete one real Critic ZeRO-1 update?

### Hypothesis
Role separation is memory-admissible without Generator block streaming; the
Student can remain a fixed role while streaming only immutable Score blocks.

### Importance
The producer-consumer schedule is meaningless unless both physical roles fit
and execute their complete real operation, not just one DiT call.

### Evidence Needed
Four G rank full nested traces with no autograd/gradient/version change; four S
rank C11→C12 updates with exact DeepSpeed batch contract, Critic LoRA gradients,
frozen Score no-gradient, memory and host admission.

### Refutation Condition
Any G nested OOM/graph/gradient/version drift, Student backward/optimizer OOM,
WORLD=8 leakage, wrong batch fields, or missing Critic LoRA gradient.

### Allowed Changes
Narrow real-model gate launchers/instrumentation and logical scheduler restore;
no pipeline ordering, prefetch, model/loss/optimizer policy or EMA change.

### Commands
`launch_critic_pool_real14b_gate.sh generator it185_gresident_gate_r1` and
Student retries `it185_student_c11_gate_r1/r2`.

### Measured Results
- Generator gate PASS on all four G ranks. Complete resident mixed BF16-frozen
  plus FP32-LoRA Generator executed rank-qualified nested traces at 384×640.
- Trace physical-call counts were 4, 10, 8 and 8 across pool ranks; every call
  was no-grad, output had no graph, and G3 SHA was unchanged.
- Generator peak allocated 33.6598 GiB, reserved 33.9668 GiB; streamed bytes 0.
- Student attempt 1 failed because the target native DeepSpeed model-state has
  `lr_scheduler=None`; the failed log is retained.
- Bounded fix loaded native module/optimizer without scheduler state and restored
  the logical scheduler from the target package.
- Student retry PASS on four S ranks: DP=4, micro=1, GAS=1, train-batch=4;
  C11/optimizer/scheduler 11→12; all 812 Critic LoRA tensors observed a gradient;
  frozen Score gradient count 0.
- Student peak allocated 11.1054 GiB, reserved at most 11.709 GiB. Captured
  immutable frozen state was 26.1809 GiB/rank and forward+checkpoint materialized
  52.3618 GiB/rank/update.

### Correctness
PASS for both isolated role gates. EMA remained explicitly skipped by the
isolated-subgraph contract; this is not official continuous step11–14.

### Performance
Not Run formally. Gate phase timings are only used for overlap-ceiling
motivation; they are not an end-to-end pipeline speedup.

### Interpretation
A resident Generator role is feasible on A100-40GB, while Student needs only
intra-role immutable block streaming. The basic pool capacity hypothesis holds.

### Files Changed
`runtime/critic_pool_checkpoint.py`, `runtime/critic_pool_helios.py`, real-14B
gate launchers/probes and tests. Foundation commit `49de6b9565...`.

### Failure or Limitation
The Student gate initially used a trace boundary tensor rather than P2P
Generator output, so integrated four-wave execution remained unverified.

### Gate Status
Real role admission PASS.

### Next Highest-Value Question
Can the exact same P2P and Student update path complete four Sequential waves
from C11 to C15?

### Next Command
Launch `it186_seq_selfnoise_a` and retain any transport failure.

## Iteration 186

### Milestone / Gate
Critic-only Pool / integrated Sequential-4G4S and same-layout noise floor.

### Question
Can 16 exact logical samples traverse resident G→P2P→Student for four serialized
Critic updates, and is the frozen 2e-3 update-delta contract above the baseline
same-layout noise floor?

### Hypothesis
Sequential execution will preserve exact identities/versions and two independent
launches will be numerically close enough to freeze a valid Pipeline tolerance.

### Importance
Sequential is both the causal performance baseline and the only scientifically
valid way to set the numerical noise band before observing Pipeline.

### Evidence Needed
8/8 success, 16 exact traces/x0/noise/timestep identities, G3 unchanged,
C11→C15, exact flush, two independent launches, raw FP32 master/Adam vectors and
chunked update/state comparisons.

### Refutation Condition
P2P layout error/deadlock, missing/duplicate wave, version drift, queue leak, or
same-layout master update relative-L2 above 2e-3.

### Allowed Changes
Shared transport correctness and post-cycle correctness-state export only; no
schedule-specific primitive or tolerance change.

### Commands
Sequential runs `it186_seq_selfnoise_a`, retry `...a_r2`,
`it186_seq_selfnoise_b`, then state-export runs `it186_seq_noise_state_a/b` and
`summarize_critic_pool_cycle_pair.py`.

### Measured Results
- First integrated run failed at x0 send: `Tensors for P2P must be non-overlapping
  and dense`. Helios nested rollout returned a non-dense logical x0 view.
- Shared fix materializes one contiguous wire x0 for both schedules and records
  model-output and wire checksums separately. Unit tests passed.
- Two no-export Sequential runs completed in 147.1136 s and 152.5269 s for 16
  samples (0.108759 and 0.104900 samples/s).
- G peak 33.661 GiB; S peak 11.106 GiB; node memory peaks 217.211/218.123 GiB.
- All 16 trace, generated x0, wire x0, noise and timestep identities were exact;
  every queue flushed; G3→G3 and C11→C15.
- Two state-export correctness runs started from bitwise-identical initial
  master/Adam partitions. Their max loss abs difference was 3.5882e-5.
- Same-layout master update cosine 0.99995394, relative-L2 0.00961119, max abs
  1.13384e-6. Final-master direct relative-L2 was only 5.34264e-6.
- Adam m/v final direct relative-L2: 0.00160969 / 0.00029218.

### Correctness
Exact input, wire, queue and version contract PASS. Frozen numeric contract FAIL:
Sequential self-repeat update relative-L2 0.009611 > absolute cap 0.002.

### Performance
Engineering timing only. State export runs are explicitly excluded from formal
timing because raw-state I/O/page-cache activity changed cycle duration.

### Interpretation
The causal baseline itself exposes an ill-conditioned update-delta/noise-floor
conflict. This was frozen before looking at Pipeline, rather than post-hoc
relaxing a threshold.

### Files Changed
Dense transport materialization, raw rank-local master/Adam export,
`summarize_critic_pool_cycle_pair.py`, and frozen self-noise manifest. Commits
`e8c1ef4f...` and `548a3ee8...`.

### Failure or Limitation
The latest user contract simultaneously requires Pipeline to remain inside the
Sequential noise band and uses 2e-3 as an absolute reject cap; the measured
baseline band already exceeds that cap.

### Gate Status
Sequential functional PASS; numerical contract conflict FAIL.

### Next Highest-Value Question
Does Pipeline preserve exact control identities and land inside the already
frozen Sequential noise band?

### Next Command
Run one state-export Pipeline correctness diagnostic; do not make a formal
speed claim if it fails.

## Iteration 187

### Milestone / Gate
Critic-only Pool / Pipeline-Q1 functional and numerical comparison.

### Question
Can a two-slot queue overlap `G(B_{i+1})` with `S(B_i)` while preserving exact
sample/version/optimizer ordering and remaining within frozen self-noise?

### Hypothesis
Only wall-clock ordering changes, so Pipeline should match exact boundaries and
its numerical difference should not exceed the Sequential self-repeat band.

### Importance
This is the central mechanism candidate: eliminate role idle time without
changing one-update-per-global-batch semantics.

### Evidence Needed
8/8 completion, queue invariants, exact input identity, G3 unchanged, C11→C15,
raw master/Adam comparison and same-primitive makespan decomposition.

### Refutation Condition
Deadlock/stale batch/duplicate update/version crossing, any exact identity
mismatch, or numeric difference above 0.009611 self-noise / 0.002 absolute cap.

### Allowed Changes
Pipeline ordering only; same models, target package, trace, ZeRO-1, offload and
transport as Sequential.

### Commands
`it187_pipeline_state_a` plus pairwise raw-state summaries against both frozen
Sequential runs.

### Measured Results
- 8/8 ranks succeeded. All 16 trace/x0/noise/timestep identities, queue flush and
  version tracks were exact; G3 unchanged; C11→C15 with four optimizer steps.
- Pipeline correctness-run makespan 111.5486 s for 16 samples, G/S GPU peaks
  33.661/11.106 GiB, node memory peak 218.446 GiB.
- Corresponding state-export Sequential makespans were 171.3047/160.8063 s;
  preliminary speedups 1.536×/1.442×. These are not formal timing claims.
- Pipeline vs Seq-A master update cosine 0.99992795 and relative-L2 0.01201414;
  vs Seq-B relative-L2 0.01464669.
- Final-master direct relative-L2 remained small (6.678e-6 / 8.145e-6), but the
  preregistered update and noise-band gates fail.
- Estimated overlapped role work was about 39.56 s, roughly 80% of the measured
  Generator work available for overlap.

### Correctness
Functional/control correctness PASS. Numerical correctness FAIL under the frozen
contract: Pipeline exceeds both 0.002 and the 0.009611 Sequential noise band.

### Performance
Preliminary only. Formal warm-up→discard→C11 restore and three-launch randomized
crossover timing were not run, as required after correctness failure.

### Interpretation
The pipeline has a large capacity/overlap signal, but the current numerical
contract forbids promoting it to a system result.

### Files Changed
Raw Pipeline artifacts and pairwise comparison manifests.

### Failure or Limitation
The P2P group was still an eager world8 group and emitted serialization warnings;
a single bounded correctness repair was preregistered before changing it.

### Gate Status
Pipeline functional PASS; numerical gate FAIL; formal performance BLOCKED.

### Next Highest-Value Question
Can one bounded, semantics-neutral communication-order repair lower the
Sequential self-noise below 2e-3?

### Next Command
Pre-register pair-specific P2P + fixed Ring/Simple correctness settings, then
rerun two Sequential exports before any new Pipeline.

## Iteration 188

### Milestone / Gate
Critic-only Pool / bounded numerical-reproducibility repair and stop decision.

### Question
Are eager world8 P2P serialization and unconstrained NCCL protocol the dominant
source of the excessive same-layout update noise?

### Hypothesis
Four pair-specific P2P groups plus fixed Ring/Simple and thread caps will reduce
two-Sequential master update relative-L2 below 0.002.

### Importance
This is the only authorized repair before deciding whether formal timing and
world16 are scientifically admissible.

### Evidence Needed
Two new Sequential state exports from exact C11, exact input/control contract,
raw master/Adam comparison under the preregistered settings.

### Refutation Condition
Same-layout update relative-L2 remains above 0.002.

### Allowed Changes
Four two-rank P2P groups, separate world8 control group, correctness-only
`NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `CUDA_DEVICE_MAX_CONNECTIONS=1` and CPU
thread caps. No model/loss/optimizer/trace/checkpoint/queue/tolerance change.

### Commands
`DISTILLSCHEDULE_POOL_DETERMINISTIC=1` Sequential runs
`it188_det_seq_state_a/b`, then raw-state comparison and full pytest.

### Measured Results
- Both repaired Sequential runs completed with exact inputs, queues and versions.
- Makespans were 153.8022 s / 165.3343 s with correctness-state export; not
  formal timing.
- Master update cosine 0.99985108 and relative-L2 **0.01727223**; final-master
  direct relative-L2 9.60586e-6.
- Max loss abs difference 7.95126e-5; Adam m/v direct relative-L2
  0.00191806 / 0.00033447.
- Repair did not reduce noise; it increased the measured update-relative value.
- Full repository suite: 207 passed, 21 warnings, 26.45 s.

### Correctness
Exact semantic contract PASS; frozen numerical contract FAIL.

### Performance
Formal timing Not Run. Preliminary Pipeline timing remains non-claim evidence.

### Interpretation
The bounded repair hypothesis is refuted. Continuing to tune communication or
silently relax the threshold would violate the preregistered budget. The
candidate remains functionally promising but is a bounded negative under the
current numerical contract.

### Files Changed
Pair-specific P2P groups, correctness-only launch settings,
`artifacts/manifests/iteration188_*`, single-node run table and
`docs/19_critic_only_pool_report.md`.

### Failure or Limitation
The evidence does not identify the entire numerical source. CPUAdam/NCCL/BF16
reduction effects remain plausible, but further attribution requires a new,
explicitly authorized contract/experiment rather than ad hoc threshold tuning.

### Gate Status
`BOUNDED_NEGATIVE_NUMERICAL_CONTRACT`. Formal timing and world16 Not Run.

### Next Highest-Value Question
Should the project retain the 0.002 update-delta cap and close this candidate,
or explicitly replace it with a measured same-layout state-space/noise-floor
contract before formal timing?

### Next Command
Wait for that contract decision; do not enter world16 or claim the preliminary
1.44–1.54× speedup.

## Iteration 189

### Milestone / Gate
Critic-only Pool / post-reduce, pre-optimizer numerical attribution.

### Question
Does the same-layout Sequential variability already exist in the ZeRO-1 FP32
gradient partition after DP reduction and before unscale/clipping/CPUAdam, or is
it first introduced by the optimizer step?

### Hypothesis
If two exact-C11 Sequential runs already differ at this pre-step gradient point,
CPUAdam is not the first source: the variability enters in backward or DP
reduction. If this gradient is exact while the master update differs, CPUAdam or
the optimizer step is implicated.

### Importance
The frozen 0.002 update-delta cap failed in the baseline itself. This diagnostic
closes causal attribution without changing the model, optimizer, schedule,
process groups, trace, package, queue, or tolerance.

### Evidence Needed
Two normal Sequential runs from the same immutable target-world4 C11 package;
wave-0 raw FP32 gradient partitions from Student ranks 4–7; exact input,
prediction/loss diagnostics; chunked cosine/relative-L2/max-absolute comparison;
and the already established first Adam/update evidence.

### Refutation Condition
The exported gradient partitions are exact while the first master/Adam update is
not, or the input/target/version contract is not exact.

### Allowed Changes
Correctness-only wave-0 gradient export, offline comparator, unit tests,
manifests/reporting. No model policy, optimizer, P2P/DP group, queue, trace,
checkpoint, tolerance, formal timing, or world16 change.

### Commands
Pre-registered as `it189_grad_seq_a` (nonce 189001) and
`it189_grad_seq_b` (nonce 189002), Sequential, SHA-256 correctness checksum,
no final-state export, and `DISTILLSCHEDULE_EXPORT_GRADIENT_WAVE=0`.

### Measured Results
Not Run.

### Correctness
Not Run.

### Performance
Not Run; these runs are invalid for formal timing because raw multi-GiB gradient
I/O occurs between backward and optimizer step.

### Interpretation
Pending.

### Files Changed
Gradient exporter, comparator, unit test, preregistration manifest, and this
append-only record.

### Failure or Limitation
The diagnostic localizes the first difference only to backward/DP-reduce versus
optimizer; it does not by itself separate BF16 kernel nondeterminism from NCCL
reduction order.

### Gate Status
PREREGISTERED_NOT_RUN; prior `BOUNDED_NEGATIVE_NUMERICAL_CONTRACT` remains
binding.

### Next Highest-Value Question
Does wave-0 post-reduce gradient variability reproduce before CPUAdam?

### Next Command
Run the two preregistered Sequential gradient exports and compare them.

## Iteration 190

### Milestone / Gate
Critic-only Pool / measured closure of the Iteration 189 pre-optimizer attribution.

### Question
Does the first same-layout numerical difference appear before CPUAdam?

### Hypothesis
Exact wave-0 inputs/forward/loss but non-exact post-DP-reduce FP32 gradient
would exonerate CPUAdam as the first source and localize the variability to
Student backward or DP reduction.

### Importance
This decides whether an optimizer repair is scientifically justified, without
changing the frozen schedule or correctness cap.

### Evidence Needed
Two pre-registered exact-C11 Sequential runs, wave-0 gradient partitions from
Student ranks 4–7, exact identities, and offline chunked vector metrics.

### Refutation Condition
The pre-step gradients are exact while the update differs, or the wave-0
input/prediction/loss identity is not exact.

### Allowed Changes
Only those frozen in Iteration 189; no model, optimizer, collective, trace,
checkpoint, queue, tolerance, timing, or world16 change.

### Commands
`DISTILLSCHEDULE_EXPORT_GRADIENT_WAVE=0` with `it189_grad_seq_a` nonce 189001
and `it189_grad_seq_b` nonce 189002, followed by
`summarize_critic_pool_gradient_pair.py --wave 0`.

### Measured Results
- Both runs: 8/8 ranks success; cgroup OOM/kill 0; G peak 36,928 MiB and S
  peak 35,024 MiB.
- All 16 input identities, queues and G3/C11→C15 versions exact.
- Wave-0 4/4 prediction hashes and all four local loss values exact.
- Four FP32 gradient partitions, 1,246,314,496 elements: cosine
  0.999999769595, relative-L2 0.000678832747, max abs 3.8146973e-6; not exact.
- Per-rank relative-L2: rank4 0.0014954064, rank5 0.0008334757, rank6
  0.0007323416, rank7 0.0001238877.
- Global preclip norm 0.2270611972 vs 0.2270615697; max norm 10 and scale 1,
  so clipping is inactive.
- Diagnostic cycle times 156.553/154.804 s and node peaks 217.974/217.863 GiB.

### Correctness
Exact semantic identity PASS. Pre-optimizer bitwise identity FAIL by design of
the attribution result: the first difference is already measurable before
CPUAdam.

### Performance
Not Run. The diagnostic cycle times are invalid for formal performance because
about 4.64 GiB raw FP32 gradient data are written between backward and step.

### Interpretation
CPUAdam is not the first divergence source. The earliest observed difference is
in Student backward or DP4 reduction. This closes the optimizer attribution but
does not split kernel nondeterminism from reduction-order effects.

### Files Changed
Gradient raw artifacts, comparison/boundary/result manifests,
`docs/20_critic_pool_numerical_attribution.md`, attribution table, and this
append-only record.

### Failure or Limitation
The frozen 0.002 update-delta cap remains unchanged and already fails for the
Sequential baseline. Formal Pipeline timing and world16 therefore remain
blocked. No direct pre-reduce local-gradient comparison was run.

### Gate Status
`FIRST_DIVERGENCE_PRE_OPTIMIZER` PASS;
`BOUNDED_NEGATIVE_NUMERICAL_CONTRACT` remains the overall candidate gate.

### Next Highest-Value Question
Should the experiment retain the ill-conditioned 0.002 update-delta cap and
close, or explicitly preregister a same-layout gradient/state-space noise-floor
contract before formal timing?

### Next Command
Wait for the contract decision; do not claim speedup or enter world16.

## Iteration 191

### Milestone / Gate
Critic-only Pool / pre-reduce local-gradient attribution.

### Question
Does same-layout variability already exist in the per-rank autograd-delivered
Critic LoRA gradients before ZeRO-1 backward-epilogue DP reduction?

### Hypothesis
Any local-gradient SHA mismatch under exact wave-0 input/prediction/loss proves
that Student backward is sufficient to create the first divergence. If every
local gradient is exact but the post-reduce partition differs in the same pair,
DP4 reduction is the first observed source.

### Importance
Iteration 190 localized the difference to backward-or-reduce. This bounded
experiment decides which subsystem should receive any future reproducibility
work and prevents an unjustified CPUAdam/NCCL repair.

### Evidence Needed
Two normal exact-C11 Sequential runs; SHA-256 for all 812 autograd-delivered
trainable gradients per Student rank at wave 0; the existing raw post-reduce
FP32 partition; exact input/prediction/loss and version/queue identity.

### Refutation Condition
The semantic identity changes, the fingerprint inventory is incomplete, or all
local and post-reduce gradients are exact, making the timing-perturbed probe
inconclusive.

### Allowed Changes
Correctness-only local-gradient hook fingerprints, existing post-reduce export,
offline comparator, tests and evidence. No model/loss/optimizer/group/queue/
trace/package/tolerance/policy change; no formal timing or world16.

### Commands
Pre-registered `it191_localgrad_seq_a/b`, nonces 191001/191002, Sequential,
SHA-256, no final-state export, with both local fingerprint and post-reduce
export enabled for wave 0.

### Measured Results
Not Run.

### Correctness
Not Run.

### Performance
Not Run; hook-side SHA-256 and raw gradient I/O invalidate timing.

### Interpretation
Pending.

### Files Changed
Local-gradient observer/comparator/tests, preregistration manifest, append-only
research record.

### Failure or Limitation
The blocking fingerprint changes kernel/communication timing. A mismatch remains
decisive; exactness may be probe-induced and is therefore not automatically
proof of uninstrumented determinism.

### Gate Status
PREREGISTERED_NOT_RUN. Existing `BOUNDED_NEGATIVE_NUMERICAL_CONTRACT` remains
binding.

### Next Highest-Value Question
Do local pre-reduce gradients differ before NCCL?

### Next Command
Run the two preregistered correctness-only Sequential experiments.

## Iteration 192

### Milestone / Gate
Critic-only Pool / measured pre-reduce backward attribution.

### Question
Does local Student backward already differ before DP4 reduction?

### Hypothesis
Any local-gradient SHA mismatch under exact wave-0 input/prediction/loss proves
backward is sufficient for the first divergence; local exactness followed by a
post-reduce mismatch would instead implicate DP reduction.

### Importance
This closes the remaining backward-versus-reduce ambiguity and prevents further
repairs against CPUAdam or NCCL without causal evidence.

### Evidence Needed
Two preregistered exact-C11 runs, complete 812-gradient inventory on each of
four Student ranks, exact wave-0 boundary identity, and paired post-reduce FP32
metrics.

### Refutation Condition
Local gradients are exact, inventory is incomplete, or boundary identity fails.

### Allowed Changes
Only the Iteration 191 correctness probe; no training or scheduling change.

### Commands
`it191_localgrad_seq_a/b` with local fingerprint and post-reduce export at wave
0, followed by the local and raw FP32 pair summarizers.

### Measured Results
- 8/8 ranks succeeded in both runs; no cgroup OOM/kill.
- All 16 inputs, queues and G3/C11→C15 versions exact; wave-0 4/4 predictions
  and all local losses exact.
- Per Student rank, 800/812 local gradient tensors differed by SHA-256; aggregate
  3,200/3,248 (98.522%). Mismatching tensors cover 98.484% of tensor numel.
- Same-pair post-reduce gradient: cosine 0.999999765776, relative-L2
  0.000684435468, max abs 3.8146973e-6.
- Diagnostic cycle times 179.971/158.409 s; invalid for performance because the
  local hook performs synchronous hashing and raw gradient export.

### Correctness
Exact semantic identity PASS. Local pre-reduce numerical exactness FAIL, which
is the positive attribution signal under the preregistered decision rule.

### Performance
Not Run. No speedup claim.

### Interpretation
Student backward itself is sufficient to create run-to-run gradient differences
before NCCL reduction or CPUAdam. Reduction may add noise but is not necessary.

### Files Changed
Local/post-reduce/boundary/result manifests, attribution report and table, and
this append-only result entry.

### Failure or Limitation
The probe does not identify the individual CUDA backward kernel and does not
measure per-element local relative-L2 because only strong hashes were retained.

### Gate Status
`BACKWARD_SUFFICIENT_FOR_DIVERGENCE`; overall
`BOUNDED_NEGATIVE_NUMERICAL_CONTRACT` remains unchanged. Formal timing and
world16 remain Not Run.

### Next Highest-Value Question
Keep the frozen 0.002 update-delta cap and close, or authorize a preregistered
same-layout gradient/state-space noise-floor contract for formal timing?

### Next Command
Await the contract decision; do not tune CPUAdam/NCCL or enter world16.

## Iteration 193

### Milestone / Gate
Critic-only Pool / resident Student admission and real Critic-backward gate.

### Question
Can each dedicated S GPU keep the complete BF16 Score backbone and Critic LoRA
module resident while completing four official-shape Critic updates, rather
than streaming every frozen Transformer block?

### Hypothesis
ZeRO-1 CPU optimizer offload plus audited activation offload leaves enough
A100-40GB capacity for resident Score/Critic parameters and a real Critic
forward/backward/step at 384x640.

### Importance
The proposed fixed-role G/S Pool is meant to remove repeated model-weight
movement. The prior executor used `streamed_block_count=0`, which means all
frozen blocks in the historical API, so its 11.106-GiB Student peak did not test
the resident-Student premise.

### Evidence Needed
One preregistered Sequential-4G4S run from exact target-layout C11, 8/8 rank
success, four Critic commits, resident parameter inventory before/after, GPU and
node memory peaks, exact trace/version/queue identities, and no OOM.

### Refutation Condition
Any S rank OOMs during Critic forward/backward/step; any module parameter leaves
its dedicated S GPU or changes storage ownership; or C11-to-C15, trace, queue,
or ZeRO-1 subgroup semantics fail.

### Allowed Changes
Remove frozen Score block streaming from the Student primary path; add a
resident-storage guard and manifest/test evidence. Keep model, loss, ZeRO-1,
CPU optimizer, activation policy, transport, trace, package, EMA isolation,
queue and version order unchanged.

### Commands
Planned: `launch_critic_pool_real14b_cycle.sh sequential
it193_resident_student_seq_gate 193001 sha256 noexport`.

### Measured Results
Not Run.

### Correctness
Not Run.

### Performance
Not Run; this is an admission/correctness gate and cannot be used as formal
timing.

### Interpretation
Pending. The existing 35.643-GiB resident Fake-Score forward is supporting
capacity evidence only; it does not substitute for this backward gate.

### Files Changed
Resident Student guard/policy in the real-14B executor, launcher policy fields,
unit test, preregistration manifest, gate note and this append-only entry.

### Failure or Limitation
The frozen 0.002 update-delta contract remains independently blocked by the
measured same-layout backward noise floor. Passing this admission gate does not
authorize formal timing or world16.

### Gate Status
PREREGISTERED_NOT_RUN.

### Next Highest-Value Question
Does the resident Student complete the real four-wave Critic cycle within
A100-40GB?

### Next Command
Run the preregistered Iteration 193 Sequential resident-Student gate.

## Iteration 194

### Milestone / Gate
Critic-only Pool / measured resident Student gate result.

### Question
Did the dedicated S GPU fail because the resident Score/Critic compute itself
cannot fit, or because another runtime allocation occurs after the update?

### Hypothesis
The preregistered resident module either completes the whole four-wave gate, or
the precise failure boundary will identify the remaining non-parameter reserve.

### Importance
This distinguishes “Critic weights must stream” from “resident compute fits but
the transport/control plane needs a bounded reserve.”

### Evidence Needed
All rank JSON failures, source-order call site, GPU/node monitor, activation
census, resident storage inventory and cgroup events.

### Refutation Condition
The module fails to materialize, Critic forward/backward fails before an update,
or host OOM invalidates the GPU attribution.

### Allowed Changes
No changes beyond the Iteration 193 preregistration.

### Commands
`launch_critic_pool_real14b_cycle.sh sequential
it193_resident_student_seq_gate 193001 sha256 noexport` at commit
`fb3bbcf96d55b0487312522f31757ad8ea192188`.

### Measured Results
- All four S ranks materialized the complete resident module: 28.9804 GiB
  unique module parameter storage each, comprising 26.6590 GiB frozen and
  2.3214 GiB live trainable module storage; block streaming was disabled.
- The first Critic forward, backward and `engine.step()` returned. All four S
  ranks then failed at `transport.send_ack()` with NCCL CUDA OOM.
- S GPU monitor peak: 40,336 MiB; G peak: 36,928 MiB.
- PyTorch S failure snapshot: max allocated 36.2577 GiB, max reserved 36.6387
  GiB, driver free about 2.4 MiB at failure.
- Wave-0 activation hook offloaded 5,052,825,600 bytes and retained
  685,260,288 bytes below the 32-MiB cutoff.
- Node cgroup peak: 236,180,393,984 bytes (219.96 GiB); cgroup OOM/kill 0.

### Correctness
Resident parameter/storage admission PASS. Resident Critic compute reached the
post-step ACK call site. Full four-wave queue/version gate FAIL because ACK was
not delivered.

### Performance
Not Run. A failed gate cannot be a speedup denominator.

### Interpretation
The experiment does not support a claim that Critic weights require streaming.
The resident compute is feasible. The bounded failure is the first
reverse-direction GPU NCCL control message under essentially zero driver
headroom. Fake-x0 data-plane transfer had already succeeded.

### Files Changed
Raw rank/event/monitor/log artifacts and Iteration 194 result manifest; gate
report and append-only research log updated.

### Failure or Limitation
Because the exception occurred before the wave record was serialized, the
C11-to-C12 counter is source-order evidence (`engine.step()` returned before
`send_ack`) rather than a final exported optimizer package.

### Gate Status
`RESIDENT_COMPUTE_PASS_ACK_GPU_ADMISSION_FAIL`.

### Next Highest-Value Question
Can a CPU/Gloo ACK control plane remove the post-step GPU allocation while
leaving x0 on identical NCCL P2P data paths for Sequential and Pipeline?

### Next Command
Preregister and run one resident Sequential gate with NCCL x0 plus Gloo ACK.

## Iteration 195

### Milestone / Gate
Critic-only Pool / resident Student with split data/control transport.

### Question
Can the tiny post-step ACK avoid GPU admission while generated x0 remains on
pair-local NCCL P2P?

### Hypothesis
The Iteration 193 failure is caused by lazy reverse NCCL ACK allocation, not by
resident Critic compute. A CPU/Gloo ACK shared by both schedules removes that
allocation and completes C11-to-C15.

### Importance
This is the smallest causal repair: the 1.1-MiB fake latent remains a GPU NCCL
data-plane tensor, while a fixed-width control record no longer competes with
resident weights/gradients for the last GPU reserve.

### Evidence Needed
8/8 rank success; launch/rank transport identities; four x0 and ACK events per
pair; exact trace/version/queue contracts; stable resident parameter storage;
GPU/node peaks and no OOM.

### Refutation Condition
Any OOM remains, x0 does not use NCCL, or semantic/storage/queue identity fails.

### Allowed Changes
Add pair-local Gloo groups only for `CriticCommitAck`; keep header/x0 on the
existing pair-local NCCL groups. No model/loss/optimizer/activation/checkpoint/
trace/queue/version/tolerance or schedule change.

### Commands
Planned: `launch_critic_pool_real14b_cycle.sh sequential
it195_resident_glooack_seq_gate 195001 sha256 noexport`.

### Measured Results
Not Run.

### Correctness
Not Run.

### Performance
Not Run; admission/correctness gate only.

### Interpretation
Pending.

### Files Changed
Hybrid transport wiring, launcher/rank policy fields, transport unit test,
preregistration manifest and append-only log.

### Failure or Limitation
Gloo ACK latency is not yet measured and cannot be used for a pipeline claim.
Both Sequential and Pipeline must use the same control plane in any later
causal comparison.

### Gate Status
PREREGISTERED_NOT_RUN.

### Next Highest-Value Question
Does the resident four-wave Sequential cycle now complete?

### Next Command
Run Iteration 195 after tests and a clean preregistration commit.

## Iteration 196

### Milestone / Gate
Critic-only Pool / resident Student + Gloo ACK measured result.

### Question
Does separating the tiny ACK control plane complete the resident four-wave
Sequential gate while x0 remains on NCCL?

### Hypothesis
The post-step failure disappears because Gloo ACK allocates no reverse GPU NCCL
buffer, while all training/data semantics remain unchanged.

### Importance
This decides whether full resident G/S role pools are actually feasible on one
8xA100-40GB node without frozen Score block streaming.

### Evidence Needed
8/8 success, 16/16 trace and x0 SHA identities, G3 immutability, C11-to-C15,
queue flush, resident storage stability, transport identities and memory peaks.

### Refutation Condition
Any rank failure, non-NCCL x0, semantic mismatch, parameter storage movement or
GPU/host OOM.

### Allowed Changes
Only the Iteration 195 preregistered ACK control-plane change.

### Commands
`launch_critic_pool_real14b_cycle.sh sequential
it195_resident_glooack_seq_gate 195001 sha256 noexport` at commit
`558fb0d2fe9fb736c63c0d28f70a98430104d132`.

### Measured Results
- Return code 0; 8/8 ranks succeeded; cgroup OOM/kill 0.
- 16/16 operation-trace identities and 16/16 generated/received x0 SHA-256
  identities matched.
- G remained G3; trainable SHA exact before/after; no trainable Generator grad.
- Every S rank advanced C11-to-C15 with optimizer and scheduler 11-to-15;
  exactly four waves committed and both slots/outstanding work were empty.
- Complete Student module stayed resident at 28.9804 GiB parameter storage.
- Generator max allocated 33.6608 GiB; Student max allocated 36.2577 GiB.
  Monitor peaks: 36,736 MiB G and 40,290 MiB S.
- Node cgroup peak 232,888,762,368 bytes (216.89 GiB).
- SHA-instrumented four-wave makespan 110.6543 s; diagnostic only. It
  corresponds to 0.14459 samples/s globally, not a formal throughput result.

### Correctness
Resident Sequential admission/semantic gate PASS.

### Performance
Not Run formally. This is one correctness launch with SHA and detailed
instrumentation, without warmup discard, repeated independent launch or node
crossover.

### Interpretation
The correction is decisive: full resident Score/Critic computation is feasible
on a dedicated A100-40GB. Frozen block streaming is not required by this pool
layout. The previous failure was the GPU ACK control plane, not the resident
weights or Critic backward.

### Files Changed
Raw rank/event/monitor/log artifacts, result manifest, resident gate report and
append-only research log.

### Failure or Limitation
The existing update-delta numerical contract remains blocked by same-layout
backward nondeterminism. This run does not authorize a Pipeline speedup claim or
world16.

### Gate Status
`RESIDENT_SEQUENTIAL_4G4S_PASS`; formal numerical/performance gate remains
`BOUNDED_NEGATIVE_NUMERICAL_CONTRACT`.

### Next Highest-Value Question
Does resident Pipeline-Q1 preserve the exact queue/input/version semantics and
complete without exceeding the tight Student memory headroom?

### Next Command
Preregister and run one SHA-instrumented resident Pipeline-Q1 functional gate;
do not call it formal timing.

## Iteration 197

### Milestone / Gate
Critic-only Pool / resident Pipeline-Q1 functional gate.

### Question
Can the intended two-slot G/S overlap execute with both models resident while
preserving queue and parameter-version semantics?

### Hypothesis
G(B_i+1) is independent of the current Student C_i-to-C_i+1 update because G
remains G3 across all four waves; the pipeline therefore completes with at
most one ready producer batch plus one consuming Student batch.

### Importance
The resident Sequential gate is now proven. This is the smallest real-14B test
of the actual scheduling idea rather than model loading or transport setup.

### Evidence Needed
8/8 success, 16 trace/x0 SHA identities, exact G3/no-grad, serial C11-to-C15,
queue counters/slots/work, transport policy and memory peaks.

### Refutation Condition
Any deadlock, OOM, stale batch, duplicate/missing wave, version crossing or
input/storage mismatch.

### Allowed Changes
None. Run the already committed `pipeline` branch with the same resident models,
checkpoint, trace, NCCL x0 and Gloo ACK as Sequential.

### Commands
Planned: `launch_critic_pool_real14b_cycle.sh pipeline
it197_resident_glooack_pipeline_gate 197001 sha256 noexport`.

### Measured Results
Not Run.

### Correctness
Not Run.

### Performance
Not Run formally; one SHA-instrumented functional launch only.

### Interpretation
Pending.

### Files Changed
Preregistration manifest and append-only log only.

### Failure or Limitation
Final optimizer-state equivalence remains blocked by the frozen ill-conditioned
update-delta contract; no formal performance claim is permitted.

### Gate Status
PREREGISTERED_NOT_RUN.

### Next Highest-Value Question
Does resident Pipeline-Q1 complete the isolated four-wave subgraph?

### Next Command
Run the preregistered Pipeline-Q1 functional gate.

## Iteration 198

### Milestone / Gate
Critic-only Pool / measured resident Pipeline-Q1 functional result.

### Question
Does the two-slot resident pipeline execute the intended overlap while keeping
all input, queue and version semantics exact?

### Hypothesis
The schedule changes only operation order: G remains G3 while S serially
advances C11-to-C15, so all four batches can be consumed exactly once and the
step-15 boundary can flush.

### Importance
This is the first real-14B result for the proposed resident G/S pipeline rather
than a streaming/offload surrogate.

### Evidence Needed
The Iteration 197 preregistered rank records, pairwise Sequential/Pipeline input
identity comparison, queue/version state and diagnostic makespan/memory.

### Refutation Condition
Any rank/identity/queue/version/memory failure.

### Allowed Changes
None beyond the preregistered schedule selection.

### Commands
`launch_critic_pool_real14b_cycle.sh pipeline
it197_resident_glooack_pipeline_gate 197001 sha256 noexport` at commit
`b3003817b1bef905d55cab6b1e722ec37e0bbe46` plus offline boundary comparison
to the Iteration 195 resident Sequential run.

### Measured Results
- Pipeline: return code 0, 8/8 ranks success, cgroup OOM/kill 0.
- Against Sequential: trace, generated x0, wire x0, noise and timestep were all
  exact for 16/16 entries.
- G remained G3 with exact trainable SHA and no gradient. Each S rank committed
  C11-to-C15 serially; all queue counters were four, both slots empty and
  outstanding work zero.
- Prediction/loss SHA exactness was 4/16 (the first wave); max later loss
  absolute difference was 3.8147e-5 and max preclip-norm relative difference
  4.3956%. Final state was not exported.
- Diagnostic SHA-instrumented makespan: Sequential 110.6543 s, Pipeline
  73.9825 s, ratio 1.4957x and 33.14% makespan reduction. This is not formal.
- Pipeline PyTorch peaks: 33.6608 GiB G, 36.2577 GiB S. Node cgroup peak
  235,135,385,600 bytes (218.99 GiB), no host OOM.

### Correctness
Input/transport/queue/version/residency contract PASS. Numerical optimizer-state
equivalence Not Run; prediction divergence after wave 0 is consistent with the
already measured same-layout backward variability but is not waived.

### Performance
Preliminary only. The 1.4957x ratio is from one SHA-instrumented launch per
schedule, without warmup discard, repetitions, randomized order or crossover.

### Interpretation
The intended legal overlap exists and is large enough to be worth formal
validation. Resident pools remove parameter movement, and Q1 overlaps role
compute without stale G or parallel Critic updates. The result is not yet a
paper speedup because the numerical and formal timing gates remain open.

### Files Changed
Raw rank/event/monitor/log artifacts, result manifest, pipeline report and this
append-only result entry.

### Failure or Limitation
No final master/Adam vectors were exported. The frozen 0.002 update-delta gate
is still invalidated by the Sequential self-noise evidence and has not been
replaced by user authorization.

### Gate Status
`RESIDENT_PIPELINE_FUNCTIONAL_PASS`; `FORMAL_TIMING_NOT_AUTHORIZED`.

### Next Highest-Value Question
Can the timing path remove SHA/CPU-copy instrumentation and enforce warmup
discard plus exact target C11 restore without accidentally launching a formal
claim?

### Next Command
Implement and test low-overhead tensor identities and a fail-closed formal-run
wrapper; leave the numerical contract status proposed/not active.

## Iteration 199

### Milestone / Gate
Critic-only Pool / low-overhead formal timing path implementation, formal run
blocked.

### Question
Can future timing avoid full tensor SHA/per-call detailed tracing and guarantee
warmup state is discarded before a fresh C11 timing launch?

### Hypothesis
GPU scalar fingerprints and lightweight call auditing retain corruption,
no-grad, call-count and stage checks without CPU-copying full outputs. Separate
warmup/formal processes make C11 restoration fail closed.

### Importance
The 1.4957x diagnostic result cannot become a paper result if instrumentation
is inside the measured region or warmup silently advances C11 to C15.

### Evidence Needed
Unit/full tests, source audit, an inactive-contract wrapper rejection before
GPU launch, and no formal experiment artifacts.

### Refutation Condition
Any SHA remains unconditional in the timed waves, detailed tracer remains in
timing mode, the wrapper can launch with a proposed contract, or tests fail.

### Allowed Changes
Timing identities/audit, validation/wrapper, proposed-not-active contract,
tests/docs only. No training or schedule primitive change.

### Commands
Implementation and tests planned; formal launcher must be tested only for its
pre-launch exit-3 rejection.

### Measured Results
Not Run.

### Correctness
Not Run.

### Performance
Not Run and not authorized.

### Interpretation
Pending.

### Files Changed
Timing identity/auditor/runtime wiring, formal validator/wrapper, proposed
contract, tests, docs and this append-only entry.

### Failure or Limitation
GPU fingerprints are non-cryptographic timing guards; SHA remains mandatory for
correctness runs. The wrapper cannot activate its own contract.

### Gate Status
IMPLEMENTATION_ONLY_FORMAL_BLOCKED.

### Next Highest-Value Question
Does the implementation fail closed and pass the full test suite?

### Next Command
Run targeted/full tests, source audit and inactive-contract wrapper rejection.

## Iteration 200

### Milestone / Gate
Critic-only Pool / low-overhead timing infrastructure validation.

### Question
Does the new path remove correctness instrumentation from timing and reject a
formal launch before GPU use while the numerical contract is inactive?

### Hypothesis
The implementation passes tests/source audit and the wrapper exits 3 without a
GPU process for `PROPOSED_NOT_ACTIVE`.

### Importance
This prevents the preliminary 1.4957x ratio from being promoted through a
contaminated or state-shifted timing protocol.

### Evidence Needed
Targeted/full tests, bash/py compile, source SHA audit, wrapper return code and
GPU process census.

### Refutation Condition
Any test failure, unconditional timed SHA, wrapper launch or GPU process.

### Allowed Changes
Only Iteration 199 infrastructure.

### Commands
Targeted pytest, full pytest, `bash -n`, grep source audit, and wrapper invocation
with the proposed-not-active contract.

### Measured Results
- Targeted tests: 32 passed.
- Full suite: 217 passed, 21 dependency/deprecation warnings.
- Shell syntax and Python compile passed.
- Remaining `tensor_sha256` call sites are correctness/post-cycle paths; the
  Generator output SHA is conditional and timing mode selects GPU identities.
- Formal wrapper returned 3 with the expected inactive-contract message before
  launching any GPU process; GPU process census remained empty.

### Correctness
Infrastructure/unit gate PASS. Real timing-mode 14B execution Not Run.

### Performance
Not Run. No overhead or speedup result was produced.

### Interpretation
The future formal protocol is now fail closed: warmup cannot leak C15 into the
formal process, and a proposed contract cannot self-activate. This is enabling
engineering, not a new performance result.

### Files Changed
Generic identity, lightweight call audit, timing-mode executor/launcher,
validator, formal wrapper, proposed contract, tests/docs/manifests and log.

### Failure or Limitation
A real timing-mode launch and paired-median profiler-overhead test remain Not
Run. They are intentionally blocked with formal timing until the user decides
the numerical contract.

### Gate Status
`FORMAL_WRAPPER_READY_BUT_BLOCKED`; numerical contract
`PROPOSED_NOT_ACTIVE`.

### Next Highest-Value Question
Should the frozen ill-conditioned update-delta cap be replaced by the proposed
same-layout exact-semantic plus measured gradient/state noise-floor contract?

### Next Command
Await explicit contract authorization; do not run formal timing or world16.

## Iteration 201

### Milestone / Gate
Critic-only resident Pool / formal numerical-contract activation and preregistration.

### Question
Can the existing proposed same-layout contract be activated without adding tolerance, then fail-closed gate formal resident 4G+4S timing?

### Hypothesis
The exact semantic boundary plus the already measured Sequential post-reduce-gradient and direct FP32 state envelope is a well-conditioned binding contract; update-delta remains a non-gating diagnostic.

### Importance
Formal Pipeline timing is scientifically usable only if correctness is checked against the baseline noise floor rather than an ill-conditioned tiny update delta.

### Evidence Needed
One fresh resident Sequential/Pipeline SHA+state+wave-0-gradient pair, followed only after PASS by three preregistered low-overhead paired timing rounds.

### Refutation Condition
Any exact identity/version/queue failure, gradient relative-L2 above 0.0006844354683410283, direct master above 5.342643191672148e-06, direct Adam-m above 0.0016096859647390276, or direct Adam-v above 0.00029217883629420346.

### Allowed Changes
Activate the existing manifest, encode its existing measured thresholds without margin, add a fail-closed validator/tests, and add formal manifests/logs/reports.

### Commands
Pre-run state audit: branch distillschedule/autoresearch, HEAD 68170804b51e7ec4596a9a81623e92e34791781d, clean tree, eight idle A100s, cgroup oom/oom_kill zero, assets present. Full pre-change test suite: 217 passed.

### Measured Results
Not Run at preregistration time.

### Correctness
Not Run at preregistration time.

### Performance
Not Run at preregistration time.

### Interpretation
Formal action order is frozen as S/P, P/S, S/P. No failed round may be dropped. GAS2/world16/Generator-update/EMA/full-Helios remain Not Run.

### Files Changed
`critic_pool_formal_numerical_contract_proposed_v1.json`, Iteration-201 preregistration, numerical-contract validator/tests, append-only log.

### Failure or Limitation
The existing envelope is deliberately strict and has no added safety margin. A failure closes this timing attempt rather than triggering threshold repair.

### Gate Status
PREREGISTERED; GPU correctness gate pending.

### Next Highest-Value Question
Does a fresh resident Pipeline remain inside the activated same-layout envelope?

### Next Command
Run fresh resident SHA/state/gradient-wave0 Sequential, then Pipeline, summarize, and apply the fail-closed numerical validator.

## Iteration 202

### Milestone / Gate
Critic-only resident Pool / activated-contract numerical preflight.

### Question
Can the fresh resident Sequential preflight reach the same-layout gradient/state gate under an idle, clean single-node launch?

### Hypothesis
The previously admitted resident 4G+4S construction will repeat, allowing a fresh Sequential/Pipeline state comparison before formal timing.

### Importance
The activated numerical contract must be checked on a fresh resident pair before any low-overhead timing result is accepted.

### Evidence Needed
Successful Sequential and Pipeline state/gradient exports, exact identities, and all binding gradient/direct-state metrics inside the frozen envelope.

### Refutation Condition
Any construction OOM, exact semantic failure, or binding numerical metric above the activated envelope.

### Allowed Changes
No model, batch, queue, trace, checkpoint, loss, optimizer, threshold, or timing-policy change. Execute the preregistered preflight and preserve its artifacts.

### Commands
`DISTILLSCHEDULE_EXPORT_GRADIENT_WAVE=0 scripts/distillschedule/launch_critic_pool_real14b_cycle.sh sequential it202_resident_contract_seq 920201 sha256 export`

### Measured Results
Return code 1. Generator ranks 1/2/3 OOMed in `model.to(device)` requesting 50/136/50 MiB. Monitor GPU peaks were `[20226,39748,39156,39748,4054,4054,4054,4054]` MiB; node peak was `360478216192` B. CUDA errors reported two process allocations on each affected G GPU. Host cgroup `oom=0`, `oom_kill=0`.

### Correctness
Not Run: the execution did not pass cold construction and produced no four-wave or numerical comparison.

### Performance
Not Run. The previous SHA-instrumented `1.495682x` remains preliminary and is not promoted.

### Interpretation
This is a fail-closed construction-admission/state-drift failure, not evidence for or against the same-layout numerical contract or Pipeline speedup. The immediate pre-launch GPU query was idle, but during construction the CUDA allocator reported duplicate/concurrent process occupancy. Those PIDs disappeared after teardown, so attribution remains unresolved.

### Files Changed
Iteration-202 result manifest, `docs/25_critic_pool_formal_attempt.md`, append-only log. Raw failed launch directory and log preserved.

### Failure or Limitation
The run cannot distinguish an external transient process from launch-local process duplication after the PIDs disappear. The strict contract forbids replacing this failed attempt with an unregistered retry.

### Gate Status
STOPPED_FAIL_CLOSED_GPU_CONSTRUCTION_OOM. Pipeline preflight and formal rounds Not Run.

### Next Highest-Value Question
Can a future separately preregistered ownership guard identify and prevent the second GPU allocation throughout construction?

### Next Command
Not Run in this iteration; stop and report per the user-authorized fail-closed rule.

## Iteration 202

### Milestone / Gate
Critic-only resident Pool / fresh numerical preflight before formal timing.

### Question
Does the first fresh resident Sequential correctness export pass cold admission so the activated same-layout noise contract can be evaluated?

### Hypothesis
The previously admitted resident 4G+4S construction will reproduce on eight initially idle A100s; SHA/state/gradient export happens outside or after admission and will not change Generator construction.

### Importance
The formal timing wrapper may only run after an exact and numerical fail-closed gate.

### Evidence Needed
Successful Sequential and Pipeline resident runs from C11 with state and wave-0 post-reduce-gradient exports.

### Refutation Condition
Any GPU/host OOM, failed rank, exact mismatch, or numerical-envelope violation.

### Allowed Changes
No model, policy, batch, trace, checkpoint, loss, optimizer, queue, or threshold change. Only append-only artifacts.

### Commands
`DISTILLSCHEDULE_EXPORT_GRADIENT_WAVE=0 launch_critic_pool_real14b_cycle.sh sequential it201_resident_numeric_seq 201101 sha256 export`

### Measured Results
Return code 1. Rank 2 failed a 50-MiB CUDA allocation with 22.81 MiB free; rank 3 failed a 136-MiB allocation with 2.81 MiB free during Generator `model.to(device)`. Monitor peaks were 20,226/39,748/38,748/39,748/4,054/4,054/4,054/4,054 MiB. Node cgroup peak was 254,846,935,040 B; cgroup OOM/OOM-kill remained 0/0.

### Correctness
NOT RUN to completion. No exact or numerical candidate result was produced.

### Performance
NOT RUN. No formal warmup or timing round was launched.

### Interpretation
The preregistered GPU-OOM stop condition fired before numerical comparison. The log reported two CUDA process allocations on each failing Generator GPU even though the pre-launch audit saw idle GPUs. This is a construction-admission/runtime-drift result, not a Pipeline speed result.

### Files Changed
Failure output retained under `artifacts/profile/m1/critic_pool_cycles/it201_resident_numeric_seq_sequential`; append-only Iteration-202 manifest, report, and research log.

### Failure or Limitation
Root cause is not established because the protocol forbids retry/repair after OOM in this round. Earlier preliminary 1.495682x remains non-formal.

### Gate Status
STOP_GPU_OOM_BEFORE_NUMERICAL_GATE_AND_FORMAL_TIMING.

### Next Highest-Value Question
Under a separately authorized recovery protocol, why did resident construction create two CUDA allocations per Generator GPU?

### Next Command
Not Run pending explicit recovery authorization; do not launch timing.

## Iteration 203

### Milestone / Gate
Critic-only resident Pool / shared-node execution-drift reconciliation.

### Question
Why did the Iteration-202 CUDA errors report two process allocations on each Generator GPU after an idle pre-launch audit?

### Hypothesis
A second independent world8 preflight entered the same eight GPUs during the first launch.

### Importance
A concurrency collision must not be misclassified as a resident-capacity OOM or used to select a retry.

### Evidence Needed
Distinct output trees with overlapping monitor timestamps and identical combined GPU occupancy.

### Refutation Condition
Only one launcher/output tree exists, or their execution intervals do not overlap.

### Allowed Changes
Evidence reconciliation only; no GPU retry and no model/policy/threshold change.

### Commands
Inspect both launch manifests, node monitors, logs, process census, cgroup counters, and repository timestamps.

### Measured Results
`it201_resident_numeric_seq_sequential` started monitoring at 1785341229.6804643; `it202_resident_contract_seq_sequential` started at 1785341233.9457853, 4.265321 s later. Both ended at approximately 1785341321 and observed the same combined occupancy. Two world8 launchers therefore overlapped on the same eight GPUs.

### Correctness
Not evaluated; both preflights are invalid due to violated exclusive GPU ownership.

### Performance
Not Run; no formal round exists.

### Interpretation
The duplicate process allocations and construction OOM are causally explained by concurrent launches, not single-run resident capacity. This is exactly the task/state drift guarded by the user instruction.

### Files Changed
Append-only reconciliation manifest, report, and log; both failed raw artifact trees retained.

### Failure or Limitation
The identity of the second executor is not established, but the overlap itself is directly measured. No retry is permitted in this round.

### Gate Status
STOPPED_CONCURRENT_LAUNCH_STATE_DRIFT.

### Next Highest-Value Question
Can a future separately authorized attempt acquire an exclusive node/GPU lock before preregistered formal rounds?

### Next Command
Not Run; report and await a new bounded authorization.

## Iteration 203

### Milestone / Gate
Critic-only resident Pool / failed-launch attribution correction.

### Question
Why did Iteration 202 report two CUDA processes on each affected Generator GPU despite an idle audit?

### Hypothesis
A second DistillSchedule preflight overlapped the user-visible launch rather than a single job duplicating model state internally.

### Importance
The distinction decides whether the failure refutes resident capacity or only the orchestration guard.

### Evidence Needed
Independent launch manifests and node-monitor intervals proving two jobs used the same GPU set concurrently.

### Refutation Condition
Non-overlapping monitor intervals or different device sets.

### Allowed Changes
Read preserved logs/monitors and append a correction; no GPU retry and no model/policy/threshold change.

### Commands
Compare `it201_resident_numeric_seq_sequential` and `it202_resident_contract_seq_sequential` launch manifests, monitor timestamps, and CUDA OOM PID evidence.

### Measured Results
`it201` ran 16:07:09.680–16:08:41.184 UTC; `it202` ran 16:07:13.946–16:08:41.190 UTC. They overlapped 87.238 seconds, used the same commit and GPUs 0–7, and both returned 1. CUDA errors named the same two host PIDs per affected GPU. cgroup OOM/OOM-kill remained 0/0.

### Correctness
Not Run: both jobs failed before the four-wave cycle.

### Performance
Not Run: no formal timing region completed.

### Interpretation
The GPU OOM is fully explained by two concurrent DistillSchedule launches. It does not refute isolated resident capacity. It does demonstrate that the pre-launch idle check was not an atomic ownership guard.

### Files Changed
Corrected Iteration-202/result reports; added Iteration-203 collision-attribution manifest; append-only correction.

### Failure or Limitation
The activated protocol still requires stopping on any OOM/state drift. A scientifically valid retry needs a separately preregistered global launch lock and may not change the numerical thresholds.

### Gate Status
STOPPED_FAIL_CLOSED_OVERLAPPING_LAUNCH_GPU_OOM. No retry; formal rounds Not Run.

### Next Highest-Value Question
Can a separately authorized launch-lock repair prevent audit-to-construction races without changing execution semantics?

### Next Command
Not Run in this authorization scope.

---

## Iteration 205: W16 three-round suite — preregistration and immutable-input gates

### Status
ACTIVE. The user approved the three-round order: W16-Critic-Matched, W16-GC-Balance, then continuous W16-Full5. Formal unprofiled timing precedes one representative Nsight capture per round.

### Preflight
Two nodes expose 8 idle NVIDIA A100-SXM4-40GB GPUs each. Shared source and assets were verified, and the suite holds the cross-node ownership lock at `artifacts/locks/w16_three_rounds_20260730.lock`.

### Measured Gates
The W16 Critic-only operation trace contains 8 producers x 4 waves = 32 rank-qualified entries. Its pool trace hash is `941b6cccaab12030b2c0c0b07be7a8049f3a848f9909fa18f5b143ba2f7c96b2`; its state-provenance trace hash is `ebce07124d5f7500b0920e4e55f43f0998c71c5e764bdfac9196f0b431b3e939`.

The world8 C11 Student target was imported, natively verified, exported, contract-validated, and loaded successfully. Source manifest hash: `d114d98dc222280f83c15b5161a819a2bc141225074c4fe54c5aa74a98a703f8`. Target manifest hash: `4edbdbfefa3272e8ccffd73d43de3096f75b716abaae9d227a7c16dc0bd505f2`.

### Correctness
Focused runtime tests: 25 passed. Python compilation, shell syntax, and `git diff --check` passed. No formal timing or profiler claim is made by these gates.

### Next Command
Commit the clean W16 substrate, re-audit both nodes, and run W16-Critic-Matched in the preregistered Sequential/Pipeline and role-local/pair-local order.

---
## Iteration 205A: W16-Critic-Matched formal result

### Measured Results
All four preregistered fixed-work cases passed. Each run used 16 A100 GPUs as 8 Generator workers + 8 Critic workers, four waves, 32 logical samples, and four Critic optimizer updates. Role-local Sequential took 116.544 s and role-local Pipeline took 72.772 s: 1.6015x speedup and 37.56% lower makespan. Pair-local Sequential took 118.857 s and pair-local Pipeline took 75.029 s: 1.5842x speedup and 36.88% lower makespan.

Role-local Pipeline was 3.01% lower makespan than pair-local Pipeline and is selected. The selected run sustained 0.4397 logical samples/s. Generator produce averaged 10.924 s; Critic forward+backward+optimizer averaged 14.581 s. The continuous-service fit predicts approximately 7G+9C, with 7G+9C and 6G+10C as the adjacent candidates subject to the fixed-semantics gate.

### Correctness
Every rank succeeded; 32 trace entries and payload identities matched; queues flushed; Generator remained G3; Critic, optimizer, and scheduler advanced C11 to C15; no new cgroup OOM/OOM-kill occurred.

### Interpretation
The role-local result supports keeping Student DP collectives inside one node. The cross-node x0 payload is not the bottleneck. Unequal G:C is not authorized merely by the service-time fit: it must preserve 32 logical samples, four optimizer updates, loss normalization, and collective participation.

---
## Iteration 205B: W16-GC-Balance result

### Allocation gate
The measured service-time fit predicts 7G+9C, with 7G+9C and 6G+10C adjacent. Both are preregistered **Not Run**, not failed: the frozen update contract is four optimizer updates, MB1/GAS1, all Critic ranks participating, and 32 rank-distinct samples. This gives `4 x C = 32`, so `C=8`. C=9 or C=10 would process 36 or 40 samples; retaining 32 would require ragged, dummy, or idle ranks and would change loss normalization or collective work. The selected faithful allocation remains 8G+8C.

### Wave sweep
Role-local 8G+8C Pipeline passed at all wave counts. W4: 72.772 s, 0.4397 samples/s. W8: 136.016 s, 0.4705 samples/s. W16: 255.272 s, 0.5014 samples/s. W16 improves throughput by 14.03% over W4. A three-point fit gives 13.14 s startup/drain overhead and 15.17 s per additional wave; fitted bubble fraction drops from 18.1% at W4 to 5.1% at W16.

### Boundary
W8/W16 are isolated throughput sensitivity traces. Logical labels cross multiples of five, but Teacher and Generator updates are intentionally absent; these runs are not continuous Helios training.

---
## Unnumbered code-archive maintenance entry (2026-08-01)

> This is source-tree maintenance, not a new Auto-Research iteration or a new
> performance result. Existing iteration numbering and frozen experiment claims
> are unchanged.

### Question
Can the accumulated DistillSchedule code area be made easier to navigate without
removing active runtime paths or deleting reproducibility evidence?

### Hypothesis
The safest useful cleanup is to archive only one-shot tools with zero active
references, while preserving divergent experimental source histories and leaving
raw artifacts untouched until a separate manifest-based retention audit.

### Evidence and preflight
The canonical checkout was clean at
`3c9c4566403107ce5a7edf5a1a7b2404c878f80c`; both GPU nodes were reachable,
had zero user GPU processes, and reported no new cgroup OOM/OOM-kill. The full
pre-change CPU suite passed: `219 passed, 23 skipped, 20 warnings in 22.08s`.

The canonical, world8-prefetch, and Pipeline-CDE trees are not duplicate clones:
the prefetch line has 20 commits unique after its merge base and the Pipeline
line has 9 commits on top of the canonical pre-cleanup head. The canonical
`artifacts/` tree is about 661 GiB and is experiment evidence, not dead source.

A reference audit found 39 early one-shot
`summarize_iteration{0..41}.py` tools (with no files for 35--37) with zero
references from active runtime, scripts, tests, configs, or top-level code. They
total 491,651 bytes and 10,392 lines.

### Actions
Before moving code, complete verified Git bundles were written under
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_archives/code/2026-08-01/`
for canonical (`3c9c456...`), prefetch (`ec77900...`), and Pipeline-CDE
(`288d71a...`). Their SHA-256 values and `git bundle verify` results are recorded
in `artifacts/manifests/post_iteration_220_code_archive.json`. Canonical refs
`archive/prefetch-baseline-node1-20260801`,
`archive/pipeline-cde-node0-20260801`, and annotated tag
`archive/pre-code-cleanup-20260801` preserve the same source states.

The 39 inactive summarizers were moved with `git mv` to
`archive/iteration_tools/000_041/`. Active summarizers for Iterations 131 and 134
remain in `scripts/distillschedule/`. Navigation and provenance are documented in
`archive/README.md`, `scripts/distillschedule/README.md`, and
`docs/34_code_archive_audit.md`.

### Validation
Post-change CPU suite (final repeat with `CUDA_VISIBLE_DEVICES=""`):
`219 passed, 23 skipped, 20 warnings in 20.72s`. A device-visible unit-test
validation also passed `242 tests` in 23.83s; it was not a 14B/model performance
experiment. All archived Python tools compile. The old-path active-reference gate returns
zero. The manifest parses and `git diff --check` passes.

### Result and boundary
**PASS_NON_DESTRUCTIVE_CODE_ARCHIVE.** No runtime/model/loss/optimizer/trace/
checkpoint/schedule semantics changed. No 14B model or performance experiment was
performed. Post-validation GPU process count was zero and cgroup OOM/OOM-kill
remained 0/0. No raw artifact, failure, Not-Run record, config, checkpoint, branch, or experiment
checkout was deleted. The 661-GiB artifact tree remains intentionally untouched.

A future pass may consolidate launcher families and classify artifacts into
`keep`, `bundle`, `recomputable`, and `external checkpoint`, but only after a
manifest-based provenance scan. A Git bundle preserves source history; it does
not replace untracked experimental evidence.

---

## Task-status consolidation maintenance (2026-08-05)

### Scope

Documentation-only maintenance. No GPU experiment, performance measurement, or
scientific gate was run. Existing iteration bodies and manifests remain
append-only and were not renumbered, merged, deleted, or reinterpreted in place.

### Decision

Project progress is now maintained by research task rather than by iteration
count. `docs/36_task_status_registry.md` groups setup attempts, bounded repairs,
failed retries, corrections, and final results under seven research tasks.
Iteration identifiers remain run-level provenance only.

### Current status

The only open task is Step15-A exact-B0 world16 capacity. Its all-40-Generator-
block diagnostic memory-envelope preflight passes and its preregistration is
recorded, but the GPU gate is Not Run. The earlier B1 full-stream/hybrid passes remain bounded
capacity/profile evidence and do not establish exact-B0 feasibility. Exact-B0
tail-1/tail-3/tail-5 failures are consolidated as one repeated 216-MiB
checkpoint-recompute capacity-negative result.

### Gate identity preservation

The open gate's original preregistration also froze three documentation hashes.
Changing those files would otherwise make the launch fail closed despite no
runtime change. A same-run, documentation-only amendment was therefore appended
at
`artifacts/manifests/iteration_262_step15a_b0compute_b1hybrid_a32_g_fullstream_memory_envelope_world16_gate_preregistration_amendment_001.json`
(SHA-256
`8e1955b0ce8a312b104e1d877698d8aaca55b5a24dc5d64470acc336f213f878`).
All non-document frozen source hashes still match the original preregistration;
GPU work remains Not Run.

### Manifest integrity quarantine

The historical Iteration-254 profile failure-evidence file contains literal
`\\n` bytes after its closing JSON object. It is preserved byte-for-byte because
later immutable preregistrations cite SHA-256
`dd03695c22079e58b758035ef630f80dab3c470f55987d986d0d90bd3a3dfb92`.
A valid, non-iteration correction sidecar now carries the structured record and
quarantine policy at
`artifacts/manifests/artifact_integrity_correction_iteration254_profile_failure_001.json`
(SHA-256
`59262d54dd421ffebb70791b2e403fa090300c0a5fb1f6b879b1b3a8a780c7e5`).
The scientific status remains profile-capture failure with no performance claim.

### Maintenance invariant

Future retries stay under the same task while the scientific objective and
claim boundary are unchanged. A new task is created only when that objective or
boundary changes. README files carry only the current task pointer; immutable
manifests retain run-level detail.

---

## Task-status consolidation correction (2026-08-05)

### Scope

Documentation-only correction to the immediately preceding maintenance entry.
No GPU work, runtime change, historical-artifact rewrite, or experiment
renumbering was performed.

### Reason for correction

The first task-level pass inspected the canonical checkout but did not include
two archived side branches and the isolated ACK-free workspaces. It therefore
misclassified Tasks 5-7 and incorrectly stated that no numbered work used
210-220. That statement is superseded.

### Correct task mapping

5. **Critic-only role pool and schedule** combines the resident-pool bring-up,
   W16 matched/wave work, corrected-trace C/D/E recapture, 4G+12C and 8G+8C
   placement studies, queue depth, and ACK removal. These are stages of one
   steps-11-14 producer/consumer scheduling question, not separate tasks.
6. **Official-baseline reinforcement and transfer diagnosis** combines the
   baseline Nsight work, official rank-control trace repair, and the
   corrected-trace P0/P1/P2 pinned/prefetch study. Trace repair receives zero
   speedup credit. P2/P0 is 1.552624x on single GPU and a bounded 1.086336x on
   world8; the latter is below the 10% positive gate.
7. **No-GAN Full5 integration and Route A/B matrix** includes Route A/B x
   Critic-only/Generator-update cells, Step15-A, the future Step15-B handoff,
   and the continuous five-step comparison. The active Iteration-262 capacity
   gate is only the open A-G subproblem, not the entire task.

### Parallel-lineage reconciliation

- `archive/prefetch-baseline-node1-20260801` (`ec77900`) preserves the line
  labeled 210-212.
- `archive/pipeline-cde-node0-20260801` (`288d71a`) preserves legacy
  `iteration_210c` artifacts later reconciled in that branch log as global
  213-220.
- Isolated 4G+12C and 8G+8C ACK-free workspaces also reused local labels
  211-219. They remain source-qualified run IDs and are not forced into a
  single fictitious global order.
- Canonical Step15 work starts at 221. Future citations of 210-220 must include
  the branch/workspace, commit, and immutable run ID.

### Evidence boundary

The old W16 1.6015x matched result predates the official rank-control repair and
is retained only as historical diagnostic evidence. The corrected 4G+4S C/D/E
pair is a one-pair descriptive 1.343607x result with disclosed numerical drift.
The ACK-free 4G+12C and 8G+8C four-update runs pass their own contracts, but
their batch sizes differ and they do not select a final allocation. Continuous
Full5, Step15-B transition, repeated matched Route-A/Route-B timing, and
automatic plan selection remain Not Run.

### Authoritative status

`docs/36_task_status_registry.md` now carries the corrected task mapping,
source-qualified evidence anchors, Route-A/Route-B matrix, and current open
gate. The earlier maintenance entry remains intact as append-only history and
is superseded only where this correction says so.

---

## Pre-cast backbone construction maintenance (2026-08-06)

### Scope

This is unnumbered initialization-path maintenance, not a new Auto-Research
iteration and not a training-throughput result. It removes repeated backbone
construction work while retaining the existing mixed-precision parameter
policy. Historical launchers and immutable preregistrations remain unchanged.

### Previous behavior

`HeliosTransformer3DModel.from_pretrained` first instantiated a complete FP32
14B model, loaded all six FP32 safetensors shards into a second full state dict,
copied the state dict into the model, and only then converted most parameters to
BF16. Each rank therefore repeated an unnecessary FP32 initialization, a
53.3179-GiB checkpoint read, and an FP32-to-BF16 conversion. Observed
construction peaks were approximately 109--136 GiB RSS per process depending
on how the raw decimal measurements were reported and which backbone was being
constructed.

### Maintained construction method

The durable construction path is now:

1. Convert each source checkpoint once, one shard at a time.
2. Store ordinary floating-point tensors as BF16 while retaining
   `time_embedder`, `scale_shift_table`, `norm1`, `norm2`, `norm3`, and
   `history_key_scale` in FP32.
3. Instantiate the Helios module under `accelerate.init_empty_weights()`.
4. Validate the artifact schema, source identity, config/index hashes, shard
   inventory, all state-dict keys, shapes, and dtypes.
5. Bind the validated CPU safetensors storages with
   `load_state_dict(assign=True)`; do not silently fall back to the legacy
   loader.

The implementation is in `runtime/precast_checkpoint.py`, the offline command
is `scripts/distillschedule/convert_helios_checkpoint_dtype.py`, and the
explicit model entry is `precast_direct_load=True`. Training selects the two
artifacts through `DISTILLSCHEDULE_PRECAST_GENERATOR_PATH` and
`DISTILLSCHEDULE_PRECAST_SCORE_PATH`.

The reusable artifacts are:

- Generator:
  `/share/project/shuchenweng/DistillSchedule_assets/precast/helios-distilled-transformer-ode-mixed-bf16-v1`
- Score:
  `/share/project/shuchenweng/DistillSchedule_assets/precast/helios-score-transformer-mixed-bf16-v1`

Generator and Score have identical layouts and configs but different parameter
values, so they must remain separate. Runtime source-identity validation rejects
cross-wiring them.

### Verification and measurements

Both source checkpoints contain 1,101 FP32 tensors and 57,249,710,336 tensor
bytes. Each converted artifact contains 976 BF16 tensors, 125 retained-FP32
tensors, and 28,683,223,168 tensor bytes. Every tensor in each artifact passed a
chunked bitwise comparison against applying the declared dtype policy to its
own FP32 source. The Generator conversion plus hashing and source-equivalence
verification took 459.346 s once; Score took 451.241 s once.

The validated direct loader took 2.83--2.92 s for Generator and 2.90 s for
Score in isolated CPU construction probes. Max RSS was approximately 0.78 GiB;
all 1,101 model state entries were materialized and no parameter or buffer
remained on meta. Existing norm post-processing added approximately 0.003 s.
Both shared artifacts passed validation from Helios node 0 and node 1. The
formal project test directory passed `296 passed, 22 warnings`.

These timings cover CPU model construction and state binding. The first
CUDA/DeepSpeed placement must still fault and transfer approximately 26.7 GiB
per full logical backbone, so 2.9 s is not a claim for complete distributed
prepare or end-to-end training startup.

### Construction-wave policy

For checkpoint-resumed runs using the validated pre-cast loader,
`DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE` must default to `0` (disabled). A wave
size of 8 on an eight-rank node is one all-rank wave: it does not reduce
concurrency and only adds synchronization barriers. Smaller waves were a host
memory admission mechanism for the legacy FP32 construction path and are no
longer justified by the measured 0.78-GiB direct-load RSS.

The resume-capable `launch_m0_single_a100.sh`, `launch_m0_multinode_dp.sh`,
`launch_m1_profile.sh`, `launch_m1_node_profile.sh`, and
`launch_m1_multinode_profile.sh` entry points now select the pre-cast Generator
and Score automatically when `DISTILL_M0_RESUME_FROM_CHECKPOINT` is set. Their
checkpoint-resume construction-wave default is 0; the former wave-4 default in
`launch_m1_node_profile.sh` was removed. Legacy construction waves remain
available only for explicit legacy-loader reproduction or diagnostics.

### Semantic boundary

Checkpoint-resumed experiments are the default use of this method because the
subsequent checkpoint restore replaces trainable adapters, optimizer state, and
saved RNG state. A full old-loader versus pre-cast distributed one-step A/B gate
has not yet been run, so end-to-end trajectory equivalence is not claimed by
this maintenance entry.

Fresh initialization is different: meta construction and ordinary FP32
construction advance CPU RNG differently before LoRA creation. The runtime
therefore fails closed for a pre-cast fresh start unless
`DISTILLSCHEDULE_ALLOW_PRECAST_FRESH_INIT=1` is explicitly set. Existing
Step15 synthetic/frozen launchers remain unchanged until their RNG contract and
immutable preregistration are deliberately superseded.

---

## Pre-cast checkpoint-resume world16 equivalence follow-up (2026-08-06)

### Scope and matched contract

This is the result follow-up to the unnumbered pre-cast construction
maintenance above. It is not a new Auto-Research iteration. The treatment ran
one logical step only, from global step 10 to 11, on two Helios nodes with eight
A100 GPUs per node (`world_size=16`). The existing clean old-loader control was
reused rather than repeated:

- Control:
  `iteration126_m0_b4fast_zero1_shared_cw4_x16_resume10_trace_authority`
- Treatment:
  `precast_resume_equivalence_world16_treatment_20260806_001`
- Shared checkpoint:
  `artifacts/runs/iteration124_m0_b4fast_zero1_shared_cw4_x16_checkpoint10/checkpoint-10`
- Shared rank-16 operation trace bundle SHA-256:
  `bd97ddc9cf0e51e791a50ce13586126599d3667fd4684505c142ed177096f12f`

The treatment used the exact control source commit
`360aa92f63c66641027276ede6ac4c8ca43fd387` plus only the pre-cast loader
integration, the same rank-to-GPU UUID assignment, ZeRO-1, shared immutable
frozen blocks, swap wave 1, and construction wave 0. Both node launchers exited
with return code 0, all 16 ranks reached
`training_complete_skip_final_save`, and neither node added an OOM or OOM-kill
event.

### Strict and numerical results

The preregistered bit-exact gate is preserved as **FAIL**. Its first difference
is rank 0's post-generator-update checksum norm:
`97.33728572065773` for the control versus `97.33728572048722` for the
treatment, a relative difference of approximately `1.75e-12`. Therefore this
run does not establish bitwise reproducibility and must not be reported as an
exact semantic-trace digest match.

The strict failure is limited to negligible floating-point differences and
runtime activation-offload duplicate accounting. A separate, explicitly
post-hoc numerical diagnostic passed all floating leaves with
`rtol=1e-3, atol=1e-6`:

- Generator loss is exact on all 16 ranks.
- Maximum critic-loss difference is `7.796287536621094e-05` absolute and
  `5.1603947749345895e-05` relative.
- Maximum post-optimizer parameter-norm relative difference is
  `1.3389362329805085e-10`.
- Maximum parameter-sum difference divided by parameter count is
  `8.597853630697048e-13` per element.
- Maximum EMA-norm relative difference is `1.258777534595803e-11`.

Activation offload retained identical global `pack_count=28512` and
`unpack_count=6534`. On every rank, `cpu_copy_count + duplicate_pack_count` is
also exact; the global total is 2,752 for both runs. The control classified
2,738 packs as real copies and 14 as duplicates, while the treatment classified
2,736 and 16. This is runtime duplicate-detection accounting variation, not a
change in the mathematical step.

### Decision

For checkpoint-resumed BF16 experiments, the two paths are accepted as
numerically equivalent but explicitly not bit-exact. The validated mixed-BF16
pre-cast direct loader is the project default, and
`DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE=0` remains the default. Construction
waves are retained only for explicit legacy-loader reproduction or diagnostics.

This decision does not extend to fresh initialization, which continues to fail
closed without `DISTILLSCHEDULE_ALLOW_PRECAST_FRESH_INIT=1`. It also makes no
training-throughput claim.

### Evidence

- Preregistration:
  `artifacts/manifests/precast_resume_equivalence_world16_preregistration_001.json`
  (SHA-256
  `db94b2ff51b80337ab433dba8378acbac5532bfff9ecc5de3b73c55b5d5f78e2`)
- Strict result:
  `artifacts/manifests/precast_resume_equivalence_world16_result_001.json`
  (SHA-256
  `f405e88edf148dff6cceb19d85989d5455bcefd53c0219c7f29f5b8039eccc5d`)
- Numerical diagnostic:
  `artifacts/manifests/precast_resume_equivalence_world16_numeric_diagnostic_001.json`
  (SHA-256
  `07e4cf18fb35c3f2d0fc5621cb02cb8ab3e6dec2229fb37fcf79a8e950c0a1c0`)
- Acceptance decision:
  `artifacts/manifests/precast_resume_equivalence_world16_decision_001.json`
  (SHA-256
  `eab1e03bd1e3062231483c203b3a74e785741e082cd0e03b3e18d7e0d1b98f5c`)
- Strict comparator:
  `scripts/distillschedule/compare_precast_world16_resume_ab.py`
- Numerical analyzer:
  `scripts/distillschedule/analyze_precast_world16_numeric.py`
- Archived treatment traces:
  `artifacts/traces/precast_resume_equivalence_world16_treatment_20260806_001/`

---

## Maintained pre-cast rollout and fresh-init RNG contract (2026-08-06)

### Scope and project decision

This is unnumbered loader-policy maintenance, not an Auto-Research iteration
and not a GPU result. The Iteration-262 exact-B0 all-block preregistration is
preserved as immutable `Not Run` history, but it is retired as a launch binding.
No Iteration-262 or successor GPU process was started. A successor for the same
Task-7 capacity question must be newly preregistered against the maintained
initialization policy.

All maintained DistillSchedule M0/M1/Step15 entrypoints now validate and select
the reusable mixed-BF16 pre-cast Generator and Score artifacts and fix
`DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE=0`. Construction waves remain relevant
only for explicit legacy-loader reconstruction or diagnostics. The upstream
`b0` branch of `launch_m0_single_a100.sh` is the sole preserved legacy-loader
exception and identifies itself as such in its launch manifest.

### Fresh-init contract

`runtime/fresh_init_rng.py` defines
`distillschedule.precast_fresh_init_rng.v1`. For an explicitly authorized
pre-cast fresh start, SHA-256-derived component seeds isolate Generator and
Score LoRA initialization from backbone-construction RNG consumption. The
context preserves Python, NumPy, and Torch CPU RNG state; after both adapters
are configured, `train_helios.py` reapplies the configured base training seed
and records the contract, base seed, and component seeds in the phase trace.

Pre-cast fresh initialization now requires both
`DISTILLSCHEDULE_ALLOW_PRECAST_FRESH_INIT=1` and an explicit training seed.
Pre-cast resume paths do not use the fresh-init adapter contract and keep the
allow flag unset; checkpoint restoration continues to replace adapters,
optimizer/scheduler/EMA state, and saved training RNG state. A configured but
missing pre-cast resume checkpoint fails closed instead of silently becoming a
fresh start.

This is the reproducibility contract for future fresh starts. It does not claim
bitwise or trajectory equality with historical FP32-construction fresh starts.

### Maintained launcher coverage

The policy is wired into:

- `scripts/distillschedule/launch_m0_single_a100.sh`
- `scripts/distillschedule/launch_m0_multinode_dp.sh`
- `scripts/distillschedule/launch_m0_node_dp.sh`
- `scripts/distillschedule/launch_m1_profile.sh`
- `scripts/distillschedule/launch_m1_multinode_profile.sh`
- `scripts/distillschedule/launch_m1_node_profile.sh`
- `scripts/distillschedule/launch_step15a_hybrid_residency_world16.sh`

Launch manifests disclose both artifact paths, whether the run is a pre-cast
fresh start, the RNG contract name, and construction wave 0.

### Validation

With CUDA hidden, the focused loader/RNG/baseline suite passed `45/45`, and the
current project test directory passed `266 passed, 33 skipped, 20 warnings`.
All seven maintained launchers passed `bash -n`; the changed Python files passed
`py_compile`; the scoped tracked diff passed `git diff --check`.

A repository-root pytest discovery attempt was not a valid test run because it
recursively collected multiple immutable `artifacts/source_snapshots/*/tests`
trees with duplicate module names. Restricting collection to the authoritative
`tests/` directory removed that pre-existing collection collision. No GPU test,
training step, or profiler capture was run during this maintenance.

---

## Task-7 exact-B0 all-Generator-block streaming successor (2026-08-06)

### Scope and launch contract

This is the unnumbered maintained-policy successor to the retired
Iteration-262 launch binding. It belongs to the existing Task-7 Route-A
Step15-A capacity question; it is not a new research task or iteration.
Iteration 262 remains immutable `Not Run` history.

One logical Generator-update step (`15 -> 16`) ran on two Helios nodes with
eight A100-40GB GPUs per node (`world_size=16`, global batch 16). The run kept
the exact `b0_vectorized_split_modulation` path and disabled FFN chunking,
LoRA-output chunking, direct LoRA-B gradient accumulation, and every other
memory-bounded compute tiling path. It streamed all 40 immutable Generator
blocks with window 1, did not block-stream Score, offloaded selected Generator
activations at the 32-MiB threshold, used shared immutable backing for both
Generator and Score, kept ZeRO-1 optimizer state on CPU, used swap wave 4, and
offloaded no trainable parameters.

Initialization used the validated mixed-BF16 pre-cast Generator and Score
artifacts, construction wave 0, and
`distillschedule.precast_fresh_init_rng.v1`. Both node launchers used the same
preregistered run ID and nonce and exited with return code 0. There was one
launch attempt and no silent retry.

### Capacity and semantic result

The aggregate gate is **PASS**. All 16 rank-result files and all 16 phase traces
are present. The validator confirmed exact B0 on both physical models, zero
compute-tiling/chunking paths, Generator streaming ownership for blocks 0--39,
no Score block streamer, balanced per-block Generator materialize/release
events, selected-Generator activation offload, Generator/Score shared immutable
backing, fresh-init component-seed evidence on every rank, and completion of the
step-15 Generator update. Neither node added an OOM or OOM-kill event.

Whole-run cgroup peaks were `478.822 GiB` on node 0 and `520.207 GiB` on node
1. Within the Generator-update gate window, the peaks were `438.480 GiB` and
`469.900 GiB`. The resource monitor observed per-GPU peaks of
`37,936--38,104 MiB` (about `37.0--37.2 GiB`) during that instrumented window.
This establishes that exact
B0 fits when all Generator blocks stream and Score remains phase-resident.

### Diagnostic timing

Across 16 ranks, `generator_update_gate_start` to
`generator_update_gate_complete` took `114.897013--114.962889 s`, with median
`114.921426 s` (about 1 minute 55 seconds). The median forward/loss interval was
`89.842378 s`; loss-end through backward-end was `21.469663 s`; backward-end
through optimizer-step was `2.510786 s`. Python main-entry to traced completion
took a median `334.744655 s`; launcher wall time including distributed startup,
construction, and cleanup was 400 s on node 0 and 397 s on node 1.

Detailed phase tracing and scalar CUDA allocator/driver snapshots around every
block materialization and release were enabled. These measurements are
diagnostic gate timing, not native latency, steady-state throughput, or an
Nsight result.

### Decision and next action

The exact-B0 capacity upper bound is now closed: G-all-40 streaming with
S-not-streaming passes. The earlier tail-1, tail-3, and tail-5 failures remain
valid lower-bound evidence at their first unstreamed recompute pressure region.
The next Task-7 action is to use this run's per-block memory trajectory to choose
and preregister a smaller Generator streaming set. Step15-B, the continuous
Full5 transition, Nsight/native timing, repeated timing, quality, convergence,
and automatic plan selection remain `Not Run`.

### Evidence

- Preflight:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_preflight_001.json`
  (SHA-256
  `d52e0972ffa6b38e1746cfb67c6cc98ba3515fa5023fdc58153577321a472758`).
- Preregistration:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_gate_preregistration_001.json`
  (SHA-256
  `087aa84f1b340dd4593849b18feeb1603eda0d6560445b75c9094254c96c649c`).
- Aggregate result:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_gate_result_001.json`
  (SHA-256
  `b6f0ec8bc303a8b484c9ba04d9b48caec19c2a5628c1533110008f41a647d3e5`).
- Timing and memory analysis:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_gate_timing_analysis_001.json`
  (SHA-256
  `759a396577b3a1de12131cc2589ecb47eb9df059f4b2e31101756842286fc897`).

---

## Task-7 exact-B0 swap-wave-8 matched gate (2026-08-06)

### Question and frozen comparison

Can the already-passing exact-B0, Generator-all-40-streaming and
Score-phase-resident policy increase whole-model swap concurrency from four to
all eight local ranks without exceeding the node memory limit, and does that
reduce the instrumented one-step Generator gate time?

The candidate kept world size 16, logical step 15, the rank-qualified operation
trace, mixed-BF16 pre-cast fresh initialization and RNG-v1 contract, exact B0
vectorized split modulation, Generator blocks 0--39 streaming with window 1,
Score with neither block nor activation offload, Generator A32, ZeRO-1 CPU
optimizer state, and per-block memory snapshots unchanged. The only intended
schedule change was `swap_wave_size=4 -> 8`. During the matched gate, the
maintained launcher kept four as its default and used a preregistered override;
the gate assertion and launch manifest bound to the resolved value. After
reviewing the passing result, the user explicitly promoted wave 8 to the
maintained default. The relevant focused CPU suite passed `50 passed, 1 skipped`
before launch.

### Capacity and correctness result

Both two-node launchers exited zero on their first attempt, all 16 rank results
and traces passed the corrected aggregate validator, and neither node added an
OOM or OOM-kill event. The worst node peak was `573.347 GiB` under the
`760-GiB` cgroup limit, leaving `186.653 GiB` of measured headroom. Wave 4 had
peaked at `478.822/520.207 GiB`; wave 8 peaked at
`506.272/573.347 GiB`, increases of `27.450/53.140 GiB`.

All 16 Generator losses are bitwise equal between the matched runs, and the
initial trainable-state hash is identical. The post-update raw BF16 hashes are
not bitwise equal. On rank 0, the checksum norm changed by only
`3.5503e-08` (`3.6473e-10` relative), while the cancellation-sensitive checksum
sum changed by `-0.0120421`. Thus capacity and finite-update correctness pass,
and the numerical observation is within the previously accepted tiny norm
drift, but literal cross-run bitwise post-update equality is not claimed.

### Diagnostic timing result

| Metric (16-rank median) | Wave 4 | Wave 8 | Change |
| --- | ---: | ---: | ---: |
| Generator gate | `114.921426 s` | `102.993347 s` | `-11.928079 s`, `1.1158x` |
| Forward/loss interval | `89.842378 s` | `78.313752 s` | `-11.528625 s` |
| Three model swaps, summed | `37.466997 s` | `31.845999 s` | `-5.620998 s` |
| Generator block H2D intervals | `42.607252 s` | `35.570478 s` | `-7.036775 s` |
| Loss-end through backward-end | `21.469663 s` | `20.964869 s` | `-0.504794 s` |

Materialization counts and bytes are identical: median 320 block uses and
`209.447327 GiB/rank`. The lower block-copy interval therefore reflects
run-level transfer/cache variation rather than fewer operations and must not be
credited solely to the wave change. The directly affected three-swap interval
did improve, while the heavy Generator-to-Score swap showed rank-dependent
bandwidth contention. This is one matched instrumented gate, not repeated
native or steady-state timing.

### Validator correction and decision

The first post-run validator invocation failed before inspecting rank evidence
because it hard-coded launch schema v4 and swap wave 4, while the parameterized
launcher emitted schema v5. That failure artifact is preserved. The validator
was corrected to accept explicit `--launch-schema-version` and
`--swap-wave-size`; validation then passed without rerunning the GPU workload or
changing its artifacts.

Wave 8 is admitted and is the stronger diagnostic performance candidate. The
subsequent explicit user decision makes wave 8 the maintained launcher and
validator default. Wave 4 remains available only through an explicit override
for historical reproduction. Do not cite the single `1.1158x` result as
steady-state throughput.

### Historical wave-4 evidence identity

- The maintained-policy exact-B0 all-40-streaming gate at `114.921426 s` is the
  direct wave-4 control for this wave-8 result. It is the only prior wave-4
  result that serves as the current exact-B0 performance baseline.
- Iteration 253 passed with full Generator and Score block streaming, but used
  the later-retired memory-bounded FFN/LoRA compute route. It is historical B1
  implementation evidence, not an exact-B0 baseline.
- Iteration 257 passed the B1-Hybrid tail-1 configuration and Iteration 258
  profiled it. Both used the memory-bounded B1 compute path and are residency or
  profiler predecessors only, not current exact-B0 timing baselines.
- Iterations 246--252 used wave 4 while establishing shared backing and block
  streaming, but their gates were failures until Iteration 253. They remain
  failure/debugging evidence.
- Iterations 259--261 used wave 4 for exact-B0 tail-1/tail-3/tail-5 attempts.
  All failed capacity and remain negative streaming-set bounds, not performance
  baselines. Iteration 262 also declared wave 4 but was never run.

### Evidence

- Preregistration:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_swapwave8_gate_preregistration_001.json`
  (SHA-256
  `12fd7a3580f9436f14882657929847dd42332d3bfba75e5bbe3aa0fa058ecf00`).
- Preserved schema-only validator failure:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_swapwave8_gate_result_001.json`
  (SHA-256
  `c9e26248f88a49680ea1371708b68470eaeac5070e82b5e5ba2d0739f795ad2f`).
- Corrected aggregate PASS:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_swapwave8_gate_result_corrected_001.json`
  (SHA-256
  `c5b23d6f5434275c2ff83efd1c16009e1c1d2297cc7182a110d974b208edb47e`).
- Matched comparison:
  `artifacts/manifests/task7_step15a_exact_b0_g_fullstream_precast_world16_swapwave8_comparison_001.json`
  (SHA-256
  `151d61e297366e030dcc7d0f9d576316d2bd40a3bb346ff61e1c19007670167a`).

---

## ZeRO-3 pipeline pre-cast construction and compute/movement profile (2026-08-06)

### Scope

This unnumbered maintenance/profile run repeated one Step15-A update with the
historical resident `8G + 4S + 4T` pipeline: Generator used ZeRO-3, Score and
Teacher were read-only resident roles, global batch was 16, Generator used two
microbatches with one accumulated optimizer update, and the correctness
rendezvous remained shared-file based. It did not adopt the exact-B0
all-Generator-block-streaming workload. The maintained mixed-BF16 pre-cast
artifacts were loaded directly with construction concurrency 8. Swap wave 8 was
recorded for configuration alignment, but this resident-role pipeline performs
no whole-model swap scheduling, so that setting has no runtime effect here.

Both node launchers and Nsight stats exited zero. All 16 source/gradient digest
chains, gradient lineage checks, finite nonzero input gradients, both
accumulation boundaries, and the single Generator update passed. Wave-0 outputs
match the Iteration-237 correctness gate. Neither node added an OOM or OOM-kill
event. The Generator checksum norm delta was
`9.284919940455438e-05`, consistent with the previously accepted tiny update
drift.

### Construction result

The candidate and Iteration-238 legacy profiles report these rank medians:

| Construction segment | Legacy | Direct pre-cast | Legacy / candidate |
| --- | ---: | ---: | ---: |
| G model load | `38.6607 s` | `17.8802 s` | `2.162x` |
| G DeepSpeed engine init | `34.9009 s` | `44.0051 s` | `0.793x` |
| S load plus device transfer | `78.0156 s` | `38.6382 s` | `2.019x` |
| T load plus device transfer | `55.6286 s` | `38.4669 s` | `1.446x` |

Direct pre-cast therefore removes most model-construction conversion time. For
G, the separately measured load plus engine-init medians improve from
`73.5616 s` to `61.8854 s` (`11.6762 s`, about `1.189x`) because concurrent
ZeRO-3 engine initialization was `9.1042 s` slower in this capture. Do not cite
the `2.162x` model-load speedup as the speedup of the complete launch.

### Generator compute and movement

The historical unprofiled Iteration-237 G pipeline remains the native-latency
control at `48.5516 s` (`3.0345 s/logical sample`). Under Nsight, the G leaf
span was `55.7510 s`, including `7.1994 s` profiler overhead. Across eight G
GPUs, median per-device non-NCCL CUDA-kernel busy time was `33.5070 s`; median
NCCL-kernel busy time was `22.8102 s`; median memcpy-engine busy time was
`0.5114 s`. Their intervals overlap and must not be summed. The median union of
all three was `47.2467 s` over the `57.7566 s` outer NVTX range.

| G phase, median/rank | Wall | non-NCCL compute | NCCL | memcpy engine | no kernel/copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave 0 rollout | `12.8208 s` | `6.1396 s` | `8.4350 s` | `0.0265 s` | `2.0087 s` |
| wave 0 backward | `17.9474 s` | `11.4711 s` | `4.4640 s` | `0.1783 s` | `1.8012 s` |
| wave 1 rollout | `12.2011 s` | `8.6159 s` | `8.5569 s` | `0.0217 s` | `0.5679 s` |
| wave 1 backward | `9.5356 s` | `7.2596 s` | `1.0917 s` | `0.1858 s` | `0.9792 s` |
| optimizer boundary | `2.1097 s` | `0.0028 s` | `0.0131 s` | `0.0776 s` | `1.9905 s` |

The optimizer's uncovered interval includes CPU-offloaded optimizer work; it is
not GPU idle that can automatically be eliminated. Likewise, other uncovered
intervals can contain CPU launch work, Python/filesystem coordination, profiler
overhead, and real waits. The two G waits for returned Score gradients were only
`0.207 ms` and `0.423 ms` median, so Score/Teacher did not create a material G
critical-path wait in this two-microbatch schedule.

Per G GPU, Nsight recorded `67.8437 GiB` D2D, `1.7848 GiB` D2H, and
`1.7531 GiB` H2D memcpy traffic. Across all eight G GPUs this is
`542.750 GiB`, `14.278 GiB`, and `14.025 GiB`, respectively. NCCL is reported
as communication-kernel busy time above, not as memcpy-engine traffic.

### Score/Teacher overlap

The S4T4 leaf span was `35.3406 s` and the outer range was `36.9484 s` under
Nsight. Each Student GPU had about `10.82--10.87 s` non-NCCL compute,
`2.77--2.82 s` NCCL, and `0.11 s` copy-engine activity. Each Teacher GPU had
about `15.92--15.98 s` non-NCCL compute, near-zero to `0.17 s` NCCL, and
`0.22 s` copy-engine activity. Student forward compute was `2.6415 s` and
`2.6387 s` median for waves 0 and 1; Teacher conditional/unconditional compute
was `1.9432/1.9296 s` and `1.9469/1.9390 s`. These four S/T lanes overlap one
another and overlap G, so summing them does not produce pipeline wall time.

### Analysis correction and evidence

The first offline analyzer result incorrectly used the lower 24-bit thread ID
from Nsight `globalPid/globalTid` and consequently emitted zero per-phase kernel
activity. That JSON is preserved as a failed postprocessing artifact. The
analyzer now joins NVTX and CUDA records with the upper global-process component
(`value >> 24`); no GPU workload was rerun for this correction. Schema-v2 also
records per-phase kernel/copy unions and explicitly labels uncovered intervals
as not necessarily removable bubbles.

- Final result:
  `artifacts/manifests/step15a_zero3_pipeline_precast_wave8_nsys_result_001.json`
  (SHA-256
  `854388579d6ebae253e1e239e74db313a34c4a147c11cb057c899cae5412346e`).
- Analyzer:
  `artifacts/manifests/step15a_zero3_pipeline_precast_wave8_analyze.py`
  (SHA-256
  `2b19c774b628b17c480483a42afcb792f3d352f253083aa5a152e64579c7c169`).
- Preserved bad postprocessing result:
  `artifacts/manifests/step15a_zero3_pipeline_precast_wave8_nsys_result_phase_pid_bug_001.json`
  (SHA-256
  `d7a879942a88434969231466c0331f203e35741b522cbbf18a551b185ce145ad`).
- Timeline figure:
  `artifacts/figures/step15a_zero3_pipeline_precast_wave8_nsys_timeline_001.png`
  (SHA-256
  `3dc426134f8e8d885b4542d203953ca363446d5151790af3a9d0e6d5ae4af8f6`).

---

---

## Route-A RA16-F5-PR12 bounded capacity experiment (2026-08-09)

### Milestone / Gate

World16 official-384x640 continuous Full5 functional admission for the
user-authorized g12 Generator partial-streaming policy. Formal timing was out of
scope.

### Question

Does streaming exactly Generator blocks
`[0,24,1,5,8,25,10,32,36,15,16,26]` with window 1 provide enough capacity for
the unchanged Route-A steps 10--14 graph while Score remains unstreamed?

### Hypothesis

The additional four blocks relative to PR8 free another 2.618 GiB/rank and are
sufficient for the Step10 Generator checkpoint backward and the subsequent
Full5 phases.

### Importance

PR8 proved that four more streamed blocks changed real allocation by exactly
their capacity but still failed Generator recomputation. PR12 is the bounded
user-selected test of whether another equal increment closes the capacity gap.

### Evidence Needed

Exact world16 topology, checkpoint-10 and rank-qualified trace; official
384x640; ZeRO-1 CPU optimizer; MB1/GAS1/global batch16; G2/C10/EMA10 to
G3/C15/EMA15; zero lifecycle residue; no silent policy change.

### Refutation Condition

Any direct CUDA/host OOM, semantic/version mismatch, missing update or failure
to complete steps 10--14 under the exact g12 plan.

### Allowed Changes

Only the preregistered g8 to g12 Generator streamed set. One non-semantic
instrumentation repair was allowed after a newly reached path exposed a trace
field-name collision. No g16, ZeRO-2, activation threshold, trace or shape
change was authorized.

### Commands

The two-host controller ran immutable IDs
`ra16_f5_pr12_20260809_001` and
`ra16_f5_pr12_20260809_002` with plan SHA-256
`7a22646c2b1741351a680487a337cdbd9729c936d60ee04134c35fee11026dcb`.
The exact recovery used runtime commit
`0d66b4237ec68e8e4bc9991e543cf40a833f4e23`.

### Measured Results

Attempt 001 passed topology, construction, checkpoint resume and the complete
Step10 Generator backward/optimizer update on all ranks with no CUDA or cgroup
OOM. It then failed at Critic entry because the trace event name and metadata
both used the Python argument `phase`. The production fix only renamed the
metadata key to `resident_phase`; 35 focused runtime tests passed on each node.

Attempt 002 again completed the Generator update, the no-grad Critic rollout
and Critic forward/loss. At `semantic_critic_pre_backward`, median allocation
was 33.738 GiB and median driver-free capacity was 2.725 GiB. During Critic
checkpoint recomputation all 16 ranks directly CUDA-OOMed: 3 requested 108 MiB,
7 requested 216 MiB and 6 requested 292 MiB; PyTorch allocation was
38.04--38.25 GiB and driver-free capacity was 96.38--272.38 MiB. Sampled node
cgroup peaks were 714.167 and 713.410 GiB, with no cgroup OOM/oom-kill
increment.

### Correctness

Inputs, plan, trace, checkpoint and distributed contract remained frozen.
Generator Step10 backward/update passed. No Critic update, EMA update,
steps 11--14 or checkpoint-15 completed, so Full5 correctness failed closed.

### Performance

Not Run. Neither failed capacity run is a timing result.

### Interpretation

The Generator sub-hypothesis is verified: g12 crosses the PR8 Generator
backward boundary. The complete Full5 hypothesis is refuted by a different,
downstream bottleneck. Before Critic forward, the phase-resident Generator
blocks returned to zero and Generator was moved off GPU; the unstreamed Score
model was then resident. Therefore increasing Generator streaming to g16 would
not release the measured Critic-backward peak. Prior resident-Student evidence
from split-role/different-layout experiments is not admission evidence for this
replicated Route-A DP16 path.

### Files Changed

Runtime commit `0d66b42` fixes only the phase-trace field collision. Curated
PR12 contracts, results and raw-evidence pointers live in the DMD-System
repository at commit `18f703c`.

### Failure or Limitation

PR12 is a bounded Full5 negative. Score streaming and a changed Critic
activation policy remain Not Run and would be new mechanisms. Formal timing is
Not Run.

### Gate Status

PR12 policy budget closed. No retry and no automatic g16.

### Next Highest-Value Question

Under a separately frozen experiment, what is the minimum Score/Critic
lifecycle change that admits Critic backward without erasing the transfer-time
benefit sought by partial Generator streaming?

### Next Command

Not Run pending explicit authorization and preregistration.

## Iteration 263: RA16-F5 typed Generator LP-state phase offload

### Milestone / Gate

Route-A world16 official-384x640 continuous Full5 admission; one-variable
functional recovery after the bounded PR12 Critic-backward OOM.

### Question

Does the inactive Generator ZeRO-1 BF16 optimizer flat state explain the extra
continuous-Full5 Critic-phase floor, and is releasing it sufficient to admit
the otherwise unchanged resident Score/Critic backward?

### Hypothesis

`bit16_groups_flat` is a replicated 2.33-GiB/rank phase-owned state absent from
the passing split-role Critic executors. Offloading only this typed state after
the Critic no-grad Generator rollout should reduce Critic-entry allocation by
at least 2 GiB and allow Step10 Critic backward.

### Importance

This tests the user's observation that Critic backward itself has passed in
other schedules without incorrectly escalating Generator block streaming or
retesting ZeRO-2. It distinguishes physical executor state from model math.

### Evidence Needed

Exact world16 topology/checkpoint/trace/384x640 contract; typed storage
inventory; exact before/after allocation; successful G update; Critic
forward/backward/update; cgroup/CUDA OOM counters; no changed workload policy.

### Refutation Condition

The typed state is not approximately 2.33 GiB, cannot be released without
state/version drift, pre-backward allocation falls by less than 2 GiB, or the
same unchanged Critic backward still OOMs.

### Allowed Changes

Only a fail-closed ZeRO-1 LP-state controller and one Critic-boundary
offload/reload action. No Score streaming, g16, ZeRO-2, activation threshold,
shape, trace, loss, optimizer-order or EMA-contract change. One small CUDA gate
and one world16 functional attempt; timing Not Run.

### Commands

The central two-host controller launched immutable run
`ra16_f5_glp_offload_20260809_001`, plan SHA-256
`c3c9c29443b64c7e4f51026781173fd91973ebb129353e1d17db61c66d09dba5`,
DMD head `56ba8eb` and Runtime head `13f39e6`.

### Measured Results

All 16 ranks inventoried and offloaded exactly 2,502,885,376 bytes. Relative to
PR12, `semantic_critic_pre_backward` allocation fell from 36,226,001,920 to
33,723,116,544 bytes/rank: an exact 2,502,885,376-byte (2.330994-GiB)
reduction. Generator backward/update completed on all ranks.

All ranks nevertheless OOMed during Critic checkpoint recomputation on a
216-MiB request. Failure allocation was 37.84--38.01 GiB and driver-free space
96.38--216.38 MiB. The immediate worst-rank deficit was at least 119.62 MiB,
but remaining completion headroom is unknown. Sampled cgroup peaks were 701.622
and 712.753 GiB with no new cgroup OOM/oom-kill.

### Correctness

The small CUDA gate passed exact offload/reload SHA, parameter-view rebind,
subsequent optimizer step and balanced handles. The real run preserved the
input and version contract through the Generator update and Critic loss. No
Critic update, EMA update, steps 11--14 or checkpoint-15 completed, so Full5
correctness failed closed.

### Performance

Not Run. A failed capacity experiment is not a speedup denominator.

### Interpretation

The suspected hidden LP state and its phase release are verified exactly, but
the sufficiency hypothesis is refuted. Checkpoint recomputation consumed the
released capacity and failed later. Generator g16 is not a causal repair
because Generator frozen-block residency is already zero at Critic entry.

### Files Changed

Runtime `13f39e6` adds the typed controller and tests. DMD `56ba8eb` adds the
typed plan, contract, launcher identity and CUDA probe. Curated result/report
artifacts are recorded in the following evidence commit.

### Failure or Limitation

This is a bounded negative only for GLP offload as the sole additional
mechanism. It is not evidence that lifecycle management is useless, and it
does not transfer isolated resident-Student admission to replicated Full5.

### Gate Status

Closed. No unchanged retry, automatic g16, or formal timing.

### Next Highest-Value Question

Which separately typed Critic-phase state or minimum selective Score lifecycle
change supplies the remaining headroom while moving materially fewer than all
40 Score blocks?

### Next Command

Not Run pending a new preregistration and authorization.


## Iteration 264: RA16-F5 selective four-block Score lifecycle

### Milestone / Gate

Route-A world16 official-384x640 continuous Full5 functional admission after
the bounded Generator LP-state offload result.

### Question

Does streaming exactly Score frozen blocks 36--39 with window 1 provide the
remaining Critic checkpoint-backward headroom without returning to all-40 Score
streaming?

### Hypothesis

Four tail blocks contain 2.6 GiB of immutable BF16 Score storage. Keeping at
most one of those blocks resident during physical forward/recompute calls will
release enough phase capacity for Critic backward while preserving the exact
Full5 execution and version contract.

### Importance

The parent GLP-only run proved a 2.33-GiB inactive Generator state could be
released but still OOMed later in Critic recomputation. This experiment tests a
minimum bounded Score-side lifecycle action rather than escalating Generator
streaming or re-entering the closed ZeRO-2 path.

### Evidence Needed

Exact world16 topology/checkpoint/trace/384x640 contract; exactly four Score
streamed blocks; exact storage and pre-backward delta; complete G/C/EMA version
trajectory; balanced lifecycle handles; checkpoint-15; CUDA/cgroup OOM deltas.

### Refutation Condition

The four blocks are not the configured physical Score blocks, storage release
does not match their captured bytes, any rank OOMs, any version/trace/lifecycle
gate fails, or steps 10--14 do not complete.

### Allowed Changes

One component-specific Score streamer set `[36,37,38,39]`, window 1. Generator
`g12`, typed GLP offload, A32 activation offload, ZeRO-1 CPU optimizer, model
math, loss, trace, batch, update order and EMA remain frozen. One real attempt;
no block sweep and no formal timing.

### Commands

The two-host controller launched run `ra16_f5_glp_score4_20260809_001` with
plan SHA-256 `1087443e7b9379647e5e3b8e6e2f858b27ecc67f22662b2a001934ee3187c0f3`,
DMD training head `6a4d42f` and Runtime head `947b28e`. The immutable raw traces
were then re-summarized after a bounded evidence-validator cardinality fix.

### Measured Results

All 16 ranks captured exactly 2,811,154,432 bytes across Score blocks 36--39.
Critic pre-backward allocation fell from the parent 33,723,116,544 to
30,911,962,112 bytes/rank, an exact 2,811,154,432-byte (2.618-GiB) reduction.
Only one block, 702,788,608 bytes, was resident at a time. Each rank performed
52 Score block materializations totaling 36,545,007,616 bytes; measured Score
H2D wall time ranged 8.08--13.68 s.

All steps completed. GPU maximum allocated/reserved were 38,775,584,256 and
39,652,950,016 bytes. Sampled node cgroup peaks were 737,171,681,280 and
766,650,753,024 bytes, with zero new OOM or OOM-kill events.

The single instrumented Full5 window was 404.042 s: step spans were 161.525,
66.581, 62.646, 62.035 and 65.487 s. This is diagnostic only. Formal repeated
paired timing is Not Run.

### Correctness

PASS after correcting a non-semantic validator cardinality bug. Exact observed
versions were G2->G3, C10->C15 and EMA10->EMA15; one G, five C and five EMA
updates occurred. Both activation scopes ended with zero live CPU bytes, both
streamers ended with zero resident bytes, trace/topology gates passed, and
checkpoint-15 passed (97 files, 99,826,069,878 bytes).

The preserved v1 summarizer reported FAIL because it expected exactly one
activation final record while the runtime correctly installed and finalized
two scopes (`generator_selected` and `critic_train`). V2 compares installed and
final component multisets and still requires every live-byte value to be zero;
it returned PASS with zero evidence errors. No run was retried and no semantic
or numerical tolerance changed.

### Performance

Measured diagnostic only. The 404.042-s window is 41.33% below the historical
688.726-s all-40 diagnostic window, but a one-run instrumented comparison is
not a formal speedup claim.

### Interpretation

The preregistered sufficiency hypothesis is verified. Selective four-block
Score streaming plus typed Generator LP-state offload turns the parent OOM into
a complete Full5 run and moves materially less Score state than all-40
streaming. Four blocks are sufficient, not proven minimal.

### Files Changed

Runtime `947b28e` adds component-specific Score streaming. DMD `6a4d42f` adds
the plan/contract; `e272541` fixes evidence validation for multiple installed
activation scopes. Curated report and manifests are in the DMD evidence commit.

### Failure or Limitation

This is one diagnostic-instrumented acceptance run. It does not establish a
formal speedup, minimum block count, or equivalence with split-role Pipeline
physical state. Node 1 approached the cgroup limit but recorded no actual OOM.

### Gate Status

Functional PASS. Attempt budget exhausted; no sweep. Formal timing remains Not
Run.

### Next Highest-Value Question

Under an explicitly authorized timing gate, does the frozen Score4 plan retain
its diagnostic advantage in repeated paired trials against the strongest
same-contract all-40 Route-A baseline?

### Next Command

Not Run pending separate formal-timing authorization.

## Iteration 265: systems-only GAN bootstrap and world16 execution contract

### Milestone / Gate
GAN world16 sequential denominator / bootstrap gate.

### Question
Can the validated pre-GAN Helios state be converted into one immutable, explicitly systems-only GAN-active initial state shared by baseline and pipeline?

### Hypothesis
A deterministic GAN-head artifact plus fully materialized ZeRO-1 Adam state can enter logical steps 1000--1004 without claiming source-faithful checkpoint-1000 continuation.

### Importance
Separate bootstrap/optimizer initialization cost from later Full5 timing and prevent baseline/pipeline from independently initializing GAN state.

### Evidence Needed
One immutable checkpoint-1000-shaped package, fixed head/optimizer hashes, 16 optimizer shards, rank-qualified 1000--1004 trace, and fail-closed contract validation.

### Refutation Condition
Missing GAN state, lazy first-step optimizer materialization, trace rank mismatch, or the artifact being mislabeled as an official checkpoint-1000.

### Allowed Changes
Bootstrap/trace builders, manifests, tests, and launcher gates only; no GAN/DMD math or schedule change.

### Commands
Built `systems_only_gan_bootstrap_v1`, audited ZeRO partition widths, then built corrected `systems_only_gan_bootstrap_v2` and GAN-aware Score pre-cast v2. Ran focused and repository tests.

### Measured Results
Bootstrap v2 artifact hash is `c528dd62b72267570f42b61c36a0926da85e1e0176930efcf72649ef8da3776e`. GAN-aware Score pre-cast hash is `d31e8f364501c925e4b831e83d9ef3b23d2920aaf90852eb839e10f764d91709`. The rank-qualified trace bundle hash is `d31d4df15ccdf4790ee6239e0354f9d051cff2c9d2e55a83c771af4fa271c170`.

### Correctness
PASS for package structure, fixed logical window, deterministic shared GAN heads, materialized optimizer state, and 16 fixed-width ZeRO-1 shards. Rank 15 contains the required 27 zero-padding elements.

### Performance
Not Run. Bootstrap/construction is excluded from formal timing.

### Interpretation
A common systems-performance starting state exists. It remains explicitly non-official and non-convergence evidence.

### Files Changed
DMD-System bootstrap/trace gates through `b3a7084`; artifacts under `/share/project/shuchenweng/DistillSchedule_assets/gan/` and `/share/project/shuchenweng/DistillSchedule_assets/precast/`.

### Failure or Limitation
The package does not prove a source-faithful training trajectory from official GAN checkpoint-1000.

### Gate Status
PASS.

### Next Highest-Value Question
Can the conservative sequential world16 runtime execute the whole GAN-active Full5 window?

### Next Command
Launch one correctness-instrumented world16 sequential run from bootstrap v2.

## Iteration 266: first real GAN world16 optimizer boundary

### Milestone / Gate
Sequential GAN baseline / first full-rank functional attempt.

### Question
Does the bootstrap cross both Generator and Critic optimizer boundaries on all 16 ranks?

### Hypothesis
The initial package contains correctly shaped optimizer state for every ZeRO-1 partition.

### Importance
A malformed tail shard can survive construction and fail only after expensive real backward work.

### Evidence Needed
All-rank optimizer receipts, exact error signature, cgroup/GPU peaks, and OOM deltas.

### Refutation Condition
Any optimizer partition shape mismatch or capacity failure.

### Allowed Changes
One bounded functional run and bootstrap builder repair only.

### Commands
Ran `gan_seq_systems_functional_20260810_004` on 2x8 A100 at 384x640.

### Measured Results
All ranks completed the Generator optimizer at step1000. Rank 15 then failed during the first Critic Adam update: optimizer state length 139,667,127 versus gradient partition length 139,667,154, an exact 27-element deficit. No CUDA or cgroup OOM occurred.

### Correctness
FAIL at the first Critic optimizer boundary.

### Performance
Not Run; failed functional attempt cannot be a denominator.

### Interpretation
The mechanism was deterministic ZeRO tail padding, not a training-math or capacity issue.

### Files Changed
Failure manifest `gan_seq_systems_functional_20260810_004_result.json`; fixed-width builder and tests in DMD-System `b3a7084`.

### Failure or Limitation
No later step executed in this attempt.

### Gate Status
Blocked by bootstrap v1; recovered by bootstrap v2.

### Next Highest-Value Question
Does corrected bootstrap v2 reach the complete Critic commit/semantic observer path?

### Next Command
Launch run005 from bootstrap v2.

## Iteration 267: GAN low-VRAM Critic observer scope

### Milestone / Gate
Sequential GAN baseline / semantic instrumentation.

### Question
Can the existing gradient observer cover the official low-VRAM joint Critic backward without changing its two backward calls or single DeepSpeed commit?

### Hypothesis
The observer only needs to begin before `_critic_loss` and close after the official function returns.

### Importance
Instrumentation must not reject or alter the very GAN path it is intended to validate.

### Evidence Needed
All-rank progress through official Critic commit, exact observer failure, tests proving unchanged backward/commit ordering.

### Refutation Condition
Training math must be reordered or an additional optimizer step is required.

### Allowed Changes
Observer scope and tests only.

### Commands
Ran `gan_seq_systems_functional_20260810_005`, then moved the existing observer boundary and ran 314 runtime tests.

### Measured Results
All 16 ranks completed Generator step1000 and the internal joint Critic low-VRAM commit. The wrapper then raised `semantic Critic gradient observation does not support GAN low-VRAM mode`. Peaks were 596,337,405,952 and 586,284,298,240 bytes cgroup, 38,276 MiB GPU; OOM delta was zero.

### Correctness
Execution FAIL due only to observer placement. Fix tests PASS; no loss/backward/commit path changed.

### Performance
Not Run.

### Interpretation
The official joint Critic path was viable; the failure was a local semantic-instrumentation ordering bug.

### Files Changed
Runtime `08291cf188bcd0318876b895c997facddcd4f833`; DMD failure manifest `gan_seq_systems_functional_20260810_005_result.json`.

### Failure or Limitation
Run005 did not emit outer Critic/EMA completion receipts.

### Gate Status
Recovered; proceed with one real verification.

### Next Highest-Value Question
Can the corrected observer survive all five logical steps?

### Next Command
Launch run006.

## Iteration 268: post-EMA Generator residency gap

### Milestone / Gate
Sequential GAN baseline / multi-step low-VRAM state machine.

### Question
Does the low-VRAM role placement remain valid across consecutive GAN Critic-only steps above `ema_start_step`?

### Hypothesis
The existing swap state at the end of one Critic-only step is sufficient for the next Generator rollout.

### Importance
The five-step denominator requires persistent correctness across 1001--1004, not just one optimizer boundary.

### Evidence Needed
Per-step version/update traces, model-move events, exact device mismatch, and one bounded prior-failure search.

### Refutation Condition
The next rollout begins with non-streamed Generator state on CPU.

### Allowed Changes
Residency-state guard and tests only; no offload policy or GAN math change.

### Commands
Ran `gan_seq_systems_functional_20260810_006`, searched prior logs/signatures, inspected rank phase traces, and added a non-streamed compute-state residency predicate.

### Measured Results
Steps1000 and 1001 completed on every rank (G update1, C update2, EMA update1). Step1002 failed at Generator `patch_embedding` with CUDA BF16 input and CPU BF16 weight. Peaks were 600,195,186,688 and 601,306,611,712 bytes cgroup, 38,276 MiB GPU; OOM delta was zero.

### Correctness
FAIL at step1002 placement. The root cause is exact: step1001 correctly skips EMA, leaving Score on GPU; the next wave-swap Critic entry incorrectly assumed Generator was already resident. No matching prior failure mechanism was found.

### Performance
Not Run.

### Interpretation
Earlier pre-EMA workloads masked this state-machine edge because EMA restored Generator after every step.

### Files Changed
Runtime `a2b25e6200ec9e37ac45119fa9718b8b3b7312bc` and two focused tests; full runtime suite 316 PASS. DMD manifest `gan_seq_systems_functional_20260810_006_result.json`.

### Failure or Limitation
The fix restores only missing non-streamed compute state; streamed frozen blocks remain CPU-backed by contract.

### Gate Status
Recovered; one real revalidation authorized.

### Next Highest-Value Question
Does the guarded state machine complete 1000--1004 without new capacity or semantic failure?

### Next Command
Launch run007 from the same immutable bootstrap.

## Iteration 269: sequential GAN world16 Full5 functional PASS

### Milestone / Gate
Sequential GAN baseline / functional acceptance.

### Question
Can the conservative fixed-offload world16 baseline complete official GAN-active logical steps1000--1004 from the shared systems-only bootstrap?

### Hypothesis
With fixed bootstrap padding, observer scope, and phase-entry residency guard, all ranks complete exact G/C/EMA update counts without OOM.

### Importance
Pipeline work and formal paired timing require a real sequential denominator that has crossed the complete five-step state machine.

### Evidence Needed
Sixteen complete phase traces, exact update/version receipts, lifecycle-zero records, checkpoint-1005, resource/OOM evidence, and strict summarizer PASS.

### Refutation Condition
Any rank crash, missing update, version drift, lifecycle leak, checkpoint failure, or OOM.

### Allowed Changes
One functional run plus non-semantic evidence-parser repair against preserved raw artifacts.

### Commands
Ran `gan_seq_systems_functional_20260810_007`; re-summarized the immutable artifacts after adding support for the runtime's categorical `operation_trace` authority, nested EMA receipt, observed component final receipts, and scientific-notation resource samples. Ran 170 DMD tests (1 skipped) and 316 runtime tests.

### Measured Results
All 16 ranks completed steps1000--1004. Every rank recorded exactly one Generator update, five Critic updates, one EMA update, and final G201/C1005/EMA801. Checkpoint-1005 contains 97 files totaling 115,639,903,431 bytes. Sampled cgroup peaks were 596,042,338,304 and 607,768,354,816 bytes; no new OOM/OOM-kill occurred. The detailed functional window was 596.417563 s (diagnostic only).

### Correctness
PASS. Trace hash/rank authority, topology, update trajectory, final observed component receipts, lifecycle zero, and checkpoint inventory all pass. The initial summarizer FAIL was parser-only; no training process was rerun and no numerical tolerance or training math changed.

### Performance
Measured diagnostic only: 596.418 s Full5, 119.284 s/step, 0.134134 samples/s under detailed correctness instrumentation. Formal low-overhead denominator is Not Run.

### Interpretation
The PR4 sequential GAN baseline is now functionally executable on real 14B models at official shape. This is the prerequisite denominator, not yet a performance claim.

### Files Changed
Runtime `a2b25e6`; DMD-System `7e000a56672e897610b484d4b27080e49fde2bde`, strict result `gan_seq_systems_functional_20260810_007_result.json`, and `docs/status/gan_world16.md`.

### Failure or Limitation
Actual-consumption fairness and compute-inventory receipts are not yet integrated; formal unprofiled timing, pipeline, Nsight, and minimum-offload search remain Not Run.

### Gate Status
Sequential functional PASS. Fairness/compute receipt gate remains open before formal speedup.

### Next Highest-Value Question
Can the existing PR5 prototype be connected to real Helios execution while preserving two-wave batch16/GAS2 semantics and the same actual-consumption contract?

### Next Command
Audit PR5 against the now-passing PR4 execution substrate and implement only the missing real executor seams.

## Iteration 270: actual-consumption history contract closes fail-closed

### Milestone / Gate
Sequential GAN baseline / fairness and compute-inventory gate.

### Question
Does the passing run007 prove that every processed-history value used by the official Generator and Critic was consumed from the rank-qualified operation trace?

### Hypothesis
The immutable trace already contains both history values and their packed-layout index tensors, but the runtime replay site consumes only the four values; a narrow replay repair can consume all eight without changing GAN, DMD, optimizer, or offload policy.

### Importance
Baseline and pipeline execute random and preprocessing operations in different physical orders.  A trace manifest or common seed is not evidence that both implementations consumed the same workload.

### Evidence Needed
A fail-closed receipt builder joining trace hashes to append-only access rows, physical DiT/Score call reconciliation from block-streaming events, focused replay tests, and the complete runtime/DMD test suites.

### Refutation Condition
Run007 already consumed all eight history tensors; replaying the indices changes model math; or the repaired runtime fails existing tests.

### Allowed Changes
Actual-consumption receipt tooling, tests, append-only evidence, and the two existing Generator/Critic trace replay sites only.  No GAN/DMD loss, update, memory-policy, batch, or schedule change.

### Commands
Built `gan_execution_receipts.py` and its CLI/tests, applied it to preserved run007, added `replay_operation_trace_prepared_history`, and ran the focused and full suites on the live server environment.

### Measured Results
The new builder rejects run007 at the first missing value: `slot=0 key=step/1000/generator/indices_hidden_states`.  Run007 remains a real functional PASS but cannot be promoted to actual-consumption fairness PASS.  The immutable bundle's internal `bundle_hash` is `d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15`; this is distinct from any aggregate directory hash.  Focused runtime tests: 8 PASS.  Full runtime `tests/`: 317 PASS.  DMD-System suite with the receipt builder: 173 PASS, 1 skipped.

### Correctness
The repair replays four floating history tensors plus four integer layout/index tensors at the same official consumption sites.  Tests verify all eight actual access rows and preserve dtype.  Existing runtime and launcher tests pass.

### Performance
Not Run.  No timing claim is made from run007, and the trace-contract repair is a correctness prerequisite rather than a speedup.

### Interpretation
Run007 proved the full 14B control path, updates, versions, lifecycle cleanup, and checkpoint, but not complete operation-keyed workload consumption.  One fresh correctness run is required; old evidence is preserved rather than rewritten.

### Files Changed
Runtime: `helios/utils/utils_helios_post.py`, `tests/test_gan_trace_replay.py`.  DMD-System: `src/distillschedule/runtime/gan_execution_receipts.py`, `scripts/distillschedule/build_gan_execution_receipts.py`, tests, and the iteration preflight manifest.  Preserved negative receipt attempt: `gan_seq_systems_functional_20260810_007/receipts_actual_consumption_attempt_20260811/`.

### Failure or Limitation
An unconstrained repository-root pytest invocation also collected archived source snapshots and produced import-name collisions; the authoritative maintained suite is `pytest tests`, which passed.  A real world16 revalidation with the repaired runtime is still Not Run.

### Gate Status
Receipt tooling PASS; run007 actual-consumption gate FAIL as expected; fresh baseline correctness run authorized.

### Next Highest-Value Question
Does one fresh fixed-offload world16 run complete steps1000--1004 and emit PASS fairness, compute, physical-call, semantic, and exact timing-structure receipts?

### Next Command
Commit and synchronize the validated repair, then launch `gan_seq_systems_functional_20260811_008` from the unchanged bootstrap and trace.

## Iteration 271: sequential GAN actual-consumption and compute gate PASS

### Milestone / Gate
Sequential GAN baseline / actual-consumption fairness and compute inventory.

### Question
Does the repaired rank-qualified trace replay complete official GAN-active
steps1000--1004 and prove the random/history workload and physical model-call
inventory actually consumed by every rank?

### Hypothesis
The narrow prepared-history replay repair is sufficient; a fresh fixed-policy
world16 run will preserve G/C/EMA semantics and reconcile all required trace
lookups and block materializations.

### Importance
The persistent pipeline changes RNG call order and physical ownership.  A
shared seed or an immutable trace manifest cannot establish a fair denominator
without evidence of actual consumption and equal logical compute.

### Evidence Needed
Sixteen completed phase traces, exact update/version receipts, checkpoint-1005,
zero lifecycle leaks/OOM deltas, 80 fairness entries, 80 compute entries, and
independent block-event reconciliation.

### Refutation Condition
Any required tensor missing or consumed by the wrong physical rank; any
logical DiT/Score call mismatch; update/version/lifecycle drift; or capacity
failure.

### Allowed Changes
One fresh correctness run and at most one bounded non-math repair to the new
receipt builder.  No model, loss, optimizer, schedule, trace, batch, or memory
policy change.

### Commands
Launched `gan_seq_systems_functional_20260811_008` on 2x8 A100-40GB from the
unchanged systems-only bootstrap.  Ran the strict functional summarizer,
actual-consumption builder, compute cross-check, exact timing validator, four
focused receipt tests, and 91 DMD refactor tests.

### Measured Results
All 16 ranks completed official-shape steps1000--1004 with G update1, C update5,
EMA update1 and final G201/C1005/EMA801.  The fairness and compute receipts each
contain 80 entries; 2,432 actual lookup rows were joined to 16 rank-qualified
trace hashes.  Physical block evidence reports per rank: Generator 31 logical
forwards plus 3 checkpoint recomputes; Score 14 logical forwards plus 11
checkpoint recomputes.  All materialize/release counts balance.  Sampled cgroup
peaks were 600,024,436,736 and 609,916,502,016 bytes; allocator maxima were
36,687,222,784 allocated and 39,210,450,944 reserved bytes.  No new CUDA or
cgroup OOM/OOM-kill occurred.

### Correctness
PASS.  Fairness digest
`b7104c0729f44d28963dff90fca6d51b293798d9dc0ba718b1d3af7d6985de66`;
compute digest
`482dfd6cc0b27747adb0ef835cfea187dd6c7769d847e443853b46ab06d2af34`.

### Performance
The exact timing receipt is structurally PASS at 609.592914539 s global Full5,
but the run is correctness-instrumented and therefore diagnostic only.  It is
not eligible as a formal speedup denominator.

### Interpretation
The sequential baseline now has a complete functional, stochastic-workload,
and physical-compute evidence chain.  Two receipt-only bugs were repaired:
rank-qualified traces use local `sample/0` tensor keys, and step1000 performs
both the differentiable Generator rollout and the Critic-data rollout.  Neither
repair changes Helios execution or math.

### Files Changed
DMD-System receipt builder/tests at `199064d` and `0fe99e1`; this append-only
log and `iteration_271_gan_seq_actual_consumption_pass_20260811.json`.

### Failure or Limitation
Low-overhead formal baseline timing, real persistent 8G+8Score execution,
paired trials, Nsight, and minimum-offload search remain Not Run.

### Gate Status
Sequential functional + fairness + compute + exact timing-structure gate PASS.
Proceed to the PR5 real pipeline executor only; do not start formal pairing
until the pipeline passes the same gates.

### Next Highest-Value Question
Can the PR5 prototype execute a real persistent 8G+8Score Full5 with two-wave
batch16/GAS2 and exact single-commit G/C semantics?

### Next Command
Merge the passing PR4 execution substrate into a new PR5 execution branch,
then implement only the missing real executor and role-local checkpoint seams.


## Iteration 272: persistent 8G+8Score standalone gate PASS

### Milestone / Gate
GAN Pipeline / real-14B standalone correctness and exact-timing gate.

### Question
Can the persistent 8G+8Score executor complete official-shape GAN-active steps1000--1004 with two-wave batch16/GAS2 and the same consumed workload as Sequential?

### Hypothesis
The target-layout logical package, DP8 role groups, bidirectional step1000 gradient service, and two-wave Critic accumulation preserve one G commit, five C commits, one EMA commit, and the frozen fairness/compute contract.

### Importance
Formal scheduling comparison is invalid until the new topology proves the same logical work and update boundaries.

### Evidence Needed
Sixteen PASS rank results; G1/C5/EMA1; two waves per step; zero lifecycle leaks and new OOM; exact actual-consumption, compute, and timing receipts.

### Refutation Condition
Any deadlock/OOM, rank or version drift, per-wave optimizer commit, missing stochastic input/Score call, lifecycle leak, or timing-receipt failure.

### Allowed Changes
Only the previously audited real executor seams and fail-closed evidence tooling; no loss, trace, batch, update, or memory-policy change.

### Commands
Ran `gan_pipeline_formal_gate_20260811_003` on 2x8 A100-40GB with all 40 frozen blocks streamed, window1, `HOST_SHARED_IMMUTABLE`, no prefetch, and the systems-only GAN bootstrap.

### Measured Results
All 16 ranks PASS. Maximum allocator observations were 33,652,875,264 allocated and 36,148,609,024 reserved bytes. No new CUDA/cgroup OOM occurred. Exact Full5 was 1014.788817054 s.

### Correctness
PASS. Generator ranks each report G1/C0/EMA1; Score ranks each report G0/C5/EMA0. Normalized fairness digest `b7104c0729f44d28963dff90fca6d51b293798d9dc0ba718b1d3af7d6985de66` and compute digest `482dfd6cc0b27747adb0ef835cfea187dd6c7769d847e443853b46ab06d2af34` match Sequential.

### Performance
Measured standalone gate only: 1014.788817054 s. It is not used alone as the speedup numerator.

### Interpretation
The persistent topology is functionally valid under the frozen fixed policy; paired timing is authorized.

### Files Changed
DMD-System execution branch through `9b2c4cf`; artifact `artifacts/formal_gate/gan_pipeline_formal_gate_20260811_003/`.

### Failure or Limitation
This gate does not isolate node/time drift and does not measure minimum offload.

### Gate Status
PASS; proceed to preregistered paired formal timing.

### Next Highest-Value Question
Does persistent role overlap beat the same-policy Sequential denominator over alternating fresh runs?

### Next Command
Run three paired trials with order B-P, P-B, B-P.

## Iteration 273: fixed-policy paired formal timing is a negative result

### Milestone / Gate
GAN Baseline vs Pipeline / formal scheduling comparison.

### Question
At identical all40/window1/shared-host/no-prefetch policy and exact workload, is persistent 8G+8Score faster than Sequential world16?

### Hypothesis
Role overlap compensates for halving each role's DP width and processing two waves.

### Importance
This is the causal Level-A versus Level-B test of pipeline scheduling itself.

### Evidence Needed
Three alternating fresh pairs; every run fairness, compute, update, OOM, and exact unprofiled timing PASS; no profiled run in the table.

### Refutation Condition
Any gate failure or median paired speedup below 1.0x.

### Allowed Changes
No code or contract changes during the six formal runs.

### Commands
Ran `gan_systems_paired_formal_20260811_002`: pair0 B->P, pair1 P->B, pair2 B->P.

### Measured Results
Pair0 656.125119729 vs 1139.194008663 s (0.575955557x); pair1 596.234346715 vs 1043.215012830 s (0.571535436x); pair2 658.338258614 vs 1170.196847944 s (0.562587619x). Sequential median 656.125119729 s; Pipeline median 1139.194008663 s. Median paired speedup 0.571535436x; geometric mean 0.569999011x. Speedup CV 1.1948%; family CVs 5.5321% and 5.9241%.

### Correctness
PASS in all three pairs: actual stochastic/DiT parity, 4 Generator Score forwards/sample, 2 Critic Score forwards/sample, exact G1/C5/EMA1, and exact timing receipts.

### Performance
Negative. Pipeline is about 1.75x slower (+74.97% wall time) at the same policy. Mean throughput: Sequential 0.125609 samples/s; Pipeline 0.071586 samples/s.

### Interpretation
Persistence alone does not pay for DP16->DP8 plus two-wave work under synchronous all40 streaming. This is a valid negative scheduling result, not an OOM or fairness artifact.

### Files Changed
Immutable formal artifact `artifacts/formal/gan_systems_paired_formal_20260811_002/result.json`.

### Failure or Limitation
The causal phase breakdown requires diagnostic profiling; minimum-offload Level-C remains separate.

### Gate Status
Formal paired gate PASS; positive speedup hypothesis REFUTED.

### Next Highest-Value Question
Which compute/copy/NCCL/idle intervals explain the slowdown?

### Next Command
Run one exact-range Nsight capture per family, excluding both from formal speedup.

## Iteration 274: Sequential exact-range Nsight PASS

### Milestone / Gate
GAN profiling / Sequential diagnostic attribution.

### Question
Can the exact step1000-first-compute to step1004-C-commit interval produce valid two-node Nsight reports without entering formal timing?

### Hypothesis
The existing node-local profiler wrapper yields 16 balanced range receipts and two non-empty CUDA/NVTX reports.

### Importance
It provides a diagnostic reference for copy, synchronization, compute, and NCCL; it is not a new speed sample.

### Evidence Needed
Sixteen exact receipts, controller ranks 0/8, two non-empty reports, non-empty CUDA API/kernel/NVTX tables.

### Refutation Condition
Missing range/report/activity or any attempt to promote profiled time to formal speedup.

### Allowed Changes
Profiler wrapper and receipt validation only; workload unchanged.

### Commands
Ran `gan_baseline_nsys_20260811_001` and generated node-local Nsight statistics.

### Measured Results
PASS. Node0 report 1,304,199,360 bytes, SHA256 `47d5efd0601eea34643f85e019cf155e61f9eff20ea93646649a2b2652cfaa5a`; node1 report 1,320,460,096 bytes, SHA256 `13f357526229f1c3972fe7af0249818804b92e6f14f4f48b50680b2235d8bda3`. On node1, summed CUDA API time: `cudaMemcpyAsync` 2808.654 s/904,065 calls (51.3%); `cudaStreamSynchronize` 2016.051 s/637,526 calls (36.8%); `cudaLaunchKernel` 457.794 s (8.4%).

### Correctness
Exact-range receipt validation PASS; logical window 1000--1004. The overlapping process sums are not interpreted as wall-clock fractions.

### Performance
Diagnostic only; formal-speedup eligible = false.

### Interpretation
The Sequential path issues very frequent copies and synchronization under all40 streaming, but this profile alone does not explain the Pipeline delta.

### Files Changed
Curated summary `artifacts/manifests/gan_baseline_nsys_20260811_001_summary.txt`.

### Failure or Limitation
No pipeline report yet; profile overhead excludes this run from formal tables.

### Gate Status
Sequential Nsight PASS.

### Next Highest-Value Question
Can Pipeline produce the same exact-range diagnostic evidence?

### Next Command
Run one Pipeline exact-range Nsight capture.

## Iteration 275: Pipeline Nsight reproduces known teardown failure; GPU work blocked

### Milestone / Gate
GAN profiling / repeated-failure audit and minimum-offload preflight.

### Question
Can the persistent Pipeline finalize exact-range node-local Nsight reports through the local-rank0 CUDA profiler API?

### Hypothesis
The exact range completes and the two controllers finalize reports after the mandatory Full5 flush barrier.

### Importance
A valid Pipeline report would causally separate role compute, P2P, NCCL, copies, overlap, and idle gaps.

### Evidence Needed
Sixteen receipts, two reports, non-empty activity tables, and clean process teardown.

### Refutation Condition
Missing controller receipts/report, post-compute distributed hang, or reproduction of the prior local-rank-only `cudaProfilerStop` failure.

### Allowed Changes
One bounded diagnostic attempt only. Under the repeated-failure rule, no repeated patch escalation after a matching signature.

### Commands
Ran `gan_pipeline_nsys_20260811_001`; observed the completed non-controller range; compared against Iteration 255; terminated the owned hung trees after bounded observation. Prepared but did not launch actual-OOM-only baseline/pipeline offload searches.

### Measured Results
Fourteen receipts (ranks 1--7 and 9--15) completed the 1000--1004 interval in 1256.232645--1256.234138 s. Ranks 0 and 8 never emitted receipts; no `.nsys-rep` finalized; five GPU processes per node remained in `futex_wait_queue_me`. Controlled termination returned [143,143].

### Correctness
Profile FAIL. The completed logical window is not promoted to a correctness PASS because the exact 16-rank/report contract failed. Formal paired correctness remains independently PASS.

### Performance
Not reportable from this profiled attempt.

### Interpretation
This matches the bounded Iteration-255 node-local `cudaProfilerStop`/distributed-teardown failure. Decision: NO-RETRY for the same mechanism. It is instrumentation failure, not GAN math failure.

### Files Changed
`artifacts/manifests/gan_pipeline_nsys_20260811_001_failure.json`, diagnostic analyzer/tests, and capacity-search proof support for a possible resident k=0 candidate.

### Failure or Limitation
A cleanup SSH command was incorrectly double-quoted: local expansion turned `$pat` into an empty remote match and terminated both job containers more broadly than intended. Both SSH endpoints now refuse connections. This operator incident is recorded separately in `gan_remote_cleanup_incident_20260811_001.json`. Baseline and Pipeline minimum-offload searches are Not Run.

### Gate Status
Pipeline Nsight FAIL (bounded known negative). GPU execution BLOCKED pending job restart.

### Next Highest-Value Question
After restoring both jobs, what are the actual-OOM capacity boundaries `k*_baseline` and `k*_pipeline`?

### Next Command
Restart/recreate both jobs, revalidate the live substrate, then run the prepared baseline and Pipeline capacity searches from fresh IDs; do not rerun the failed profiler mechanism.

## Iteration 276: role-aware streaming and Generator no-grad phase residency implemented locally

### Milestone / Gate
GAN Pipeline Level C / role- and phase-specific immutable-parameter lifecycle gate.

### Question
Can the persistent 8G+8Score Full5 remove the dominant repeated frozen-block materialization without changing the official steps1000--1004 math, batch16/GAS2, trace, or update/version contract?

### Hypothesis
The fixed-policy Pipeline's 1043--1170 s Full5 is dominated by applying all40 synchronous streaming independently to every physical Generator and Score call.  Keeping the already-managed Generator blocks resident only across the post-G-update no-grad Critic-rollout phase, streaming only Score tail blocks 36--39, and retaining the persistent Generator allocator cache will remove most of that movement while leaving Step1000 differentiable execution unchanged.

### Importance
The formal same-policy scheduling result is 0.5715x, so persistence alone is not useful.  The requested positive target is a real, correctness-gated 300--400 s Full5; achieving it requires eliminating redundant lifecycle work rather than changing GAN/DMD computation.

### Evidence Needed
A fresh 2x8 A100 capacity/correctness run with all 16 ranks PASS; unchanged fairness and compute receipts; exact G1/C5/EMA1; no CUDA/cgroup OOM; balanced lifecycle; one Generator phase enter/exit; Generator materializations 600/rank and Score materializations 200/rank for candidate A; then independent low-overhead formal timing.

### Refutation Condition
Any model/loss/trace/batch/update drift, phase entry while autograd is enabled or backward state remains live, unbalanced release, deadlock, actual CUDA/cgroup OOM, fairness/compute failure, or a verified Full5 result remaining above the requested range after the bounded candidates.

### Allowed Changes
Only immutable frozen-block ownership/residency, role-specific streamed-block sets, persistent allocator-cache policy, lifecycle receipts, and launcher evidence.  No GAN/DMD loss, Score-call count, stochastic input, precision, global batch, GAS, optimizer, scheduler, EMA, or parameter-version change.

### Commands
Revalidated both local worktrees and the immutable paired result.  Both historical BAAI job endpoints authenticated at the gateway but closed before command execution, so real GPU work is Not Run.  Implemented and ran:

```text
PYTHONPATH=src:. UV_CACHE_DIR=/tmp/uv-cache-distillschedule \
  uv run --with pytest --no-project pytest -q \
  tests/test_refactor_gan_execution_receipts.py \
  tests/test_refactor_gan_systems_formal_trial.py \
  tests/test_refactor_gan_pipeline_offload_search.py
```

The preregistered next real command is recorded in
`iteration_276_gan_pipeline_role_phase_optimization_preregistration.json`.

### Measured Results
Measured historical formal inventory: each Generator rank performs 62 logical forwards plus 6 checkpoint recomputes per streamed block, or 2720 materializations and 1,911,585,013,760 bytes across all40; each Score rank performs 28 logical forwards plus 22 recomputes, or 2000 materializations and 1,405,577,216,000 bytes.  Candidate A predicts 600 Generator materializations (421,673,164,800 bytes, -77.94%) and 200 Score materializations (140,557,721,600 bytes, -90.0%).  These are deterministic accounting predictions, not performance measurements.  Local focused tests: 20 passed.  The broader no-Torch refactor subset previously passed 119 tests.  Real-14B CUDA result: NOT_RUN because both job endpoints are unavailable.

### Correctness
Local fail-closed receipt and launcher tests PASS.  Step1000 differentiable Generator rollout/backward remains on the existing all40 streaming path.  The new phase scope requires an enclosing `torch.no_grad()`, rejects pending checkpoint-backward releases, materializes each selected immutable block once, and releases all blocks on normal or exceptional exit.  Real-14B correctness remains NOT_RUN.

### Performance
NOT_RUN.  No 300--400 s claim is made from predicted bytes or the No-GAN Route-A Score4 diagnostic.

### Interpretation
The highest-value bounded candidate is now executable: G all40 during differentiable Step1000, one full-G resident scope for all ten no-grad Critic rollouts, and Score tail4 streaming.  This directly targets the repeated physical work revealed by the formal receipts while preserving the frozen semantic fingerprint.

### Files Changed
Local DMD-System execution branch: role-aware formal launcher, execution-receipt reconciliation, CLI, tests, and immutable Iteration-276 preregistration manifest.  Runtime execution tree: the existing GAN probe now consumes role-specific policies and the persistent Generator rollout can skip legacy per-rollout `empty_cache()`.  Pre-existing unrelated dirty runtime files were preserved.

### Failure or Limitation
The two original BAAI job allocations are no longer command-capable: direct SSH authenticates to the gateway and the remote closes the session.  CUDA/Torch tests and all real capacity/timing evidence are therefore NOT_RUN.  Candidate A may still OOM when all Generator blocks are phase-resident; candidate B is allowed only if that exact capacity failure occurs.

### Gate Status
GO for the preregistered real Candidate-A capacity/correctness gate; GPU execution pending restored/recreated 2x8 A100 jobs.  The 300--400 s goal is not yet achieved.

### Next Highest-Value Question
Does G40/S4 with one post-update Generator no-grad resident phase fit A100-40GB and preserve every formal receipt, and what exact Full5 wall time does it produce?

### Next Command
After receiving live job endpoints, revalidate HEAD/status/assets/GPU/cgroup state, safely stage only the hash-bound source files, then run `GAN-P-ROLEPHASE-G40-S4-R1` from a fresh immutable result path.  Run candidate B only if candidate A fails specifically on Generator phase capacity.

## Iteration 276 Addendum: failed-capacity evidence is now fail-closed

Before any GPU candidate was launched, the controller was tightened so a failed
Pipeline run preserves launcher/rank/lifecycle artifacts locally and classifies
capacity failure only from an explicit CUDA OutOfMemory signature or a positive
cgroup `oom`/`oom_kill` delta.  NCCL errors, timeouts, nonzero return codes,
missing files, trace drift, and code errors cannot trigger Candidate B.  Two new
classifier tests raise the focused local total from 20 to 22 PASS.  This is
controller evidence handling only; real-14B status remains NOT_RUN and the
candidate sequence is unchanged.

## Iteration 277: GAN Pipeline G40/S4 role-phase capacity gate

### Milestone / Gate
GAN Pipeline Level-C role/phase lifecycle optimization.

### Question
Can the preregistered G40/S4 policy keep Score blocks 0--35 resident and keep all Generator blocks resident across the Critic-only phase while preserving the frozen world16 Full5 contract?

### Hypothesis
Removing 77.94% of Generator and 90% of Score repeated block materializations will pass capacity and preserve all exact receipts.

### Importance
The fixed-policy Pipeline median is 1139.194 s and is dominated by 520.390 s Generator plus 311.933 s Score streamer H2D wall accounting. This gate tests the direct mechanism needed for the requested 300--400 s target.

### Evidence Needed
Real 16-rank 384x640 execution, explicit CUDA/cgroup capacity evidence, exact trace/update/queue/lifecycle receipts, and role-specific streamer counters.

### Refutation Condition
Any CUDA/cgroup OOM, exact receipt failure, deadlock, or frozen-contract drift.

### Allowed Changes
Only the preregistered role-specific streamed block counts, Generator Critic-phase residency, and removal of per-rollout allocator-cache clearing. No math, trace, batch, update, topology, or precision changes.

### Commands
`run_gan_systems_formal_trial.py --family pipeline --capacity-search --generator-streamed-block-count 40 --score-streamed-block-count 4 --generator-critic-phase-resident ...`

### Measured Results
**Measured capacity failure.** All eight Score ranks failed in Critic denoising checkpoint-recompute backward. Representative failure: a 292 MiB allocation with 266.38 MiB driver free, 35.70 GiB PyTorch allocated, and 2.13 GiB reserved-but-unallocated. Some ranks failed on 108 MiB. CUDA OOM signatures were explicit; cgroup OOM/OOM-kill deltas were zero. The Generator ranks were cancelled after the independent Score-pool failure, so Generator phase-residency capacity was not decided by this run.

### Correctness
Not reached. No performance or correctness claim.

### Performance
Not Run; the capacity-search run is diagnostic-only.

### Interpretation
S4 is one frozen block short of robust GAN Critic-backward admission. The preregistered R0 fallback is not causal because Generator and Score occupy disjoint GPUs. The minimal one-variable next test is S4 -> S5, freeing 702,788,608 bytes per Score rank while leaving the optimization mechanism and workload unchanged.

### Files Changed
Failure result/evidence under `artifacts/g_offload_search/gan-rolephase-20260812/candidate_A/`; live preflight manifest; Iteration-278 amendment.

### Failure or Limitation
No rank completed, so no actual materialization or timing receipts were produced. Missing block-streaming directories are a consequence of failure before normal finalization, not evidence against the OOM classification.

### Gate Status
REFUTED_S4_CAPACITY; GO_ONE_BOUNDED_S5_RETRY.

### Next Highest-Value Question
Does tail-5 streaming supply enough Score-backward headroom while Generator phase residency also admits?

### Next Command
Run the immutable G40/S5 capacity/correctness candidate once.

## Iteration 278: GAN Pipeline G40/S5 role-phase capacity gate

### Milestone / Gate
GAN Pipeline Level-C role/phase lifecycle optimization.

### Question
Does tail-5 Score streaming provide enough headroom for the real Critic backward while preserving G40 and Generator Critic-phase residency?

### Hypothesis
One additional nonresident Score block (702,788,608 bytes/rank) crosses the S4 capacity boundary without changing the workload.

### Importance
This is the minimal causal correction to the S4 failure and preserves the planned reduction in repeated Score materialization.

### Evidence Needed
A real world16 384x640 run with no CUDA/cgroup OOM and all exact correctness, update, lifecycle, and queue receipts PASS.

### Refutation Condition
Any CUDA/cgroup OOM, semantic receipt failure, deadlock, or contract drift.

### Allowed Changes
Only Score streamed tail count 4 to 5. Generator G40/phase residency and the complete frozen training contract remain unchanged. A separate cleanup-only launcher fix terminates exact descendant PIDs across nested process groups and does not touch model execution.

### Commands
Ran the preregistered `run_gan_systems_formal_trial.py --family pipeline --capacity-search --generator-streamed-block-count 40 --score-streamed-block-count 5 --generator-critic-phase-resident ...` after live GPU/cgroup/lock checks and 128 passing refactor tests.

### Measured Results
**Measured capacity failure.** All observed Score failures remained inside the first Critic denoising checkpoint-recompute backward. Unlike S4, S5 crossed the earlier 108/292 MiB pressure point and reached a later LoRA FFN allocation of 362 MiB. Per-rank driver-free memory was 250.38--326.38 MiB, a measured shortfall of 35.62--111.62 MiB. PyTorch reported 37.40 GiB allocated and 381.29--441.29 MiB reserved-but-unallocated. Explicit CUDA OOM signatures were present; cgroup OOM/OOM-kill deltas remained zero.

### Correctness
Not reached. No correctness or performance claim.

### Performance
Not Run; capacity search is diagnostic-only.

### Interpretation
S5 made causal progress but is still one resident Score block too aggressive. Streaming one more block releases 702,788,608 bytes/rank, comfortably above the measured 112 MiB worst shortfall. This justifies one bounded S6 attempt. Disabling Generator phase residency remains rejected because G and Score occupy disjoint GPUs.

### Files Changed
Immutable S5 failure result/evidence; exact process-tree cancellation helper and focused test; Iteration-279 S6 amendment manifest.

### Failure or Limitation
The run did not complete its first Critic update, so no formal timing exists. The controller SSH multiplex session ended with a broken-pipe message after failure collection, but the immutable result was written, both nodes had zero cgroup OOM events, and the new exact-PID cleanup left no GPU workers.

### Gate Status
REFUTED_S5_CAPACITY; GO_ONE_BOUNDED_S6_RETRY.

### Next Highest-Value Question
Does G40/S6 admit the whole Full5 and preserve all exact receipts?

### Next Command
Run the immutable G40/S6 capacity/correctness candidate once.

## Iteration 279: GAN Pipeline G40/S6 role-phase capacity gate

### Milestone / Gate
GAN Pipeline Level-C role/phase lifecycle optimization.

### Question
Does S6 provide enough Score-backward headroom for the full world16 GAN Full5?

### Hypothesis
The additional 702,788,608-byte nonresident Score block will cross the S5 LoRA-FFN allocation boundary.

### Importance
S6 retains 85% fewer Score materializations than all40 while testing the smallest next capacity change.

### Evidence Needed
All 16 ranks PASS with exact workload/update/lifecycle receipts and no CUDA/cgroup OOM.

### Refutation Condition
Any CUDA/cgroup OOM, exact receipt failure, deadlock, or contract drift.

### Allowed Changes
Only Score streamed tail count 5 to 6; all other policies and semantics frozen.

### Commands
Ran the immutable G40/S6 capacity/correctness candidate after both GPUs/locks/cgroups were clean.

### Measured Results
**Measured capacity failure.** S6 crossed the S5 LoRA-FFN allocation and reached a later `second_loss.backward()` checkpoint recomputation. It then failed while materializing a streamed Score block tensor of 134--136 MiB. Driver-free memory ranged from 2.38 to 124.38 MiB across ranks; PyTorch reported 37.67--37.80 GiB allocated. Explicit CUDA OOM signatures were present and cgroup OOM/OOM-kill remained zero.

### Correctness
Not reached; no correctness claim.

### Performance
Not Run; diagnostic capacity candidate only.

### Interpretation
The memory policy again made causal forward progress, but tail6 lacks robust block-rematerialization headroom. To avoid serial one-block probing, the next bounded candidate skips S7 and uses S8, adding about 1.31 GiB capacity while retaining an 80% reduction in Score materialization count versus all40.

### Files Changed
Immutable S6 failure evidence and Iteration-280 S8 amendment manifest.

### Failure or Limitation
No Critic update completed, so timing is not reportable. The failure remains isolated to the Score pool; Generator fallback remains non-causal.

### Gate Status
REFUTED_S6_CAPACITY; GO_ONE_BOUNDED_ROBUST_S8_RETRY.

### Next Highest-Value Question
Does G40/S8 admit the complete Full5 and expose the actual optimized time?

### Next Command
Run the immutable G40/S8 capacity/correctness candidate once.

## Iteration 280: GAN Pipeline G40/S8 role-phase capacity gate

### Milestone / Gate
GAN Pipeline Level-C role/phase lifecycle optimization.

### Question
Does a robust two-block capacity increment (S6 to S8) admit the real Score Critic backward?

### Hypothesis
About 1.31 GiB additional static headroom crosses the S6 block-rematerialization boundary.

### Importance
S8 still removes 80% of fixed-policy Score materializations and avoids inefficient one-block search.

### Evidence Needed
All 16 ranks and every exact correctness/lifecycle/update receipt PASS without CUDA/cgroup OOM.

### Refutation Condition
Any CUDA/cgroup OOM, exact receipt failure, deadlock, or contract drift.

### Allowed Changes
Only Score streamed tail count 6 to 8; all semantics and Generator policy frozen.

### Commands
Ran the immutable G40/S8 capacity/correctness candidate after clean GPU/lock/cgroup preflight.

### Measured Results
**Measured capacity failure.** S8 successfully materialized recompute block39, then failed within that block's attention/LoRA compute on a 134 MiB allocation. Most ranks had 70.38--128.38 MiB driver free; PyTorch reported 37.72 GiB allocated and 274.64--294.64 MiB reserved-but-unallocated. All eight Score ranks emitted explicit CUDA OOM and cgroup OOM counters stayed zero.

### Correctness
Not reached; no correctness claim.

### Performance
Not Run; diagnostic capacity candidate only.

### Interpretation
Increasing Score streaming continues to move the failure monotonically deeper into the same first Critic backward. The next and final coarse capacity count is S10, adding another approximately 1.31 GiB while preserving 75% fewer materializations than all40. If S10 still fails in this backward, count-by-count patch escalation stops and the transient lifetime peak becomes the next causal target.

### Files Changed
Immutable S8 failure evidence and Iteration-281 S10 amendment manifest.

### Failure or Limitation
No Critic update completed and no timing is claimable.

### Gate Status
REFUTED_S8_CAPACITY; GO_FINAL_COARSE_S10_RETRY.

### Next Highest-Value Question
Does G40/S10 complete Full5, and if so what is its measured lifecycle/time envelope?

### Next Command
Run the immutable G40/S10 capacity/correctness candidate once.

## Iteration 281: G40/S10 capacity gate closes linear block-count escalation

### Milestone / Gate
GAN Pipeline Level-C transient-lifetime diagnosis.

### Question
Does the final coarse S10 capacity point admit the first real GAN Critic backward?

### Hypothesis
Two additional nonresident Score blocks provide sufficient margin after S8.

### Importance
This is the preregistered stopping point before shifting from static-weight capacity to the transient lifetime peak.

### Evidence Needed
A complete 16-rank Full5 with exact receipts and no CUDA/cgroup OOM.

### Refutation Condition
Any actual OOM or contract failure.

### Allowed Changes
Only Score streamed tail count 8 to 10; all other policy and semantics frozen.

### Commands
Ran the immutable G40/S10 candidate on the clean two-node substrate.

### Measured Results
**Measured capacity failure.** S10 progressed farther but still failed in the first Critic `second_loss` checkpoint-recompute FFN on a 362 MiB allocation. Driver-free memory was 190.38--266.38 MiB on reported ranks; PyTorch allocated 37.61 GiB and held 243.16--283.16 MiB reserved-but-unallocated. Explicit CUDA OOM was present; cgroup OOM/OOM-kill stayed zero.

### Correctness
Not reached; no correctness claim.

### Performance
Not Run.

### Interpretation
Static block-count escalation has now failed at S4, S5, S6, S8, and S10 while monotonically moving the failure deeper into the same first backward. Per the preregistered rule, no S11/S12/S16 linear patching follows. The next causal state class is the packed fake/real/R1 checkpoint-boundary activation.

### Files Changed
Immutable S10 failure evidence. No training-math change.

### Failure or Limitation
The precise minimum weight-only offload count remains unknown, but finding it by linear search is no longer the highest-value route to the 300--400 s goal.

### Gate Status
WEIGHT_COUNT_ESCALATION_CLOSED; GO_SELECTIVE_CRITIC_ACTIVATION.

### Next Highest-Value Question
Can the existing audited activation primitive admit S8 by offloading only large Critic-phase saved tensors?

### Next Command
Run G40/S8 with Score Critic activation threshold 256 MiB.

## Iteration 282: Selective Critic activation A256 census gate

### Milestone / Gate
GAN Pipeline Level-C Critic saved-activation lifecycle.

### Question
Can a 256 MiB Critic-only saved-tensor threshold admit G40/S8 without affecting Score gradient service?

### Hypothesis
The packed fake/real/R1 checkpoint boundaries exceed 256 MiB and can be moved to pinned CPU memory.

### Importance
This pivots from repeated weight-count escalation to the transient state class actually colliding with backward.

### Evidence Needed
Nonzero audited CPU copies only in the Critic phase, complete world16 Full5, exact receipts, zero live handles, and no OOM.

### Refutation Condition
No selected tensors, capacity failure, host failure, activation leak, or semantic drift.

### Allowed Changes
Existing audited saved-tensor hooks installed only after both Score gradient-service waves; 256 MiB threshold; G40/S8 otherwise frozen.

### Commands
Implemented the fail-closed phase-scoped primitive, added exact scope/copy/live-byte gates, passed 129 DMD refactor tests plus 33 runtime lifecycle tests, and ran G40/S8/A256.

### Measured Results
**Hypothesis refuted by measured census.** The hook opened two Critic model scopes before failure but made zero CPU copies. The 40 key packed-GAN checkpoint boundaries are 133.59375 MiB each, two additional tensors are 168.75 MiB, and the 256 MiB threshold selected neither. The run therefore reproduced S8's 134 MiB attention/LoRA OOM. cgroup OOM counters remained zero.

### Correctness
Not reached. Phase ordering itself passed source/tests: hooks are installed after the two Score gradient-service waves and before `_critic_loss`.

### Performance
Not Run.

### Interpretation
The mechanism was not disproved; its threshold was. The low-overhead failure census gives a non-post-hoc boundary: 128 MiB selects the intended 133.59/168.75 MiB state while excluding the 126.56 MiB and smaller classes.

### Files Changed
Runtime phase-scoped activation policy and stats, launcher flag/receipt gate/tests, immutable A256 evidence, Iteration-283 amendment.

### Failure or Limitation
No full step completed. A128 may impose material pinned-memory/copy overhead even if it admits.

### Gate Status
REFUTED_A256_SELECTION; GO_ONE_CENSUS_DRIVEN_A128_RETRY.

### Next Highest-Value Question
Does A128 admit G40/S8, remain leak-free, and expose a useful Full5 time?

### Next Command
Run the immutable G40/S8/A128 capacity/correctness candidate once.

## Iteration 283: A128 exposes the GAS2 second-wave capacity boundary

### Milestone / Gate
GAN Pipeline Level-C joint Score parameter/activation lifecycle.

### Question
Does the census-selected 128 MiB threshold admit G40/S8, remain bounded, and preserve the frozen world16 Full5 contract?

### Hypothesis
Offloading the measured 133.59/168.75 MiB Critic saved-tensor classes removes the transient peak that defeated the weight-only candidates.

### Importance
This is the first real test in which the Critic-only activation primitive selects the intended physical state. It determines whether the previous OOM was actually activation-driven rather than another resident-weight boundary.

### Evidence Needed
Nonzero Critic-only CPU copies, forward/backward progress beyond the former S8 boundary, bounded CPU live bytes, all 16 ranks PASS, exact receipts, and no CUDA/cgroup OOM.

### Refutation Condition
Zero selected tensors, no causal progress, any capacity failure, activation leak, exact receipt failure, deadlock, or contract drift.

### Allowed Changes
Relative to A256, only the Critic saved-tensor threshold changed from 256 MiB to 128 MiB. G40/S8, synchronous pinned copies, no prefetch, placement, trace, batch, updates, math, and precision remained frozen.

### Commands
Ran `run_gan_systems_formal_trial.py --family pipeline --capacity-search --generator-streamed-block-count 40 --score-streamed-block-count 8 --generator-critic-phase-resident --score-critic-activation-offload-min-mib 128 ...` after clean GPU, cgroup, and lock checks.

### Measured Results
**Measured causal progress followed by CUDA capacity failure.** On representative Score ranks the hook issued 44 CPU copies totaling 6,311,116,800 logical bytes (5.8777 GiB), with 5,603,328,000 bytes (5.2185 GiB) peak live CPU activation storage. Three Critic model scopes opened: the complete first GAS wave's denoise and packed GAN-D/R1 forward/backward paths completed, then the second wave's denoise forward completed and its backward failed. The former S8 134 MiB attention/LoRA boundary was therefore crossed. Six captured Score ranks failed on a 292 MiB allocation with 150.38--270.38 MiB driver free and 37.45 GiB PyTorch allocated; the measured immediate shortfall was 21.62--141.62 MiB before any safety margin. CUDA OOM signatures were explicit and cgroup OOM/OOM-kill deltas were zero.

### Correctness
Not reached because the second GAS wave failed before the first Critic optimizer commit. The trace, math, topology, and update contract were unchanged. Completed scopes show the activation policy was confined to `score_critic_train`; Score DMD/GAN-G gradient service remained outside the hooks.

### Performance
Not Run. This capacity-search run is diagnostic-only. Its wall time cannot enter formal speedup results.

### Interpretation
A128 validated the targeted activation mechanism and moved the failure from the first wave's packed-GAN backward to the second wave's denoising backward, where accumulated first-wave gradients reduce available headroom. Threshold-only tuning is stopped: lowering to 126 MiB would be a post-hoc narrow-margin choice. A joint G40/S10/A128 candidate is instead justified because two additional nonresident Score blocks release 1,405,577,216 bytes (1.3090 GiB) per Score rank, well above the measured 142 MiB deficit while retaining 75% fewer Score materializations than all40.

### Files Changed
Immutable A128 result/failure evidence under `artifacts/g_offload_search/gan-rolephase-20260812/candidate_S8_A128/`; Iteration-284 preregistration. No training-math change.

### Failure or Limitation
The failed process could not prove final live handles equal zero; process teardown reclaimed all storage and left both nodes GPU-idle with zero cgroup OOM events. A successful run must still pass the existing `scope_count=20`, `cpu_copy_count>0`, and `live_cpu_bytes=0` gates.

### Gate Status
VALIDATED_ACTIVATION_MECHANISM; REFUTED_S8_HEADROOM; GO_ONE_JOINT_S10_A128_GATE.

### Next Highest-Value Question
Does the preregistered joint S10/A128 policy provide robust GAS2 backward headroom and complete exact Full5?

### Next Command
Run one immutable G40/S10/A128 world16 capacity/correctness candidate; do not resume linear S11/S12/S16 probing.

## Iteration 284: Joint G40/S10/A128 real-14B capacity and correctness PASS

### Milestone / Gate
GAN Pipeline Level-C optimized lifecycle admission.

### Question
Does the preregistered joint Score tail-10 plus 128 MiB Critic activation policy complete exact world16 Full5 while preserving Generator Critic-phase residency?

### Hypothesis
The additional 1.309 GiB static Score headroom is sufficient for the GAS2 second-wave denoising backward exposed by A128.

### Importance
Formal timing cannot begin until one immutable non-default lifecycle policy passes the complete 16-rank workload and all exact gates.

### Evidence Needed
16/16 PASS; G=1, Critic=5, EMA=1; exact stochastic and compute receipts; DP8/MB1/GAS2/B16; no lifecycle/activation leaks; zero CUDA/cgroup OOM.

### Refutation Condition
Any rank failure, wrong update count, receipt mismatch, queue/lifecycle leak, or capacity failure.

### Allowed Changes
Relative to Iteration 283 only Score streamed tail count changed from 8 to 10. A128, G40, Generator phase residency, synchronous pinned copies, no prefetch, and the frozen training contract remained unchanged.

### Commands
Ran the immutable `G40/S10/A128` world16 capacity/correctness candidate after exact source-SHA, GPU, lock, cgroup, and focused-test preflight.

### Measured Results
**PASS.** All 16 ranks completed steps 1000--1004. Generator ranks each reported exactly one Generator and one EMA update; Score ranks each reported exactly five Critic commits. DP8/MB1/GAS2/global-batch16 and operation-keyed stochastic/DiT/Score-call receipts passed. GPU maximum allocated was 39,111,174,144 bytes and maximum reserved was 40,290,484,224 bytes. Both nodes had zero cgroup OOM and OOM-kill increments. Generator materialization fell to 600 calls / 421,673,164,800 bytes per rank; Score used 500 calls / 351,394,304,000 bytes per rank. All streamers ended with zero resident bytes. Score activation hooks completed 20 scopes; representative rank 8 issued 413 unique CPU copies, offloaded 58,591,641,600 logical bytes, peaked at 5,603,328,000 live CPU bytes, and ended at zero live bytes.

### Correctness
PASS for every currently registered exact fairness, compute-inventory, batch, update, EMA, queue, activation, and streamer-lifecycle gate. Fairness receipt SHA256: `069069ba2ba5f15bc75ed532c25c5ed3d456dc681f44786d8f0f35c269dc3b96`; compute receipt SHA256: `52746478d75d7112955db0413f001df258c76a6058725fd0f233ebc527659a42`.

### Performance
**Diagnostic only:** the detailed-event capacity run measured 868.488686064 s for the global Full5 window. It is not formal-speedup eligible because 16 ranks synchronously opened and appended tens of thousands of shared-filesystem JSON events. It already improves the old fixed-policy Pipeline median 1139.194 s, but this value must not be used as the final result.

### Interpretation
The joint physical-lifetime policy is now admitted. The next run freezes the exact policy by manifest/result hashes and removes only detailed per-pack/unpack/materialization event logging; counters, scope summaries, exact timing receipts, and all training semantics remain. This is the required low-overhead formal measurement, not a new model-policy search.

### Files Changed
Immutable capacity result and diagnostic events under `candidate_S10_A128/`; hash-bound validated-policy manifest; formal promotion guard and tests.

### Failure or Limitation
The 39.11 GB allocated peak leaves little margin, so any source/policy drift must fail closed. The capacity gate proves execution correctness/admission, not that the requested 300--400 s target is already achieved.

### Gate Status
PASS_CAPACITY_CORRECTNESS; GO_HASH_BOUND_LOW_OVERHEAD_FORMAL.

### Next Highest-Value Question
What is the real unprofiled Full5 time after removing detailed shared-filesystem event logging while keeping the admitted physical policy exact?

### Next Command
Run one formal G40/S10/A128 launch with the immutable validated-policy manifest; retain scope/summary and exact timing receipts, but no detailed events.

## Iteration 285: Low-overhead G40/S10/A128 model execution passes but receipt aggregation fails

### Milestone / Gate
GAN Pipeline Level-C low-overhead formal promotion.

### Question
What is the real unprofiled Full5 window after disabling only detailed shared-filesystem lifecycle events for the hash-bound admitted policy?

### Hypothesis
Detailed per-pack/unpack/materialization JSON was the dominant avoidable measurement overhead; summary counters and exact timing receipts are sufficient to retain the gate.

### Importance
This is the first timing attempt after a complete real world16 capacity/correctness PASS for the non-default physical policy.

### Evidence Needed
All 16 model workers and timing receipts PASS, aggregate fairness/compute/lifecycle receipts PASS, no OOM, and an exact global Full5 window.

### Refutation Condition
Any model/rank/trace/update/capacity failure, missing exact timing receipt, or failure to close the aggregate receipt gate.

### Allowed Changes
Disable detailed lifecycle JSON only. Keep G40/S10/A128, phase residency, synchronous pinned copies, no prefetch, topology, trace, math, batch, updates, and precision frozen.

### Commands
Ran one hash-bound low-overhead world16 launch from the immutable validated-policy manifest. No additional launch was made after the user requested a stop.

### Measured Results
All 16 model workers returned PASS and all 16 unprofiled exact timing receipts returned PASS. The exact global window, defined as maximum rank stop minus minimum rank start, was 824.685005847 s. No CUDA or cgroup OOM occurred. The wrapper then failed during postprocessing with `eager block evidence inventory drift` because formal low-overhead JSON contained only streamer install/remove summaries while the receipt builder still required per-block materialization rows.

### Correctness
Model-level update counts remained Generator=1, Critic=5, EMA=1. The overall formal correctness gate is not closed because the aggregate receipt builder failed after execution; the run remains non-claimable.

### Performance
Measured preliminary only: 824.685006 s for one exact unprofiled Full5 window, 0.097007 samples/s for 80 logical samples. Relative to the old fixed Pipeline median it is 27.61% shorter, but it is 25.69% slower than the Sequential median. Three paired formal trials were Not Run.

### Interpretation
The lifecycle policy reduces the old Pipeline time materially but does not approach the 300--400 s goal. The immediate failure is a postprocessor mode mismatch, not a model or capacity failure. It must not be repaired by fabricating missing detailed events.

### Files Changed
Preserved raw rank results/timing receipts/failure result; added a preliminary timing summary and source snapshot for review.

### Failure or Limitation
One launch only; receipt aggregation FAIL; no formal speedup claim.

### Gate Status
MODEL_EXECUTION_PASS_POSTPROCESS_FAIL; STOP_FOR_IMPLEMENTATION_REVIEW.

### Next Highest-Value Question
Which pipeline scheduling/overlap assumptions should be corrected before another capacity gate?

### Next Command
None. GPU experimentation is stopped.

## Iteration 286: Publish implementation and measured handoff for correction

### Milestone / Gate
GAN Pipeline implementation handoff.

### Question
Can the exact executed code and bounded evidence be packaged for review without promoting incomplete timing?

### Hypothesis
A focused draft PR with source identity, capacity PASS, raw timing, and explicit failure boundaries is sufficient for the user to correct the pipeline implementation.

### Importance
Continuing launches against an implementation the user intends to change would waste two-node GPU time and mix incompatible variants.

### Evidence Needed
Idle nodes, no cgroup OOM, exact local/remote source SHA match, curated immutable artifacts, relevant tests, and a draft PR with no claim inflation.

### Refutation Condition
Any active workload, source mismatch, missing raw timing evidence, accidental staging of unrelated changes, or a PR that labels 824.685006 s formal.

### Allowed Changes
Documentation, manifests, source snapshot, curated existing artifacts, launcher/process-safety code already exercised, and tests. No model execution or training-math change.

### Commands
Rechecked both nodes, recovered exact existing timing receipts, curated capacity and failure evidence, and prepared the GitHub review branch.

### Measured Results
Both nodes were idle with `oom=0` and `oom_kill=0`. Local and remote SHA-256 values matched for the DMD launcher/process wrappers and the five executed runtime/Helios source files.

### Correctness
No new model run. Existing capacity gate remains PASS; existing low-overhead attempt remains non-claimable.

### Performance
No new timing. 824.685006 s remains a single measured preliminary exact window.

### Interpretation
The next change should correct the legal overlap/state machine first, then restart at one capacity/correctness gate. It should not inherit a formal performance claim from this handoff.

### Files Changed
DMD-System status report, Iteration-286 manifest, exact source snapshot, curated capacity/timing/failure evidence, and append-only research log.

### Failure or Limitation
The external DistillSchedule runtime repository has no GitHub remote; its exact executed files are mirrored into the DMD-System PR as a review snapshot rather than silently rewriting repository ownership.

### Gate Status
STOPPED_FOR_PIPELINE_IMPLEMENTATION_CORRECTION.

### Next Highest-Value Question
User review of the persistent 8G+8Score implementation.

### Next Command
None until a corrected schedule is preregistered.

## Iteration 287: PR8 fixed-Score node1 role-phase worker binding

### Milestone / Gate
GAN role-phase prototype / node1 code-adaptation preflight.

### Question
Can node1's eight fixed Score GPUs be bound to Phase-A 8Score/B16 and Phase-B C12/B12 without reconstructing or streaming the 14B Score backbone at the phase boundary?

### Hypothesis
Global ranks 8--15 can retain the same physical resident Score/Critic module, serve two read-only Phase-A gradient waves, then rebind it to the validated PR6-style DP12/B1/GAS1 ZeRO-1 CPU-optimizer backend. Only the logical optimizer partition changes; fixed node1 ranks require no whole-backbone parameter redistribution.

### Importance
PR8's 8G/8Score to 4G/12C schedule is only useful if the fixed Score half avoids unnecessary model movement and every node1 rank joins exactly the official phase-local batch and collective geometry.

### Evidence Needed
Live node1 audit, exact PR8 source identity, deterministic process-group construction, eight node1 rank bindings, DP12/B1/GAS1/ZeRO-1 fail-closed checks, targeted tests, zero GPU allocation/OOM, and an explicit Not Run boundary for real 14B/world16 execution.

### Refutation Condition
Any mapping other than Phase-A slots `(lane,lane+8)` and Phase-B slots 4--11, any ZeRO-2/GAS2/batch16 drift in Phase B, process-group creation at the phase boundary, repeated Score frozen-block streaming, or a failed low-cost test.

### Allowed Changes
PR8 process-group inventory, a node1-specific real-worker adapter, pure contract helpers/tests, source snapshot/status evidence, and documentation. No model math, trace, loss, batch contract, optimizer hyperparameter, EMA predicate, node0 worker, 14B launch, or timing.

### Commands
Revalidated PR8 HEAD `c221bc9f673f17dc26e4a07cd39b908c6b2b8bd1`, the isolated runtime base `171e3bf38605b790b26341496d714cbf3766fb03`, node1 GPU/process/cgroup state, and the previous PR6/PR7 worker paths. Staged new source under fresh shared paths, compiled it, ran focused DMD/runtime tests, and emitted all eight contract-only rank receipts.

### Measured Results
DMD PR8 focused tests: 29 passed. Runtime node1 contract tests: 4 passed. Contract receipts: 8/8 PASS; Phase-B node1 slots were exactly 4--11 and sources were exactly `[0,1,2,3,0,1,2,3]`. All eight A100s remained at 0 MiB and cgroup `oom=0`, `oom_kill=0`. One first test invocation failed only because the two repositories both expose a top-level `runtime` package and were placed in the wrong PYTHONPATH order; the preserved second invocation separated the DMD and external-runtime test roots and passed without source changes.

### Correctness
Contract/static correctness PASS. The worker explicitly keeps transport micro-wave 0 while calling the official one-wave Critic path with `pipeline_micro_wave=None`, so each DP12/B1 call commits once instead of inheriting PR7's two-wave GAS2 defer rule. Real Phase-A gradients, DP12 collectives, C1000->C1005, and reverse handoff are Not Run.

### Performance
Not Run. No 14B model was constructed and no CUDA timing was performed.

### Interpretation
The fixed node1 half does not need PR6 pinned target materialization: its physical Score backbone remains resident and is rebound from read-only service to DP12 ZeRO-1. PR6-style materialization remains necessary only for flex ranks 4--7 on node0, which actually change Generator->Critic physical role. The next load-bearing task is the node0 companion and one full world16 correctness gate.

### Files Changed
`runtime/gan_rolephase_node1.py`, `scripts/distillschedule/probe_gan_rolephase_node1.py`, `tests/test_gan_rolephase_node1.py`; PR8 deterministic execution groups/test; node1 staging roots and immutable local logs/report.

### Failure or Limitation
This node1-only worker cannot be launched as an independent world8 and is not a complete Full5 executor. Node0 companion, flex materialization, Generator/EMA work, commit notification, reverse handoff, full trace receipts, real capacity, and timing remain Not Run.

### Gate Status
NODE1_CODE_ADAPTATION_PASS; WORLD16_REAL14B_NOT_RUN.

### Next Highest-Value Question
Can the node0 companion reuse PR6's validated flex target materialization while preserving PR8 Phase-A overlap and Phase-B DP12 one-call/one-commit semantics?

### Next Command
Implement the node0 companion against the same pre-created group inventory, then run no-model world16 topology/transport tests before any 14B construction.

### Iteration 287 Addendum: call inventory terminology corrected before GPU work

The contract receipt now distinguishes Python-level service/loss invocations
from actual Score-backbone forwards. Each node1 rank serves two Phase-A logical
samples; the frozen official inventory is four Score forwards per sample, so
the receipt records two gradient-service calls and eight expected Score
forwards. Each Phase-B rank makes one `_critic_loss` call per step, containing
one denoising Score forward and one packed GAN-D/real/R1 Score forward, so the
receipt records one loss call and two expected Score forwards. No model math,
schedule, batch, or measured result changed; real 14B remains `NOT_RUN`.

### Iteration 287 Addendum: node0 implementation status corrected by user

The node1-only scope of this iteration reflected GPU availability, not a
missing node0 implementation. The node0 execution path already exists and must
be reused; no new node0 companion should be implemented. The next execution
gate is a single world16 integration run that launches the existing node0 path
with this node1 adaptation, first validating group/transport/phase contracts
and then, if admitted, the real 14B correctness path. What remains `NOT_RUN` is
that world16 integration and its flex transition evidence, not node0 code
construction.


## Iteration 289: DP12 tail10+A128 bounded capacity gate

### Milestone / Gate
GAN role-phase world16 / Phase-B capacity and correctness before timing.

### Question
Can the unchanged DP12/MB1/GAS1/B12 Phase-B Critic finish C1000→C1005 when the already-active A128 pinned activation offload is paired with the previously audited Score tail10 block policy?

### Hypothesis
Changing only the streamed immutable Score tail from blocks 36–39 to blocks 30–39 frees about 3.927 GiB of additional static GPU storage per Critic rank, enough to cover the measured 292/362-MiB first-backward requests. The existing A128 policy remains unchanged.

### Importance
The new role-phase topology has not completed a single Critic commit. Capacity correctness is a hard prerequisite for transition optimization or formal timing; repeating resident/tail4 or pretending A128 was absent would repeat a documented failure.

### Evidence Needed
Sixteen PASS rank receipts; twelve DP12 Critic ranks; five commits; G201/C1005/EMA801; exact trace/call/queue/reverse contracts; exactly ten A128 scopes and nonzero saved-tensor copies per Critic rank; tail10 blocks 30–39 with window1 and HOST_SHARED_IMMUTABLE; exact materialize/release reconciliation; zero live activation/parameter handles; no CUDA or cgroup OOM.

### Refutation Condition
After receipts confirm both tail10 and A128 actually executed, the same first Critic checkpoint-recompute backward still OOMs, or any correctness/version/queue/lifecycle gate fails.

### Allowed Changes
Only register Phase-B tail10, preserve the existing synchronous A128 primitive, and add fail-closed policy/lifecycle receipts, tests, launcher/controller validation, manifest, status, and append-only log. No model math, trace, batch, loss, optimizer, EMA, placement, prefetch, async copy, or timing-policy change.

### Commands
Rechecked local HEAD/status and the Iteration 288/284 evidence; inspected the active worker and confirmed A128 is installed after both resident and streaming branches. Added tail10 dry-run and lifecycle validation. Local Python compile, shell syntax, git diff check, controller dry-run, 31 focused role-phase tests, and 184 CI refactor tests pass in a disposable CPU Torch environment. The full maintained tests report 367 passed, 10 skipped, and two pre-existing failures because this worktree lacks archived Iteration-281 manifest files; neither failure touches the P0 code path. Both SSH endpoints authenticate but remote commands time out, so GPU/process/cgroup/assets and remote tests remain Not Run.

### Measured Results
Not Run. No GPU experiment has been launched. Static policy delta is six additional streamed blocks, 4,216,731,648 bytes (3.927 GiB) per Critic rank based on the prior audited block inventory; this is evidence-derived, not a new measurement.

### Correctness
Not Run. New fail-closed summary requirements are implemented but await the single real world16 gate.

### Performance
Not Run. This iteration is not a timing claim.

### Interpretation
GO for exactly one tail10+A128 capacity/correctness attempt once live substrate checks pass. If the confirmed policy reproduces the same first-backward capacity failure, stop; do not try S11/S12. If it passes, the next iteration may transplant the validated host-safe transition substrate and acquire stage timing.

### Files Changed
Worker, launcher, controller, summarizer, focused tests, this preregistration manifest, status report, and append-only research snapshot.

### Failure or Limitation
The current BAAI SSH command channels time out on both nodes after authentication, so live GPUs, cgroup OOM counters, assets, remote tests, staging, and the real world16 run cannot yet be verified. Local CPU tests pass, but they do not substitute for the remote CUDA/DeepSpeed gate.

### Gate Status
PREREGISTERED; REAL14B NOT RUN; FORMAL TIMING NOT RUN.

### Next Highest-Value Question
Does exact DP12 tail10+A128 pass the first Critic backward and close the full correctness/lifecycle gate?

### Next Command
After SSH recovery: revalidate HEAD/status, GPU ownership, cgroup OOM counters, assets and remote tests; stage an immutable source identity; then launch one correctness run with `--phase-b-streamed-blocks 10`.
