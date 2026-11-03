# Open workload program: semantics, not a prewritten optimization recipe

## Public entry

A user Python factory returns `Workload(Program(...), bindings, state, feeds)`.
It describes model calls, loss/backward/update functions, logical reads/writes,
backward provenance and actual inputs. It does NOT provide measured winners,
optimization candidates, placement choices or a model-specific scheduler.
Python loops/branches instantiate the actual workload from parameters before
planning. Unresolved data-dependent control flow is not guessed.

For a bounded Student1/FakeN detached-score loop, the optional
[DMD semantic builder](#open_workload_dmd_builder) constructs these same calls
from user-owned forward, noising, loss and history hooks; it selects no strategy.

```bash
python scripts/distillctl.py workloads native program auto \
  --factory examples/open_workloads/video_input_gradient.py:build \
  --parameters examples/open_workloads/video_parameters.json \
  --output /tmp/open-video-run --memory-limit-bytes 1000000 \
  --samples 1 --max-program-runs 8 --execute
```

The example uses small random Conv3d weights and actual input gradients/Adam on
CPU. It is an interface example, not an original DMD performance result.
To change frames, width or update count, change the JSON; no planner edit or new
launcher is required. `describe`, `calibrate`, metadata-only `plan`, and `run`
are available separately. Factories are trusted user code and model construction
requires explicit --execute; no implicit Hub download or cluster allocation.

## What the shared system does

- `workloads/program.py` compiles calls into the existing ExecutionGraph. It
  derives version RAW/WAR edges and keeps backward/replay dependencies alive.
- The generic Torch backend matches immutable parameter identity (content,
  shape, dtype, layout) and explicitly replay-safe gradient-bearing regions.
  It can choose physical immutable aliases and nonreentrant checkpointing.
  Adapters/optimizers remain separate; mutable parameter aliases are rejected.
- Calibration runs the actual supplied math from paired cloned model/Adam
  states. The numerical comparison result is recorded separately.
- Native save-attention/save-matmul variants require **observed** matching
  operator calls and retained bytes on every rank; unavailable options are
  recorded as inapplicable rather than ranked as optimizations. Checkpoint
  replay callbacks are not required: PyTorch 2.14 may reuse saved native
  results without replay dispatch. Native forward-operation counts plus full
  gradient/optimizer parity test that recomputation is genuinely avoided.
- `search/service.py` reuses the existing event-frontier DP/beam across admitted
  candidates. It is independent of model names, loss functions and update ratios.
- The selected sequence runs through PreparedRolePlan. Stored timing and memory
  predictions are reconciled against measured components before execution.
  Logical work, including Python/Torch RNG, remains tied to call IDs.
- Costs bind code, parameters, state, feeds, dtype/device and environment. New
  parameters require matching calibration. Budget truncation is explicit.

## Actual boundaries

The open callable backend now executes local programs and explicitly allocated
multi-process data-parallel programs. For admitted module-forward contracts it
matches replicated post-backward reduction, DDP asynchronous reduction and FSDP2
reshard/keep-forward layouts, plus topology-admitted HSDP2 and per-module layout
assignments. These use the SAME Program, calibrator, search
service and call executor, with extra completion edges for reducer lifetimes.
See [Distributed open workloads](#open_workload_distributed) for exact
contracts, CPU/Gloo coverage, examples and remaining restrictions. The separate
[RoleTask/RoleFunctions route](open_role_pools.md) implements bounded role-pool
partitioning and cross-pool VJP transport; arbitrary Workload graphs are not
automatically lowered to it. A100/H100 validation of these generic capabilities
is NOT_RUN. Original workload-specific routes remain available; no five-model
GPU parity is implied.

`immutable_modules` and `replay_safe` are semantic promises verified where
possible, not automatic inference of all Python side effects. All relevant
state must pass via ctx.state/inputs. No external autograd graph, hidden model
closure, in-flight accumulated gradients or silent optimizer reset is admitted.
Use backward_of for differentiated values, including input gradients through a
frozen critic. Incorrect state declarations are outside the ABI.

CPU memory is tracked tensor/storage accounting, not an RSS guarantee. CUDA
allocator accounting is implemented but NOT GPU validated here. Calibration
clones model state, so peak preparation capacity differs from steady execution;
large-model lifecycle/offload requires its appropriate backend. CPU timings are
not claimed GPU speedups. A successful small-model test is not full pretrained
Wan/Helios/H3/LiveWan numerical or quality evidence.

This branch is stacked on PR #33 and preserves its original-protocol routes.
The full five-workload capability migration and GPU validation remain unsatisfied goals.

<a id="open_workload_distributed"></a>
## Generic workload execution: replicated and sharded layouts

Implementation owners: `open_workload/`, `search/` and the native backend adapters.
PR #35's alternative ABI was inspected but is not copied into a second public
system. Its branch is preserved; this PR is the continuation line.

### User contract versus system decisions

A factory supplies ordinary model, loss, backward and update functions through
`Workload(Program(...), bindings, state, feeds)`. Model names and known-winner
recipes are not queried by the public capability matcher or shared DP service.
Parameters must change the actual factory computation/graph, not just metadata.
For data parallelism add `ReplicaSemantics(module_bindings, backward_calls,
local_batch_size)`: ordinary trainable-root forward interfaces, parameter
backward call IDs and an equal local-mean objective. These are training facts,
not optimization choices. Inputs may differ across ranks, but initial models,
optimizer states, logical graph, dtype and input geometry must agree.

The system currently matches and measures:

| Capability | Actual execution |
|---|---|
| Post-backward mean | SUM/world-size on parameter gradients after backward |
| DDP overlap, 1/128 MiB buckets | Native DDP plus the EXISTING `runtime.wan13b_fair.ddp_sum_then_divide_hook` |
| Immutable aliases | Same frozen parameter storage, independent adapters/optimizers |
| Nonreentrant checkpoint | User-declared replay-safe regions; backward still runs |
| FSDP2 reshard | Native `fully_shard`, declared trainable roots, parameter/gradient/optimizer sharding |
| FSDP2 keep-forward | Same sharding with `reshard_after_forward=False` |
| HSDP2 reshard/keep-forward | Native replicate-by-node, shard-within-node mesh, admitted only for complete equal multi-rank nodes |
| Per-module layouts | Bounded combinations of admitted replicated/DDP/FSDP2/HSDP2 layouts for separate declared model roots |

FSDP roots include their frozen base parameters, but other frozen-only model
roots remain resident. Aliases across separate roots are NOT combined with
FSDP2; unsupported mixed-dtype root groups and non-pointwise optimizer state are
rejected. These restrictions generate fallback candidates, not fake support.
FSDP2's new Parameter identities are installed into the EXISTING optimizer groups;
existing dense Adam moments are distributed rather than initialized to zero.

All choices have measured per-call costs from paired executions. The same
`search/service.py` ranks candidates under the declared per-device capacity.
The planner adds native forward/backward-completion dependencies before search;
an order that is only mathematically valid is insufficient for a stateful reducer.
Rank zero chooses and publishes one plan. The executor reconstructs the native
constraints and all ranks agree on identity/order before model collectives.

### Runnable CPU example (no models downloaded)

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,native]'
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/distillctl.py workloads native program auto \
  --factory examples/open_workloads/replicated_input_gradient.py:build \
  --parameters examples/open_workloads/replicated_parameters.json \
  --distributed gloo --group-timeout 180 --output /tmp/new-program-run \
  --max-candidates 8 --samples 1 --max-program-runs 11 \
  --memory-limit-bytes 10000000 --execute
```

The example has random Conv3d models, a frozen input-differentiable critic and
actual Adam. `width` is feature channels; `height`/`image_width` are spatial
sizes. Change these, `frames` or `updates` in the JSON: the actual tensors and
logical calls change and costs must be recalibrated. No launcher, capability or
planner source edit is required. An unsupported arbitrary Python program is
not guessed into this ABI. Factories are trusted explicit user code.

`plan --calibration ... --memory-limit-bytes ... --output ...` is metadata-only
and can execute without torchrun or constructing a model. `describe`, `calibrate`
and `run` remain available individually. No `--execute` means no factory/group
execution. Each fresh `auto` uses an explicit total program-run budget including
the reference, candidate repetitions and selected-plan confirmation. Limited
candidate budgets are labeled incomplete enumeration, not global optimality.

For allocated CUDA ranks, `--distributed nccl` sets LOCAL_RANK before the factory;
the factory must return tensors on that rank's CUDA device. CUDA/NCCL source
paths exist but have NOT been GPU-validated in this work. Do not use CPU timing
or this example as video-model speedup or training-quality evidence.

### Evidence and limits

Tests use actual 2-process Gloo, native DDP/FSDP2, autograd and Adam, not mocked
collectives. They compare parameters and optimizer states to independently
concatenated single-process global-batch references. Two trainable roles retain
separate adapters/optimizers; frequency and input-shape changes are tested.
Tests also exercise public torchrun -> measured calibration -> common DP ->
selected execution, rank-specific stale input rejection and completion edges.
A separate measured FSDP candidate is replayed even when the CPU timing winner
is a replicated strategy; that replay is not called an automatic FSDP win.
Four-process CPU/Gloo coverage also exercises HSDP2 on two **logical** nodes on
one host, including warmed Adam moments, a global-batch reference and measured
common-plan replay (`test_open_workload_hsdp.py`). This does not establish
physical multi-node NCCL behavior or A100/H100 GPU validity (NOT_RUN).

CPU FSDP tests require Torch >=2.10. The distributed CI pins 2.10 so these tests
are executed rather than silently skipped. Older CPU environments retain the
replicated path and explicitly decline FSDP2. Original canonical paths are not
migrated to a newer Torch runtime by this test workflow.

This is still an admitted finite-program executor, NOT an unrestricted production
trainer: HSDP2 requires complete equal multi-rank node groups, and this callable
data-parallel path does not automatically lower arbitrary role pools or general
activation offload. Bounded role pools have a separate
[RoleTask/RoleFunctions contract](open_role_pools.md). Mid-run topology changes
and resumable streaming training remain unsupported. Factories initially
construct dense state and calibration clones it; sharding does not yet solve
large-model initialization capacity. CPU capacity uses storage accounting with
explicit DDP/FSDP staging allowances, not an RSS bound. GPU allocator behavior,
full pretrained workloads and end-to-end throughput remain unverified.

Calls stay serial across the data-parallel group. DDP overlaps gradient
communication WITHIN backward; it is not Generator/Score pipeline overlap.
Parameter `.backward()` and optimizer updates must be separate semantic calls;
input/latent gradients are not averaged. Unequal per-rank batches, hidden
stateful closures, sparse/higher-order parameter gradients and undeclared
side effects require their own dependency contracts.

### A100/H100 hardware-specific calibration (opt-in public backend)

For an already allocated A100/H100 CUDA/NCCL group, the same `program auto`
command performs bounded native candidate calibration on the **current** GPUs.
It fingerprints GPU family/SM (A100 SM80, H100 SM90), total VRAM, device UUID,
Torch/CUDA/NCCL runtime and allocated rank topology. Cost receipts cannot be
replayed on another accelerator/environment without new measurements. GPU
family is a **capability identity**, not an instruction to pick a fixed winner.

Actual per-call costs remain the synchronized **host wall** times (including
completion fences and measured NCCL effects); separate CUDA events expose
on-stream GPU execution time. Automatically collected, scratch-only 4 MiB
pinned H2D and 1/8 MiB NCCL all-reduce samples are saved under
`calibration.json:hardware_probe`. Those microbenchmarks **never** replace
actual generator/backward/optimizer cost data or claim to represent an
end-to-end communication critical path. Use `--no-hardware-probe` to disable
the diagnostic probes while preserving actual candidate timing.

In CUDA `auto`, `--memory-limit-bytes` is optional. The default budget is the
minimum per-rank currently free + current process allocator-reserved reusable
VRAM, minus a 2 GiB reserve. Use `--gpu-reserve-gib N` (0..16) to adjust
reserve, or provide an explicit smaller `--memory-limit-bytes`. The budget and
per-rank snapshot are retained in the plan; execution checks current real
headroom again before attempting a selected CUDA plan. CPU still requires an
explicit memory budget (tracked tensors are not whole-process RSS).

These features do NOT supply an H100-only kernel, automatic role-pool lowering
of arbitrary Workload graphs, layerwise bandwidth roofline, full GPU Nsight trace or verified
A100/H100 numerical throughput. Hardware-specific strategy selection is
conditioned on **real** candidate measurements, not assumed winners. Mixed
GPU models within a rank group and inter-node transport require dedicated
admission and GPU verification before performance claims.

#### Deep trace only after selecting a validated plan

`--deep-profile` on `program auto` or `program run` wraps only the selected
execution, not every calibration candidate. It emits a per-rank PyTorch Kineto
Chrome/Perfetto trace (`torch-profiler-rankXXXX.json`) with logical-call labels,
CPU/CUDA operator activity and native memory events. Paths and SHA256 digests
are embedded in `execution.json` and never reused as speedup evidence. This
flag adds instrumentation overhead; run a separate **unprofiled**, matched
train/optimizer/EMA validation when reporting speedups. CUDA traces require
supported local PyTorch/CUPTI installation; if unavailable, fail explicitly.

The `--deep-profile` flag is not a substitute for Nsight Systems on multi-GPU
critical paths or Nsight Compute on kernel roofline/MFU. Those require external
tools and valid access to the actual A100/H100 cluster. MFU is not inferred
from torch.profiler's partial FLOP coverage.

#### Profile-guided bottleneck and upper-bound report

A metadata-only `program plan` now emits `opportunity_report` for its selected
measured candidate: per-call median wall time, share of the currently SERIAL
plan, maximum-to-median rank slowdown and per-call Amdahl *what-if* bounds for
halving or eliminating that call. Binding-level cost shares reveal whether the
large opportunity is in a forward, backward, communication-containing update,
or data movement; the report does not claim to isolate NCCL without profiling.
Amdahl numbers use the **observed**, single-allocation model and assume all
other work unchanged. They are not measured speedups, not the upper bound of a
future cross-role overlapping plan, and do not count incomplete FLOPs as MFU.
Selectively run `--deep-profile` for the most expensive unclear call, rather
than optimizing small kernels opportunistically or tracing every candidate.

##### Standalone A100/H100 probe without loading 14B model weights

On an existing single GPU: `python scripts/distillctl.py workloads native
program hardware --output /shared/gpu-profile.json --execute`. For multi-GPU
communication, launch the **same command** under an existing 2-node torchrun
allocation with `--distributed nccl` and one local GPU per rank. It performs
bounded scratch H2D and NCCL all-reduce operations, collecting a per-rank
hardware-fingerprint/probe receipt. It does not initialize Wan/Helios/H3
weights, attempt any optimization, or pretend that a microbenchmark is a
complete training-workload cost model. CPU testing is supported but labeled
`NOT_GPU_PROFILE` rather than claiming an A100/H100 measurement.

<a id="open_workload_dmd_builder"></a>
## User-owned DMD math, shared open-program planner

`distillschedule.open_workload.dmd.build_dmd_program` builds the existing
`Program`/`Call` graph for a **bounded Student1/FakeN detached-score DMD loop**.
It is a semantic onboarding helper, not another solver or a model registry.
`build_dmd_workload` additionally binds user models and optimizers into an
ordinary `Workload`; return that object from the same external factory used by
`workloads native program auto`. No CLI, calibrator, capability or planner changes
are needed. Factory construction still requires the existing explicit execution
authorization. Nothing here downloads weights or allocates GPUs.

### Supply math, not a winner

The caller supplies inspectable functions (or `DMDHook(function, reads=...,
writes=..., replay_safe=...)`) in `DMDHooks`:

| Hook | `dmd_inputs(ctx)` semantic values, in addition to `feeds` | Gradient mode |
| --- | --- | --- |
| `student_forward` | optional prior-cycle `history` | grad for Student update; no-grad for fresh Fake samples |
| `history` (optional) | detached `current`, optional `previous` history | no-grad |
| `noise` | detached `generated`, optional current `history` | no-grad |
| `teacher_forward` | `generated`, `noise`, optional `history` | no-grad |
| `fake_forward` | `generated`, `noise`, optional `history` | no-grad for Student scoring; grad for Fake training |
| `student_loss` | **live** `generated`, detached `noise`/`teacher`/`fake`, optional `history` | grad |
| `fake_loss` | detached `generated`/`noise`, **live** `fake`, optional `history` | grad |
| `student_backward`, `fake_backward` | `loss` | grad; defaults to `loss.backward()` |
| `student_update`, `fake_update` | `backward` completion token | no-grad; defaults to optimizer step then zero-grad(None) |

Every hook also receives `ctx.attributes['cycle']`, `['phase']` (`student` or
`fake`), and `['fake_index']` (None for Student). `feeds` is the complete supplied
feed mapping. Hook return values may be tensor trees/mappings, so noise packages,
conditioning, conditional/unconditional predictions and custom objectives are
not forced into a particular tensor signature. The builder contains **no**
noise schedule, DMD gradient formula, guidance scale, loss reduction or surrogate
objective. The CPU tests bind the existing `runtime.dmd_boundary` math explicitly.

Minimal factory wiring, after constructing the user's models, optimizers, inputs
and Context hooks:

```python
from distillschedule.open_workload.dmd import DMDHook, DMDHooks, build_dmd_workload

hooks = DMDHooks(
    student_forward=DMDHook(my_student, replay_safe=True),
    fake_forward=DMDHook(my_fake, replay_safe=True),
    teacher_forward=my_teacher,
    noise=my_noising_contract,
    student_loss=my_dmd_objective,
    fake_loss=my_denoising_objective,
    history=my_previous_current_history,  # optional, output must be read-only
)
workload = build_dmd_workload(
    student=student, fake=fake, teacher=already_frozen_teacher,
    student_optimizer=student_optimizer, fake_optimizer=fake_optimizer,
    hooks=hooks, feeds=feeds, cycles=2, fake_updates=5,
    fake_sample_policy='fresh_student', history_feed='initial_history',
    # replica_batch_size=local_batch_size,  # optional equal-rank-mean contract
)
return workload
```

Inside a hook, use `dmd_inputs(ctx)` instead of hard-coding graph call IDs, and use
`ctx.state['student']`, `['fake']` or `['teacher']` for models. Context exposes only
declared/inferred state reads. Do not capture another model, optimizer or tensor
in a closure/global. `replay_safe` is a caller purity/RNG assertion, not automatic
proof; default is false. Optional hook state effects must name `extra_state`
entries (or other existing logical state) explicitly. Extra state must use the
normal backend's supported signature types (tensors, modules, optimizers,
mappings/sequences and scalar data), not opaque scaler/scheduler objects. State
that cannot be represented by that ABI requires another backend/contract. The
default updates do **not** silently add clipping, scaling, EMA or decay changes.
Preserve such algorithmic behavior in declared custom hooks.

For direct ABI use, call `build_dmd_program(hooks=hooks, feed_names=tuple(feeds),
...)`, then bind `hooks.bindings(program)` and all declared state in `Workload`.
The reserved logical state names are `student`, `fake`, `teacher`,
`student_optimizer`, `fake_optimizer`, `student_grad`, `fake_grad`. Optional
`initial_versions` bind the current logical versions; default is zero.

### Two explicit, different workload contracts

`fake_sample_policy` is required, never selected as an optimization:

- `reuse_student`: each cycle's Fake updates reuse the detached Student sample
  from **before** that cycle's Student optimizer update.
- `fresh_student`: after the Student update, every Fake substep calls the updated
  Student under no-grad. Its noise/forward/loss hooks receive that fresh sample.

Student scoring reads the current Fake version. Fake updates have their own
separate backward and optimizer calls. Teacher has no updates. G/F/T can have
different internal widths/depths; compatible latent/noise/loss I/O is the caller's
contract, not inferred from architecture names. The helper rejects a trainable
Teacher or optimizers that do not exactly own each role's trainable parameters.

### Detached previous/current history, not causal-cache resume

An optional history hook combines the detached current Student output and prior
history. All history/noising/scoring outputs cross explicit no-grad graph nodes;
the backend recursively detaches returned tensors. Student's live output is
passed separately to its loss. History returned by the hook is read-only and may
share storage: this is **detach**, not an implicit clone or permission to mutate
the live activation. Student consumes the *previous cycle's* history; fresh Fake
rollouts use that same previous context. Their temporary histories do not replace
the main history that feeds the next cycle. Final main history and every scalar
loss are program outputs.

There is no persistent KV cache, causal attention state reconstruction, checkpoint
resume, or hidden multi-call buffer lifecycle. A real persistent causal workload
needs its own declared state and parity proof. `build_dmd_workload` also requires
all tensors nested in the supplied initial `history_feed` to have
`requires_grad=False` and no `grad_fn`; even a grad-enabled leaf is rejected at
construction. It never silently detaches caller-supplied external history graphs.
External autograd graphs and in-flight gradients are still rejected by the
existing backend. The compiler
rejects backward/replay across a changed live model version; no-grad history must
not be relabeled differentiable to bypass that guard.

### Validation and limits

`tests/test_open_workload_dmd_builder.py` specifies two complete CPU cycles with
Student1/Fake1 and Student1/Fake3, both sample policies, cross-scale model widths,
frozen Teacher, detached histories, real DMD boundary math, parameter updates and
full Adam state against an independent sequential loop. Ordinary no-history
cases exercise both sample policies through that same oracle. It also exercises
baseline/full-checkpoint variants, nested initial-history admission and stale
provenance rejection. These are test specifications, not an execution receipt.
It adds no
target-GPU, large-model, distributed-numerical or performance claim.

The helper instantiates at most 511 calls, as required by `Program`. It describes
this explicit loop, not all DMD variants. GAN branches, other update orderings,
dynamic ratios, differentiable score paths, stateful forward-buffer mutations,
nonstandard backward/optimizer lifecycles or additional Teacher updates may need
direct `Program`/`Call` construction. The existing external factory path remains
available for those real workloads; no registry/solver edit is necessary.
