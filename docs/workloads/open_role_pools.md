# Generic two-role training: capability boundary and current progress

The workload provides **training facts**, not a layout: a generator function,
score objective, optimizer factories, input/condition loading and explicitly
versioned update boundaries. `RoleTask.program()` compiles producer forward,
latent transfer, score input VJP, gradient return and generator parameter commit
using the same versioned `Program` graph. An optional Score optimizer commit is
strictly after all Generator VJPs; currently the executable Score train path
supports **one** such update, not arbitrary GAN/DistillSchedule protocols.

`role_program.enumerate_layouts()` derives contiguous G/Score rank splits and
bounded pipeline-window candidates from **allocated rank capacities**. The
factory supplies no 8+8 rule, no optimization rankings and no model-name
whitelist. Unsupported topology/size combinations are rejected, not remapped.

`RolePoolExecutor` is an actual two-role distributed **execution primitive**:
G and Score models are created only on their assigned ranks, explicitly
initialized process groups own each role, tensor activation is sent from G to
Score, Score computes `autograd.grad(score_loss, latent_leaf)`, the resulting
input-gradient is sent back, and G applies `latent.backward(VJP)` to its own
original forward graph. Generator parameters are never stepped until ALL input
VJPs have completed. Role-local gradients are SUM-reduced after normalization by
global logical microbatch count. A separate optional Score parameter update
runs after the Generator commit. Nonblocking point-to-point sends and a bounded
in-flight G window enable G-forward/Score-evaluate overlap on disjoint ranks.
NCCL requires tag-free send/recv; messages are ordered by each source/target
pair. Runtime enforces output shapes, dtype, finite values, disjoint physical
pool ranks and agreement among role-model replicas.

**Verified:** actual 2-, 3- and 4-process CPU/Gloo transport, 1:1 and asymmetric
pool splits, pipeline windows 1/2, both frozen-guidance and separate Score
optimizer paths. Final model tensors are checked against an independently
computed single-process global-microbatch reference. These are real PyTorch
forward, input autograd, backward, Adam and point-to-point collectives, not
mocked callbacks.

**Not yet:** this primitive is not by itself the common cost-driven selected
plan. Production factory/CLI onboarding, per-layout cost calibration, measured
search/selection, native CUDA/NCCL validation, cross-node 16-GPU performance,
full original model semantics, CPU offload/prefetch, and asynchronous
communication/copy profiling require their own admission. Do not call the
supported CPU transport proof a measured A100/H100 speedup.

## Measured common-search integration

`role_planning.calibrate()` enumerates budgeted role pools/windows from the
current allocated rank capacities, constructs only the model for each rank's
selected role, and performs actual full-cycle Gloo/NCCL role execution. All
rank receipts include native model-state identity, model/optimizer storage and
wall time. Observed per-layout full-cycle wall durations and a 0-duration finish
are stored as content-hashed `CostModel` component receipts; calibration fails if
repeated initial-state replays diverge. A declared input/cohort identity and
inspectable callable source are mandatory. Old costs must match source, hardware,
training semantics and data identity.

`role_planning.search()` uses the EXISTING shared `search.service.select` and
event-frontier solver on measured **whole-cycle envelopes** plus source-bound
physical G/Score state, declared transport occupancy and per-rank capacity. The
native execution supports bounded FIFO windows; it does **not** lower arbitrary
DP event interleavings. Internal G/Score overlap comes from native nonblocking
communication and is already INCLUDED in observed cycle wall time. The planned
cycle is not the sum of serial G/S call times, and no synthetic overlap discount
is applied. `run_selected()` verifies the ranked choice and source-bound receipts
before executing the selected real native worker. Hardware changes invalidate
these costs rather than assuming A100 and H100 have identical winners.

The measured cost is a *diagnostic native cost*, not a verified speedup: the
original pretrained-model numerical reference, model-specific DMD quality,
optimizer/EMA trajectory and CUDA/NCCL performance still require independent
validation. Tensor/storage accounting does not bound all unseen autograd
activations, CUDA allocator behavior, future changing inputs or nonlinear
transfers. This new role interface does not automatically translate original
Wan, Helios, H3 or LiveWan training semantics; those require adapters that
declare and invoke their **actual** original functions.

## Saved-activation CPU offload and prefetch (opt-in candidate)

`PoolLayout.activation_policy` is a planner choice (`resident` or
`cpu_offload_prefetch`), not a workload hint. Admission requires explicit
positive CPU host capacity on each producer rank. A `HostBudget` guards the
aggregate actual saved-tensor bytes across all in-flight Generator graphs.
PyTorch `saved_tensors_hooks` stages saved intermediate tensors to CPU without
relocating resident module weights, and checks in-place version mutations where
the original storage is still alive. On CUDA, an explicit copy stream writes to
pinned host memory and records copy completion events; a producer may begin
prefetching retained saved activations before waiting for the Score VJP. The
backward stream waits on completion, and pinned leases are released only after
all associated transfers are complete. The Score input-gradient still travels
via explicit native P2P and is applied to the original Generator graph.

CPU/Gloo tests exercise the same hook and return-gradient math against the
independent global-batch Adam reference. CPU copies are synchronous correctness
fixtures: only CUDA can exercise actual async pinned-transfer overlap. GPU/NCCL
A100/H100 validity, pinned-memory budget under the host NUMA topology and
large video attention replay are **NOT_RUN**. No fallback silently enables
reduced batch/steps or changes Score gradient semantics when capacity is absent.

`role_planning.calibrate(..., allow_offload=True)` considers offload only with
compatible host resource facts, measures it as a distinct native layout, and
rejects a candidate unless saved tensors were actually packed, prefetched,
and used in the backward on the producer ranks. GPU residency predictions from
tracked role state and declared transport are conservative **in scope**, not a
complete upper bound for unobserved intermediate autograd/allocator memory.

## Public new-workload entry (factory, parameters, no optimization recipes)

For a new conforming two-role training algorithm, implement a trusted local
`build(parameters)` factory that returns `(RoleTask, RoleFunctions)`. Functions
specify model constructors, optimizers, data/cohort identity, generator forward,
Score objective and (optionally) a single later Score parameter update; they do
**not** prescribe pool sizes, checkpoint/offload policies or scheduler winners.
`role_storage_estimate_bytes` supplies only conservative per-role static memory
facts so the planner can reject impossible model placements before execution.
It does not guarantee CUDA activation or allocator peak capacity.

An example previously unknown to the planner is
`examples/open_workloads/unseen_score_pipeline.py:build` with parameters in
`examples/open_workloads/unseen_score_pipeline.json` (small Conv3D, no training
speed/quality claim). To describe the semantic graph (trusted factory is run):

```bash
python scripts/distillctl.py workloads native program role-describe \
  --factory examples/open_workloads/unseen_score_pipeline.py:build \
  --parameters examples/open_workloads/unseen_score_pipeline.json --execute
```

To calibrate, search and execute automatically under an **already allocated**
CPU/Gloo group:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 scripts/distillctl.py \
  workloads native program role-auto \
  --factory examples/open_workloads/unseen_score_pipeline.py:build \
  --parameters examples/open_workloads/unseen_score_pipeline.json \
  --distributed gloo --memory-limit-bytes 10000000 \
  --host-memory-limit-bytes 10000000 --max-candidates 3 \
  --samples 1 --max-native-runs 4 --output /tmp/role-auto --execute
```

`role-auto` emits content-bound component receipts, a common frontier/DP plan,
selected runtime evidence and a summary; changing frames, call count or the
inspectable factory invalidates previous calibration. `role-plan` can operate on
calibration metadata without loading models. `role-run` verifies the saved
search/identity again before executing it. `--disable-offload` excludes CPU
saved-activation offload; without `--execute` no user factory or model is run.
On already allocated A100/H100 multi-rank NCCL, use `--distributed nccl`
and supply device-appropriate CUDA model/data factories; automatic VRAM headroom
uses the reserve, but cross-role CUDA P2P, pinned offload and 14B training are
**NOT YET GPU VALIDATED**. No spontaneous GPU allocation or scheduler submission
is performed. This CLI represents **one** supported producer-latent→Score-VJP
family, not a compiler for arbitrary GAN/DiffSynth/H3 video protocols.

Automated CPU integration includes a genuinely new Conv3D workload whose input
sizes and microbatch frequency are changed without editing planner logic, real
two-rank Gloo producer/Score P2P, optimizer work, measured candidate selection,
and rejection when changed task/data attempts to replay old plan receipts.
This is an interface/execution validation, not an independent full model DMD
numerical or video-quality reference.

## Bounded candidate coverage on a 16-GPU allocation

Small calibration budgets must not be interpreted as searching only the first
contiguous split. Formerly a four-candidate limit tried only variations of
`1G+15Score`, excluding `8G+8Score` regardless of the hardware and workload.
The generic enumerator now first checks the **actual per-rank role capacity**,
then deterministically samples across generator/score fractions, in-flight
windows and available host-offload policies using farthest-point coverage.
It starts with a feasible approximately-balanced resident window-1 baseline,
then maximizes diversity in the bounded family. Neither a balanced split nor
an offload choice is assumed to be fastest; every admitted choice must be
calibrated and numerically checked. The remaining legal choices are reported as
`SEARCH_BUDGET`, not conflated with `ROLE_CAPACITY` or `HOST_CAPACITY`.
A bigger explicit `max_candidates` budget enumerates all legal combinations.
Supported physical topology remains contiguous rank cuts; arbitrary role
reshuffles, hetero-GPU model remapping and dynamic role migrations are not
implemented. CUDA/NCCL effectiveness requires real A100/H100 profiling.

## Opt-in Generator-pool FSDP2 (bounded native prototype)

The same role-pool runtime can now instantiate a real PyTorch FSDP2 mesh
inside the Generator pool while Score stays replicated. The explicit P2P
Score-latent VJP returns to the original Generator graph. Optimizer parameter
identities and initialized Adam moments are migrated using the existing
generic FSDP2 helper. All role ranks create the DeviceMesh in a common order.

Admission requires more than one Generator rank, equal per-rank microbatch
counts, a window that holds all local forwards before any backward, and
resident saved activations. Only homogeneous dense, contiguous parameters
and supported pointwise optimizer layouts are admitted. This conservative
phase-order restriction prevents divergent FSDP collective schedules.
FSDP2's default group-average is compensated exactly once at the Generator
VJP backward boundary. A second gradient allreduce is forbidden.

The four-process CPU/Gloo test checks sharded params, score P2P, true input
VJP and Generator/Score Adam updates against an independently computed
global-microbatch reference. `allow_generator_fsdp=True` explicitly enables
additional search candidates; the existing default family is unchanged.

This does NOT establish A100/H100/NCCL correctness, heterogeneous BF16/FP32
root grouping, Score sharding, sharded activation offload, or reliable 14B
allgather capacity. Candidate admission retains the full dense Generator
footprint until device-specific memory peaks have been measured.

### Bounded role-FSDP2 calibration and communicator lifetime

`role_planning.calibrate(..., allow_generator_fsdp=True)` admits the
above strictly defined FSDP2 alternatives into the **same measured candidate
search**; the public `role-auto --allow-generator-fsdp` flag enables it.
Mixed-dtype roots are still refused and all unvalidated GPU capacities retain
the conservative dense full-parameter memory requirement.

The sharded Generator DeviceMesh reuses the already created role group; it
does not create a second communicator. PyTorch 2.10 can cache DTensor mesh
names across calibration arms, so a bounded sharded Generator group is kept
alive until the parent `torchrun` WORLD is destroyed, instead of prematurely
removing its named process group. An independent 4-process/Gloo test runs
calibration across different role layouts, then a real sharded candidate
through two fresh-input cycles, verifying final model and Adam presence.
This does not yet implement model-weight restart checkpoints or A100/H100
CUDA validation.
