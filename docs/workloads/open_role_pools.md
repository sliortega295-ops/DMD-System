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
