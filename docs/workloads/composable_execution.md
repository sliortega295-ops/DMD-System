# Composable native execution: incremental acceptance ledger

This extends PR #28. It does not replace the five canonical profiles or claim
completion of the full five-workload planner objective.

## Implemented: common prepared actions

`runtime/cycle_plan.py` emits the existing `ExecutionGraph` representation:
one G update, five separately prepared B1 rollouts, five ordered C commits,
and the workload's explicit EMA positions. A retained-rollout window 1..5
generates legal preparation/update interleavings without recipe-name lookup.
Preparation order remains fixed, preserving operation-qualified trace ownership.

`StaticRoleRunner.prepare` compiles and validates the selected order before the
cycle timer: dependencies, required/produced versions, handlers and input
lifetimes. A one-shot execution releases each latent/context after its last
consumer. It adds no Torch operation, synchronization, collective or value read.
Dispatch receipts alone are not model-correctness evidence. Cross-pool actions
without a native transport lowering are rejected, not silently skipped.

## Implemented: reusable cost identity

`ComponentContract` binds operation family, work/shape/calls, precision,
placement, local implementation and resource/residency context. It does not key
on an entire candidate or an absolute cycle/operation ID. Source, assets and
hardware remain bound by the existing `CostModel` and content-hashed receipts.
Changing concurrent NCCL, cold optimizer state, shape, precision or local kernel
requires new calibration. Serial measurements do not prove contention costs.

CPU fixtures demonstrate reuse/rejection, **not measured prediction accuracy**.
Native component calibration and held-out schedule prediction remain unfinished.

## Native Wan action binding

New opt-in IDs `joint-wan2-{fsdp|resident}-c{chunk}-e{0|1}-w{1..5}` carry the
window through profile resolution into both trajectory and two-cycle timing
workers. For example, window 3 lowers to:

```
G; R0 R1 R2; C0 EMA0; R3 C1 EMA1; R4 C2 EMA2; C3 EMA3; C4 EMA4
```

The adapter calls the original G loss/update, exact-trace B1 rollout, Critic
loss/backward/optimizer and EMA helpers. No model mathematics is reimplemented.
Logical batch16, source steps10..14, G1/C5/EMA5, precision and trace-derived DiT
exits/calls remain native gates. Window is retained-context depth, not batching.
Persistent-G unsharding is rejected until represented as a multi-action scope.
Default/`joint-wan1` execution remains unchanged. New IDs cannot consume legacy
14-stage calibration as if it were independent action costs.

Both rank summaries additionally verify the exact dispatched order, engine
versions, retained depth and complete receipts. Failed action prefixes and
CUDA/world-fence failures are retained. The two-cycle timing worker prepares
the program before START and materializes its successful receipt after STOP.

## Full goal: not completed

| Workload | Next native requirement |
|---|---|
| Wan | Generated-order world16 validation; measured reusable action costs and held-out search execution |
| LiveWan | Separate actual G/C operations from coupled-commit envelopes; legal pool/wave/transport lowering |
| H3 | Operation-owned RNG/sample bindings for interleaving; shared adapter/recompute exclusion; measured action costs |
| Helios No-GAN | Native action binding and genuine same-process two-cycle trace/version generalization |
| Helios GAN | Same, retaining FSDP-safe GAN gradient service and canonical B16 (not B12 rolephase) |

Backbone sharing, pool sizing, ordering and kernel choices must eventually be
searchable and actually consumed together; registering profiles is insufficient.
User-approved finite numerical drift may be admitted as explicit diagnostic
exploration while retaining original strict metrics. It does not waive sample,
RNG, version, call/update census, finite, source, resource or completeness gates.
