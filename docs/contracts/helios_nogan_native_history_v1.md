# W2 native rank-local amplified history, version 1

This is an opt-in **continuation input/execution contract**, not a new
`OriginalHeliosRequest`, canonical benchmark, saved-plan dimension or trainer.
The normal `original_helios_cohort_stream.v1` route is unchanged.

## Identity

- Input kind: `original_helios_native_history_cohort_stream.v1`.
- Stream schema: `distillschedule.original_helios_native_history_cohort_stream.v1`.
- Cohort schema: `distillschedule.helios_native_history_cohort.v1`; explicitly
  carries `native_policy=source_native_amplified_v1`.
- Execution: `distillschedule.helios_native_history_execution.v1`.
- Checkpoint policy: `distillschedule.helios_native_history_checkpoint_policy.v1`.
- Fixed original W2 geometry: 384×640, 33 frames, BF16 input, physical B1,
  global B16, world16, G1/C5/EMA5; original Step10 G2/C10/EMA10 bootstrap.
- Same complete-cycle checkpoint and same-source/assets/stream/topology resume
  rules as the existing original W2 lifecycle. Phase materializations are
  transient, are released before a checkpoint, and are rebuilt on restore.

A stream's kind, policy, source inventory, source-code fingerprint, samples,
seeds, per-rank controls and full task digest are bound together. Old input kinds
reject new stream/task schemas. Original benchmark request enums are not widened.

## Original sampling and work

The existing native dataset, `wan_capture.prepare_history`, isolated input RNG
streams, block noise and `wan_capture.stage_control` remain authoritative.
Preparation supplies **actual** post-corruption history zero predicates and the
actual rank to the latter. Initial normal exits still agree across ranks. A
zero-history rank then keeps its original independent amplified exits and
*doubled scheduler budget*: 4 rather than 2 at each pyramid stage. Neither the
number of calls nor the rank-local exception is synchronized, padded, dropped,
resampled or replaced by a normal branch.

The shared pure all-rank schedule validator is reused, but its old Resident-only
execution label is not: this contract requires
`FSDP2_ALIGNED_MATERIALIZATION_WITH_ORIGINAL_REDUCE_AND_STEP`. Before model
construction, each worker independently checks its real CPU tensor history
predicate, original seeded exits and exact executed scheduler grid. Runtime
receipts check that worker's actual calls, shapes, BF16-projected timesteps,
backward/commit counts and original update versions.

## Actual execution

Only the existing reference backend is admitted: world16 FSDP2, chunk512, SDPA,
B1/window1, original AdamW/scheduler/EMA, old persistent-unshard flag OFF.
Manual SafeBundle, saved plans, unbound policy overrides and GAN are rejected.

At each aligned Generator graph action all ranks materialize the same original
FSDP root and **40 complete `HeliosTransformerBlock` checkpoint units** in the
same order. Interior PEFT Linear sharding or an equal count of other units is
not equivalent. The installed active checkpoint owner must use the original
`fsdp2_metadata_tolerant_checkpoint` or its admitted original partial.

For G update the phase encloses the entire existing helper: native rollout,
Score calls and G backward. Its engine view releases full parameters before
calling the **original** engine step, clipping, AdamW, scheduler and zero-grad.
Detached Critic-input rollout actions get their own aligned no-grad phase;
the old no-grad-only persistence scope is not stretched over a G backward.
Critic updates and EMA keep their original helpers and order.

Each actual action receipt records acquire/release/forward/backward/commit and
restoration counts with the native task SHA. Exceptions release owned full
parameters and cannot publish a successful phase/complete checkpoint.

## Validation boundary

This source capability does not itself admit a full-size world16 run or
establish memory capacity. Full Generator materialization must fit alongside
original Score state and activations under unchanged resource checks. Target
Torch2.10 CPU integration and actual GPU capacity/numerical validation remain
separate from local metadata/protocol fixtures. No performance result is implied.
