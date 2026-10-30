# W2 native-history + full SafeBundle continuation v1

Explicit preset: `helios_nogan.native_history_safe_ed8.v1`.
Example: `recipes/helios_nogan/training.native-history-safe.json`.

This version composes the existing original rank-local history input ABI with
node-HSDP 2×8 and the **entire** `safe-nogan-ed8-v1` bundle. It does not change
`safe_stream_ed8.v1`, the reference/native-history policy, a benchmark plan, or
the original sampler/training helpers. No persistent-unshard, merged LoRA, old
stride2 attention replay, extra padding, truncated exits or new random draws.

## Physical and algorithm contract

- Original 384×640×33, BF16 inputs, physical B1/global16, G1/C5/EMA5 per cohort.
  The source-qualified Step10 G2/C10/EMA10 bootstrap and original Adam/LR/EMA
  remain mandatory; this is not a cold bootstrap or quality-training claim.
- Source-qualified `original_helios_native_history_cohort_stream.v1`, including
  normal synchronized initial exits and rank-local amplified exits with the
  original doubled denoising budgets. The prepared finite horizon is immutable.
- Exactly the existing SafeBundle implementation: node-HSDP row-major 2×8,
  C512, rollout window5, p0, stride0, pinned local FA2, original scale1/add,
  norm/residual/LoRA fusion, dispatch warmup and LoRA checkpoint replay scope.
  All original installation checks and **per-cycle positive effect deltas**
  remain required. Merely selecting the preset cannot satisfy that census.

## Collective-safe actions

The existing `fsdp2_variable_forward_phase` owns the same 41 FSDP units: root
and 40 **complete original Helios checkpoint blocks**, never interior Linears.
Before each aligned G action or detached `r0`…`r4`, every rank explicitly
materializes these units in the same order on its original HSDP shard mesh.
Unequal rank-local no-grad prefix lengths therefore do not issue parameter
all-gathers within that variable region. The original checkpoint/backward and
HSDP gradient reduction remain authoritative. G has one original backward and
commit; full parameters are released **before** original clipping/Adam/LR.
Detached actions do not update and release at action exit, including failures.
The ordered window5 graph is unchanged: all five rollouts precede the first C
update, and each output is freed after its actual consumer. No `r_all`/GAN
persistent owner is borrowed or nested.

The runtime additionally verifies the real FP32 DTensor 2×8 ownership before
using this capability. It records new phase schema, mesh, task digest and actual
acquire/release/backward/commit census. Full-safe native actions reject a
reference-phase receipt, even if its numerical counters happen to match.

## Restart and current output

New training identity:
`distillschedule.helios_native_safe_stream_identity.v1`.
New policy:
`distillschedule.helios_native_safe_stream_checkpoint_policy.v1`.
Both bind the exact new execution descriptor, source fingerprint, immutable
assets, input stream/horizon, per-rank state/layout and cursor. The existing
all-rank atomic checkpoint, state restore, stop control and output-only owner
are reused. No temporary full materialization, rollout or kernel cache is
serialized. Old reference/SafeBundle checkpoint identities are not relabelled.
Regenerate only source-bound stream metadata/controls through the existing
producer from retained inputs and **the same** original config and seeds; never
rehash old-source traces into a current stream or resample exceptions.

## Qualification

Local CPU tests exercise actual metadata producer/consumer, typed lowering,
worker identity, construction capability, phase dispatch, failure cleanup,
checkpoint publication and refusal. A separate actual Torch CPU16 2×8 HSDP
component test has 40 tiny complete blocks, rank-local unequal forward counts,
checkpointed autograd, original gradient reduction/Adam/scheduler and five
no-grad phases. It is not the original full Helios model or GPU kernel proof.

Pinned target Torch2.10/upstream dependency tests, actual full-size source-bound
input admission, FA2/LoRA/norm effect census, native HSDP capacity and C1/save →
fresh-process C2/current-output remain **NOT_RUN**. CPU/scalar/synthetic kernel
fixtures cannot promote them. Existing accepted long trajectories are retained,
not repeated or reclassified by this opt-in source integration.
