# Wan parameterized native-action bridge — local checkpoint, 2026-10-12

## Status
Implementation and CPU handler tests only. NOT pushed to GitHub. No GPU run.
Starting upstream source: `088b8ad53f16051577aebb5d75eb6e2f5370fc13`.

## Completed step
`runtime/wan_cycle_plan.py` now has one native handler binding implementation
shared by the unchanged Full5 entry and an explicitly contracted parameterized
entry. Both call the original `_run_nogan_generator_update_p0`,
`_precompute_critic_rollouts`, `_run_nogan_critic_precomputed`, and `ema.step`.
No Euler protocol, model, loss, optimizer or new launcher was added.
Critic result retention and collection now use the declared count, not 5.

The new entry checks the original no-GAN protocol, source-step identity,
explicit EMA positions, physical B1, world16, nonpersistent unshard and
operation-qualified trace declaration. It rejects truncating a Full5 trace to
pretend it describes a new task. The original generator helper still assumes
source entry 10, so the new entry explicitly preserves that limit.

## Validation
87 tests passed: new parameterized bridge tests plus existing Wan prepared
execution and parameterized-cycle tests. Tests use CPU native-handler fixtures,
not Wan tensors or distributed collectives. They cover counts 1/2/3/5/8,
window 1/full, EMA all/none/last, Full5 dispatch equality, source/backend
rejections and missing native optimizer commits. These tests do not establish
original-model numerical parity or new GPU parameter support.

## Remaining work (critical)
The parameterized entry is not yet exposed by production launchers. Existing
StaticWorld16Trace, worker census, import-bound helpers and native calibration
still have Full5/shape assumptions. Add an actual validated parameterized trace
adapter, then connect existing placement/async-reducer/chunk/epilogue choices
before construction, corresponding cost collection and worker execution.
Do not mark new shapes or update frequencies GPU-supported merely because
this binding accepts a metadata contract. Do not claim all targets achieved.

## Resume discipline
Restore authenticated GitHub write or the development workspace connection.
Inspect remote branch and apply this isolated patch without overwriting others.
Commit/push/verify this step before another large edit. Preserve canonical
profiles, old evidence identities and the naive-FSDP2 baseline. No force push.
