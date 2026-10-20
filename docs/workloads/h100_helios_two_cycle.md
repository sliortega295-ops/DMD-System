# H100 Helios C2 runtime unit — CPU/source verified, hook not installed

This is an executable orchestration unit around the original native helpers,
not a calibrated cost model or a claim of completed H100 execution. No H100
connection, GPU work, source commit, campaign restart, or native checkout edit
was performed. The earlier Full5 bridge remains frozen and **does not** accept
this new runtime as an executable/cost-admitted C2 plan.

## What executes when an authorized future caller installs the hook

`distillschedule.runtime.h100_helios_two_cycle.run_two_cycles` receives the
already-built native Generator/Score engines, their optimizers/schedulers,
EMA, trace, policy, negative embedding and diagnostics. It calls the exact
8fc logical helper bodies, without calling `main`, rebuilding or reloading:

1. `_run_logical_generator_update_p0` once per cycle.
2. `_precompute_logical_critic_rollouts` once per cycle, each retaining the
   native two independent B1 scopes, slots `(rank, rank+8)`.
3. `_run_logical_critic_update_p0` for source steps 10 through 14, with the
   original EMA recurrence after every committed Critic update.

These native helpers use the original `MicrobatchCommitEngine`: each B1 loss
is divided by two exactly once; only the second B1 commits the optimizer.
Native engine/optimizer/scheduler/EMA methods are temporarily wrapped solely
to count their **actual calls**, then restored even on exceptions. Per cycle,
each physical rank must execute G backward2/commit1/optimizer1/scheduler1,
C backward10/commit5/optimizer5/scheduler5, and EMA5. Eight ranks still produce
eight receipts, not sixteen manufactured rank receipts.

Actual physical and logical No-GAN counters must agree and advance:
`G2/C10/EMA10 → G3/C15/EMA15 → G4/C20/EMA20`. Object identities remain the
same across cycles. Both cycles replay the immutable source cohort 10–14;
this is **not** a claim of new step15–19 samples or source-faithful fresh RNG.
Native trace input/noise identities are audited unchanged per slot and call.

The adapter preserves the manual within-cycle latent lifetime, releases those
outputs before the final cycle fence, and rejects any outstanding hardened
Critic trace-cache entry before entering C2. It never resets model, optimizer,
scheduler, EMA, or global RNG state. The native 41-unit retain/restore scopes
and recorded interior fence behavior must match the selected registered arm.

## Source and policy binding

Only the three original `best`, `h100-nograd-retain`, and
`h100-nograd-retain-nosync` policy fingerprints are accepted. They retain
local-FA2, stride2 selective replay, HSDP2×4, CUDA EMA/residuals, original
384×640 trace/latent1×16×9×48×80 and unpacked Score. The newer planner's
stride0 recompute and FA3 are **not** silently substituted or declared equal.
The prior numerical REVIEW remains REVIEW.

The module checks the original worker SHA after removing only the exact hook,
three logical-helper source hashes, native auditor/commit-engine/math/trace/
profile dependency hashes, and clean native Git status. The actual new head,
worker hash, adapter hash, and dependency identities are recorded. A clean
source with these files is necessary, but is **not** a registered deployment or
GPU authorization; a new launcher/auditor source-admission gate is still needed.

The original native CPU auditor is reused for complete per-call trace identity,
shape/dtype/autograd/denoise census and persistence verification. Its JSON-key
normalization is performed only after the measured region. Missing child,
fence, call, slot, or version fields fail closed; unsupported geometry requires
a new explicit adapter rather than a loose override.

## Timer and evidence boundary

Source checks, native policy/asset parsing, added receipt audits, JSON
normalization, object checks and added EMA checksum probes are outside both
timers. Each native timer is bounded by the explicit outer barrier/CUDA fence
and final CUDA synchronization. The original helpers' mode-specific diagnostics
and required method-count increments are inside. Only `timing` uses the native
non-diagnostic helper mode. This accounting scope is recorded, not hidden.

The rank receipt includes separate C1/C2 timers, physical/logical versions,
native child records, observed method census, retention, interior/outer fences,
source identities and empty-cache checks. It never emits `full5_seconds` for
the combined run. No seconds are hard-coded and no CPU timing is calibration
evidence. Single-rank C2 seconds are not automatically the eight-rank maximum.

`launcher_auditor_gate` remains
`NOT_IMPLEMENTED_NEW_C2_SOURCE_ADMISSION_REQUIRED`; numerical/full-state parity
and `claimable_speedup` are not promoted. Historical **94.311432633 seconds is
one Full5**, not C2, and cannot fill `predicted_cycle2_ns`.

## Exact default-off seam, not applied

The accompanying `h100_helios_two_cycle_hook.patch` targets only the original
8fc worker. It inserts the `HOOK` constant after its existing initial-state
receipt and before the first Full5 timer (original line1813). It passes the
existing initialized locals to the new runtime and returns through the normal
hardened atomic writer and worker teardown. It does not replace world16 guards,
modify A100 runtime paths, rerun `main`, or duplicate training math.

Default mode is `off`; the explicit environment variable is
`DISTILLSCHEDULE_HELIOS_H100_TWO_CYCLE_MODE` with only `correctness` or `timing`
accepted by the runtime. **Setting it on the old launcher does not enable C2**:
the original launcher clears all inherited `DISTILLSCHEDULE_*` variables before
reintroducing its frozen environment, and its original summary rejects this
two-cycle schema. This is intentional fail-closed behavior, not a workaround.

Future integration, still required and not performed here:

- Copy the runtime module into a separately reviewed native source revision and
  apply the hook; bind its new head and file hashes. Keep the old Full5 adapter
  pinned to unmodified 8fc, or explicitly create a different registered cohort.
- Extend a **new** bounded launcher prepare path to freeze the C2 mode, shape,
  hardware allocation and source identity. Do not add arbitrary environment
  overrides or bypass fresh native asset/capability/correctness gates.
- Add a C2-specific raw eight-rank summary/admission validator: reconcile
  source/assets/environment, source-step/RNG identities, all16 distinct slots,
  C1→C2 versions, real method and retention/fence receipts, health/teardown and
  C2 world-max timing. Old Full5 summaries must be rejected.
- Only after actual future authorized H100 runs, fresh numerical comparison and
  source-bound C2 calibration may search/lowering/run-selected use these costs.
  The stopped campaign remains stopped until explicitly authorized to resume.

## CPU verification performed

- Original bridge: 17/17 stdlib tests, including real clean native three-arm
  freeze/argv compilation, no Torch imported.
- C2 unit: 13/13 stdlib tests. The test imports the original logical helper
  bodies by AST, the actual native `MicrobatchCommitEngine`, and original pure
  rollout/persistence auditor functions. Only model computation/Tensors/CUDA
  and asset payloads are fixtures, always labeled CPU-only. It checks two
  cycles, once-only construction, half-loss scaling, outer commits, all16 slots,
  state continuation, fences, retained units, bad shape/missing fields/early
  commits/missing optimizer calls, and cache leaks preventing C2.
- Fresh isolated interpreter imported the **actual** native auditor and checked
  all three real frozen policy digests without importing Torch.
- Exact hook AST parse and reversible source hash PASS; `git apply --check`
  against the real clean 8fc checkout PASS. The patch was **not applied**.

Native GPU C2 execution, full-state parity, current host availability, measured
planner/manual C2 proximity and public C2 search admission: **NOT_RUN**.
