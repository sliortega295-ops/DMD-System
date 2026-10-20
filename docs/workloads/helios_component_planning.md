# Helios No-GAN / GAN: measured components and native plans

This opt-in adapter joins the existing `workloads search --strategy model-based`
entry point. It does not change the five canonical profiles or their historical
results. The old P0 helpers remain the model/loss implementation; the selected
versioned graph, not the old one-cycle loop, now dispatches those helpers.

## Bounded implementation space

| Workload | Placement | Chunk tokens | Retained rollout window | G unshard persistence |
|---|---|---|---|---|
| Helios No-GAN | world16 FSDP / node HSDP 2x8 | 512 / 2048 | 1..5 | off |
| Helios GAN | world16 FSDP | 512 / 2048 | 5 | off / on |

For the legacy `native0` candidate, other canonical kernels, B1/global16,
unpacked DMD Score, both CUDA residuals, CUDA EMA and local FA2 are frozen.
The separate explicit complete GAN manual bundle is pinned-SM80-FA3/down40,
with merged-LoRA/N70 execution OFF; it is not a FA2-only candidate and requires
its own source-bound calibration. There are no new sharing aliases or TP/PP
pools. The existing Score/Teacher adapter alias remains. GAN's official packed
fake/real/R1 Score input is physical B3; it is still logical B1, not searched
batching. R2 is inactive.

No-GAN lowers G update, individual R/C actions and EMA after each C. GAN keeps
one `r_all` action around all five B1 rollouts so persistence is acquired and
released once. It then performs five C commits and EMA only after C0. Each GAN
C helper retains its two backward calls and one commit. Splitting GAN's
persistent scope is not admitted by this cost model.

Each launch initializes once and replays its fixed immutable source cohort for
two cycles, carrying real model/optimizer/EMA versions forward. This is explicitly
`fixed_source_cohort_replay`, not new trace steps 15..19 or 1005..1009.

| Variant | Source cohort | Initial G/C/EMA | After two cycles |
|---|---|---|---|
| No-GAN | 10..14 | 2/10/10 | 4/20/20 |
| GAN | 1000..1004 | 200/1000/800 | 202/1010/802 |

The GAN rows are **logical schedule versions**, not physical Adam counters.
The pinned systems-only bootstrap restores physical G2/C10/EMA800; after
two cycles its physical counters are G4/C20/EMA802. The adapter validates
the immutable bootstrap's `systems_only_control` mapping, then applies the
fixed offsets only to its version ledger and logical receipt labels. It never
writes G200/C1000 into the optimizer or scheduler. Both physical and logical
counters are recorded and independently rechecked at action/cycle boundaries.
No-GAN has an identity mapping. A mismatch fails before the first cycle.

## Cost and execution provenance

`helios_implementation.py` validates a typed assignment and lowers it into the
existing source-bound launcher. `helios_planner_cycle.py` prepares the common
`ExecutionGraph` / `StaticRoleRunner`, invokes the native helpers and records
actual application forwards, trace membership, optimizer commits, versions,
retained output lifetimes and effective choices. Assignment/receipt mismatch
fails rather than selecting the nearest historical recipe.

Correctness, calibration and native timing are distinct modes of the same
opt-in executor. Canonical invocations leave the adapter off. Tensor-value
loss reads and global-coordinate parameter/Adam/EMA sketches are outside the
cycle timer. Sampled parity remains REVIEW, not full-state numerical proof.
Correctness must precede calibration/native execution at the same source,
assets and assignment. Runtime receipts are revalidated, not accepted by exit
code alone.

Calibration fences each native action. Actual rank maximum action walls supply
component samples; internal NCCL/compute overlap is already included. Chunk,
placement, persistence, actual shape/call work and first/second-cycle state bind
the cost key. Reordering windows can reuse matching component costs. No
historical wall or synthetic throughput estimate fills missing observations;
unsupported/missing combinations remain `UNCALIBRATED`.

Memory uses observed post-setup static allocation and per-action peak deltas,
plus separately measured retained latent lifetimes. Other trace/cache/state
growth remains conservatively inside workspace. The allocator margin defaults
to exactly 2 GiB; it is not a substitute for native capacity validation.
Construction-memory peaks are outside this post-setup model. Serial world16 action
envelopes do not invent additional inter-action GPU overlap.

An explicit Helios-only `assemble-calibration --helios-allocator-margin-mib N`
sets a nonnegative integer margin in MiB. The API keyword is
`allocator_margin_bytes`; the optional layout field has that same name and
rejects booleans, negative values and nonintegers. Omission preserves old layout
fields and the exact 2 GiB default. Per-rank caps remain derived from actual
driver total minus maximum observed nonallocator usage (`total-free-reserved`,
floored at zero) minus this margin, taking the minimum across fragments.
Arbitrary caller memory limits are not accepted: native validation rebuilds
from the hash-bound layout, and manual-match protocol budgets must equal it.

This input does not redefine `gpu_reserve_bytes`, the independent sampled
driver-free gate, and neither gate proves continuous free-memory extrema.
A different margin requires authorization and fresh source-bound preregistration
before new match runs. It cannot make a prior strict-2-GiB rejection pass or
relabel older native receipts under a new HEAD, authorize GPU execution, or
establish numerical equivalence or formal speedup.

`helios_model_adapter.py` constructs the common `PlanningProblem`, verifies raw
source-linked costs, and lowers the selected plan. `helios_native.py` rechecks
16 rank receipts, effective launcher, runtime identity, resource/OOM evidence,
and owned teardown. Cycle 2 is the maximum rank-local CUDA-fenced wall outside
probes; cycle 1 is warm-up. Calibration/profile walls are never native speedup.

## Entry points

```bash
python scripts/distillctl.py workloads search --strategy model-based \
  --workload helios14b_nogan_best --dry-run
python scripts/distillctl.py workloads search --strategy model-based \
  --workload helios14b_gan_best --dry-run

# After NativeTrialRunner has collected matched correctness and component costs:
python scripts/distillctl.py workloads assemble-calibration \
  --workload helios14b_gan_best --result <baseline-result.json> <manual-result.json> \
  --expected-head <same-source-head> --include-helios-gan-manual \
  --helios-allocator-margin-mib 2048 --output <new-assembled-directory>

python scripts/distillctl.py workloads search --strategy model-based \
  --workload helios14b_nogan_best --planning-layout <layout.json> \
  --cost-model <cost-model.json> --assets-config <assets.json> \
  --run-id <fresh-id> --emit-native-plan <native-plan.json> \
  --output <new-search-result.json>
```

The existing `NativeTrialRunner.calibrate()` collects one implementation;
`helios_model_adapter.assemble_calibrations()` assembles those real receipts.
The existing explicit `--execute-native` validation path executes selected
plans. A CPU fixture, dry-run, or emitted plan is not GPU success.

## Completed two-node closure (2026-09-08)

Both adapters have now completed real world16 correctness, component collection,
common CPU search and native two-cycle validation. This is no longer a CPU-only
capability inventory. The source-bound results are in
[the Helios closure report](../experiments/DMD_HELIOS_COMPONENT_CLOSURE_20260908.md).

- No-GAN: 40 observed components, 20 assignments, 10 feasible and 10
  `UNCALIBRATED`. The generated HSDP/chunk512/window1 plan actually dispatches
  `g,r0,c0,e0,...,r4,c4,e4` on all 16 ranks. Confirmed native cycle2: 91.617 s.
- GAN: 22 observed components, four assignments, one feasible, two
  `UNCALIBRATED`, one predicted capacity rejection. The selected nonpersistent
  chunk512 plan completes two native confirmations: 126.902 / 126.762 s.
  These are the **same implementation**, not an optimization speedup.
- Both: semantic/rank/trace/version/finite gates PASS, numerical `REVIEW`, no
  CUDA/cgroup OOM, owned teardown PASS. No canonical winner was overwritten.

The original native sources remain separate: No-GAN `85be7e4e36de`, GAN
`cc7e451bff73`. Later collector/controller fixes and this publication do not
relabel those measurements as execution of a newer HEAD. Costs from a different
HEAD still fail admission; obtain matching calibration before a new-source run.

### Predicted-rejected incumbent

A previously runnable incumbent need not satisfy the current modeled safety
margin. GAN C25 passed correctness/calibration without OOM but its 37.006 GiB
peak exceeds the model's unchanged limit after reserving 2 GiB and observed
non-allocator usage. It is retained as `PREDICTED_CAPACITY_FAIL`, **not CUDA OOM**.
Its native validation and speedup are `NOT_RUN`/null; calibration time is not
substituted. Baseline must remain feasible and the new-candidate budget does
not increase when an incumbent is skipped.

`NativeTrialRunner` now carries an admitted-candidate set through preparation,
calibration and timing. It cannot implicitly bootstrap a rejected peer. Existing
source/asset-bound correctness preparation can be reused; absent that preparation
and another feasible peer, a fresh validation fails closed before launching GPUs.
The cluster controller used the already verified calibration preparation. This
is an explicit prerequisite, not a fallback to a historical fixed script.

A 2 GiB **modeled allocation** margin is not a 2 GiB continuously sampled
**driver-free** guarantee: the GAN native run sampled as little as 440 MiB free
while allocator reservation was high. The report preserves both measurements.
