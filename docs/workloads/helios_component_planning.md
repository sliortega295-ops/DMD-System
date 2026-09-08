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

Other canonical kernels, B1/global16, unpacked DMD Score, both CUDA residuals,
CUDA EMA and local FA2 are frozen. There are no new sharing aliases or TP/PP
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
growth remains conservatively inside workspace. A 2 GiB model capacity margin
is not a substitute for native capacity validation. Serial world16 action
envelopes do not invent additional inter-action GPU overlap.

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

Initial acceptance: 972 CPU refactor tests passed, 17 skipped; both capability
dry-runs and compile/shell checks passed. Real Helios GPU correctness,
calibration, search feedback and timing are **NOT_RUN at this documentation
checkpoint**. Subsequent experiment reports must state exact runtime HEADs.
