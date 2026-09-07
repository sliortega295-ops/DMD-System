# Wan generated actions and reusable component costs

## Preserved completed evidence — no rerun

`jointp_wan_actions_20260907_r1_w3_correctness`, source
`b5b47bba288d4a693665ba40e14d813260a958e3`, already completed:
16/16 ranks, two cycles / 32 rank-cycle receipts, final G4/C20/EMA20,
native schedule/call/sample/trace gates PASS, no cgroup OOM increment and owned
teardown PASS. It executes the generated window3 order through native worker
helpers. This is correctness evidence, not a timing trial or a model-selected
winner. Its original receipts and source checkout are preserved.

## Newly completed implementation

- Opt-in calibration of **32 native actions**: two cycles, each containing
  one G update, five individual B1 rollouts, five C updates and five EMA updates.
- Reusable costs are keyed by actual tensor layouts/call inventory, precision,
  local implementation, placement and cold/warm context, **not a full recipe ID
  or rollout-window number**. Unrelated kernel switches do not invalidate EMA
  costs. First-cycle observations do not replace steady-state costs.
- Retained outputs use deduplicated CUDA storage metadata, inspected **outside**
  the measured producer range, with no tensor-value read or device copy.
  Workspace excludes previously retained inputs. Output lifetimes start at
  producer exit and end at the owning C consumer; producer workspace includes
  creation. Both terms reconcile and cannot count the output twice.
- The common planner covers 35 implemented legal placement/kernel/window
  combinations. Unknown component keys remain `UNCALIBRATED`; fixtures and
  historical whole-cycle timings cannot fill them.
- The source-bound assembler pools matching component observations, retaining
  and rechecking every original receipt SHA. It rejects source/context changes,
  mismatched fragment/cost data and nested aggregation; identical source
  observations are deduplicated.
  Raw Nsight, checkpoints and native traces stay remote.

## Commands (new immutable output directories required)

Collect one explicitly selected, **not previously completed** native component
calibration with the existing collector command. Omitting `--execute` is a
source-only plan; actual execution retains current-source correctness gates.

```bash
python scripts/distillschedule/calibrate_wan_native_stages.py \
  --implementation joint-wan2-resident-c2048-e1-w5 \
  --expected-head "$(git rev-parse HEAD)" \
  --run-id <NEW_RUN_ID> --output <NEW_CONTROLLER_DIRECTORY>
```

After native collection, run the metadata-only assembler beside its persistent
source receipts. This command never starts or repeats GPU experiments:

```bash
python scripts/distillschedule/assemble_wan_action_calibrations.py \
  --calibration-result <FSDP_COMPONENT_RESULT_JSON> \
  --calibration-result <RESIDENT_COMPONENT_RESULT_JSON> \
  --expected-head "$(git rev-parse HEAD)" --output <NEW_ASSEMBLY_DIRECTORY>
```

Use the resulting `problem.json` and `cost-model.json` with the existing
model-based search CLI and native lowering. Calibration must be measured before
the library or a resulting plan is described as GPU-validated.

## Claim boundaries / unfinished work

CPU tests exercise graph execution, timing observer ordering, storage lifetime
reconciliation, per-rank evidence validation, pooling, search and native adapter
lowering. They are **not** new GPU performance data.

This serial-world16 model deliberately predicts **no hypothetical speedup from
inter-action overlap**. Identical component counts produce equal predicted
window latency, while retained-output memory differs. A selected held-out
window still requires a new native capacity and unprofiled cycle-2 confirmation.
Calibration barriers perturb cross-stage overlap and its duration is not a
performance denominator. Numerical REVIEW remains distinct from strict parity.

The following section records the subsequent real action-cost collection and
held-out validation at that source milestone. Existing Wan macro search,
LiveWan macro search and H3 correctness receipts remain preserved, not rerun or
relabelled as component search. LiveWan/H3 component execution and the two
Helios generated adapters remain separate unfinished units; this is not a
five-workload completion claim.

## Native component closure — only previously missing work

Executed source: `ceb2921985e5a855ff93d6c9d7e21e7528b6ef5a`.
Run prefix: `jointp_wan_components_20260907_r1`.
The later publication/capability metadata change is not the executed source.

Two new calibration runs, FSDP/c512/epilogue-off and
Resident/c2048/epilogue-on, both used window5. Each collected 32 native actions
over two cycles on all 16 ranks: **1,024 rank-action observations** in total.
Pooling equivalent physical work produced **24 distinct component costs**, not
a table assigning one whole-cycle time to each recipe.

Each calibration uses current-source correctness preparation; the two runs
share that preparation. A third correctness run covers only the held-out
window2. No completed window3 canary, old seven-arm matrix, baseline native
timing or incumbent native timing was repeated. No Nsight was run.

### Search and held-out prediction

The common CPU oracle evaluated 60 assignments: 25 rejected by the implemented
contract, 25 left `UNCALIBRATED`, and 10 predicted feasible. These are 35 legal
compositions, **not 35 GPU-tested configurations**. Five window orders reuse
each of the two measured placement/kernel component sets.

| Held-out Resident window2 | Seconds |
|---|---:|
| Model-predicted cycle 2 | 24.513250866 |
| Native cycle 1, warm-up | 27.615927894 |
| Native cycle 2 | 24.368434417 |
| Prediction minus measurement | +0.144816449 (+0.59428%) |

The measured order was **not** included in calibration. Its typed lowering,
selected candidate, source/asset/precision binding and actual native execution
are checked against the saved model prediction. The old FSDP correctness
preparation is reused, not launched again.

All five Resident windows predict equal serial action-sum latency; window2 is
a deterministic tie-break, not a uniquely faster schedule. This validates one
held-out composition, not arbitrary pool overlap, a global optimum, or another
speedup over a newly measured baseline. Calibration sums cannot serve as that
baseline. No canonical winner was changed.

### Calibration decomposition — not native performance denominators

The following values sum max-rank synchronized action envelopes in cycle 2.
They are neither kernel unions nor exact cycle wall clocks; barriers and fences
perturb cross-action overlap. Repeated same-work observations are pooled by the
cost model, so predicted totals need not equal these observation sums exactly.

| Component family | FSDP c512/e0 (s) | Resident c2048/e1 (s) |
|---|---:|---:|
| G update | 8.415994 | 5.671255 |
| Five individual B1 rollouts | 11.966236 | 7.521978 |
| Five independent C updates | 15.784440 | 11.251533 |
| Five EMA updates | 0.245246 | 0.096291 |

### Memory, semantics and teardown

- All native gates: 16/16 ranks, two cycles, final **G4/C20/EMA20**;
  G2/C10/EMA10 actual commits over both cycles, unchanged trace/B1/global-B16,
  rollout call inventory, dtype and finite checks.
- Numerical comparison remains **REVIEW**, not strict equivalence. The legacy
  coarse comparison status includes `50C` in its name; these runs executed
  exactly **two** cycles, not fifty. No threshold was relaxed.
- Post-setup allocator residency: FSDP 991,354,368 B; Resident 19,071,560,192 B.
  Calibration peak allocated: FSDP 5,814,859,776 B; Resident 23,513,754,624 B.
- Each retained rollout/input bundle: **17,128,176 B**. Window5 retains at most
  **85,640,880 B**, with ten observed producer-to-consumer lifetimes over two
  cycles. Final retained bytes **0**, calibration tensor-value reads **0**.
  The main G update peak dominates, so reducing this small retention window
  does not reduce the overall predicted peak in this experiment.
- Held-out peak allocated **21.89889 GiB**, reserved **22.37305 GiB**;
  sampled minimum driver free **16.22498 GiB**; observed node cgroup peak up to
  **54.05648 GiB**. These are not continuous hardware-counter peaks.
- CUDA OOM absent; both cgroup OOM/oom-kill deltas **0**. Native teardown and
  independent readback show no GPU applications, free node locks and no listener
  on experiment port 31527. No environment installation or foreign cleanup.

### Reproduction evidence

Small bundle: `artifacts/experiments/joint_planner_wan_components_20260907/`.
It contains source/run identities, native measurement and calibration summaries,
per-action CSV, lowering hashes, commands/scripts and SHA256 checksums. The
scripts are records of this immutable server run, not commands to reuse its IDs.

Remote full component fragments, pooled receipts, problem, cost model and
lowered plan:
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/component_costs_r1/analysis_r1/`.

Local detailed preflight/failure/controller evidence:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/artifacts/component_costs_r1/`.

The front-door capability report now explicitly exposes the prepared-action
adapter and 35 legal compositions, while preserving the older macro adapter.
Capability metadata requires source-bound receipts; it never auto-promotes
historical measurements to current-source validation.
