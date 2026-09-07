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

Actual action-cost calibration and held-out model validation remain NOT_RUN at
this source milestone. Existing Wan macro search, LiveWan macro search and H3
correctness receipts are preserved, not rerun or relabelled as component search.
LiveWan/H3 component execution and the two Helios generated adapters remain
separate unfinished units; this is not a five-workload completion claim.
