# W4 — Self-Forcing DMD / LiveWan C51

**Original protocol:** 14B teacher -> 1.3B student; continuous sigma; G1/C5/EMA0.

Current C51 opt-in must be explicit. The five-profile compatibility registry still locks C34; do not silently dispatch C34 for this recipe.

For continuous full-C51 training use [C51_CONTINUATION.md](C51_CONTINUATION.md).
For current-Student checkpoint export and new-input generation with the actual
LiveWan streamer and Wan VAE, use the [native output guide](../../docs/guides/livewan_native_output.md).
The separate CPU Program recipe is not substituted for either native owner.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads candidates --workload livewan_crossscale_pipeline_best --include-livewan-c51
```

Continue through the existing [native runbook](../../docs/runbooks/livewan_c51_planner_v1.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- [`src/distillschedule/search/livewan_implementation.py`](../../src/distillschedule/search/livewan_implementation.py)
- [`src/distillschedule/runtime/livewan_crossscale_worker.py`](../../src/distillschedule/runtime/livewan_crossscale_worker.py)

The compatibility profile `livewan_crossscale_pipeline_best` is preserved but is not silently substituted for this guide's explicit cohort.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: [source-bound record](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json), status `MATCHED_DIAGNOSTIC_TARGET_MET`. Its fixed training source is `5cdffc2efab969ad35a8f69dd7b94ad061cc8c86`.

H100 measured source: `sliortega295-ops/DMD-System@4d7c589acd01a4a6c90ca141da266b3090df2b68`. Published source: `1b8236c243323ff21023935f3b8192ab3e5d136d`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived [measurement index](../../artifacts/publications/20261019/h100/measured_results.csv) defines the actual denominator and numerical status.

Read the [six-workload correction](../../docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md) before interpreting any number. No new GPU or quality result is created by this recipe.

## Public training configuration

[`training.json`](training.json) selects `self_forcing.c51.v1` with explicit
`mode: manual` and `execution.preset`; supply your qualified original assets
and output directory, **not a historical saved plan**. The existing typed
implementation is bound against the current source/assets and enters the same
native runner and gates. This is exactly two fixed-contract cycles with no
checkpoint/resume, not continuous training. REVIEW; full C51, never C34; no C51 drain/checkpoint/resume.

Read the [direct preset guide](../../docs/guides/native_presets.md) for the
complete source/mechanism/runtime mapping and the
[native training guide](../../docs/guides/native_workloads.md) for separate
reference/fresh-input and saved-plan paths. CPU admission does not certify
GPU execution, numerical parity, quality or speedup.

For reference-first finite training rather than selected C51 replay, use
[`training.reference.json`](training.reference.json) and the
[original reference guide](REFERENCE_TRAINING.md). This separately versioned
2..64-cycle route uses genuine zero-step assets, original model/update owners,
fresh TeacherLatents cohorts and complete-cycle same-topology restart. It is
not a C51 pipeline checkpoint implementation and has CPU/source evidence only.

For the separately bounded original LiveWan **CPU1 FP32,32px/81-frame**
training/current-G export/new-input generation route, see
[CPU_PUBLIC.md](CPU_PUBLIC.md). Native latents work without a decoder; actual
video requires a separately pinned original Wan2.1 codec and supplied safe
weights. Random full-class CPU tests do not establish pretrained quality,
native world16/C51 qualification or performance, and are not the separately
integrated official `guandeh17/Self-Forcing` route.
