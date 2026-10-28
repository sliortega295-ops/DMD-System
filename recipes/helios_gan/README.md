# W3 — Helios14B GAN-on

**Original protocol:** DMD + GAN-G/GAN-D; G1/C5/EMA1.

No merged-LoRA promotion. N70-OFF timing has a preserved post-timer FAIL and numerical REVIEW; historical 113.59 s is N70-ON.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads candidates --workload helios14b_gan_best --include-helios-gan-manual
```

Continue through the existing [native runbook](../../docs/runbooks/native_planner_workflow.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- [`src/distillschedule/search/helios_model_adapter.py`](../../src/distillschedule/search/helios_model_adapter.py)
- [`src/distillschedule/runtime/helios_gan_manual/__init__.py`](../../src/distillschedule/runtime/helios_gan_manual/__init__.py)

The compatibility profile `helios14b_gan_best` is preserved but is not silently substituted for this guide's explicit cohort.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: [source-bound record](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json), status `REUSED_MATCHED_DIAGNOSTIC_TARGET_MET`. Its fixed training source is `5d5b354d48a0eca728ce61e2c98b61fcb5ce26ed`.

H100 measured source: `sliortega295-ops/DMD-System@624515e7c8d1511aa688e95e69cf94f1a2a273f7`. Published source: `624515e7c8d1511aa688e95e69cf94f1a2a273f7`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived [measurement index](../../artifacts/publications/20261019/h100/measured_results.csv) defines the actual denominator and numerical status.

Read the [six-workload correction](../../docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md) before interpreting any number. No new GPU or quality result is created by this recipe.

## Public training configuration

[`training.json`](training.json) selects `helios_gan.fa3_down40_nomerge.v1` with explicit
`mode: manual` and `execution.preset`; supply your qualified original assets
and output directory, **not a historical saved plan**. The existing typed
implementation is bound against the current source/assets and enters the same
native runner and gates. This is exactly two fixed-contract cycles with no
checkpoint/resume, not continuous training. REVIEW; merged-LoRA/N70 remains off and runtime merge counts must be zero.

Read the [direct preset guide](../../docs/guides/native_presets.md) for the
complete source/mechanism/runtime mapping and the
[native training guide](../../docs/guides/native_workloads.md) for separate
reference/fresh-input and saved-plan paths. CPU admission does not certify
GPU execution, numerical parity, quality or speedup.

For the full fresh-stream recipe, complete native checkpoint → current
unmerged Student + Wan2.1 VAE → independent new-prompt decoded video:
[NATIVE_OUTPUT.md](NATIVE_OUTPUT.md). This includes source/CPU acceptance;
pretrained world16/GPU output and generation quality are separate checks.
The older CPU-only diagnostic remains at
[PORTABLE_INFERENCE_CPU.md](PORTABLE_INFERENCE_CPU.md).
