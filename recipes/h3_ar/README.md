# W6 — Autoregressive Audio-Video DMD (Original DiffSynth H3)

**Original protocol:** Previous+current AV chunks; continuous sigma; 8-step; G1/F1; LoRA64.

Original AR adapter, not the generic flow example and not W5. A100 zero3-native/AP and H100 native-FSDP/FA2 are separate contracts.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads native validate --task configs/distillschedule/original_h3_ar/w6_reference.json
```

Continue through the existing [native runbook](../../docs/runbooks/original_h3_ar_native.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- [`src/distillschedule/strategies/h3_ar_cli.py`](../../src/distillschedule/strategies/h3_ar_cli.py)
- [`src/distillschedule/strategies/h3_ar_session.py`](../../src/distillschedule/strategies/h3_ar_session.py)
- [`src/distillschedule/strategies/h3_ar_planning.py`](../../src/distillschedule/strategies/h3_ar_planning.py)

W6 uses its original native task schema; it is not added by cloning a W5 legacy profile.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: [source-bound record](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json), status `MATCHED_DIAGNOSTIC_TARGET_MET`. Its fixed training source is `e88d21c168ce54be99134cecd195c961319cb45a`.

H100 measured source: `sliortega295-ops/DMD-System@0419b96b9ed9a8dd76737042e6c5020c80f6673a`. Published source: `0419b96b9ed9a8dd76737042e6c5020c80f6673a`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived [measurement index](../../artifacts/publications/20261019/h100/measured_results.csv) defines the actual denominator and numerical status.

Read the [six-workload correction](../../docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md) before interpreting any number. No new GPU or quality result is created by this recipe.

## Public training configuration

Use [`training.json`](training.json) with your explicit qualified native inputs
and saved selected plan. Read the [native training guide](../../docs/guides/native_workloads.md)
for actual cycle limits, CPU admission, launch ownership and unsupported lifecycle
operations. This binding does not replace the original protocol or certify GPU execution.


The separate [actual-original cached-AV CPU reference](CPU_REFERENCE.md)
provides finite fresh-cache training and safe complete-cycle/new-process
recovery through the public Program lifecycle. It uses the maintained licensed
cached-only source, not the incomplete full-SFT archive, and does not alter the
native two-update ZeRO-3/selected-plan contract. No VAE/decoder, pretrained,
GPU, quality or performance claim follows from that local tiny-model proof.
