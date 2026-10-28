# Budgeted resident-owner training

The public `distillctl train` lifecycle can keep a native model owner in place
while admitting an explicit checkpoint policy, calibrating alternatives, or
executing a saved plan. It uses the same serial snapshot/restore protocol for
all three modes. There is no whole-owner CUDA clone and no separate trainer.

## Supported scope

This initial portfolio is the source-qualified, single-device FP32 **tiny
native** SD1.5, SDXL and official Self-Forcing configurations produced by their
recipe fixture builders. They instantiate the actual upstream model classes.
SD uses B1, resolution 16, latent resolution 8 and Fake:Student=2; Self-Forcing
uses 21×4×4 latents, three-frame blocks and Fake:Student=5. Self-Forcing must
keep Generator checkpointing ON; only the noncausal Fake checkpoint policy is
an alternative. The native sampling and update rules are unchanged.

Use each recipe's pinned dependency stack and explicit local source/assets.
Pretrained-size constructors, AMP, distributed owners and other model families
do not inherit this admission. Their reference and optimized-profile entry
points retain their existing contracts. CPU and CUDA qualification are recorded
separately in the [support matrix](support_matrix.md).

## Configure an explicit budget

Add `execution.resident_budget` to the normal training configuration. All
amounts are positive integer **bytes**. This example describes a 4 GiB CUDA
trial envelope, not a recommended budget for pretrained models:

```json
{
  "schema_version": "distillschedule.training_resident.v1",
  "construction_limit_bytes": 134217728,
  "host_staging_limit_bytes": 134217728,
  "trial": {
    "host_limit_bytes": 2147483648,
    "snapshot_limit_bytes": 536870912,
    "result_limit_bytes": 536870912,
    "scratch_bytes": 134217728,
    "reserve_bytes": 134217728,
    "host_growth_bytes": 134217728,
    "device_limit_bytes": 4294967296,
    "device_state_growth_bytes": 134217728,
    "device_gradient_bytes": 67108864,
    "device_activation_bytes": 536870912,
    "device_workspace_bytes": 134217728,
    "device_restore_bytes": 67108864,
    "device_reserve_bytes": 134217728
  }
}
```

For CPU, the `trial` object contains only `host_limit_bytes`,
`snapshot_limit_bytes`, `result_limit_bytes`, `scratch_bytes` and
`reserve_bytes`. Do not copy CUDA fields into a CPU configuration. Host limits
cover managed tensors and staging, not the entire Python process RSS.

The factory checks source, architecture, weight headers, precision, topology
and construction capacity **before allocating the model**. Training also checks
fresh host/cgroup and CUDA headroom. A budget does not turn an unsupported
model into a supported one. `construction-admission.json` records the forecast
and actual managed storage; these variable observations are not checkpoint
identity.

## Direct policy, automatic planning and saved plans

For direct SD1.5/SDXL checkpoint execution, use:

```json
{
  "adapter": "program",
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": false,
    "checkpoint_bindings": [],
    "owner_execution": {
      "native_cycle": {"gradient_checkpointing": true}
    }
  }
}
```

Add the budget object above to this execution object. For official
Self-Forcing, replace the native option with
`{"fake_gradient_checkpointing": false}`; retain its Generator-ON model
setting. Direct mode needs no calibration file.

Automatic mode uses `"mode": "auto"`, an explicit
`"memory_limit_bytes": 2147483648`, `"max_candidates": 2`, `"samples": 1`
and `"max_program_runs": 3`, together with the same resident budget. It writes
real measured costs and `planning/selected-plan.json`. To reuse that plan from
its bound initial state, use `"mode": "plan"` and
`"plan": "/path/to/selected-plan.json"`. A source, state, input or policy
mismatch is an error, not a fallback to a hand-written command.

```bash
distillctl train train.json --cycles 3 --output runs/first
distillctl train train.json --cycles 6 --resume /path/to/checkpoint --output runs/resumed
distillctl export train.json --checkpoint /path/to/checkpoint --output exports/student
```

The first policy admission compares reference and candidate from one snapshot
and accepts exactly **one** real cycle. Calibration trials restore their initial
state and do not advance the data iterator. Later cycles reuse the admitted
owner and bind fresh inputs and parameter versions. Native effect counters are
written into the cycle metrics.

## Resume and failure behavior

Use the same source, device, topology, mathematical configuration and resident
budget when restoring a complete-cycle checkpoint. The lifecycle reinstalls
the selected physical policy before restoring roles, Adam, schedulers, EMA,
RNG, data positions and native clocks. Transient snapshots and caches are
rebuilt. A resumed selected owner re-admits its next actual cycle; it does not
consume an extra sample or search again.

Admission receipts are published before a speculative cycle is accepted.
Publication or numerical failure restores the snapshot. A failure during an
ordinary continuation invalidates that live owner and prevents publishing a
new checkpoint; restart from the last complete checkpoint. Closing the public
model lifetime revokes its resident admission.

The inference export contains the current Student (not the optimizer or a
resumable training state). Existing recipe generation and decoding commands
consume it unchanged.

## Validation

The public path has run on A100 with the actual tiny SD1.5, SDXL and official
Self-Forcing classes. For each family, three consecutive cycles through
automatic selection match the native reference state, and direct execution
matches an uninterrupted run after a fresh-process checkpoint restore. Current
Student exports decode to images or video. The shared saved-plan/restart path
is additionally exercised with SD1.5, including changed-plan rejection.

The tests retain one native owner, check parameters/Adam/scheduler/EMA/RNG/data
positions and clocks, and leave the unrelated device's RNG unchanged. See the
[CPU and A100 acceptance record](../status/FRAMEWORK_PUBLIC_RESIDENT_ACCEPTANCE_20261028.md)
for source and test scope. These checks use small randomly initialized native
models; they do not measure pretrained quality or performance.
