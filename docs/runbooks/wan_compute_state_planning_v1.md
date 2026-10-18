# Wan bounded checkpoint-retention planning, v1

Status: opt-in source/CPU integration. Current-source GPU qualification,
calibration and manual-versus-saved native C2 are **NOT_RUN**. Historical
retention experiments are provenance, not costs for this source.

## Why a distinct compute identity is necessary

The existing `WanStrategy` can retain the last four Critic blocks in original
autograd instead of checkpointing them. It can also use isolated or shared
Generator graph scratch. These choices change memory residency and measured
action costs, while `strategy.backend` is unchanged. Matching costs or runtime
receipts by backend alone would alias them. The old manual-only guard prevented
this experimental path from reaching a saved-plan GPU run; this is not evidence
that historical default planner runs executed the wrong policy.

`compute_policy` binds the full versioned state policy, and `compute_key` binds
it together with the existing backend. Rollout-window size is still a DAG
decision, not part of this key. State-qualified actions include the policy in
their component context. This conservatively separates every action's costs,
including effects from persistent graph allocations. The planner and worker
select exactly one corresponding calibration variant; rehashing a plan cannot
substitute retention or a scratch policy that was not measured.

## Explicit inventory and runtime path

```bash
python scripts/distillctl.py workloads native validate \
  --task configs/distillschedule/original_wan/full5_reference.json \
  --include-incumbent-bundle --include-critic-retention
```

Use the actual task path in the checkout; the task must be exactly the original
`OriginalWanTask`. The incumbent graph bundle retains its existing immutable
assets, precision, shape and bootstrap restrictions. No new-resolution or
variable-update graph support is claimed.

The inventory adds only Critic blocks 26–29 with
`retention_execution=planner-retain4.v1`. It does not enumerate failed tail6 or
tail8, an attention-replay prototype, LoRA merging, a different batch size,
optimizer or mathematical workload. Defaults and legacy strategy JSON are
unchanged. Old `manual-only.v1` retention is still rejected in calibration/run;
it does not silently inherit planner permission.

Calibrate each explicit strategy using the existing native `calibrate` worker,
assemble real all-rank receipts, and use the existing `plan` and `run --plan`
commands. A retained candidate is not a preselected winner: measured action
costs, workspace limits, output lifetimes and the existing frontier search
decide. Unknown costs are not filled from historical ~20-second endpoint times.
The automatic streaming-campaign inventory is not extended in this version.

The saved strategy reaches `OriginalWanSession`, which invokes the existing
`checkpoint_block_retention.install` on the original Critic. Selected blocks
use their original function and normal autograd; the other 26 use the original
checkpoint dispatcher. Generator checkpoint dispatch is unchanged. Original
execution/version/census, source/asset, capacity and numerical checks remain.

## Evidence boundary

CPU tests exercise two same-backend policies through cost construction, common
search, serialization, policy substitution rejection and the actual Session
writer/installer path. A tiny CPU model verifies output, gradients, RNG, Adam
and 30-block dispatch counts. All synthetic timing/asset fixtures are test-only,
not GPU evidence. No strict tolerance, golden result or historical gate is
relaxed. Production source fingerprint changes require fresh calibration.
