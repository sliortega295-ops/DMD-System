# Same-source manual reference versus a saved native Wan plan

This opt-in benchmark is **not a new optimizer, model, or replacement for the
saved-plan runner**. `run` still requires its validated source/assets/cost-bound
plan. Canonical defaults, costs and existing admission checks are unchanged.

`benchmark-manual` makes the missing unprofiled reference explicit. It constructs
the same `OriginalWanSession` and original action handlers from an admitted
strategy, uses their direct native order and existing two-cycle CUDA-fenced
timer, and does not attach calibration scopes or search for a schedule. Its rank
receipt says `mode=manual`, `MANUAL_REFERENCE_DIRECT_NATIVE_ORDER`,
`search_performed=false` and `plan_sha256=null`. It cannot be relabelled as actual
saved-plan execution.

```bash
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native benchmark-manual \
  --task "$RUN/task.json" --assets "$RUN/assets.json" \
  --implementation "$RUN/manual-strategy.json" --cycles 2 \
  --output "$RUN/manual-native" --execute
```

Invoke this only under the existing authorized world16 launcher and atomic node
owner, not as a standalone world1 shortcut. Without `--execute` it reports source
capabilities only; no asset load or model construction occurs. Exactly two
fixed-trace cycles are permitted. Plans, fresh streams and ambiguous strategies
are rejected before allocation. A window-five reference preserves the original
five-rollout cluster; the selected window/order is a legal measured decision.

For comparison, separately calibrate the current source, assemble observed costs,
search and persist a plan, then use the unchanged `run --plan ... --execute`.
Both arms must share source, assets, trace, precision, environment/hardware,
resource policy and native timer. C1 is warm-up; compare world-max C2, not an
average. Record actual kernel effects, calls, samples, RNG, versions and state
comparison. REVIEW remains REVIEW. A single pair supports only a descriptive
three-percent matching target, not statistical equivalence or a global optimum.

The accepted incumbent bundle remains opt-in and original-task-bound; merged
LoRA, score graphs and altered math are not enabled. No historical duration is
inserted into calibration or marked measured by this interface.
