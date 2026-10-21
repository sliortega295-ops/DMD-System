# H3 cache planner diagnostics with a control repeat

The default H3 sampled comparator and `DEFAULT_P0_PARITY` do not change.
When both the cross-cache comparison and a repeated control fail that comparator,
an explicit, source-bound diagnostic policy can qualify bounded planner testing.
It cannot authorize strict numerical equality, training quality or a formal
speedup claim.

`distillctl workloads calibrate` accepts `--h3-noise-policy <policy.json>`.
The policy has exactly these fields:

```json
{
  "schema_version": "distillschedule.h3_noise_diagnostic_policy.v1",
  "source_head": "<exact current 40-character commit>",
  "reference_candidate_id": "joint-h3v3-h480-w832-f124-cpu_staged_trim-d2-n128",
  "candidate_id": "joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128",
  "max_cross_self_delta_l2_ratio": 2.0,
  "purpose": "PLANNER_DIAGNOSTIC_ONLY",
  "formal_timing_eligible": false,
  "default_gate_unchanged": true
}
```

Only cache storage/trim can differ: shape, checkpoint policy, tile, staging
depth, rollout schedule and precision stay matched. The controller runs a fresh
control, candidate and independent repeated control at that source. The original
strict comparison is saved unchanged before the diagnostic decision. Every arm
must complete sixteen ranks, both cycles, the sample/RNG and replay contracts,
and matched finite state-sample coordinates.

Qualification uses the L2 distance of each cycle's Student/Fake loss samples and
the four sampled parameter/Adam components. Each cross-cache distance must be at
most the explicit limit times its control-repeat distance. Zero self distance
admits only zero cross distance. The allowed limit is at most two. This is a
bounded diagnostic rule, not a statistical confidence interval or a proof that
systematic error is absent. A single repeat and local-shard samples do not prove
trajectory or full-gradient equivalence.

The resulting `REVIEW_NOISE_QUALIFIED_DIAGNOSTIC` receipt includes all three
hash-bound sixteen-rank witness sets. Assembly and measurement re-read those
files and recompute qualification. Edited metrics, stale witnesses, missing
ranks and aliased repeat evidence are rejected. The original failed comparison,
thresholds, `numerical_status=REVIEW` and `formal_timing_eligible=false` remain
visible throughout the calibration provenance.

Use a new immutable output/run prefix for each attempt. An earlier failed
preparation is evidence, not a directory to overwrite. H3 model execution and
the node ownership/cleanup rules are unchanged by this CPU-side admission path.
