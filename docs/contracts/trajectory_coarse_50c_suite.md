# Five-workload 50-cycle trajectory suite

This is a **post-run evidence contract**. It neither launches GPU work nor
changes the strict `DEFAULT_P0_PARITY` gate. Numerical trajectory thresholds
are evaluated only after both arms have completed all 50 cycles.

## Fixed matrix

| `workload_id` | Comparison | Claim boundary |
|---|---|---|
| `wan1_3b_nogan` | world16 FSDP2 vs Resident | single-seed coarse semantic trajectory |
| `helios14b_nogan` | baseline vs optimized | single-seed coarse semantic trajectory |
| `helios14b_gan_on` | baseline vs optimized | single-seed coarse semantic trajectory |
| `livewan_crossscale` | sequential vs pipeline | single-seed coarse semantic trajectory |
| `minimax_h3_exact_vs_grid1000` | continuous sigma vs Grid1000 | **approximation, never exact equivalence** |

Every `PASS` matrix row must point to explicit baseline and candidate roots.
Each root must contain exactly one `rank_*.jsonl` file for every rank 0--15,
and every file must contain cycles 1--50. The underlying
`trajectory_coarse_v1` comparison verifies the per-rank operation, sample,
RNG, version, update-count, call-count, shape, and dtype contract.

## Append-only matrix manifest

The manifest is JSONL. Every row uses
`distillschedule.trajectory_run_matrix_record.v1`. A row records one attempt
for one canonical workload. The latest attempt is summarized, while all older
attempts remain part of the hashed history.

The machine-readable JSON Schema is
`configs/distillschedule/trajectory/trajectory_run_matrix_record.schema.json`;
the Python loader additionally enforces the workload-specific comparison kind,
append ordering, and immutable identity rules that JSON Schema cannot express.

Required pair-level fields are:

- `suite_id`, `workload_id`, monotonically increasing `attempt`;
- ISO-8601 `recorded_at_utc` ending in `Z`;
- `execution_status`: `PASS`, `FAIL`, or `NOT_RUN`;
- non-empty `reason` and exact shared asset identities;
- baseline and candidate `run_id`, variant, exact 40-character git HEAD,
  arm-specific assets, and (for `PASS`) an absolute rank-JSONL root.

Optional pairwise state comparison JSON can be attached using
`state_comparison_paths`. A file may contain one raw
`TrajectoryStateComparison`, a list, or a `state_comparisons` wrapper. State
anchors may be required at cycles 1, 10, 25, and 50. Missing optional state
evidence is reported as not provided; it is not silently manufactured.

Within a `suite_id`, source HEADs and asset identities are immutable. A source
or asset change requires a new suite ID. The writer rejects a truncated file,
attempt rewind, duplicate attempt, or identity drift.

Example record (paths and hashes are illustrative):

```json
{
  "schema_version": "distillschedule.trajectory_run_matrix_record.v1",
  "suite_id": "trajectory_50c_seed20260817_v1",
  "workload_id": "wan1_3b_nogan",
  "comparison_kind": "semantic",
  "attempt": 1,
  "recorded_at_utc": "2026-09-03T12:00:00Z",
  "execution_status": "PASS",
  "reason": "both arms completed",
  "expected_world_size": 16,
  "expected_cycles": 50,
  "baseline": {
    "run_id": "wan_fsdp2_50c",
    "variant": "world16-fsdp2",
    "source_head": "1111111111111111111111111111111111111111",
    "assets": {"config_sha256": "aaaaaaaa"},
    "rank_jsonl_root": "/persistent/wan_fsdp2_50c/trajectory",
    "implementation_expectations": {"placement": "world16-fsdp"}
  },
  "candidate": {
    "run_id": "wan_resident_50c",
    "variant": "resident",
    "source_head": "2222222222222222222222222222222222222222",
    "assets": {"config_sha256": "bbbbbbbb"},
    "rank_jsonl_root": "/persistent/wan_resident_50c/trajectory",
    "implementation_expectations": {"placement": "resident"}
  },
  "shared_assets": {
    "model_sha256": "cccccccc",
    "bootstrap_sha256": "dddddddd",
    "trace_sha256": "eeeeeeee"
  },
  "state_comparison_paths": [],
  "required_state_cycles": [],
  "curve_noise_floor": {}
}
```

Append a record without rewriting history:

```bash
python scripts/distillschedule/record_trajectory_suite_run.py \
  --manifest-jsonl /persistent/suite/run_matrix.jsonl \
  --record-json /persistent/suite/wan_record.json \
  --suite-id trajectory_50c_seed20260817_v1
```

Once all five workloads have a recorded status, generate one JSON/CSV/Markdown
and PNG matrix:

```bash
python scripts/distillschedule/summarize_trajectory_suite.py \
  --manifest-jsonl /persistent/suite/run_matrix.jsonl \
  --output-dir /persistent/suite/report
```

The summary is `FAIL` for malformed/incomplete/contract-invalid evidence,
`NOT_RUN` while a recorded arm is not run, `REVIEW` for a completed numerical
trajectory outside the coarse thresholds, and `PASS` only when every pair has
complete, contract-matched evidence within the post-run thresholds.
