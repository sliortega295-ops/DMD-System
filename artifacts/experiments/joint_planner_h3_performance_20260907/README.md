# H3 action-model performance closure

See `docs/experiments/DMD_H3_ACTION_PERFORMANCE_CLOSURE_20260907.md`.
Actual GPU runtime source: `bf59e39b8a6fe158ad15ebae07be7df71369f5fd`.
Later report and simulator serialization fixes are not retroactive runtime identities.

`reports/performance-closure.json` is the derived outcome.
`reports/search-r2/` is the admitted search from the persisted component model.
`reports/search/` preserves the first CPU plan that failed round-trip admission.
`manifests/*failure*` and the red regression preserve failure evidence; GPU was
not launched by those failed preflight attempts. Run IDs are immutable.

Raw calibrated rank JSON, source component receipts, node logs and launch
manifests remain on persistent remote storage at the absolute paths in these
hash-bound receipts. Only small derived evidence is included here. No model,
Grid table, environment binary, Nsight or SQLite is included.

Scripts are exact execution/analysis provenance with old immutable run IDs,
not instructions to overwrite those runs. Use a fresh run ID and source-bound
assets when generating a new plan. Final numerical status remains REVIEW.

The two verbose executable plans are losslessly gzip-compressed (`mtime=0`).
`plan-archives.json` records original byte sizes and SHA256. `gzip -dc` restores
the exact originals for inspection; do not overwrite old run IDs.

Raw pytest failure output preserves its original trailing spaces and checksum;
source whitespace checks exclude these immutable log bytes.
