# Six-workload source organization

Base checkout: `0e8cbe5c006fceda36a7547b3eda60d6daea1eb8`, after the shared `ea291293` runtime and `c823cb0` immutable publication data were integrated in `7bb5c89` on PR #34. The unapplied five-profile migration was retired, not applied to the newer tree.

## Public ownership

`workloads/catalog.json` enumerates six **protocols** and the matching local native entrypoints, documentation, tested source and hardware provenance. It is packaged so discovery works outside a checkout. `catalog.py` only reads/displays/verifies metadata; it cannot execute, contact a node, manufacture costs or choose a winning plan. Actual calibration/search/execution still uses existing `strategies`, `search`, `native` and `runtime` implementations.

Six recipe guides expose original Wan, the Helios safe/manual cohorts, LiveWan C51, full-sequence H3 Grid1000 and original H3 AR. Long legacy profile IDs and their five byte-locked JSON files remain compatible. `workloads list --legacy` makes the distinction explicit. The task schemas, allowed run modes, numerical thresholds, source/asset fences and defaults have not changed.

## File moves and source verification

[`repository_organization.json`](repository_organization.json) is the exact relocation map. Standalone asset/profiling/canary scripts move only when outside the source/config/launcher reference closure. Historical baseline controllers and model workers keep their audited paths. Mutable runbooks and tests follow moved paths; immutable publications, contracts, upstream code and raw receipts are not edited.

The earlier source checker failed because 32 legitimately updated files were still checked against first-import hashes. Those files were independently compared byte-for-byte to `ea291293` before this pass. The immutable `source_import_map.json` is retained. An explicit [`source_layout.lock.json`](../workloads/source_layout.lock.json) records each updated or relocated target, original imported hash, prior reviewed-source hash, exact provenance commit and current expected hash. Source checks reject mismatches; this is not disabling a gate or refreshing all hashes without an origin.

`runtime/` stays at its source-audited path, respecting the compatibility boundary. Setuptools now includes it in the same wheel; imports remain `runtime.*`. No duplicate copied package or path-injection shim is introduced. Its model/math bytes are unchanged.

## Packaging and validation

The old wheel check inherited `PYTHONPATH=src:.`, so pip could see checkout metadata and skip the install. `scripts/check_wheel_install.py` removes Python path/home injection, creates a new virtual environment, installs the wheel with isolated Python outside the checkout and checks real site-packages imports. The assertion is retained and strengthened, not skipped.

Run the catalog, import-map, relocation and legacy regression tests, plus an isolated wheel check. A later GPU validation requires newly qualified source/assets/plan receipts; no archived performance result is assigned to this source-layout commit.

## Remaining boundaries

The H100/manual derivative index includes separately pinned source trees and immutable external packets. They are not blindly merged into shared code or silently made defaults. W5's latest sampled backward FAIL still blocks its manual/saved-plan timing pair; W3 N70-OFF keeps the post-timer failure record. Final hardware promotion, broader script extraction and full numerical/long-run quality validation are separate tasks. Historical negative evidence is not deleted to make the tree look cleaner.
