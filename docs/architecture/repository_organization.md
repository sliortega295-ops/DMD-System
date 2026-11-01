# Repository organization

## Working directories

| Directory | Purpose |
| --- | --- |
| `src/distillschedule/` | Python package: model/data adapters, training lifecycle, Program execution and planning |
| `recipes/` | Model/workflow configurations and user guides |
| `skills/dmd-system-optimizer/` | Standalone Codex Skill, public experience library and lookup tools |
| `runtime/` | Shared compatibility components used by current native training paths |
| `tests/` | Unit, integration, CPU and opt-in GPU tests |
| `docs/guides/` | Current user documentation |
| `docs/contracts/`, `docs/runbooks/` | Native execution contracts and operating procedures |
| `docs/status/`, `artifacts/` | Dated validation reports and curated experiment records |
| `tools/`, `scripts/` | Developer utilities and command-line helpers |
| `benchmarks/`, `profiling/` | Isolated performance tests and profiling tools |
| `third_party/`, `licenses/` | Third-party sources, attribution and licenses |

See [framework architecture](framework.md) for the boundaries inside the Python
package. New training integrations start from the [extension guide](../guides/extensions.md).

The workload catalog identifies W1–W6 for benchmark reproduction; model discovery
and [recipes](../../recipes/README.md) also cover official DMD2, official
Self-Forcing and additional flow-DMD models. Five older locked profiles are
available through `distillctl workloads list --legacy`.

## Moving or updating source-sensitive files

Some imported native implementations and experiment records carry source hashes.
Before moving a file, inspect imports, launch commands, tests and the source maps.
In particular, top-level `runtime/` is packaged as `runtime.*` and remains a live
dependency; it is not an archive directory.

Source-checkout metadata records previous file moves in
`docs/architecture/repository_organization.json`, original imported identities in
`docs/workloads/source_import_map.json`, and reviewed changes in the append-only
`docs/workloads/source_layout.lock.json`. Each layout entry retains the original,
previous reviewed and new expected hashes. Keep provenance when updating a source-sensitive path; do not
rewrite original experiment receipts to describe a new implementation.

[Route-A/Route-B retirement](legacy_route_retirement.md) documents removed
launchers and their exclusive code. It does not retire the FSDP2, offload or
role-scheduling mechanisms still used by current implementations.

## Packaging checks

`pyproject.toml` declares package code, licenses and installed documentation.
Public guide links must resolve either within that installed documentation or
to an explicitly named source repository; development scripts are not all
included in the wheel.

From a source checkout, run:

```sh
python -m pytest -q tests/test_repository_organization.py tests/test_workload_catalog_validation.py
python scripts/check_wheel_install.py
```

The wheel check creates an isolated environment outside the checkout and removes
inherited `PYTHONPATH`/`PYTHONHOME` settings. This checks installation and imports;
model-specific execution and numerical validation use their separate test lanes.
