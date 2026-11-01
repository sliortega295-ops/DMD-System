# Documentation

## Start here

| Task | Guide |
| --- | --- |
| Install the framework | [Installation](guides/installation.md), [model-specific environments](guides/model_families.md) |
| Run a first example | [CPU training tutorial](../examples/framework/README.md) |
| Select a model and workflow | [Recipes](../recipes/README.md), [support matrix](guides/support_matrix.md) |
| Prepare inputs | [Data formats and preparation](guides/data.md) |
| Train, save, resume and export | [Training lifecycle](guides/training.md) |
| Use optimized official workflows | [DMD2 and Self-Forcing](guides/official_optimized_training.md) |
| Optimize another project with Codex | [Optimizer Skill](guides/optimizer_skill.md) |
| Diagnose a failed run | [Troubleshooting](guides/troubleshooting.md) |

## Development and execution

[Architecture](architecture/framework.md) explains code ownership.
[Extensions](guides/extensions.md) covers model, algorithm and execution plugins;
[CONTRIBUTING](../CONTRIBUTING.md) describes how to submit and test changes.

For distributed policies and automatic planning, see
[execution capabilities](guides/optimization_capabilities.md),
[Program](workloads/open_workload_program.md), [role pools](workloads/open_role_pools.md)
and [native planning](runbooks/native_planner_workflow.md).
The [repository map](architecture/repository_organization.md) lists developer tools.

## Validation and research records

The [support matrix](guides/support_matrix.md) links current validation records.
Dated reports under `status/` describe the code, hardware and test scope at the
time of each run; they are not a substitute for the current recipe guides.

Historical performance and source indexes are available for
[A100 and saved-plan experiments](status/GITHUB_SOURCE_SNAPSHOT_20261019.md) and
[H100 experiments](status/H100_SOURCE_SNAPSHOT_20261019.md).
Read the [six-workload results](status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md)
with the [W6 metadata correction](status/SIX_WORKLOAD_RESULTS_ERRATA_20261031.md):
r46 restored metadata reads, rather than removing them. These records retain
numerical REVIEW/FAIL states and distinguish matched measurements from historical comparisons.

[Legacy profiles](workloads/BEST_KNOWN_WORKLOADS.md),
[retired Route-A/Route-B implementations](architecture/legacy_route_retirement.md)
and [retired unlinked reports](architecture/retired_unlinked_reports_20261031.md)
remain documented for historical investigations. Their provenance is preserved.
