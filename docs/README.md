# Documentation

## Start here

| Task | Guide |
| --- | --- |
| Install the framework | [Installation](guides/installation.md), [model-specific environments](guides/model_families.md) |
| Run a first example | [CPU training tutorial](../examples/framework/README.md) |
| Select a model and workflow | [Recipes](../recipes/README.md), [support matrix](guides/support_matrix.md) |
| Prepare inputs | [Data formats and preparation](guides/data.md) |
| Train, save, resume and export | [Training lifecycle](guides/training.md) |
| Choose an execution strategy | [Native presets](guides/native_presets.md), [official DMD2 and Self-Forcing](guides/official_optimized_training.md) |
| Read measured results | [Six workloads, official workflows and H100](guides/performance_validation.md) |
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

## Results and source records

[Performance](guides/performance_validation.md) collects the six workloads,
official workflows and H100 measurements, with baseline definitions and sources.
[A100 source records](status/GITHUB_SOURCE_SNAPSHOT_20261019.md),
[H100 source records](status/H100_SOURCE_SNAPSHOT_20261019.md) and the
[W6 metadata correction](status/SIX_WORKLOAD_RESULTS_ERRATA_20261031.md)
provide the historical source mapping.

Developer history: [legacy profiles](workloads/BEST_KNOWN_WORKLOADS.md),
[retired implementation recovery](architecture/legacy_route_retirement.md).
