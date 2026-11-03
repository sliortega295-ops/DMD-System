# Documentation

## Start here

| Task | Guide |
| --- | --- |
| Install the framework | [Installation](guides/installation.md), [model-specific environments](guides/model_families.md) |
| Run a first example | [CPU training tutorial](../examples/framework/README.md) |
| Select a model and workflow | [Recipes](../recipes/README.md), [support matrix](guides/model_families.md#support_matrix) |
| Prepare inputs | [Data formats and preparation](guides/data.md) |
| Train, save, resume and export | [Training lifecycle](guides/training.md) |
| Choose an execution strategy | [Native presets](guides/native_workloads.md#native_presets), [official DMD2 and Self-Forcing](guides/native_workloads.md#official_optimized_training) |
| Read measured results | [Six workloads, official workflows and H100](guides/performance_validation.md) |
| Optimize another project with Codex | [Optimizer Skill](guides/optimization_capabilities.md#optimizer_skill) |
| Diagnose a failed run | [Troubleshooting](guides/installation.md#troubleshooting) |

## Development and execution

[Architecture](architecture/repository_organization.md#framework) explains code ownership.
[Extensions](guides/extensions.md) covers model, algorithm and execution plugins;
[CONTRIBUTING](../CONTRIBUTING.md) describes how to submit and test changes.

For distributed policies and automatic planning, see
[execution capabilities](guides/optimization_capabilities.md),
[Program](workloads/open_workload_program.md), [role pools](workloads/open_role_pools.md)
and [native planning](runbooks/native_planner_workflow.md).
The [repository map](architecture/repository_organization.md) lists developer tools.

## Versioned native contracts

Use the recipe's exact policy, not a similarly named model's ABI:
[output/publication](contracts/native_output.md),
[Helios history and checkpoints](contracts/helios_native.md),
[H3 requests and AR state](contracts/h3_native.md),
[LiveWan inputs and reference state](contracts/livewan_native.md),
[construction and retention](contracts/state_retention.md), and
[dependency overlays](contracts/python_dependency_overlays_v1.md).

## Performance

[Performance](guides/performance_validation.md) contains the six workloads,
official workflows and H100 measurements, grouped by baseline definition.
For profile inspection and reproducible commands, use the
[versioned profile guide](workloads/BEST_KNOWN_WORKLOADS.md).
