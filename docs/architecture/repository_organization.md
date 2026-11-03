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
| `artifacts/results/` | Measured results and their source identities |
| `tools/`, `scripts/` | Developer utilities and command-line helpers |
| `benchmarks/`, `profiling/` | Isolated performance tests and profiling tools |
| `third_party/`, `licenses/` | Third-party sources, attribution and licenses |

The ownership sections below describe the boundaries inside the Python package. New training integrations start from the [extension guide](../guides/extensions.md).

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

[Route-A/Route-B retirement](#legacy_route_retirement) documents removed
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

<a id="framework"></a>
## Framework ownership and lifecycle

DMD-System separates the model, distillation algorithm, training state and
execution strategy. The user-facing command is `distillctl`; the Python package
is `distillschedule`.

```mermaid
flowchart TD
    Config[Configuration] --> Data[Prepared inputs]
    Config --> Model[Model and codec adapter]
    Data --> Workflow[Algorithm / Program]
    Model --> Workflow
    Workflow --> Training[Training session]
    Config --> Strategy[Reference, manual, auto or saved plan]
    Strategy --> Runtime[Execution backend / native worker]
    Training --> Runtime
    Runtime --> State[Parameters, optimizer, EMA, history, RNG]
    State --> Training
    Training --> Output[Checkpoint, export and generation]
```

### Code ownership

| Concern | Main interface | Responsibility |
| --- | --- | --- |
| Model | `training.ModelComponents` and model factories | Role modules, trainable parameters, prediction convention, codecs and export |
| Data | `data.PreparedDataset` and `InputStreams` | Prepared tensors, sample identity and stream progress |
| Algorithm | `DMDHooks`, `Program`, `Call`, or a native workflow | Loss, sampling, role updates, EMA schedule and dependencies |
| Training state | `training.session` and checkpoint APIs | Persistent state, metrics, complete-cycle save and recovery |
| Strategy | `ExecutionPlan`, capabilities, calibration and `search/` | Placement, state lifetime and legal execution order |
| Execution | `ExecutionBackend`, `StaticRoleRunner` and native workers | Device initialization, model calls and communication |

`DMDModelAdapter` is the typed native-invocation interface; `ModelComponents`
provides state to the public training lifecycle. Model-specific work belongs in
these adapters or factories rather than model-name branches in the generic solver.
A model's prediction convention (epsilon, velocity, flow or x0) must match its
algorithm. Official DMD2 and Self-Forcing retain their native update rules; the
common flow-DMD workflow is reused by the corresponding image/video/AV adapters.

### One cycle, persistent state

A Program describes one update cycle. Training retains parameters and optimizer
objects across cycles, advancing versions and data cursors. The 511-call Program
bound applies to a cycle, not the total training duration.

Calibration trials instead clone state and inputs together to isolate each
candidate and preserve optimizer-to-parameter references. After selection,
continuation uses new data and evolving parameters with the same validated
execution structure.

Checkpoints include role state, optimizer/scheduler/EMA, RNG and data progress.
Recovery is supported at complete-cycle boundaries with the same topology;
partial updates and arbitrary topology migration are not resumable checkpoints.
See [training and recovery](../guides/training.md).

### Execution strategies

`reference` selects the workflow's reference path. `manual` selects an explicit
supported implementation. `auto` measures candidates, searches feasible plans
and checks the selected result. `plan` executes a saved selection after checking
its source, state, input and cost identities. Support varies by model/backend;
see the [support matrix](../guides/model_families.md#support_matrix).

Generic Program training uses one device per process and explicit distributed
groups. Resident/DDP/FSDP2/HSDP state persists across cycles. Role pools and native
recipes have their own supported layouts and synchronization requirements.
Benchmark presets may be limited to two cycles; continuation recipes are separate.

A native call that contains an optimizer update is atomic and cannot be replayed
as a pure forward. Model-specific execution descriptors expose compatible
optimizations, such as DMD2 UNet checkpointing, without making the entire owner
generically shardable. Runtime records track the selected implementation and
executed effects; catalog discovery alone does not enable an optimization.

### Source layout

`training/` owns the public lifecycle, `open_workload/` owns Program execution,
and `search/` owns physical selection. `adapters/` and model-native workers keep
model-specific behavior. Top-level `runtime/` contains shared compatibility
components still used by current paths; the package's internal `runtime/` holds
framework execution policies. Check imports before moving either namespace.

`recipes/` contains user configurations, `examples/` extension samples, and
`tools/` and `benchmarks/` developer utilities. Metadata discovery stays lightweight
and imports no model libraries. Some native recipes need an external pinned
source tree or the source checkout; the common Program lifecycle also works
from an installed wheel. The [installation guide](../guides/installation.md)
describes the dependency groups and packaging boundaries.

Current source maps and measured result data are retained for reproduction.
Changes to source-locked paths use the source-layout update procedure.
Old experiment reports and process receipts are archived outside the published tree.

### Design references

[ReaLHF](https://openpsi-project.github.io/ReaLHF/arch) and
[HybridFlow](https://verl.readthedocs.io/en/latest/hybrid_flow.html) are references
for separating model implementations, algorithm control and distributed execution.
They are not runtime dependencies. See [extensions](../guides/extensions.md) for
DMD-System's own model, workflow and capability interfaces.

<a id="legacy_route_retirement"></a>
## Retired execution routes

Start from [current recipes](../../recipes/README.md) and `distillctl workloads list`.
Retired Route-A/Route-B runners are not part of the public runtime. Current shared
checkpoint, offload and multi-role helpers remain at their audited paths.
The [retirement manifest](legacy_route_retirement.json) records removed paths,
checksums and retained dependencies for source maintenance.
