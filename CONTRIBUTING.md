# Contributing

Bug reports, documentation fixes, model integrations and performance improvements
are welcome. For a larger change, open an issue describing the use case and the
proposed approach before starting implementation.

## Report a problem

Include the DMD-System revision, reproduction command, expected result and actual
error. For runtime issues, add the model/recipe, Python/PyTorch versions and
hardware configuration. Small log excerpts are usually sufficient; remove
credentials and private data before posting. A plan or experiment manifest is
useful when available, but is not required to report a bug.

## Development setup

Work in a source checkout and a separate Python environment:

```sh
python -m pip install -e '.[test]'
```

This supports source-only tests. For training or model changes, install the
[appropriate dependency group](docs/guides/model_families.md) in a compatible
environment. The standalone Skill uses only the standard library.

## Where code belongs

| Change | Location |
| --- | --- |
| Model factories and training state | `src/distillschedule/training/` |
| Model-native adapters | `src/distillschedule/adapters/` |
| Input preparation | `src/distillschedule/data/` |
| Program execution and planning | `src/distillschedule/open_workload/`, `search/` |
| Optimizer Skill | `skills/dmd-system-optimizer/` |
| User configurations and guides | `recipes/`, `docs/guides/` |

See [architecture](docs/architecture/repository_organization.md#framework) and
[extensions](docs/guides/extensions.md) before adding a model or backend.
Top-level `runtime/` still contains components used by current recipes; check
callers before moving them. Legacy Route-A/Route-B runners are retired.

## Tests

Automatic CI runs the fast core suite on Python 3.12: configuration, data,
command entry points, small-model updates, checkpoint recovery and packaging.
From the repository root, with `.[train,test]` installed:

```sh
# Skill installation, lookup and workflow tests (standard library only).
python -m unittest discover -s skills/dmd-system-optimizer/tests -v

# Core training and framework checks.
python scripts/run_framework_cpu_tests.py

# Installation and the public lifecycle outside the checkout.
python scripts/check_wheel_install.py
python scripts/check_framework_install.py --output installed-framework.json
```

Model-specific suites are not part of this repository's CI. Use the relevant
recipe and dependency group for actual training checks; do not install all model
extras into one environment. Some historical Skill checks need the exact Git
revisions recorded in its catalog.

## Performance and source-sensitive changes

Preserve workload semantics, gradients, optimizer/EMA updates and data order.
Compare performance using the same workload, hardware, warmup and timing scope;
include numerical checks for the affected path. A CPU test does not establish a
GPU speedup. Changes to the algorithm or precision need an explicit description
and their own comparison baseline.

Some native paths bind source or plan identities. When changing those paths,
update the relevant versioned contract and source-layout record as documented in
[repository organization](docs/architecture/repository_organization.md). Keep
original experiment receipts unchanged; add a correction or new measurement.

## Pull requests

Describe the user-visible change and how it was tested. Keep unrelated refactors
separate, use readable code rather than compressed one-line statements, and link
to shared documentation instead of copying it into several guides. AI-assisted
contributions are reviewed under the same correctness and maintenance standards.
Keep licenses and source attribution. Weights, datasets, raw traces, optimizer
states and private conversations belong outside Git.
