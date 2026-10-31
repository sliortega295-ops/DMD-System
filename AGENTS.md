# Working in DMD-System

Start from the user's goal; do not read every historical report or build another
trainer, planner or experiment-management layer.

- **Optimize an existing workload:** use `skills/dmd-system-optimizer/SKILL.md`.
  The repository's `.agents/skills/` entry points to that same copy. Use the
  project's runner/checker and WORK_LOG; the experience library is optional help.
- **Change training:** use [framework ownership](docs/architecture/framework.md)
  and the relevant recipe. Public lifecycle is in `src/distillschedule/training/`,
  Program execution in `open_workload/`, physical selection in `search/`, and
  model-native compatibility remains at its audited paths.
- **Change a model or workflow:** start from [extensions](docs/guides/extensions.md).
  Algorithm mathematics, state ownership and physical execution have different owners.
- **Verify a change:** follow [CONTRIBUTING](CONTRIBUTING.md) and the affected
  workflow under `.github/workflows/`. Skill-only tests use Python's standard
  library; model test lanes require their own pinned environment.

Reuse current implementations first. Do not resurrect retired Route-A/Route-B
or remove a live runtime merely because it is old. Keep source receipts,
licenses, numerical checks and user data intact. Report the scope actually tested;
CPU fixtures are not evidence of GPU speed or complete training quality.
