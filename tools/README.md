# Development tools

`install_optimizer_skill.py` installs the self-contained Codex optimization
skill without replacing an existing version. See the
[optimizer guide](../docs/guides/optimization_capabilities.md#optimizer_skill).

`assets/` owns standalone trace/data/bootstrap preparation. `profiling/` owns Nsight parsing, plots, profiling commands and transfer probes. These are explicit developer tools, not alternative training implementations.

Run from the repository root. Moving a file does not authorize it to contact a cluster or use GPUs; consult `--help` and the owning runbook. The [source layout](../docs/workloads/source_layout.lock.json) records current path overrides; the [ownership guide](../docs/architecture/repository_organization.md#moving-or-updating-source-sensitive-files) describes updates. Historical receipts remain in Git history and the local archive.
