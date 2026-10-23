# Development tools

`assets/` owns standalone trace/data/bootstrap preparation. `profiling/` owns Nsight parsing, plots, profiling commands and transfer probes. These are explicit developer tools, not alternative training implementations.

Run from the repository root. Moving a file does not authorize it to contact a cluster or use GPUs; consult `--help` and the owning runbook. The [migration map](../docs/architecture/repository_organization.json) records old paths and original hashes. Immutable historical receipts keep their original paths as provenance.
