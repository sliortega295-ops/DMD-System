# Migration from experiment branches

Do not checkout historical experiment branches to run a canonical workload. Initialize pinned dependencies once (`git submodule update --init --recursive`), prepare an assets JSON from the profile example, then use `distillctl workloads` or `run_best_known_workload.py`.

Historical branch launchers remain provenance. Canonical profiles own all performance-affecting flags and fail closed on missing assets, source drift, uninitialized gitlinks at actual run time, or noncanonical variants.
