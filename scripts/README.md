# Runtime entrypoints

Use `distillctl` or `python scripts/distillctl.py`. `workloads list` enumerates six current protocols; `workloads list --legacy` enumerates five historical locked profiles.

`distillschedule/` retains native launchers, required comparison/evidence helpers and source-audited baseline controllers. Standalone analysis/preparation/canary entrypoints have moved to `tools/` and `benchmarks/kernels/`. Those paths are developer-only, not new public trainers. `check_wheel_install.py` tests installed packaging outside the checkout.
