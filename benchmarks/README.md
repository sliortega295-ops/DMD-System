# Benchmark and diagnostic entrypoints

`kernels/` contains explicit standalone kernel/state canaries relocated out of the native launcher directory. A microbenchmark success is not an end-to-end speedup or an accepted training recipe.

Required historical baseline controllers and configuration fragments remain at their audited paths in this pass. They are not removed solely because a filename contains an old experiment number: some remain dependencies of parity tests and native plans. Existing immutable result receipts stay under `artifacts/`; this directory is for code, not fabricated reruns.
