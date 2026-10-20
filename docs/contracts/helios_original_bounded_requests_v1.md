# Original Helios 14B bounded requests (opt-in)

This adapter executes the existing No-GAN/GAN training helpers, not Wan1.3B or
a generic DMD/Euler replacement. Canonical profiles and old generated identities
are unchanged. A request has its own identity, input bundle, measured costs and
serialized native plan; historical seconds cannot be reused as its costs.

Admitted dimensions: 384×640, 17 or33 video frames, active three-stage budgets
(1,1,1) or(2,2,2), world16, physicalB1/global16, G1/C5. No-GAN has EMA5;
GAN has EMA1 and retains its physical/logical checkpoint-version mapping.
Different ratios and physical8/global16 are rejected until their original
training implementations are supported, not relabelled as ordinary world8.

`OriginalHeliosRequest` is propagated through candidate IDs/assignments,
profile semantics, component cost keys, fresh trace metadata, actual helper
geometry, generated model inputs, observations and final receipts. GAN's
logical B1 fake/real/R1 pack remains physical B3 with half-spatial crops.

The exact asset identity is `trace/bundle.json` file SHA256. Its embedded
content hash is independently verified against16 request-qualified rank
manifests and bound to launch arguments/environment. Local-launch admission
compares against that verified hash, never an unconstrained environment
exception or an old canonical trace sentinel.

Default scope `source_normal_only` rejects all-zero-history amplified traces.
Explicit `benchmark_v4_strict_global` retains original sampled history and
per-rank native (possibly doubled) scheduler grids, but uses the historical
rank0-synchronized physical exit/call-count benchmark scope. This is not a
claim to reproduce upstream rank-local amplified exit probabilities. Each
trace records the complete native shifted grid, executed prefix, history
predicate and request hash. Before model construction, the worker recomputes
the grids from pinned original source and verifies hash-checked history sums.
Actual model timesteps/shapes are checked after fenced cycles, outside timers.

Native GPU validation is a separate gate. CPU tests, plans and this document do
not establish capacity, numerical equivalence or speedup for a new request.
