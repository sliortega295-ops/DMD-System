# W6 native capacity policy v2: explicit node host limits

This versions only capacity metadata for the original H3 AR adapter. The task,
training math, geometry, update ratio, timer, source/asset identity gates and
other workload defaults do not change. New calibration and saved execution
must use one unchanged source HEAD; older GPU evidence is not rewritten.

The existing unversioned v1 policy remains accepted with exactly four scalar
integer fields: `gpu_limit_bytes`, `gpu_reserve_bytes`, `host_limit_bytes` and
`host_reserve_bytes`. It still rejects unsupported capacity observations.

V2 has exactly five top-level fields:

- `schema_version`: `distillschedule.original_h3_ar_capacity.v2`;
- shared `gpu_limit_bytes` and `gpu_reserve_bytes`;
- `host_limits`: exactly string keys `"0"` and `"1"`, positive integer bytes;
- shared `host_reserve_bytes`.

No default limit or reserve is supplied. Every rank resolves its node using
`rank // 8`; an absent, invalid or mismatched node is rejected. Receipts retain
the complete identical policy on all16 ranks and record explicit node/host-limit
bindings inside memory evidence. Assembly, calibrated hardware/capacity binding,
saved-plan validation and execution preserve the full map, including the peer
node's limit; node-local resolution never replaces the saved common policy.

A missing `memory.peak` remains `None`/`UNAVAILABLE`, never current-as-peak. Each
node must independently prove its declared limit equals its **fresh observed
finite hierarchy hard limit**, with maintained host reserve 0 and unchanged
OOM/oom_kill counters. A historical process `memory.max` snapshot, physical
capacity alone, or `min(node0,node1)` is not a replacement for that observation.
Available lifetime peaks retain their existing current/highwater/reserve checks.
The separate original 64 GiB entry headroom gate remains unchanged. GPU capacity,
shared reserves and every numeric gate retain their meaning; v2 does not relax
or infer any number.

CPU tests cover distinct exact limits with unavailable peaks, explicit wrong-node
rejection even for equal limits, v1 compatibility, schema rejection, full-policy
assembly/plan tamper rejection, and the actual worker saved-plan branch on both
node ranks. No target GPU or full-model numerical result is established here.
