# H3 full local LoRA/Adam diagnostic v1

Default off; `H3_PLANNER_FULL_LOCAL_STATE=1` is legal only in an explicitly
source-bound `h3_divergence` correctness plan. It is exclusive with the raw
backward-repeat and full cache roundtrip instruments. The original native
Student1/Fake5 loop, Grid1000 policy, FA3, RNG, updates and numerical gates do
not change. This diagnostic grants no timing or numerical admission.

Each rank exports all of its named Student/Fake LoRA parameter **local shards**
and native Adam `step/exp_avg/exp_avg_sq` at initial, cycle1 and cycle2 boundaries.
Initial lazy optimizer state is honestly absent. It never calls `full_tensor`,
state_dict, all_gather, additional backward/model calls, or RNG mutation. Local
and global shapes, dtype, placements and original local stride are retained.
Non-LoRA parameters and unexpected optimizer fields fail closed.

A metadata-only prepass enforces a 2GiB tensor-byte ceiling per rank/snapshot.
Only then are detached copies made to CPU and serialized. Export is before
cycle1 or after the original cycle stop fence, outside every timer. Default
Torch CPU/CUDA RNG fingerprints must be unchanged across export. Both tensor
and metadata files are immutable, hash-bound and retained under the remote run
root. The collector requires all three real snapshots for every rank, validates
source/run/arm/rank, layout, path, byte count and file hash, and never writes
canonical numerical/timing admission. Large tensor files must not be downloaded.

This provides full **rank-local trainable/state** vectors, not a reconstruction
of the global model. Replicated optimizer step scalars must be reported
separately, never allowed to dominate a relative norm. A CPU comparison should
use matched depth2 control A/repeat B/depth1 arms and compare parameter error to
the bootstrap-to-current parameter movement, as well as Adam moments, at both
cycles. Self-noise and cross-depth metrics must be shown together. Two cycles
cannot establish long-term convergence or unbiasedness. No threshold is changed.

The copies can perturb inter-cycle scheduling. Native seconds must be stripped
from diagnostic receipts and cannot be used as C1/C2 performance results. The
old prefetch-depth numerical FAIL remains until an independently justified
conclusion; neither a matching hash nor a large same-depth noise floor alone
changes it.
