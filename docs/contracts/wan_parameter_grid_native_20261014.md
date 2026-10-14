# Native Wan main-experiment parameter grid

Explicit workload admission, not an optimization of the workload itself:

- resolutions 384x640 / 448x832 / 704x1280;
- one Generator update and 1 / 3 / 5 independent Fake updates, one original
  EMA update per Fake update;
- configured pyramid budgets 1/1/1 or 2/2/2;
- world16, independent B1/rank, BF16 frozen base and FP32 LoRA/reduction;
- same current best plan, no tuning, merging, packing or profiler;
- fresh native source-generated trace per point, model/bootstrap fixed;
- common native modulo-aligned step30 starts the first G update;
- one process initialization, C1 warm-up, C2 only claimable.

The canonical five-update trace reader, worker, distribution default and
graph-validator default are unchanged. A separate typed workload/reader
admits new ratios. Graph execution counter expectations derive from the
actual fresh trace and update count; numerical tolerances do not change.

Preserve the source history-amplification branch. If native generation
produces rank-local different DiT inventories incompatible with the current
FSDP worker, mark trace admission FAIL and preserve it. Do not choose more
convenient exits, disable amplification, truncate the old F5 trace or enter
a simulated time in the paper. A completed CPU driver test is not a native
model or distributed GPU acceptance result.

Current status: source/CPU preparation; full native generation, Torch CPU
tests, held-node admission and real GPU matrix results pending.
# Shared-source static3D arm

The same grid spec and native trace can be passed to
`probe_wan13b_static_3d_world16.py --workload wan13b-grid`, with explicit
workload/source-admission paths and hashes. Defaults for canonical Wan and
Helios remain unchanged. Each DP replica still consumes eight independent B1
samples, TP4×DP2×PP2 remains fixed, and mean8/DP2 clipping/Adam/EMA boundaries
are unchanged. Ratios and calls derive from this trace, not canonical F5 counts.
This optional binding is NOT a targetGPU or timing result.

The newly approved historical A100 normal/global-synchronized benchmark scope
excludes rank-local amplified-history exits. Source-generation receipts must
disclose this restriction and differential-check every generated stage against
the original normal branch. It is NOT full native local-exception equivalence.
Every paired arm consumes the same fresh trace. Do not overwrite the earlier
failed unrestricted-native generation or reuse resized old tensor payloads.

