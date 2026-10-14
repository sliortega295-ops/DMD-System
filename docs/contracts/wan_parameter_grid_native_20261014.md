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
