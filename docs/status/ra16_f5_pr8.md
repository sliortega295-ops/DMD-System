# RA16-F5-PR8 status

## Current status

**Bounded negative: Step10 Generator backward CUDA OOM.** Formal timing is
`Not Run`.

The live preflight on 2026-08-09 found the exact DMD source at
`9e66a7c4291b118ebd994d362b63f134db52f76a`, runtime at
`494229b99f66b36d6153a85c6a34e2445b667331`, clean checkouts, 59 passing
focused tests, no GPU compute process, and unchanged cgroup OOM counters on
both nodes. Node 1 has one old PID-1-owned `accelerate` zombie (`Z` state), but
it has no live command line and no GPU allocation; it is not treated as a
running workload.

PR4 and PR4-EC are immutable bounded negatives. Both failed at the same
Step10 Generator checkpoint recomputation allocation (216 MiB). PR8 asks a
new falsifiable capacity question by increasing only the streamed Generator
set from four to eight blocks.

## Measured run

`ra16_f5_pr8_20260809_001` passed the dual-host source/config smoke (63 focused
tests per host), world16 topology, construction, checkpoint-10 resume and
Step10 Generator forward/loss. Nine ranks then directly reported the same
216-MiB FP32 residual/FF allocation failure in checkpoint recomputation;
seven peers were terminated after the distributed failure. No Generator,
Critic or EMA update completed, steps 11--14 did not run, and checkpoint-15 was
not created. Neither node added a cgroup OOM/oom_kill event. Sampled cgroup
peaks were 738.983 GiB and 719.240 GiB.

The policy did take effect. Relative to PR4, median pre-backward allocated
memory fell from 32.787 to 30.169 GiB: exactly 2.618 GiB, the capacity of the
four additional streamed blocks. Median observed block materializations before
termination advanced from 20 to 47. PR8 therefore moved the failure farther
through checkpoint recomputation, but later gradients/recompute state refilled
the released capacity. More block streaming alone moved rather than removed
the backward peak.

## Per-rank memory analysis

The saved-tensor runtime's cumulative pack counter is 2.557 GiB/rank, but it
counts aliased saves. Storage-ID dedup gives 1.274 GiB/rank over 138 unique
storages. Forty BF16 `[1,2940,5120]` storages account for 1.122 GiB; each is
28.711 MiB, narrowly below the current 32-MiB offload threshold. These
one-backward-use activations are the strongest measured state class with lower
reuse value than a repeatedly materialized frozen block.

At the same boundary, 4.694 GiB is already allocated before the logical step,
and another 0.436 GiB of forward increment remains after subtracting unique
kept activations. Their ownership is not yet measured, so neither is labelled
redundant. In particular, the persistent floor may include trainable/private
or DeepSpeed buffers. A future owner census must establish whether any portion
belongs to inactive Score state during Generator backward before changing it.

The preflight's 2.557-GiB wording is explicitly corrected here: it was a pack
volume, not unique physical GPU storage. This correction came from alias-ID
evidence before drawing a policy conclusion.

Formal timing, allocator snapshot attribution, selective 28.711-MiB activation
offload and persistent-floor owner census remain `Not Run`. The one-attempt
PR8 budget is exhausted; no PR8 rerun or automatic successor search is
permitted. A later explicit user decision separately authorized one
`RA16-F5-PR12` attempt; that does not modify this bounded conclusion.

Curated evidence:

- `artifacts/manifests/ra16_f5_pr8_20260809_001_failure.json`;
- `artifacts/manifests/ra16_f5_pr8_20260809_001_result.json`;
- `artifacts/tables/ra16_f5_pr4_pr8_rank_memory_comparison_20260809.csv`;
- raw root recorded in the failure manifest.
