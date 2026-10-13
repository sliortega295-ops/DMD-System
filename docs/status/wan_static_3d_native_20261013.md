# Wan native static 3D baseline — pending cluster validation

This isolated branch implements the requested **ReaLHF-like placement
baseline**, not ReaLHF itself and not a timing simulation. No default profile
or planner candidate is changed.

## Frozen contract

- Two nodes, TP4 inside a node, DP2, PP2 across nodes.
- Eight independent B1 microbatches per DP replica; logical global batch 16.
- Source-qualified Wan/Helios transformer: 30 blocks, split 15/15.
- Original Generator update followed by five `(Generator rollout, Critic
  update, EMA)` operations, without phase clustering.
- Frozen base BF16; unmerged trainable LoRA rank 256 and reductions FP32.
- Immutable bootstrap and all 16 rank-qualified traces. Rank-local history
  choices, noise, timestep values and schedule budgets are **not** replaced by
  rank0's values.
- Every physical model invocation is B1. The eight-row transport container is
  split into eight B1 GPipe calls, not executed as a B8 DiT.
- No merged shadow, CUDA Graph, fused-LoRA epilogue, split-modulation cache,
  DMD-specific kernel candidate, batching or schedule reorder is enabled.

## Execution seam

`probe_wan13b_static_3d_world16.py` calls the original B1 host samplers through
`run_native_B1_cohort`. Only matching DiT boundaries rendezvous on the owning
thread. Each sample retains its own sampler and immutable operation trace.
The transformer forward is split mechanically from the SHA-qualified source;
prefix and postfix trainables have unique PP ownership. Five native tensor
fields and their gradients cross the PP seam.

Native losses average eight samples per DP replica, followed by DP SUM / 2.
The external-loss GPipe bridge does not divide those gradients by eight again.
Global clipping counts replicated TP trainables once; Adam and EMA retain the
reference equations. Checkpoint replay is not an application-level call.

## Evidence boundary

Both-node CPU fixtures cover TP LoRA/RMSNorm, PP forward/backward,
multi-role external Score loss, pyramid input gradients, and rank-local native
samplers. A separate actual-runtime fresh-process import/asset gate is required
before a GPU launch. CPU fixture timings are not training timings.

The native world16 two-cycle capacity run and C2 baseline timing are **NOT_RUN**
at this commit. Numerical trajectory revalidation is deferred by the user, not
promoted to PASS. A completed launch additionally requires 16 final receipts,
unchanged call/update/sample/version contracts, finite values, no new OOM and
both sealed node-owner cleanup receipts.

Evidence root:
`/home/lyy/Helios/research_artifacts/DMD-System/main_experiments_20261013/`.
