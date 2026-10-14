# PP2 + FSDP2 + replicated DP reference

This is a new opt-in placement experiment, **not** the previous TP4/DP2/PP2
baseline and **not** a result for the RealHF implementation.

The bounded capacity ladder is PP2/shard4/replica2, then PP2/shard8/replica1
only after an actual CUDA capacity failure. Both use sixteen owned A100s,
eight independent data lanes and two physical B1 waves. Source sample IDs are
lane0..7 then8..15; a PP peer is the same sample, never an extra sample.
Logical global batch16, loss mean16, gradient clipping over both stages,
original optimizer/EMA update boundaries and per-sample call census remain
unchanged. Increasing shard degree does not change source sample mapping.

FSDP units remain native blocks, bottom-up, plus the uniquely owned stage
root. The root's prefix preparation is a registered FSDP forward method;
stage1 must also enter the root's forward hooks. No root weight access may
bypass all-gather/reshard. No TP projection, adapter merge, model swap or
new activation policy is installed. Existing workload-specific policies must
come from exact source receipts, not be guessed from another model.

Timing is one unprofiled two-cycle process: C1 warm-up, C2 claim denominator.
Construction, serialization and correctness bookkeeping remain outside the
timer. Every cycle must account for sixteen independent logical source
samples per update and one original optimizer commit. All sixteen rank and
two owner/cleanup receipts are required. OOM is reported, never hidden by a
lower resolution, shorter sequence, fewer calls or changed batch.

Initial native bindings target Wan and Helios No-GAN. GAN feature taps,
LiveWan causal/KV state, packed H3 Grid1000, and H3 AR previous/current inputs
require their own real source-bound stage adapters. **An implemented layout
or CPU test is not a six-workload GPU result.** The registry must preserve
NOT_BOUND/NOT_RUN until these adapters and their GPU receipts exist.

Cluster status at initial implementation: **NOT_RUN**. Numerical rechecking
remains separately labelled under the user's existing main-experiment policy;
runtime census PASS is never silently promoted to full-state parity.
