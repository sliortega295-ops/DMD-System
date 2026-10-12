# Wan1.3B merged-LoRA shadow noise study (opt-in)

This study replaces the single-row P0 numeric gate **only for this explicitly
registered mechanism**. Default correctness, P0 thresholds and trajectory
execution remain unchanged. It does not approve any candidate before results.

All arms use one exact HEAD, world16 B1/global16 G1/C5/EMA5, immutable bootstrap
and rank-qualified trace, BF16 frozen base with FP32 training LoRA, fused LoRA
epilogues and normalization ON. CUDA Graph is OFF in every arm: it is a separate
mechanism and cannot be fairly combined with merged execution weights.

A/A' are identical controls. B rebuilds round-to-nearest merged execution weights
from the frozen base after each G commit, only for 27 post-G no-grad DiT calls.
Optional C uses stochastic BF16 rounding with a separate per-rank/version/layer
`torch.Generator`; no training RNG consumption. Four original FP32 base modules
remain FP32. G gradient path, Critic, reducer, optimizer and EMA are unchanged.

`--shadow-trajectory-study --trajectory-cycles 30` enables rank0 evidence only:
full FP32 trainables at cycles 0/5/10/20/30, a fixed original/unmerged G probe after
every cycle, and rounding sufficient statistics per layer/refresh. These probes,
full snapshots, numeric anchors and serialization are outside the diagnostic
training-window timer. All ranks participate in barriers around evidence, not in
extra training collectives. Probe calls have a separate census and are included
in raw fused-epilogue accounting, never in the application training census.

The externally hashed preregistration fixes the 2x self-noise bound, sustained
anchor-growth rule, bias and lost-ratio tests, zero-noise handling, frozen input
identities and the two-hour GPU budget before any GPU workload. The existing
`trajectory_coarse`/Wan comparator supplies losses, gradient-norm curves and
optimizer/EMA sketches; the study supplements it with full LoRA parameter and
original-path probe comparisons. No absolute numeric thresholds are relaxed.

Timing is permitted only with a hashed `shadow_noise_acceptance.v1` PASS embedded
in the source comparison receipt, binding this HEAD, accepted arm IDs, complete
20/30-cycle receipts and immutable preregistration. Legacy timing bindings still
require the original two-cycle correctness run. Native timing itself remains
exactly two cycles; no study probes, rounding audit or snapshots run in its timer.

Large snapshots stay on the server. No weights, generated environments, or raw
checkpoints are committed. Local study commits are not pushed by this task.
