# Experiment protocol and persistent state

## State without a second research operating system

Reuse the project's current run roots, receipts, checkpoint and decision ledger. The optional
`campaign.template.json` and `opportunities.template.json` are lightweight indexes, not a
replacement for the existing schemas. Keep raw files outside Git. Small final source/claim
manifests can follow repository conventions. Only one coordinator updates shared decisions.

Persist goal interpretation; original baseline; current verified control; workload/semantic/
timer/hardware/source identities; evidence references; remaining budgets; candidate status;
forecasts versus measurements; failed mechanisms and re-entry conditions. Reuse unchanged
records across sessions. Branch names and a new agent do not create a fresh budget.

Never edit frozen canonical profiles/locks or imported-source digests to conceal a performance
experiment. Prepare a distinct candidate/config using the repository-supported extension or
an isolated worktree. Where the native path deliberately forbids an override, first propose
a reviewed candidate interface preserving the existing canonical contract. Do not bypass its
validator, silently unlock the registry, or put performance knobs in assets JSON.

## Before allocating GPU time

Declare which question this run answers and the cheapest adequate evidence tier:

- **Static/CPU**: source and dependency checks, version DAG, hash and config validation,
  source/import audit, shape/memory/traffic analysis. No 14B construction just to test a launcher.
- **Primitive/trace probe**: bounded local performance, memory or scheduling uncertainty.
  Can be smaller than the final workload, but state what extrapolation remains unverified.
- **Functional target run**: real rank topology, workload calls, gradients/updates, numerical
  gates, checkpoint lineage and lifecycle. Functional PASS is not timing PASS.
- **Matched unprofiled timing**: only after appropriate functional/numerical acceptance;
  use the exact timer, warmup and aggregation of the workload.

Reuse valid previous evidence. Full cluster startup is not obligatory after each textual edit.
First reject impossible capacity/traffic models, then validate a representative mechanism,
then spend on the complete workload. Do not use mock/cached outputs or skipped updates for
final timing, and do not time only Critic when the target is Full5.

## Correctness questions specific to DMD

- Which operation consumes G@v/C@v/EMA@v? Can prefetch/overlap cross the version boundary?
- Does a shared Real/Fake Score base retain independent LoRA activation, mutable buffers,
  autograd graphs and mode state? A global adapter toggle cannot race with an in-flight call.
- Are frozen G and Score truly the same checkpoint? For Helios they are not.
- Does gradient flow through the intended G path? Has no-grad, detach or cache reuse severed it?
- Cache keys cover sample/shape/dtype/device, prompt/conditioning, timestep/sigma, weights and
  adapter versions, train/eval mode and randomness whenever they affect the cached value.
- Does batching change loss reduction, RNG draws, optimizer order, mixed-precision scaling,
  EMA timing, samples, staleness or update counts? Prove allowed equivalence; otherwise make
  a separately authorized semantic experiment with its own baseline and claim class.
- Can no-grad frozen weight/LoRA fusion be invalidated correctly at every update? Training
  LoRA gradients must not be silently folded away.
- Are CPU copies/pinned buffers reused only after consumers complete? Include double buffers,
  transient all-gathers, activation retention, fragmentation and checkpoint peaks in capacity.

## Fair timing and uncertainty

Use identical inputs/workload semantics, hardware allocation, software versions and timer
scope. For changed execution layouts, fix total resources and logical work; separately report
what physical plan differs. Time the window required by the contract, including relevant
fill/drain and boundary completion. Cold start and amortized warm state are separate metrics.
Never warm up by silently advancing the checkpoint that should start the measured window.

Functional instrumentation, FLOP counting and nsys/ncu overhead are not part of unprofiled
claim timing. Count independent runs, not correlated steps, as repeats. Predeclare repeat
budget and minimal meaningful effect based on control variability. Interleave matched A/B
when possible; retain all valid results, individual values and dispersion. Tiny or ambiguous
changes are INCONCLUSIVE, not a reason to select the fastest run. A single diagnostic run
may guide a next probe but cannot meet stronger repeat/claim requirements by itself.

The CPU helper's `compare` reports descriptive medians and paired differences only; it does
not produce significance or grant promotion. Changing a base, shape, window or relevant
hardware/software invalidates affected evidence and requires a fresh comparison.

## Promotion, budgets and safety

Per-direction default: two expensive full-scale attempts before re-ranking. Configure actual
GPU-hour/full-run/wall-time budget from the user's resource grant; absent grant = no autonomous
expensive launch. Give probes an explicit stopping rule. Failure, compilation and authorized
allocated idle time cost resources too; queue time costs wall time but not unallocated GPU-hours.

One runner owns an allocation, run-id, rendezvous port range, cache/output namespaces and child
process group. Record both hosts and all ranks; terminate only task-owned processes following
the existing coordinator. Do not kill another experiment, change system drivers, install
packages cluster-wide or send private artifacts to an external model/provider without permission.

GPU-memory pressure and a requested host reserve are not identical to actual CUDA/cgroup OOM.
Preserve the active contract's memory-admission rule. Failed/partial/missing-rank runs cannot
be promoted. Preserve numerical REVIEW and approximate labels; passing CPU integration does
not validate cluster behavior. Do not lower thresholds or replace goldens to pass.

After each completed batch, compare forecast with observed gain and explain discrepancy.
Retain the last verified control if a candidate is slower, wrong or inconclusive. Re-entry
requires new evidence or a changed cost structure, not another candidate name. Stop a doomed
route promptly, but do not mistake a poor first implementation for a physical impossibility.
