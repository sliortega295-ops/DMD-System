# Route-A Full5 world16 typed revalidation (2026-08-08)

The immutable workload is now defined in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md). The current evidence ledger is [`../status/routea_world16.md`](../status/routea_world16.md). This document retains the preregistration and attempt narrative for the August 8 integration cycle.

## Iteration preflight

- **Authorized hypothesis:** the typed plan/executor can reproduce the frozen Obsidian Route-A Full5 world16 workload and remain within a preregistered ±10% diagnostic timing band.
- **Frozen reference:** `task7_full5_routea_exact_b0_gs_fullstream_wave8_gate_20260806_001`; 688.7257 seconds for steps 10–14, or 137.7451 seconds per logical step.
- **Exact workload:** 2×8 A100-40GB, world16, 384×640, rank-qualified trace v3, checkpoint-10, steps 10–14, ZeRO-1 CPU optimizer, Generator/Score 40-block shared-immutable streaming window 1, Generator-selected A32 offload, swap wave 8 and EMA bootstrap updates.
- **Rejected paths:** single-A100 revalidation; reusing an old launcher that bypasses the typed code; ZeRO-2; mini workload substitution.
- **Falsifiable question:** after exact trace/config/lifecycle checks, does typed execution complete G2/C10/EMA10 → G3/C15/EMA15 without OOM and within ±10% of 688.7257 seconds?
- **Claim boundary:** one diagnostic-instrumented compatibility run, not a repeated performance distribution or formal speedup. Cold construction is excluded from the five-step timing window.

## Compatibility boundary

The typed code owns plan composition, protocol/version graph, lifecycle environment, rendered Helios config, node qualification and immutable compiled artifacts. The model math remains in the pinned validated `DistillSchedule/train_helios.py` compatibility entrypoint. This is a strangler-path parity experiment, not a claim that all Helios operations have been reimplemented.

A dry run exposed one semantic ambiguity before GPU launch: legacy Helios performs EMA bootstrap updates before the decay transition at `ema_start_step=750`. The compatibility schema therefore uses `ema_bootstrap_before_start=true`, yielding typed final versions G3/C15/EMA15 for steps 10–14. See the contract document for the exact interpretation.

## Launch attempt 001

**Failed before model construction.** The typed wrapper invoked the frozen Accelerate executable without prepending the same environment's `bin` directory to `PATH`; Accelerate could not locate `deepspeed`. Both node OOM deltas were zero and no training step ran.

## Launch attempt 002

**Failed at checkpoint world-size gate; no training step ran.** Each host independently formed a local world8 group. The world16 ZeRO-1 checkpoint correctly rejected the mismatch. This was not the known ZeRO-2 double-reduction failure.

The bounded recovery selected an explicit `standard` multi-node Accelerate configuration whose referenced DeepSpeed config remains ZeRO-1. The repository cleanup additionally requires a cheap 16-rank topology/NCCL probe before model construction.

## Launch attempt 003

At the moment Issue #2 was created, Attempt 003 had reached real world16 checkpoint resume. It had not produced a complete 16-rank trace, checkpoint-15 inventory or accepted summarizer output. Its repository status is therefore **Not Run**, not success and not a timing result. Creating the GitHub issue and adaptation branch did not interrupt the server process.

A later result may supersede this label only by linking an immutable curated manifest and updating the authoritative status page.
