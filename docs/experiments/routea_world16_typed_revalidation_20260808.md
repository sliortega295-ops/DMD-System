# Route-A Full5 world16 typed revalidation (2026-08-08)

## Iteration preflight

- **Current gate:** compatibility validation of the cleaned DMD-System runtime.
- **Authorized hypothesis:** the new typed plan/executor can reproduce the frozen Obsidian Route-A Full5 world16 workload and remain within a preregistered ±10% diagnostic timing band.
- **Frozen reference:** `task7_full5_routea_exact_b0_gs_fullstream_wave8_gate_20260806_001`; 688.7257 s for steps 10–14, 137.7451 s/logical step.
- **Exact workload:** 2×8 A100-40GB, world16, 384×640, rank-qualified trace v3, checkpoint-10, steps 10–14, ZeRO-1 CPU optimizer, Generator/Score 40-block shared-immutable streaming window 1, Generator-selected A32 offload, swap wave 8, EMA bootstrap updates.
- **Rejected paths:** single-A100 revalidation (scope mismatch); reusing the old launcher (does not test new code); ZeRO-2 (known double-reduction mechanism).
- **Falsifiable question:** after exact trace/config/lifecycle checks, does the typed execution complete G2/C10/EMA10 → G3/C15/EMA15 without OOM and within ±10% of 688.7257 s?
- **Decision:** GO only after tests, clean commit, live two-node GPU audit, and immutable preregistration.

## Compatibility boundary

The new code owns plan composition, protocol/version graph, lifecycle environment, rendered Helios config, multi-node rank qualification, immutable compiled artifacts, launch locks, and result summarization.  The model math remains in the pinned validated `DistillSchedule/train_helios.py` compatibility entrypoint.  This is a strangler-path parity experiment, not a claim that all Helios operations have been reimplemented.

A dry run exposed and fixed a real semantic gap before GPU launch: Helios executes EMA bootstrap updates before `ema_start_step=750`, while the initial generic protocol treated that field as an operation start.  `ema_bootstrap_before_start=true` now yields the correct final typed version `G3/C15/EMA15`.

## Claim boundary

This is one diagnostic-instrumented revalidation launch. It can establish functional and approximate timing consistency, not a formal repeated performance distribution or speedup. Cold construction remains excluded from the five-step comparison.
