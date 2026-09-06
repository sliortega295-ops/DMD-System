# Best-Known Workload Source Map

This file records provenance only. GPU/server revalidation on the integration branch is **NOT_RUN**. Historical seconds are not measurements of the integrated HEAD.

| Workload | Source | Branch | Publication HEAD | Performance commit | Historical result | Boundary |
|---|---|---|---|---|---:|---|
| wan13b_nogan_resident_best | sliortega295-ops/DMD-System | agent/wan13b-lora-cast-20260824 | 631bc082c819d0728059e96663cdf31cd66d5b0b | 67b2880901abece947732ba5de45a5c2685ab274 | 26.227822833 s | single unprofiled diagnostic |
| helios14b_nogan_best | sliortega295-ops/DMD-System | agent/helios14b-modnorm-materialization-20260825 | 9c53d8e75b0981a3a9299e4f3bec2819ea105013 | ad9e16b83800463c069e87b562f2d07a41bbeab3 | 94.461699722 s | single unprofiled correctness-gated trial |
| minimax_h3_grid1000_best | sliortega295-ops/H3-DMD-for-A100 | agent/h3-grid1000-fa3-pageable-staged-cache-20260827 | 9100ec51cf741e9f01a836a8428a28a3d81838ec | cfecbb4188e1f3820451657f744aa2c823e24da4 | 772.332 s | single unprofiled matched run; approximate Grid1000 |
| helios14b_gan_best | sliortega295-ops/DMD-System | codex/gan-cuda-baseline-opt-20260829 | a78827099c89795416f9ac6c77284dd5bacb0383 | d865cdc0e5d7ee99fbe6102a5965de08c8c81b9d | 125.258735931 s | single unprofiled C25 trial |
| livewan_crossscale_pipeline_best | sliortega295-ops/DMD-System | agent/livewan-continuous-c21-20260901 | 54351707288c8d8a8d2efd789ece708852a07033 | c2a1ee389f7fe93dd9c289ab32068e41e5b9a90b | 54.441603646 s | actual global-B16 single measured middle cycle |

## Integration policy

- Shared helpers are selected by explicit file ownership; no branch is wholesale cherry-picked.
- Helios shared runtime uses the C25-compatible helper versions while canonical No-GAN flags keep GAN-only features disabled.
- MiniMax-H3 is ported into `src/distillschedule/backends/training/minimax_h3`; H3-DMD-for-A100 is not a runtime dependency.
- External code that remains external is pinned as a gitlink (`lightx2v`, `livewan`, `wan21`).
- C27 is not enabled by any canonical profile.
