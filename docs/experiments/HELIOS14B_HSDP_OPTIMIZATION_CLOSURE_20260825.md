# Helios14B node-HSDP optimization closure (Iterations 368-382)

## Status and claim boundary

This report closes the current Helios14B optimization phase after the earlier
Iterations 351-367 ladder. The frozen workload is unchanged: 2 nodes x 8
A100-SXM4-40GB, No-GAN Full5 steps 10-14, 384x640 BF16, B1/rank and global
batch 16, Score packing off, node-HSDP `(replicate=2, shard=8)`, CUDA
Generator/Critic residuals, CUDA EMA, no parameter offload, streaming or model
swap, one Generator/five Critic/five EMA updates, and final `G3/C15/EMA15`.

The prior tested point was Iteration 367 at `96.529019750 s`. The new validated
winner is Iteration 380 at source
`ad9e16b83800463c069e87b562f2d07a41bbeab3`:

- strict correctness: 16/16 PASS;
- one formal unprofiled Full5: **94.461699722 s**;
- no CUDA/cgroup OOM;
- no repeat and no median were computed for this candidate.

This improves the Iteration 367 observation by `2.067320028 s` (`2.141657%`,
`1.021885x`) and the historical world16 FSDP2 median by `24.085057172 s`
(`20.316926%`, `1.254972x`). The historical medians and the newer one-trial
observations have different sampling strength and must not be presented as a
matched three-trial comparison.

## New experiment ladder

| Iteration | Candidate or audit | Formal Full5 | Decision |
|---|---|---:|---|
| 368 | post-materialization Nsight attribution | diagnostic only | compute-dominant; reject blind NCCL work |
| 369 | bypass checkpoint in 10/40 blocks for both roles | NOT_RUN | CUDA OOM in Generator grad path |
| 370 | Critic-only checkpoint bypass 10/40 | NOT_RUN | CUDA OOM in first Critic graph |
| 371 | Critic-only checkpoint bypass 4/40 | 96.341895211 s | fits, but only 0.1939% gain; reject as negligible |
| 372 | Generator no-grad history K/V cache | NOT_RUN | incompatible with live `restrict_self_attn=false` contract |
| 373 | exact per-rollout cross-attention K/V cache | 96.497110971 s | correct, only 0.0331% gain; reject |
| 374 | frozen-base QKV projection fusion | NOT_RUN | live long-sequence canary did not improve |
| 375 | host/launch-gap localization | diagnostic only | no bounded removable host class above 1% |
| 376 | selective checkpoint save of varlen FA2 output | NOT_RUN | target op absent from live graph |
| 377 | selective save of fixed FA2 output | NOT_RUN | target count zero; runtime was SDPA fallback |
| 378 | live attention-wrapper/backend audit | diagnostic only | found missing `LOCAL_KERNELS` export and SDPA fallback |
| 379 | explicit pinned local FlashAttention-2 | 95.725156637 s | **GO**, saves 0.803863113 s vs Iteration 367 |
| 380 | local FA2 + Critic selective FA2 output retention | **94.461699722 s** | **GO / current winner** |
| 381 | post-winner complete 16-rank Nsight attribution | diagnostic only | compute/kernel efficiency remains primary target |
| 382 | direct BF16 frozen cross-attention LayerNorm | 95.538186953 s | correct but 1.1396% slower than winner; reject |

Iteration 371 produced one additional timing receipt while the timing convention
was being corrected. Only `96.341895211 s` is the primary observation; the
additional receipt is preserved but excluded from all claims. All later formal
candidates use exactly one timing, as requested.

## Effective methods

### Explicit pinned local FlashAttention-2

Iteration 378 proved that the launcher had not exported `LOCAL_KERNELS`; the
supposed Flash path therefore printed `Flash Attn 2 / 3 is not installed!` and
ran PyTorch SDPA. Iteration 379 added a fail-closed backend selector and receipts
for the pinned extension identity. All 16 ranks observed `local-fa2`; exactly
one formal timing improved `96.529019750 -> 95.725156637 s`.

Primary code:

- `src/distillschedule/runtime/fsdp2_p0.py`
- `scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh`
- `scripts/distillschedule/probe_static_fsdp2_p0_world16.py`
- `scripts/distillschedule/summarize_static_fsdp2_p0_world16.py`
- `tests/test_refactor_helios_attention_backend.py`

### Critic selective FA2 checkpoint-output retention

With the actual local FA2 backend active, Iteration 380 retained the fixed-output
FA2 result for Critic blocks selected at stride 2 and reused it during
non-reentrant checkpoint replay. The per-rank fail-closed census was:

- 100 selected Critic checkpoint contexts;
- 200 original FA2 saves;
- 200 replay reuses;
- zero Generator selections and zero checkpoint bypasses.

The candidate preserved the original model calls, losses, RNG, optimizer and
update order. It improved `95.725156637 -> 94.461699722 s` in one formal run.
Peak allocated was `23.539903 GiB/rank`, maximum reserved
`34.527344 GiB/rank`, and minimum final driver free `1.762085 GiB/rank`.
Because headroom is narrow, denser retention was not attempted.

## Rejected and bounded-negative methods

- Checkpoint bypass trades recompute for activation residency. Ten retained
  blocks OOM for both all-role and Critic-only scopes. Four Critic blocks fit
  but saved only `0.187124539 s`, so fraction search was stopped.
- The source history K/V cache requires restricted self-attention, which is not
  the frozen live model. Enabling that mode would alter the attention contract.
- Exact cross-attention K/V reuse exercised 200 fills and 880 reuses per rank
  with zero mismatches, but saved only `0.031908779 s`.
- One fused QKV GEMM did not improve the dominant M6144 shape, so it was stopped
  before world16 implementation.
- The complete trace did not support another timing-clean/host-sync patch; the
  overwhelming no-traced complement remained unattributed rather than proven
  removable CPU idle.
- The direct BF16 cross-attention LayerNorm was bitwise in its isolated canary
  and 5.7-6.35x faster there, but the complete Full5 regressed to
  `95.538186953 s`. The microbenchmark result is not promoted to a system claim.

## Post-winner Nsight result

Iteration 381 profiled the exact Iteration 380 workload. The profiled controller
wall (`209.276159257 s`) is diagnostic-only and does not replace the formal
`94.461699722 s` result. Both node reports and SQLite exports were complete and
all 16 ranks reconciled.

Rank-median dominant phases were:

| Phase | Wall | non-NCCL compute union | NCCL-only | no traced activity |
|---|---:|---:|---:|---:|
| G no-grad rollout | 30.397814 s | 29.110184 s | 0.387856 s | 0.845623 s |
| Critic backward | 28.891887 s | 27.416802 s | 0.076851 s | 1.179409 s |
| Critic forward | 10.153473 s | 9.912788 s | 0.092420 s | 0.124490 s |
| G backward | 9.558409 s | 9.062887 s | 0.113042 s | 0.303112 s |
| Score/DMD forward | 7.673366 s | 6.071395 s | 0.037340 s | 1.502358 s |

Most NCCL activity overlaps compute. For example, pooled G no-grad all-gather
activity is `96.842199 GPU-s`, of which `90.611103 GPU-s` overlaps non-NCCL
kernels and `6.231096 GPU-s` does not. GPU-seconds are not wall seconds, and
non-overlap is not asserted to be causal waiting.

The pooled captured-activity complement is `7.946%`; only `0.062%` overlaps a
recorded blocking CUDA synchronization and `0.003%` a stream wait. The rest is
`UNKNOWN_NO_GPU_ACTIVITY`, not proven CPU idle. H2D is small and D2H is
effectively absent. Thus the current phase closes without another blind FSDP
prefetch, offload, or checkpoint-density experiment.

## Final result

| Variant | Evidence | Full5 | Peak allocated | Status |
|---|---|---:|---:|---|
| historical world16 FSDP2 | three-trial median | 118.546756894 s | -- | PASS |
| node-HSDP control | two-trial median | 117.304613315 s | -- | PASS |
| Iteration 367 fused materialization | one trial | 96.529019750 s | 19.7455 GiB | PASS |
| Iteration 379 local FA2 | one trial | 95.725156637 s | 19.7490 GiB | PASS |
| **Iteration 380 local FA2 + selective replay save** | **one trial** | **94.461699722 s** | **23.5399 GiB** | **PASS / WINNER** |
| Iteration 382 BF16 cross norm | one trial | 95.538186953 s | 23.5400 GiB | REJECTED REGRESSION |

The 85-second target was not reached. This Helios phase is intentionally paused
at Iteration 380 rather than escalating small or repeated negative hypotheses.
The next project focus is MiniMax-H3.
