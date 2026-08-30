# Iteration 380: C25 exact history/current add-and-cat fusion

## Selection and contract

C25 returns to the valid C21 `125.814004444 s` substrate after C24 closed the
allocator no-flush path.  The C22 profile and pinned Helios source identify an
exact repeated data-movement chain in every non-NaViT guidance-cross-attention
block: add attention into the current suffix, concatenate the unchanged
history prefix, and later split/copy that gradient during backward.

The candidate changes only that chain.  One Triton forward writes the unchanged
history prefix and BF16 sum suffix directly.  The custom backward returns the
full upstream gradient for the pre-split hidden tensor and its suffix for the
attention output.  Attention, norm, NaViT, FSDP, checkpoint, trace, loss,
update, version and timer contracts remain frozen.

## Locked-A100 canary

`iter380_gan_cuda_c25_history_current_canary_r1_20260830` ran on one owned
A100-SXM4-40GB in the validated Torch 2.10 environment.  Its immutable result
SHA-256 is
`e12c180b489da73e8d3d50a684f3867b58c91ef3c12486abada0f8ff16302c38`.

| Shape | Native forward+backward | C25 | Speedup | Output / hidden-grad / attention-grad |
|---|---:|---:|---:|---|
| B1, H512+C2048, W5120 | 0.357854 ms | 0.225314 ms | 1.5882x | bitwise equal |
| B3, H512+C2048, W5120 | 1.388100 ms | 0.662699 ms | 2.0946x | bitwise equal |

The preregistered `1.15x` all-shape gate passes.  The source rewrite also
passes against the actual pinned Helios transformer file SHA
`5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`,
and 116 focused tests pass in the exact remote environment.

## Decision

**PROMOTE_TO_WORLD16_GATES**, not yet a performance promotion.  The next steps
are validate-only, a fresh 16/16 correctness/capacity run with live shape/call
receipts and zero fallback, then one unprofiled formal timing only on PASS.

## World16 gates and formal result

All world16 runs used exact code HEAD
`d865cdc0e5d7ee99fbe6102a5965de08c8c81b9d`, the frozen C21 policy, and no
Nsight instrumentation.

| Gate | Run | Result |
|---|---|---|
| validate-only | `iter381_gan_cuda_c25_history_current_validate_r1_20260830` | 16/16 PASS |
| correctness/capacity | `iter381_gan_cuda_c25_history_current_correctness_r1_20260830` | 16/16 PASS |
| unprofiled timing | `iter381_gan_cuda_c25_history_current_timing_r1_20260830` | **125.258735931 s** |

- Validate summary SHA-256:
  `5652bd0fdddefc830e702c6387c997aea1bc66212d1c71b86b5851485b67fc8d`.
- Correctness summary SHA-256:
  `1712b0ef48743f3e840787eed6efd1715ee44b25d1d9d536f8e46adcbfa3db11`.
- Timing summary SHA-256:
  `d30f7beab5b72d710c9fcf9d9c9023a3aadcd0e24d0f27dd7882bc81e1fc7e20`.
- Every rank finishes G201/C1005/EMA801 with G1/C5/EMA1 updates.
- Timing receipts report 2,360 fused calls per rank: 1,120 grad-enabled and
  1,240 no-grad.  All observed B1/B3 shapes are supported and reference
  fallback is zero.
- The C21 persistence contract remains intact: 41 units fully unsharded inside
  the bounded rollout phase, 41 explicit reshards, and zero unsharded units at
  phase exit.
- Maximum rollout allocated/reserved memory is approximately
  `37.0057/38.5938 GiB`; CUDA and cgroup OOM deltas are zero and both nodes are
  clean after teardown.

## Performance decision

| Denominator | Full5 seconds | Change |
|---|---:|---:|
| C0 | 173.014483004 | -- |
| C21 | 125.814004444 | -- |
| C25 | **125.258735931** | -0.555268513 s / -0.4413% vs C21 |
| 30%-reduction target | 121.110138103 | C25 remains +4.148597828 s |

C25 is promoted as the fastest correctness-qualified intermediate result.  It
reduces C0 by `27.6022%` (`1.38126x`) but does not yet meet the registered 30%
target.  The end-to-end gain is deliberately reported from the formal run, not
extrapolated from the much larger isolated-kernel canary speedup.
