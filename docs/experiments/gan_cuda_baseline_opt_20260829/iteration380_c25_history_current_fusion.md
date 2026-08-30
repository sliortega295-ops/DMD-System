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
