# Iteration 307: explicit Transformer-block ZeRO-3 leaves

**Status:** exact correctness PASS and one preliminary low-overhead timing PASS;
formal repeats Not Run.

The scalar 8M threshold was rejected because it also grouped mixed-dtype
embedding modules. The accepted candidate instead uses DeepSpeed's explicit
leaf API for exactly the 40 `HeliosTransformerBlock` instances. No input,
dtype, graph, loss, optimizer, trace, version or placement changed.

## Exact continuous Full5

Run: `rb16_f5_iter307_explicit_block_leaf_correctness_20260810_001`

- all 16 ranks PASS
- actual leaf inventory: 40/40 Transformer blocks and no other class
- G3 / C15 / EMA15
- 60/60 payloads
- OOM delta: zero
- summary SHA:
  `1ff4b4a11d81053f3149d0a7f24bf9b7d6b51bc0bfc7f18ad8b22ea969f12377`

| Boundary | seconds |
|---|---:|
| Step10 Generator-rank max | 102.884 |
| Step10 world boundary max | 106.780 |
| Transition | 21.522 |
| Critic | 168.798 |
| Full5 | 300.395 |

## Low-overhead Step10 diagnostic

Run: `rb16_step10a_iter307_explicit_block_leaf_timing_20260810_001`

- Generator-rank median: **87.995 s**
- Generator-rank max: 87.995 s
- world boundary max: **92.606 s**
- node cgroup peaks: 356,848,963,584 and 79,602,139,136 bytes
- GPU peaks: 32,712 and 36,754 MiB

Relative to the Iter297 best preliminary Generator median (115.490 s), this is
23.81% less time / 1.312x throughput. Relative to the strict world-max anchor
129.004 s, the world boundary is 28.21% shorter / 1.393x throughput. The
85--105 s engineering target is met.

These are not yet three-run formal results. The next gate is an independently
launched three-repeat low-overhead Full5 evaluation bound to the exact summary.
