# Iteration 366: C13 exact Generator rollout static context

## Hypothesis

C13 changes one default-off policy on top of C12. Inside each existing
Generator no-grad B1 rollout it reuses the immutable history patch/RoPE prefix,
the projected text tensor, and current-token RoPE keyed by the exact pyramid
layout. It still computes the changing current-latent patch and preserves all
DiT, attention, FFN and FSDP calls.

## Gates and evidence

| Gate | Run | Result |
|---|---|---|
| focused CPU tests | exact `02517f1` | `52 passed, 42 skipped` |
| validate-only | `iter366_gan_cuda_c13_staticcontext_validate_r1_20260830` | `16/16 PASS` |
| correctness/capacity | `iter366_gan_cuda_c13_staticcontext_correctness_r1_20260830` | `16/16 PASS` |
| formal timing | `iter366_gan_cuda_c13_staticcontext_timing_r1_20260830` | `138.243935311 s` |

The correctness summary SHA-256 is
`52447262f380a41ab7ddfd80a7b33ec38c4a29eacc53106138a8e7efea6932c7`.
Every rank reports G1/C5/EMA1 and final G201/C1005/EMA801. Each of the five
rollout batches passes the uncached B1 reference with zero max-absolute and
normalized-L2 error. The cache records five rollout scopes, five immutable
history/text fills, 22 reuses, 15 current-layout RoPE fills, 12 reuses, and
zero mutation, identity mismatch, unsupported layout, or grad-enabled use.
Maximum allocated/reserved memory is `24.963089466/38.443359375 GiB` per rank;
both cgroups retain `oom=0` and `oom_kill=0`.

The timing summary SHA-256 is
`78f530adbc8cad0755bd5c80a2ea5a730607200c7d4583d84bafac74a4000542`.
The result is `0.067852767 s` (`0.049106%`) slower than C12's
`138.176082544 s`. This is a correctness-valid negative performance result:
the exact static preparation removed by C13 is too small to overcome cache
bookkeeping/residency at Full5 scale. C13 is therefore **not promoted**;
C12 remains the fastest substrate, `17.065944441 s` above the requested target.

Raw rank receipts and the 93 MB summaries remain in the remote persistent run
directories. Only compact summary pointers are retained locally.
