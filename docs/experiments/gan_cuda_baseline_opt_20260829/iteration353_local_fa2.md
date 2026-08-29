# Iteration 353: pinned local FlashAttention-2

## Frozen single-variable arm

C3 keeps the C2 world16 GAN-on workload and optimizer warmup unchanged.  Its
only variable is the attention implementation: the historical PyTorch SDPA
fallback is replaced by the pinned local FlashAttention-2 extension already
validated for this Helios/Torch stack.

Pinned identity:

- Kernel root: `kernels-community/flash-attn2` snapshot
  `6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c`.
- Extension SHA-256:
  `c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7`.
- Torch/CUDA: `2.10.0+cu128` / CUDA `12.8`.

## Gates

- Focused tests: `75 passed`.
- Validate-only: `iter353_gan_cuda_localfa2_validate_r1_20260829`, 16/16 PASS,
  summary SHA-256
  `9cb63d7300c7918761ea261fd195fa28331cd507ef88f0174f6ce3755f49b414`.
- Correctness: `iter353_gan_cuda_localfa2_correctness_r1_20260829`, 16/16 PASS,
  summary SHA-256
  `d239b73493f16df5ab5e703963ec0f440a6ee819421b21094a934bfd08557bbc`.
- Updates and versions: G1/C5/EMA1 and G201/C1005/EMA801.
- Generator Score inventory: four logical requests and four physical forwards;
  Score packing remains off.
- CUDA/cgroup OOM delta: zero.

## Formal timing

- Run: `iter353_gan_cuda_localfa2_timing_r1_20260829`.
- Summary SHA-256:
  `4a2dc68d633df66140149a3d5fa63bd209ca7d0c0affd291421ce90842815805`.
- Full5: `143.379703687 s`.
- Rank range: `143.369592862-143.375959625 s`.

| Comparison | Saved | Reduction | Speedup |
|---|---:|---:|---:|
| C3 vs C2 (`144.621745256 s`) | `1.242041569 s` | `0.8588%` | `1.0087x` |
| C3 vs C0 (`173.014483004 s`) | `29.634779317 s` | `17.1285%` | `1.2067x` |

C3 is promoted as the current fastest correctness-qualified substrate, but the
30% target is not met.  The remaining gap is `22.269565584 s`.
