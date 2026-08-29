# Iteration 354: modulation chunk 2048

## Single-variable contract

C4 keeps the C3 GAN-on world16 FSDP2 workload, pinned local FA2 backend,
optimizer-dispatch warmup, exact fused kernels, trace, RNG, losses, updates,
residency, and timer unchanged.  It changes only the token grouping used by the
existing modulation, residual, and cross-normalization helpers from 512 to
2048, with setup-only warmup at the selected shape.

## Gates

- Real A100 B1/B3 CUDA canary: PASS.  Output, hidden gradient, and residual
  gradient are bitwise; timestep-gradient drift passes the frozen multi-metric
  P0 gate.  Receipt SHA-256:
  `aa2a661cc7543fd74f6181f0ae189f607437d17cf5e57d16bd5c1a37aad61726`.
- Focused tests: `78 passed`.
- Validate-only: `iter354_gan_cuda_chunk2048_validate_r1_20260829`, 16/16 PASS,
  summary SHA-256
  `51394b3a01ccc4953624bfc7398c18c6991a6a28acdf45721f280a13f3067dd9`.
- Correctness: `iter354_gan_cuda_chunk2048_correctness_r1_20260829`, 16/16
  PASS, summary SHA-256
  `8ba3331d6f716c956e705e38a963a65884cd4cbf361527c0e7df60fae347458c`.
- Updates/final versions: G1/C5/EMA1 and G201/C1005/EMA801.
- Generator Score inventory: four logical requests and four physical forwards;
  Score packing remains off.
- All residual/modulated-norm fallbacks and CUDA/cgroup OOM deltas are zero.

## Formal timing

- Run: `iter354_gan_cuda_chunk2048_timing_r1_20260829`.
- Summary SHA-256:
  `8733318616ca4ef9a1112fb7fab36435fbcccec9996b1764bbbc2fd596702091`.
- Full5: `143.206713110 s`.
- Rank range: `143.195667180-143.202194473 s`.

| Comparison | Saved | Reduction | Speedup |
|---|---:|---:|---:|
| C4 vs C3 (`143.379703687 s`) | `0.172990577 s` | `0.1207%` | `1.0012x` |
| C4 vs C0 (`173.014483004 s`) | `29.807769894 s` | `17.2285%` | `1.2081x` |

C4 is the current fastest correctness-qualified substrate, but its isolated
gain is tiny and based on one formal run.  The requested 30% threshold remains
`22.096575007 s` away, requiring another `15.4298%` reduction from C4.
