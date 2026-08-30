# Iteration 363: exact modulated-norm backward

## Contract and implementation

- Parent: C8 at `141.235767135 s`.
- Source: `f3c5ed480001fb0007914de25264edd8439e9f50`.
- Frozen workload: GAN-on steps 1000--1004, world16 FSDP2,
  B1/global-B16, Real-Score packing off, local FA2, chunk 2048, Critic
  selective-Flash checkpoint stride 2, exact Generator no-grad cross-attention
  K/V reuse, CUDA residuals and EMA, G1/C5/EMA1, final G201/C1005/EMA801.
- Sole change: the already-installed exact modulated-norm custom Function keeps
  the same forward and saved tensors, but its backward calls native FP32
  LayerNorm backward and computes scale/shift timestep gradients explicitly.
  It no longer constructs a fresh autograd graph for every token chunk.

## Low-cost gate

- A100 canary:
  `iter362_gan_cuda_c11_modnorm_backward_canary_r2_20260830`.
- B1/B3 output and hidden gradients: bitwise equal.
- Timestep gradients: unchanged P0 parity PASS; worst normalized L2
  `0.002174`, cosine `0.9999976`, max-absolute `1.0`.
- Warm forward+backward: B1 `4.341077 -> 3.164160 ms` (`1.37195x`), B3
  `10.837334 -> 7.937024 ms` (`1.36542x`).
- Canary report SHA-256:
  `36008126ce3567551474431d5ab3c81af679afca0c31e2c53f08fe0f5592e31c`.
- Focused source tests: 22 PASS with 36 pre-existing CUDA Triton cases skipped
  under the lightweight system test interpreter.  The real pinned Torch 2.10
  A100 path is covered by the canary and the world16 gates below.

## World16 gates

- Validate-only:
  `iter363_gan_cuda_c11_modnormbackward_validate_r1_20260830`, 16/16 PASS;
  summary SHA-256
  `22a4329431d0211c0b816a0f578fe58521172b0a7957e5e5f5dad853c1d3e8ba`.
- Correctness:
  `iter363_gan_cuda_c11_modnormbackward_correctness_r1_20260830`, 16/16 PASS,
  `claimable_correctness=true`; summary SHA-256
  `f101d5fc8f6f5c6d6b416f14236d487cf7c5d61bc1925270e9dc0201efdc4623`.
- Each rank records exactly 1,120 explicit-backward calls and 4,720 chunks,
  all `phase_local`, with zero source fallback.
- Final state on every rank: G201/C1005/EMA801; physical G3/C15; updates
  G1/C5/EMA1.
- Maximum allocated/reserved per rank: `24.9651/38.4023 GiB`.  Both node
  cgroups remain at `oom=0`, `oom_kill=0`; teardown leaves no GPU process.

## Formal timing

- Run: `iter363_gan_cuda_c11_modnormbackward_timing_r1_20260830`.
- Status: PASS, 16 ranks, `claimable_timing=true`.
- Matched unprofiled Full5: **`138.462140212 s`**.
- Summary SHA-256:
  `aadb6d482bc4cf166816afe8a3e14f5efa2cd3d44e0b1d43fe8daa45ceeb0f10`.
- Versus C8: `-2.773626923 s` (`-1.963828%`).
- Versus fresh C0 (`173.014483004 s`): `-34.552342792 s`
  (`-19.970781%`, `1.24955x`).
- Requested 30%-reduction threshold: `121.110138103 s`; remaining gap:
  `17.352002109 s`.

## Decision

**PROMOTED_INTERMEDIATE_PASS_TARGET_NOT_MET.**  C11 is correctness-qualified
and faster, so it becomes the new substrate.  Its isolated improvement is
about 2%; the requested 30% baseline reduction still requires a new bounded
mechanism backed by source-frequency and timeline evidence.
