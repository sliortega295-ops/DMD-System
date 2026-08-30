# Iteration 367: C14 fused BF16 LayerNorm and modulation

## Frozen comparison

C14 starts from C12, with the rejected C13 static-context cache disabled.  It
changes only the registered `_TembAwareChunkedNorm` forward tile: the BF16
input read, affine-free width-5120 FP32 LayerNorm, source table/component FP32
adds, modulation formula, and BF16 store execute in one Triton kernel.  The C11
explicit backward, world16 FSDP2 topology, official GAN steps 1000--1004,
B1/global-B16 workload, trace/RNG, physical model-call inventory, loss/update
order, local FA2, selective Flash-save stride 2, CUDA residual/EMA residency,
and formal timer are unchanged.

## Evidence

| Gate | Run | Result |
|---|---|---:|
| Locked-A100 B1/B3 canary | `iter367_gan_cuda_c14_fusedln_canary_r2_20260830` | PASS, `5.535x` / `6.186x` warm forward |
| Validate-only | `iter367_gan_cuda_c14_fusedln_validate_r1_20260830` | 16/16 PASS |
| Correctness/capacity | `iter367_gan_cuda_c14_fusedln_correctness_r1_20260830` | 16/16 PASS |
| Formal unprofiled Full5 | `iter367_gan_cuda_c14_fusedln_timing_r1_20260830` | **136.084681777 s** |

The correctness summary SHA-256 is
`a95a0a6bf6ae193dab67143eca050803ac1e9c5006d1f13568e96bdfa4cc5c86`.
Every rank records G1/C5/EMA1, final G201/C1005/EMA801, 6,880 fused
LayerNorm calls, 26,240 fused chunks, 1,120 explicit-backward calls, zero
reference fallback, and a setup-only B1/B3 warmup.  Maximum final-run allocated
and reserved HBM are `24.963570118 GiB` and `38.294921875 GiB` per rank; no
CUDA/cgroup OOM is admitted by the 16-rank summary.

## Decision

C14 improves C12 by `2.091400767 s` (`1.513577%`) and improves the fresh
`173.014483004 s` GAN-on baseline by `36.929801227 s` (`21.344919%`).  It is a
real promoted improvement, but remains `14.974543674 s` above the requested
30%-faster threshold of `121.110138103 s`.

The result also bounds the standalone canary: the fused tile is much faster in
isolation, but it owns only part of Full5 and overlaps other GPU work.  Do not
extrapolate the `5.5--6.2x` microkernel speedup to the full workload.
