# Iteration 358: C7 dominant Critic GEMM NCU diagnostic

## Claim boundary

- Formal performance remains C5's unprofiled
  `142.295341636 s`; the NCU run is diagnostic-only.
- Run ID:
  `iter358_gan_cuda_c5_ncu_critic_gemm_p1_20260829`.
- Exact source HEAD on both nodes:
  `262d34474e29b0352c1630aaba9659749b5fd3dc`.
- Frozen workload: GAN-on steps 1000--1004, world16 FSDP2, B1/global-B16,
  Score packing off, CUDA residuals/EMA, pinned local FA2, chunk 2048,
  optimizer dispatch warmup, exact residual/modulated-norm/LoRA fusions, and
  Critic selective Flash replay stride 2.
- All 16 rank receipts pass with G1/C5/EMA1 and final
  G201/C1005/EMA801.  The NCU report contains exactly one requested kernel on
  node0 GPU0 inside `FSDP_BLOCK_FORWARD/critic/block_28`.

Raw NCU evidence remains on remote shared storage.  No `.ncu-rep` or large
profile was downloaded locally.

## Kernel result

Target kernel:

`ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn`

| Metric | Value |
|---|---:|
| Duration | 2.158560 ms |
| Compute/SM throughput | 88.322976% |
| SM busy | 92.149139% |
| Tensor HMMA active, elapsed | 88.322976% |
| Tensor HMMA active, active cycles | 92.149139% |
| DRAM throughput | 25.233768% |
| Memory throughput | 392.287006 GB/s |
| L2 hit rate | 83.396928% |
| Achieved occupancy | 12.496465% |
| Active warps per SM | 7.997738 |
| Issue active, elapsed | 20.130654% |
| Registers per thread | 238 |

NCU classifies the kernel as a compute-pipeline bottleneck.  Its low occupancy
is real and is limited by both registers and shared memory, but the small warp
set already keeps the Tensor pipeline active for about 88.3% of elapsed time.
DRAM reaches only 25.2%, so this is not a bandwidth-bound GEMM.  The main stall
classes are fixed-latency execution dependency and math-pipe throttle; both are
consistent with an already highly tuned Tensor-Core kernel.

Iteration 356 counted `226.94062 GPU-s` for this family across node0's eight
GPUs, or about `28.3676 GPU-s/GPU`.  Even the impossible bound of raising every
occurrence from 88.323% to 100% would recover only about `3.31 GPU-s/GPU`, and
not all of that is guaranteed to be critical-path wall time.  Hand-tuning this
single cuBLAS kernel cannot close the remaining `21.185203533 s` target gap.

## Decision

The NCU evidence rejects direct GEMM replacement as the next primary arm.
The next candidate must remove repeated exact compute rather than make this
already Tensor-bound GEMM marginally faster.  The highest-value bounded path is
per-rollout Generator no-grad cross-attention encoder K/V reuse: conditioning
and Generator version are invariant across the six physical DiT calls of one
rollout, while five rollouts still remain distinct.  A source-pinned prototype
already exists at commits `89b7efa` and `945a9f3`, but it has not been measured
on GAN-on C5.  Port only this mechanism, strengthen the input-identity gate,
and require B1 reference parity plus exact cache lifecycle receipts before
timing.

## Remote evidence

`/share/project/shuchenweng/yongyan_liu/DMD-System_gan_cuda_baseline_20260829/artifacts/runs/iter358_gan_cuda_c5_ncu_critic_gemm_p1_20260829/`

- `ncu/node_0/critic_block28_bf16_relu_gemm.ncu-rep`
  - SHA-256:
    `74c558956ced77f498c0557422a9076fa2b442c2b1ce6d8abf4c66e6dc98bbb6`
- `ncu/node_0/ncu_details.csv`
  - SHA-256:
    `f1495567ba0d826d88525273d9392bea9b4088058e15d45a5324e480a6163e8a`
- `ncu/node_0/ncu_raw.csv`
  - SHA-256:
    `ab521eaecbbf56a43aaa4df410331bc03d9dd859143f55dfb350d1977e45a0fc`
- `ncu/range_receipts/rank_000.json` through `rank_015.json`
