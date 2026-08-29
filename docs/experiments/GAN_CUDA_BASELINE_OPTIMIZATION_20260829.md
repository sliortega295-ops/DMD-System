# GAN CUDA-residual baseline optimization task contract

## Objective and frozen denominator

- Objective: accelerate the exact Helios GAN-on static world16 FSDP2 workload by
  at least 30% without changing its mathematical or random-input contract.
- Formal baseline: `iter348_gan_cuda_timing_r1_20260829`, 173.014483004 s.
- Target: at most 121.110138103 s in one matched, unprofiled Full5 run. A final
  claim also requires a fresh correctness receipt for the promoted source.
- Source substrate at baseline: DMD-System
  `55d8884cf394ecb2ddd9b762844466a82055b983` plus the recorded offline-kernel
  compatibility-only Helios runtime patch; the effective attention backend is
  the historical SDPA fallback.

## Correctness requirements

- Two nodes x eight A100-40GB, world16 FSDP2 FULL_SHARD.
- GAN-on logical steps 1000-1004, B1/global batch16, Score packing off.
- Generator and Critic residual storage are both `cuda_out_of_place`; EMA is CUDA.
- No parameter offload, parameter streaming, model swap, topology change, or
  workload-count reduction.
- Preserve the pinned trace and logical-package hashes, RNG, precision, losses,
  optimizer/update order, Score/DiT physical inventory, and rank identities.
- Exactly one Generator, five Critic, and one EMA update; final versions are
  G201/C1005/EMA801 on all 16 ranks.
- CUDA OOM, cgroup OOM, missing ranks, or an incomplete receipt rejects a candidate.

## Allowed implementation approaches

- Remove or hoist demonstrably redundant host/runtime work.
- Fuse equivalent pointwise, residual, norm, cast, or optimizer operations.
- Improve legal compute/communication overlap or tensor residency without
  changing values, calls, topology, or collective semantics.
- Add instrumentation-only NVTX/Nsight compatibility patches that do not add
  synchronization or enter formal timing claims.
- Kernel work must retain BF16/FP32 behavior and pass the existing multi-metric
  correctness gate before promotion.

## Evidence workflow

1. Capture one diagnostic-only exact-range Nsight Systems profile with the same
   policy and a pinned Nsight 2024.6.2 binary.
2. Run `nsys-ai doctor`, query actual NVTX/kernel names, build JSON evidence,
   and generate the timeline view. Treat overlapping unions as non-additive and
   never use profiled wall time as the speed denominator.
3. Rank candidate mechanisms by exposed critical-path time and source ownership.
4. For a kernel candidate, use representative NCU evidence and the
   kernel-design-agents loop: implement one candidate, validate correctness,
   measure one matched unprofiled run, then promote or reject it explicitly.

## Commands and promotion criteria

- Correctness: `run_static_fsdp2_p0_world16.py --variant gan --mode correctness`
  with the frozen policy above.
- Evaluation: the same controller in `--mode timing`, pinned to the candidate's
  correctness summary; Nsight/NCU runs are diagnostic only.
- Promote only if all semantic gates pass, no OOM counter changes, and the
  unprofiled wall improves. The requested goal is achieved only at <=121.110138103 s.

## Initial risks and candidate ordering

- The baseline already reaches about 37.7 GiB per rank, so extra residency must
  be capacity-gated.
- First obtain overlap-aware evidence for compute, NCCL, H2D/D2H, optimizer/EMA,
  host gaps, and GAN-specific Score/Critic phases.
- Do not assume prior No-GAN bottlenecks transfer to GAN-on; reuse an optimization
  only after the fresh profile shows the same mechanism.
- Candidate records and rejected paths will be appended after profiling rather
  than guessed in advance.
