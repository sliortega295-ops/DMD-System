# Helios GAN-on CUDA-resident optimization draft

## Task contract

- Objective: reduce the matched unprofiled GAN-on Full5 time by at least 30%.
- Formal denominator: `iter348_gan_cuda_timing_r1_20260829`,
  `173.014483004 s`.
- Promotion threshold: `<= 121.110138103 s` with a fresh 16-rank correctness
  receipt.
- Frozen workload: world16 FSDP2 FULL_SHARD, 2x8 A100-40GB, 384x640 BF16,
  logical steps 1000-1004, B1/global batch16, Score packing off, CUDA
  Generator/Critic residuals, CUDA EMA, no parameter offload/streaming/model
  swap, G1/C5/EMA1 and final G201/C1005/EMA801.
- The candidate may change only exact implementation mechanics. It may not
  change trace/RNG, model calls, losses, precision, update order, physical
  Score/DiT inventory, topology, or collective semantics.

## Baseline evidence

- Correctness run `iter348_gan_cuda_correctness_r2_20260829`: 16/16 PASS,
  final G201/C1005/EMA801, no CUDA/cgroup OOM.
- Formal unprofiled run `iter348_gan_cuda_timing_r1_20260829`:
  `173.014483004 s`, rank spread `0.029706817 s`.
- Diagnostic run `iter348_gan_cuda_profile_p0_20260829` used pinned Nsight
  Systems 2024.6.2 and is not a performance denominator.
- Representative device overlap from `nsys-ai` is stable across both nodes:
  compute/NCCL overlap about 86.8-87.5%, exposed NCCL about 8.82-9.28 s, and
  the compute-or-NCCL complement about 15.09-15.53 s over the diagnostic
  174.4-174.7 s device span.
- Host/device traffic is not the main issue: node0 contains about 1.81 GiB H2D
  and only 5,190 bytes D2H in the whole eight-GPU report. The dominant copy
  class is device-to-device FSDP materialization, not CPU offload.
- The top-kernel inventory is compute and tiny-kernel heavy. Across eight GPUs,
  the profile contains millions of FP32/BF16 add, mul, direct-copy, cast, norm,
  split, and materialization kernels. This is the same mechanism targeted by
  the previously validated Helios exact residual/modulation fusion ladder.

## C1.1 measured result

- Exact source HEAD: `42c0cc6aae6c7039ebcf9944eb1d821c38371671`.
- Correctness run: `iter350_gan_cuda_fusions_b3_correctness_r1_20260829`,
  summary SHA-256
  `90ba7b7c377d1b9a85ac0ab2c04af15198dddb4e94cd9611a355cacb5a6046c7`.
- Correctness: 16/16 PASS, G1/C5/EMA1, final G201/C1005/EMA801, exact
  packed-B3 physical inventory, zero residual or modulated-norm fallback, and
  no CUDA/cgroup OOM increment.
- Formal unprofiled timing:
  `iter350_gan_cuda_fusions_b3_timing_r1_20260829`, `151.851949039 s`.
- Improvement from C0: `21.162533965 s` (`12.23%`, `1.1394x`).
- Remaining gap to the requested 30% threshold: `30.741810936 s`; C1.1 is an
  intermediate promoted substrate, not completion of the task.

## Risks and unknowns

1. Prior fusion evidence is No-GAN/node-HSDP, while this denominator is
   GAN-on/world16 FSDP2. The mechanisms match, but the gain and correctness
   must be re-established on the frozen GAN workload.
2. The baseline is near the 40-GB capacity boundary; a candidate that retains
   new tensors is rejected before timing.
3. NCCL union is large but mostly overlaps compute. Blind all-gather changes
   cannot explain the requested 51.9 s reduction.
4. The current attention backend is the historical PyTorch SDPA fallback.
   Switching to the separately pinned local FA2 path is not part of the first
   candidate and must be declared as a separate arm if later tested.

## Ranked candidates

1. **C1 exact compute-fusion substrate**: port the already validated exact
   LoRA epilogue, Generator no-grad residual, timestep-gradient staging,
   modulated-norm, and modulation-materialization fusions through commit
   `6005c03`, while preserving the current world16 launcher and GAN contract.
   This directly attacks the measured pointwise/copy/norm kernel families.
2. **C2 attention backend arm**: only after C1, evaluate the pinned local FA2
   backend as an explicitly separate, correctness-gated implementation arm.
3. **C3 GAN-specific kernel follow-up**: use a bounded NCU capture on the
   remaining C1 hotspot before implementing any new kernel. Do not infer MFU
   from Nsight Systems kernel coverage.

## Validation and evaluation

- Focused tests: the static P0, LoRA epilogue, residual fusion, modulated norm,
  launcher, summarizer, and Nsight compatibility tests.
- Correctness: frozen world16 GAN-on correctness controller; require 16/16,
  G201/C1005/EMA801, G1/C5/EMA1, complete trace/inventory receipts and no OOM
  counter change.
- Evaluation: one matched unprofiled Full5 timing from the candidate's fresh
  correctness summary. Profiled wall time is diagnostic only.
- Promote a candidate only when correctness passes and formal time improves.
  The goal is complete only at `<=121.110138103 s`.

## C2 selected hypothesis

Iteration 351 closes the C1.1 profile loop over all 16 ranks.  The formal
result is still `151.851949039 s`; its diagnostic Full5 range has a 16-rank
median of `151.753837642 s`.  The largest clean setup-shaped bubble is not
parameter communication: G optimizer has `3.810982 s` and Critic update 0 has
`4.172721 s` of no-traced GPU activity, while later Critic updates fall to
`0.291-0.416 s`.  Full5 exposed NCCL is only `6.811882 s`.

C2 therefore keeps world16 FSDP and all training semantics frozen and extends
the previously exact-restoring optimizer dispatch warmup to the world16
`(Shard(),)` DTensor layout.  It is rejected if local parameter or Adam-state
bytes do not restore exactly, gradients remain attached, any rank/fingerprint
gate changes, or formal timing fails to improve.
