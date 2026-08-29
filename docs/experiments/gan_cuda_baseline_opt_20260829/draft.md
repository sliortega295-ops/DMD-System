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

## C2 measured result

- Exact source HEAD: `9bb21c0a833c7bc53d675df6be1792200ac5777f`.
- The first bounded implementation failed closed in
  `iter352_gan_cuda_optwarm_correctness_r2_20260829`: Critic trainables used two
  different DeviceMesh wrapper objects that described the same logical
  world16 `(Shard(),)` mesh.  The rejected run is retained; no timing was
  reported from it.
- The repair compares the logical mesh signature and still rejects different
  device types, rank sets, dimensions, names, or placements.  It does not
  weaken the frozen topology contract.
- Correctness run `iter352_gan_cuda_optwarm_correctness_r3_20260829`, summary
  SHA-256
  `c84b839371aec84eedce4593a9ac21fa9f5d07b3d5abc4382303b66e66b8849b`:
  16/16 PASS, G1/C5/EMA1, final G201/C1005/EMA801, exact Score inventory, no
  CUDA/cgroup OOM, and exact restoration of local parameters, Adam state,
  optimizer step, and gradients after warmup.
- Formal unprofiled timing run
  `iter352_gan_cuda_optwarm_timing_r1_20260829`, summary SHA-256
  `000742835774fd96bfe1996dff6a46b58ae94da2ec6f801cd7b36793d129fd4d`:
  `144.621745256 s`.
- Improvement from C1.1: `7.230203783 s` (`4.7614%`).
- Improvement from C0: `28.392737748 s` (`16.4106%`, `1.1963x`).
- Remaining gap to the requested 30% threshold: `23.511607153 s`; another
  `16.2573%` reduction from C2 is still required.

The next preregistered arm is C3, which changes only the attention backend from
the historical PyTorch SDPA fallback to the previously validated pinned local
FA2 build.  Selective checkpoint-output retention, HSDP, or a new kernel remain
separate later arms.

## C3 measured result

- Exact source HEAD: `86d819813a800181ba5294b2dfa72a09a2f41442`.
- Focused tests: `75 passed`.
- Validate-only run `iter353_gan_cuda_localfa2_validate_r1_20260829`: 16/16
  PASS; every rank loaded the pinned local FA2 extension with SHA-256
  `c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7`.
- Correctness run `iter353_gan_cuda_localfa2_correctness_r1_20260829`, summary
  SHA-256
  `d239b73493f16df5ab5e703963ec0f440a6ee819421b21094a934bfd08557bbc`:
  16/16 PASS, G1/C5/EMA1, final G201/C1005/EMA801, four logical and four
  physical Generator Score calls, exact fused-kernel receipts, and no OOM.
- Formal unprofiled run `iter353_gan_cuda_localfa2_timing_r1_20260829`, summary
  SHA-256
  `4a2dc68d633df66140149a3d5fa63bd209ca7d0c0affd291421ce90842815805`:
  `143.379703687 s`.
- C3 saves `1.242041569 s` (`0.8588%`) versus C2 and `29.634779317 s`
  (`17.1285%`) versus C0.
- The requested threshold remains unmet by `22.269565584 s`; C3 needs another
  `15.5319%` reduction.

The result matches the historical No-GAN finding that changing only the
attention backend is useful but small.  The next high-value work must target
the repeated block compute/launch path or a separately gated logical-equivalent
packing mechanism; exposed NCCL alone cannot close the remaining gap.

## C4 selected hypothesis

The pinned Helios transformer already exposes
`_distillschedule_modulation_chunk_tokens`, with source default 512.  C3's
correctness receipts report 63,280 modulated-norm forward chunks and tens of
thousands of residual forward/backward chunks per rank.  C4 changes only that
physical grouping to 2048 while retaining C3's exact fused kernels and all
workload semantics.  It includes setup-only warmup at the selected chunk size
so first-use compilation cannot contaminate the formal timer.  Promotion still
requires a real B1/B3 CUDA parity canary, focused tests, validate-only, 16-rank
correctness, and a faster matched unprofiled Full5.

## C4 measured result

- Exact source HEAD: `9736b433b2f8653c12ed1fd495efd2004306d8fe`.
- Real A100 B1/B3 CUDA canary: PASS.  Output, hidden gradient, and residual
  gradient are bitwise identical between chunk 512 and 2048; timestep-gradient
  drift remains inside the existing P0 multi-metric gate (worst normalized L2
  `0.004255535`, worst cosine `0.999990880`).
- Focused tests: `78 passed`.
- Validate-only run `iter354_gan_cuda_chunk2048_validate_r1_20260829`: 16/16
  PASS, summary SHA-256
  `51394b3a01ccc4953624bfc7398c18c6991a6a28acdf45721f280a13f3067dd9`.
- Correctness run `iter354_gan_cuda_chunk2048_correctness_r1_20260829`: 16/16
  PASS, G1/C5/EMA1, final G201/C1005/EMA801, four logical/four physical
  Generator Score calls, and zero kernel fallbacks or OOM deltas.  Summary
  SHA-256
  `8ba3331d6f716c956e705e38a963a65884cd4cbf361527c0e7df60fae347458c`.
- Per-rank chunk counts fell from 63,280 to 18,640 for modulated norm; the
  residual/grad forward/backward receipts are 7,600/9,440/4,720 chunks.
- Formal run `iter354_gan_cuda_chunk2048_timing_r1_20260829`: unprofiled Full5
  `143.206713110 s`, summary SHA-256
  `8733318616ca4ef9a1112fb7fab36435fbcccec9996b1764bbbc2fd596702091`.
- C4 saves only `0.172990577 s` (`0.1207%`) versus C3.  This is a tiny,
  one-run gain and should not be generalized as the Wan1.3B-sized launch
  improvement.  It is retained as the fastest correctness-qualified substrate,
  while the requested threshold remains unmet by `22.096575007 s`.

The next candidate must therefore change a larger repeated execution unit, but
the broader failure-index search closes both obvious packing routes.  Real-Score
packing previously failed strict parity.  More importantly, the later
schedule-compatible B2 implementation already ignored non-runtime metadata,
batched only identical physical schedules, and checked every row against its B1
reference.  Its world16 correctness run still failed before update on 15
preserved ranks with normalized-L2 about `0.058--0.071` and cosine about
`0.9977--0.9984`, far outside the frozen gate.  Repeating B2 or B5 on C4 is
therefore **NO-GO**, not a new experiment.

C5 instead ports the previously correctness-qualified selective checkpoint
policy that saves only the exact pinned FA2 result for every other Critic block
(`stride=2`, blocks `0,2,...,38`) and reuses it during checkpoint replay.  It
does not bypass a block, alter a Score call, change rollout grouping, or change
the optimizer.  The material new question is whether the same exact mechanism
fits and helps the official GAN-D physical-B3 Critic graph.  Capacity and exact
original-save/replay receipt counts are mandatory before timing.
