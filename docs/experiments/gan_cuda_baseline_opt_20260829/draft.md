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

## C5 measured result

C5 passes all frozen gates at
`e75c80e95198690ee070a8d7f77c6db9bfb85ce9`.  Each rank reports 220 selected
checkpoint contexts, 440 exact FA2 saves, and 440 replay reuses; G1/C5/EMA1
and G201/C1005/EMA801 remain unchanged.  The matched unprofiled Full5 is
`142.295341636 s`, saving `0.911371474 s` (`0.6364%`) from C4 and
`30.719141368 s` (`17.7552%`) from C0.

The candidate is memory-tight: peak reserved is `38.533203 GiB/rank`, with a
minimum end-receipt driver-free value of only `0.012085 GiB`.  Do not infer
that retaining all 40 blocks is safe.  C5 is promoted as the fastest substrate,
but remains `21.185203533 s` above the requested threshold.  A fresh C5 profile
must now select the next mechanism; neither known physical batching failures
nor exposed NCCL alone can supply the missing time.

## C8 measured result

C8 preserves the C5 workload and reuses only exact cross-attention encoder K/V
inside one Generator no-grad rollout.  At
`9cee963a55175aad02d94da105f32ad0690451d4`, validate-only and correctness both
pass on 16 ranks.  Every rank records 200 cache fills and 880 reuses; the cache
is cleared at every rollout boundary and correctness compares against an
uncached B1 reference.

The matched unprofiled run
`iter359_gan_cuda_c8_crosskv_timing_r1_20260829` completes in
`141.235767135 s`.  This saves `1.059574501 s` (`0.7446%`) from C5 and
`31.778715869 s` (`18.3677%`) from C0.  C8 is the new fastest substrate, but
the 30% target remains unmet by `20.125629032 s`.  The small result bounds this
particular context-hoisting opportunity; the next candidate must address a
larger exact-compute or launch-gap mechanism rather than widening the cache
without evidence.

## C9 selected hypothesis

The pinned Helios self-attention processor contains a separate, exact cache for
the invariant history prefix.  On the first denoising call it stores the
history K/V and history-only attention output; later calls still compute all
current-token Q/K/V and attention but avoid repeating history projection,
history rotary work, restricted-history LoRA, and history-only attention.  This
is distinct from C8's text cross-attention K/V cache.

C9 ports the already source-pinned wrapper from `5dc7e5c` onto C8 and removes
only its historical No-GAN admission restriction.  It is active solely in one
Generator no-grad B1 rollout, clears all processor state afterward, and keeps
the C8 cache and every workload/inventory contract.  Correctness executes an
uncached B1 reference for each of the five rollout rows.  Any miss, leaked
state, mutation, parity failure, or capacity regression rejects the candidate
before formal timing.

## C9 measured result

C9 is a bounded negative result, not a performance arm.  Its validate-only run
passed on 16 ranks, but the first real no-grad rollout failed closed immediately:
the frozen Helios assets omit `restrict_self_attn`, so the pinned model default
is `False` for all 40 blocks.  The upstream history-prefix cache is therefore
not semantically available in this workload.  Enabling restricted attention
would change model math, and enabling the wrapper on zero blocks would provide
zero reuse.  C9 formal timing is `NOT_RUN`; C8 remains the fastest valid arm at
`141.235767135 s`.

## C10 selected hypothesis

The valid C5 Nsight profile contains `439,436` BF16 add-kernel launches on
node0.  The C8 correctness receipts independently count `26,870` enabled
scale-1 LoRA epilogues per rank (`21,962` Generator plus `4,908` Critic), or
`214,960` across the eight node0 ranks.  Thus about half of that launch family
has a direct source-level owner: the source-pinned PEFT path computes the LoRA-B
GEMM and then launches a separate BF16 `base + update` kernel.  This is stronger
evidence than the previously considered cross-attention residual add, whose
static call count is far smaller.

C10 tests one bounded exact-workload rewrite: for supported contiguous BF16,
scale-1 LoRA linears, use a single `addmm`-style LoRA-B GEMM with the base result
as the beta input.  LoRA-A, LoRA-B, base GEMM, adapter selection, model-call
inventory, FSDP placement, and every workload semantic remain unchanged.  The
candidate may change BF16 rounding because a GEMM epilogue can add before the
source path's intermediate BF16 rounding.  Therefore it is **not** admitted to
world16 merely from static reasoning: a real A100 B1/B3 forward plus input,
LoRA-A, LoRA-B, and base-result gradient canary must first pass the frozen P0
multi-metric thresholds and show a CUDA-event improvement.  Failure rejects
C10 without relaxing the gate.

The existing world16 explicit-next FSDP2 prefetch result is a matching bounded
negative, not a new GAN hypothesis: exact No-GAN control was `118.267264657 s`
and explicit prefetch was `118.737632053 s` (`+0.398%`).  C10 therefore does not
combine or retry FSDP prefetch.

## C10 measured result

C10 is rejected by its preregistered one-GPU gate; no world16 run or formal
timing was launched.  Run
`iter361_gan_cuda_c10_lora_beta_canary_r1_20260830` passes the frozen P0 output
and all-gradient thresholds for all four registered B1/B3 attention/FFN
shapes.  The beta-add GEMM changes output rounding only (worst normalized L2
`0.001065207`, worst max-absolute `0.03125`), while input, LoRA-A, LoRA-B, and
base gradients are bitwise identical in this canary.

It does not, however, provide a consistent latency gain.  Warm no-grad forward
is `1.0171x`/`1.0081x` for B1 attention/FFN but `0.9971x`/`0.9950x` for B3;
forward+backward is `1.0375x`, `1.1227x`, `0.9862x`, and `1.1953x`.  The
dominant Generator no-grad B1 path therefore gains only about one percent in
the isolated epilogue while two registered shapes regress.  This cannot close
the `20.126 s` target gap and fails the all-shape promotion gate.  C8 remains
the active substrate.

## C11 selected hypothesis

C10 closes the simple LoRA beta-add route.  The next source-qualified launch
cluster is the exact modulated-norm backward.  The pinned implementation
rebuilds a PyTorch autograd graph and calls `torch.autograd.grad` separately for
every token chunk.  With chunk2048, the formal receipt contains about `4,720`
gradient chunks per rank.  Each chunk rematerializes two modulation components,
runs LayerNorm forward and backward, casts BF16/FP32, reduces timestep
gradients, and copies the hidden gradient.  This directly explains a repeated
subset of the profile's layernorm, layernorm-grad, FP32 add/mul, cast/copy, and
reduction kernels; unlike a broad kernel-name guess, the owner and frequency
are executable.

C11 first tests a source-equivalent explicit backward that calls the installed
native LayerNorm backward directly and computes the two timestep-component
gradients without rebuilding autograd.  It changes neither forward nor saved
tensors.  A locked A100 B1/B3 PhaseLocal canary must compare hidden and every
timestep-input gradient under the frozen P0 thresholds and show a warm
forward+backward improvement.  Only a passing canary can authorize a
default-off runtime implementation and world16 gates.

The admissible canary
`iter362_gan_cuda_c11_modnorm_backward_canary_r2_20260830` passes that gate.
Output and hidden-state gradients are bitwise equal for B1 and B3.  The worst
timestep-gradient result remains inside the unchanged P0 contract
(`normalized_l2=0.002174`, `cosine=0.9999976`, `max_abs=1.0`).  Warm
forward-plus-backward improves from `4.341077 ms` to `3.164160 ms` for B1
(`1.37195x`) and from `10.837334 ms` to `7.937024 ms` for B3 (`1.36542x`).
The earlier `r1` did not start the workload because the standalone command
omitted the pinned `LOCAL_KERNELS` mapping; it is retained as environment-only
failure evidence, not counted as a mechanism test.  Report SHA-256:
`36008126ce3567551474431d5ab3c81af679afca0c31e2c53f08fe0f5592e31c`.

The world16 implementation also passes.  Validate-only and correctness are
16/16 PASS; every rank records 1,120 explicit-backward calls, 4,720 PhaseLocal
chunks, and zero fallback, ending at G201/C1005/EMA801.  The matched unprofiled
run `iter363_gan_cuda_c11_modnormbackward_timing_r1_20260830` is
`138.462140212 s`: `2.773626923 s` (`1.963828%`) faster than C8 and
`34.552342792 s` (`19.970781%`) below C0.  C11 is promoted, but remains
`17.352002109 s` above the requested threshold.

## C11 post-fix profile and C12 selected hypothesis

The valid Iteration 364 profile confirms that the current GAN workload is
primarily repeated 14B compute rather than removable transfers.  Across the 16
rank-local exact ranges, median compute union is `126.112 s`; NCCL occupies
`61.524 s` but overlaps compute for `54.422 s`, leaving `7.121 s` exposed.
Copies expose only `0.501 s`.  The median no-traced complement is `5.903 s`.
Because Nsight did not export the GPU model, these are timeline occupancy
measurements, not SM utilization or MFU.

The profile plus source audit nevertheless identifies one clean launch-gap
candidate.  The measured P0 path executes a Generator-update
`torch.cuda.empty_cache()` and post-rollout synchronization, then executes an
additional post-rollout synchronization after each of five no-grad B1
rollouts.  None of the immediately following shape/dtype/autograd checks need
completed GPU data, the downstream PyTorch work carries its own dependencies,
and the exact timer already synchronizes at the outer boundaries.

C12 therefore adds a default-off policy that skips only those six in-window
host-blocking synchronizations and the allocator flush.  It does not remove
framework/FSDP synchronization, alter streams, prefetch parameters, or change
model math.  Correctness must prove exact policy propagation and all existing
16-rank semantic receipts before one unprofiled timing.  The experiment does
not assume the full `10.416 s` rank0 synchronization API wall is recoverable;
the question is intentionally bounded and falsifiable.

## C12 measured result

C12 passes all gates, but the formal gain is deliberately classified as tiny.
The timing is `138.176082544 s`, only `0.286057668 s` (`0.206596%`) below
C11.  All 16 ranks record zero in-window rollout synchronization and allocator
flush counts while retaining the outer Full5 boundary synchronization,
G1/C5/EMA1, final G201/C1005/EMA801, and the same HBM/capacity envelope.

This result is important negative evidence: the source-owned fences were
mostly waiting for work that the next phase required anyway, so rank0's
`10.416 s` aggregate synchronization API wall was never a valid removable-time
estimate.  Further broad synchronization deletion is closed.  The remaining
`17.065944441 s` requires a source-qualified repeated compute,
materialization, or structural execution mechanism.

## C13 exact rollout-context result

C13 proves that immutable history patch/RoPE, text projection and the three
current-token RoPE layouts can be reused exactly inside a Generator no-grad
rollout. All 16 ranks pass five bitwise uncached B1 references and retain
G1/C5/EMA1, final G201/C1005/EMA801, the physical call inventory and zero OOM.
The optimization is not useful at Full5 scale: `138.243935311 s` is
`0.067852767 s` slower than C12. The exact context preparation was therefore
not a material remaining bottleneck, and C13 is kept as negative evidence
rather than becoming the next substrate.
