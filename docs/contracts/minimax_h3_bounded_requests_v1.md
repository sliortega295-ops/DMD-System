# Bounded original H3 Grid1000 requests, v1

`distillschedule.h3_request.v1` admits exactly 768x1344 and 480x832, each with
56 or 124 frames. The default remains 768x1344x124. This is a versioned request contract, **not** a replacement or
sixth canonical best-known profile. The five canonical JSON files and their
lock hashes remain unchanged. Each noncanonical request has a distinct semantic
fingerprint and resolved profile. Its historical section contains no inherited
canonical seconds.

Pinned LightX2V `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be` H3
`packing.py::video_latent_num_frames` requires `num_frames % 17 == 5` and
returns `(num_frames - 5) // 17 * 5 + 2`. Thus 56 and 124 are valid native
packing lengths; 60 is invalid. This contract admits only the two bounded
lengths, not arbitrary native lengths or Wan's frame lattice. Audio packing
retains the native `round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND)` rule.

Everything except the explicitly selected resolution/frame count and physical cache policy
retains the original canonical H3 protocol: approximate discrete Grid1000,
world16 on 2x8 A100, B1/global16, BF16, SP1, LoRA128, four rollout evaluations,
fixed end step 3, Student1/Fake5, segment1, activation policy none, FA3 splits2,
all 50 replay blocks, seed20260817, no EMA. The native two-cycle adapter runs
one process lifetime, advances G2/F10, and does not reset models/RNG between
cycles. This does not support autoregressive H3 or continuous-exact sigma.
Eight-GPU topology requests and arbitrary shapes remain unsupported. The
CPU-staged matched control is still approximate Grid1000, not a continuous
reference baseline.

## Physical implementation identities

Legacy `joint-h3v1`/`joint-h3v2` retain CPU-staged cache plus trim, their old IDs,
and their default behavior. New `joint-h3v3` IDs include the request and either:

- `cpu_staged_trim`: pageable reservoir and bounded pinned staging, trim on;
  inflight depth 1 or 2, LoRA epilogue N64 or N128.
- `cuda_no_trim`: existing GPU-resident FA3 out/LSE replay, trim off;
  inactive D2H depth fixed at 2, LoRA epilogue N64 or N128.

Other storage/trim combinations are not admitted. The CLI inventory exposes
the native macro family (no new rollout reordering). A v3 optional `-r1..5`
retains the separately gated action family; it requires action-qualified
calibration and prior action proof. It cannot enter the macro collector.

## Invariants and evidence

The selected ID, request, resolved model config and environment must agree
before model construction. The selected frame count enters both
`contract.frames` and `training.dmd.num_frames`; a 124-frame loaded config
cannot satisfy a 56-frame candidate. After native setup the actual installed cache
registration must match; v3 per-cycle receipts include its policy and native
counters. CUDA must have zero CPU cache/staging/transfer and trim counters and
nonzero native replay forward/backward execution. The original workload-owned
per-cycle FA3/Grid1000/census/version gates remain unchanged.

Request/shape, storage/trim and kernel choices enter cost keys. H3 v3 macro
planning requires joined, hash-bound native calibration results. Missing costs
remain unavailable; neither parent canonical seconds nor short external
480x832 observations are imported as calibrated costs. Serialized lowering
rechecks original rank observations and the selected cost model. Cross-request
comparison and cross-workload dispatch fail closed.

CPU fixtures establish wiring and rejection behavior only. A new source HEAD
needs matched target-GPU correctness and calibration, then independent native
confirmation. Sampled two-cycle numerical REVIEW is not full-state or
long-horizon equivalence, and diagnostic timing is not formal speedup proof.
