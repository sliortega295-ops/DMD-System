# Capability and verification matrix

The public lifecycle and original-native integrations are different entry
adapters to the existing runtime. This table describes implementation support,
not throughput or pretrained-model quality.

For packaging/migration and the exact source/CPU versus actual-GPU boundary,
see the [bounded release notes](release_notes.md). The whole maturation plan
and its pretrained/GPU milestones are not complete.

**W1 native update:** the fixed-trace `wan_ar.incumbent21a.v1` preset has now
completed actual public-entry execution on 16 A100s (32 rank-cycles), with
real existing optimization counters and version/call/finite checks. This is
pretrained-backbone plus synthetic feature replay, **not** numerical parity
or quality. A subsequent explicit Resident stream source also completed two
native RNG cohorts and fresh-process checkpoint continuation on all 16 ranks,
including rank-local amplified zero-history. Its strict tensor comparison
remains **REVIEW**; restart distances are no larger than the measured
same-start repeat noise for the replicated role/optimizer/EMA state. These
cohorts reuse synthetic features: this is not real-data or long-run quality
acceptance. See [GPU acceptance](../status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).

| Entry | Data | Execution | Continuous progress | Save/resume | Generate/export | Held-out validation | This integration's verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Public Program factories | Generic raw media + explicit encoder, or prepared native feature tree | Local and distributed reference/manual; existing measured auto/saved-plan path with admitted DDP/FSDP2/HSDP | Persistent models/wrappers/optimizer, versioned one-cycle Program and explicit rank-strided cursor | Complete cycle, same topology; all-rank atomic local-shard checkpoint | Same-topology distributed checkpoint `evaluate` and full-weight `export`; local-only `generate --weights` on new prepared inputs with source/layout-bound loading; PNGs require an explicit model decoder | Optional periodic local-only `validate_batch` diagnostics on explicit held-out records; no inferred quality gate | Real local and multi-process CPU autograd, fresh-process continuation, local-shard restart, distributed outputs and installed-wheel new-input inference |
| W1 Wan | Original rank-qualified trace/cohort stream; separate complete-source-trace CPU reference | Native FSDP/Resident/saved plan; rank-local amplified history is Resident-stream only | Native stream: 2..64 explicit cohorts; CPU reference: finite original cohorts within fixed scheduler horizon | Native stream: all-rank atomic shards and actual world16 new-process restoration. Strict tensor comparison REVIEW; exact RNG/cursor/version/census. CPU reference: actual full mixed model and named state | CPU reference: current full unmerged G and new-prompt original native latent diagnostics; no native-stream decoder/export | Not exposed by either W1 binding | Fixed-trace and two-cohort full preset actually executed on world16. All-rank checkpoint comparison plus same-start repeat noise diagnostic; synthetic source features, no real-data/quality/long-run acceptance. Separate original tiny CPU update/restart/export remains qualified |
| W2 Helios No-GAN | Original encoded assets/request or fresh cohort stream; separate original full-geometry CPU input owner | Selected plan; direct FSDP2/SDPA reference; new explicit SafeBundle fresh-stream preset; separate CPU1 rank0 per-sample original warm No-GAN Program | Original selected preset: two cycles. References/new SafeBundle stream: 2..64 fresh cohorts from explicit real G2/C10/EMA10 warm state | Complete cycle/same topology; original named G/C Adam, LR, SparseEMA, clocks, source cursor and inner/outer RNG. CPU Program stages all owners before restore | CPU reference: current full unmerged G export and new-prompt original native latent diagnostics; no decoder | Not exposed by native binding | Actual original full tiny mixed-BF16/FP32 model, LoRA256, original warm updates, fresh-process exact restart and independent current-G output; SafeBundle original-helper CPU/Gloo source/state tests with simulated CUDA telemetry/counters. Pretrained/world16 numerical/CUDA/quality NOT_RUN |
| W3 Helios GAN | Native GAN/ODE/text request; separately, actual original history19 primary/secondary feature producer | Selected native plan; separate CPU1 FP32/full-original dense-SDPA Program reference | Selected: two cycles. CPU reference: finite five-row cycles within the explicit original scheduler horizon | CPU reference: safe common checkpoint with original G/C Adam, schedulers, SparseEMA, native/outer RNG and consumed-source chain; exact fresh-process continuation | CPU reference: full current unmerged G export and explicit original native latent diagnostic on new prompt tensors; no VAE/decoded video | Not exposed by native binding | Actual original full tiny models/PEFT/GAN/EMA, both backward policies, source-only imports and fresh-process restart/export. Pretrained/GPU/quality NOT_RUN |
| W4 Self-Forcing / LiveWan | Original C51 continuous-sigma inputs; finite native Teacher cohorts; separately versioned exact FP32 public cohorts | C51 role pools; original world16 reference; separate CPU1 actual-original Wan2.1/LiveWan Program | Selected: two cycles. Native reference: 2..64 fresh G1/C5/EMA0. CPU Program: finite fresh five-record cycles | Native reference: all-rank complete-cycle. CPU Program: actual original G/C/Teacher, named Adam, inner/outer RNG, source cursor and versions, exact new-process continuation | CPU Program: current full unmerged G export; native four-step new-condition rollout; exact original Wan2.1 VAE optional decoder gives actual 81-frame video | No automatic quality gate | Actual original CPU FP32 classes/update/inputs/restart/current-G and optional codec tests; not world16/BF16/C51 pipeline recovery or pretrained quality |
| W5 full-sequence H3 | LightX2V native AV assets; Grid1000 explicitly approximate. Separate fresh cached-condition inventory | Existing selected FSDP2/cache policy; separate explicit original reference config, with optional optimized paths OFF | Selected plan: two cycles. Original reference: 2..64 finite fresh-data G1/F5 cycles | Original reference only: complete cycle/same topology, real two-LoRA Adam/schedulers, versions, deterministic sampler cursor and RNG; EMA0 | Native upstream tools, not unified callbacks | Not exposed by native binding | Actual original Grid loop/data/Adam on tiny CPU construction fixtures, real Gloo failure and restart tests. Pretrained runtime/capacity/numerics NOT_RUN; optimized backward FAIL unchanged |
| W6 original H3 AR | Native rank-qualified previous/current AV; separate original flat/split-packed tensor caches | Native ZeRO-3 reference/manual/plan; separate CPU1 FP32 actual-original cached-AV Program reference | Native: two updates. CPU reference: finite inventory within explicit <=10,000-cycle capacity | CPU reference: both original LoRAs, GAN heads, named Adam/options/clocks, source/input chain, inner/outer RNG and progress; staged safe restore | CPU reference: current full-base + unmerged Student export and original eight-step AV latent diagnostic on new cached conditions; no encoder/decoder | Not exposed by native binding | Actual original tiny H3/8-step/G1F1/LoRA64/continuous sigma; flat+packed fresh-process continuation, current-Student export and exact native rollout across a labeled real GAN transition. Pretrained/GPU/quality NOT_RUN |

Portable weights are not resumable checkpoints or a bundle of model Python
source. New-input generation admits matching factory source, effective
arguments, metadata, module layout and artifact hashes before loading weights;
it needs no original training data or run directory. Distributed checkpoint
outputs still require the original training dataset, request and topology.
Periodic held-out validation is opt-in and single-process only; distributed or
original-native requests reject rather than silently skipping it. Metrics and
decoded frames do not change `quality_validation: NOT_RUN`. See
[training/output contracts](training.md#generate-and-export) and
[held-out validation](training.md#optional-periodic-held-out-validation).

The original trainers and their benchmark bindings are not interchangeable.
W2's [direct reference lifecycle](../../recipes/helios_nogan/REFERENCE_TRAINING.md)
and W5's [original Grid lifecycle](../../recipes/h3-grid-original-training.md)
are separately versioned opt-in routes. They do not widen the selected-plan
worker's two-cycle contract, turn arbitrary historical evidence into restart
state, or promote a failed optimized candidate. W4 additionally has a separately
versioned [original reference lifecycle](../../recipes/self_forcing/REFERENCE_TRAINING.md),
not a change to its bounded C51 worker. W3/W6 retain their bounded **native benchmark** lifecycle. Their separately
versioned CPU Program references are not new CUDA backends or changes to those
frozen selected plans; see [W3 training](../../recipes/helios_gan/REFERENCE_TRAINING_CPU.md)
and [W6 cached AV](../../recipes/h3_ar/CPU_REFERENCE.md).

The separately versioned [W1 CPU reference](../../recipes/wan_ar/CPU_REFERENCE.md)
uses the actual original model and P0 Full5 actions, not W3 with GAN disabled.
It admits the complete source trace but executes rank0 B1 on CPU; actual
world16 gradient reduction is not being qualified. Its exported current
unmerged G can generate explicit native latents from new prompt tensors
without old training payloads. The original native W1 recipe remains separate.

The separate [W1 complete-incumbent stream preset](../../recipes/wan_ar/INCUMBENT_STREAM.md)
now binds that original native session, its complete Graph/cast/fusion bundle,
fresh 2..64-cohort input owner and common all-rank checkpoint path. CPU scalar
autograd/Adam/EMA and fresh-process restart tests exercise the actual dispatcher
and controller lifecycle with explicitly fake CUDA capture drivers. The later
[actual world16 continuation](../status/FRAMEWORK_A100_ACCEPTANCE_20261024.md#resident-stream-follow-up-actual-world16-continuation)
is separate evidence, not a relabeling of those CPU tests. Captures are bounded to the first cycle
of each process; a valid stream with a later novel signature is rejected rather
than silently falling back. The original fixed-trace v1 preset stays unchanged.

The separately versioned [W2 CPU reference](../../recipes/helios_nogan/CPU_REFERENCE.md)
preserves its distinct warm G2/C10/EMA10 initialization, LoRA256, mixed native
precision, original clipping and G1/C5/EMA5 updates. It admits all source controls
but executes only rank0 B1, not global-B16 reduction. A generated actual-original
CPU warm fixture is labeled separately from an operator-supplied historical
Step10 package; synthetic-gradient reader fixtures are rejected for training.
Current-G export and new-prompt native latents require no old training payloads.

The separate [full-original Helios model provider](native_workloads.md#explicit-full-original-helios-model-reference)
has tiny FP32 CPU full-class dense-SDPA flow/GAN/PEFT/Adam and model-serialization
tests. It reuses the unchanged original transformer in a private namespace and
rejects unsupported NaViT/varlen execution. The separate [W3 public reference](../../recipes/helios_gan/REFERENCE_TRAINING_CPU.md)
now owns actual original input/update/state through Program. Its CPU proofs do
not establish pretrained quality, CUDA/FSDP/world16 correctness or performance.
The [inference companion](../../recipes/helios_gan/PORTABLE_INFERENCE_CPU.md) exports
current full unmerged G weights without old training payloads; explicit native
latents are not decoded media or quality evidence.
The separate [cached H3 inference companion](../../recipes/h3_ar/PORTABLE_INFERENCE_CPU.md)
exports the restored current full base and unmerged Student LoRA, without fake
score, discriminator or optimizer state. Its new-input generation reuses the
original parser and eight-step rollout; named video BCTHW and stereo audio SCT
latents are explicit diagnostics, not decoded media.

W4's [finite original input owner](livewan_inputs.md) is independently usable
through `distillctl data prepare-livewan-inputs`. Its source-qualified original
dataset, independent record/prompt identity and logical cursor are CPU-verified;
the [receipt](../../artifacts/framework/20261021/original_livewan_inputs_cpu.json)
does not qualify C51 physical two-wave consumption, pipeline drain or model
checkpoint/resume. The reference route's
[separate CPU receipt](../../artifacts/framework/20261021/original_livewan_reference_cpu.json)
qualifies actual state-owner/input contracts, not pretrained execution or C51
transport drain, shadow reconstruction and physical two-slot recovery.

The generic media preparer supplies decode/import/provenance; each production
model still supplies the appropriate native VAE, text and audio encoder. The
packaged statistics encoder is deliberately a tiny media fixture. It is never
used in a W1-W6 production recipe.

## Hardware

Existing A100/H100 implementation and historical evidence identities remain in
`distillctl workloads describe --workload Wn --hardware a100|h100`. The bounded
A100 acceptance above validates only its named public paths, not those entire
historical hardware tables; H100 was not contacted or revalidated.
Native admission continues to require the real environment/source/assets,
supported backend and capacity policy. An A100 recipe is not silently relabeled
as H100. H100 historical derivatives remain separately identified where not
integrated into the common runtime.

## What to test next for a production deployment

Use model-native inputs and environment; run the recipe's original physical
and numerical gates. Validate restart/generation adapters when adding them.
Run distributed CUDA/NCCL correctness and convergence separately from the CPU
framework checks. Unsupported combinations raise explicit errors rather than
running a smaller workload, another backend or a hidden default winner.


## Additional model and workflow integrations

These native model/workflow adapters use the same `distillctl`, Program,
checkpoint and export interfaces without changing the immutable W1–W6
identities. Choose the [family recipe](model_families.md#choose-a-compatible-integration)
and its [admitted execution policy](training.md#explicit-or-automatic-strategy).
The rows retain actual CPU evidence; named CUDA receipts below qualify only
their exact source snapshots, models, precision and topology. Source-exposed
paths and pending pretrained/quality gates are recorded separately.

**Actual GPU update (2026-10-24 UTC):** official DMD2 SD1.5/SDXL and official
Self-Forcing additionally passed all seven random-tiny CUDA FP32 reference
tests on A100/Torch 2.10. This includes original-trainer oracle comparison,
new-process recovery and current-G generation/export under an explicit
deterministic test policy. These are single-device real-model-class tests,
not full pretrained training or qualification of the separate native
distributed owners described below. The table's CPU evidence stays separate. See
[exact scope and failures](../status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).

| Integration | Actual CPU reference | Data/output and continuation | Not established |
| --- | --- | --- | --- |
| SD1.5 official DMD2 | Original SDUniModel/SDGuidance, native G1/C10 clock; actual UNet/CLIP/VAE | Prompt-only + independent guidance/real/denoising streams as selected; actual raw VAE intake; exact complete-state new-process recovery; new-prompt current-G image output. Separate native FSDP2 reference/Resident-DDP source paths below | Public native distributed GPU training/restart/output qualification, pretrained CUDA capacity/quality, full production GAN acceptance |
| SDXL official DMD2 | Original dual CLIP/added conditions, four-step and explicitly initialized single-step399; real unmerged LoRA | Independent native stream counts; actual VAE/CLIP, safe full-state resume and current-G images; separate CPU/Gloo native role-DDP continuation | Pretrained CUDA capacity/quality, CUDA native role-DDP qualification, AMP/FSDP2; SD1.5 optimized profile does not apply |
| Official Self-Forcing | Actual CausalWan/Wan/DMD classes, native 21-latent-frame block3 random-exit/KV/EMA rules | Actual native T5 preparation or exact embeddings; independent G/C streams; exact continuation; current-G full-step native inference/VAE. Separate native FSDP2 reference/optimized source paths below | >21-latent-frame decode/reencode training, I2V, uncached flex, public native FSDP2 GPU training/restart/output qualification, pretrained CUDA capacity/quality |
| LTX-2.5 | Actual bounded native AV Transformer, Gemma4/connectors and native AV codecs; generic explicit G1/F5 flow DMD, full or separate LoRA | Actual raw media/prompts/exact native features; original-format tiny safetensors import; clock-aware exact recovery; current-G new-input AV decode; common measured auto/saved-plan CPU fixture with real immutable sharing. Actual A100 single-device FP32 reference: native oracle and full/LoRA fresh-process recovery/export, 4/4 PASS | Full monolithic 22B pipeline, native architectures outside the documented bounds, distributed ownership, CUDA trial planning, pretrained capacity/quality/performance |
| FLUX.2 klein Base 4B T2I | Actual Flux2/Qwen3/Flux2 VAE tiny classes; same-family unmerged LoRA; common flow-DMD G1/F5 and exact immutable sharing | Native raw RGB/prompts/hash-bound features, actual empty text, four-step generation, exact fresh-process restart and portable current-Student native PNG; actual A100 FP32 reference native-oracle/update/restart/export, 2/2 PASS; separate reference/manual Gloo2/NCCL2/world16 replica receipts below | CUDA/distributed planning; full pretrained CPU/GPU execution, other FLUX variants, full-parameter, CFG>1, edit/causal, AMP/FSDP/HSDP/DDP-overlap, long-run/quality/performance |
| Original Qwen-Image T2I | Actual QwenImage/Qwen2.5-VL/Qwen VAE tiny classes; native template/crop34/masks; common flow-DMD and exact immutable sharing | Fixed native masked right padding without silent truncation, raw/preencoded/prompts, exact restart and current-Student decoded new-prompt PNG; actual A100 FP32 reference complex-RoPE ownership/native-oracle/restart/export, 2/2 PASS; separate reference/manual Gloo2/NCCL2/world16 replica receipts below | CUDA/distributed planning; full pretrained CPU/GPU execution, edit/new generations, full-parameter, native norm-rescaled CFG>1, causal/extra time/3D RoPE, AMP/FSDP/HSDP/DDP-overlap, quality/performance |
| Wan2.2 TI2V-5B T2V subpath | Actual tiny Wan Transformer/UMT5/T5 tokenizer/residual patch2 VAE; common G1/F5 unmerged LoRA and immutable sharing | Actual raw RGB video/frames, native geometry/prompt/features, exact fresh-process recovery and current-Student portable RGB5; common measured auto/saved-plan CPU fixture; actual A100 single-device FP32 reference native oracle/restart/export, 3/3 PASS | CUDA/distributed planning; video replica GPU qualification; full pretrained CPU/GPU/capacity, I2V, dual A14B noise experts, CFG>1, AMP/FSDP/HSDP/DDP-overlap, production geometry, long-run/quality/performance |
| HunyuanVideo-1.5 T2V | Actual tiny Hunyuan15/Qwen2.5 text-only + T5/slow ByT5/temporal4 VAE; native crop108/1000 + quoted-glyph256 or native no-glyph sentinel; common G1/F5 unmerged LoRA and real immutable sharing | Raw RGB videos/frames, prompt geometry, bound safe features; native forward/gradient/Euler/codec parity; exact fresh-process recovery and portable current-Student decoded novel-input RGB5 after removing training assets; common measured auto/saved-plan CPU fixture; actual A100 single-device FP32 reference native oracle/restart/export, 3/3 PASS | Tencent model license is not Apache; CUDA/distributed planning; video replica GPU qualification; full pretrained CPU/GPU/capacity, I2V/meanflow/distilled/causal, CFG>1, AMP/FSDP/HSDP/DDP-overlap, production geometry, long-run/quality/performance |

The common [export-only input preparer](export_inputs.md) now supplies native
prompt/raw-media preparation for the four admitted modern-flow image/video
adapters, with public CLI decoded-output proofs after deleting all training
assets/checkpoints. Other adapters have no encoding hook and refuse. This is
actual tiny-class **SYNTHETIC_TEST_ONLY** CPU evidence. The separately
[qualified FLUX/Qwen CUDA reference](../../artifacts/framework/20261024/modern_image_cuda_acceptance.json)
also generates new-prompt PNGs from exported current-Student weights after
retiring training assets. [Wan2.2/Hunyuan CUDA qualification](../../artifacts/framework/20261024/native_video_cuda_acceptance.json)
likewise exercises actual native video roles/codecs and export-only new-prompt
RGB frames, with exact current-Student CPU relocation. None of these tiny-class
receipts establishes pretrained capacity/quality or changes a frozen workload
contract; CUDA planning remains separately unqualified.

The five common-flow families above (LTX LoRA, FLUX, Qwen-Image, Wan2.2
and HunyuanVideo) additionally have an explicit **single-CUDA manual immutable
sharing** path through the existing Program runtime. The
[separate native acceptance receipt](../../artifacts/framework/20261024/native_flow_cuda_immutable_sharing.json)
records ten actual tiny-class CUDA tests: three-cycle reference/manual complete
state, gradients, RNG and output parity; real frozen-parameter aliases and
lower tracked resident storage; fresh-process restart and current-Student
portable output. This is not CUDA auto/planning, AMP, multi-rank, pretrained
capacity or quality. All roles still materialize before sharing, so reduced
resident storage does **not** establish a lower construction peak. User runs
retain `numerical_status: NOT_RUN` unless independently checked. The R1 video
test receipt's policy-label error is preserved and corrected only by a later
CPU-tested writer change, not relabeled as another GPU run. See the
[exact policy](optimization_capabilities.md#direct-single-cuda-immutable-sharing).

FLUX and Qwen-Image additionally support explicit **equal-B1 post-backward
replication**, reference or exact manual immutable sharing. The
[replica acceptance](../../artifacts/framework/20261024/native_flow_image_replicated_acceptance.json)
contains actual native Gloo2 and A100 NCCL2 evidence: an independent three-cycle
gradient-mean oracle, full-rank fresh-process recovery, current-Student export
and new-prompt PNG output. This is not FSDP/DDP-forward wrapping, AMP, CUDA
planning or pretrained-capacity/quality evidence. A
[separate world16 follow-up](../../artifacts/framework/20261025/native_flow_image_world16_acceptance.json)
now verifies both image families on two A100 nodes: 16/16 independent-B1 oracle
ranks, all-rank full-state new-process restart, and exact reference/manual state
across three cycles. New-prompt decoded output is checked CPU-side from the
current Student export. The additional-model rows above describe single-device
evidence; this paragraph is the explicitly qualified replication exception for
the two image families, not the other model families.

Wan2.2 and HunyuanVideo also expose explicit local-B1 post-backward replica
admission in source and their [Wan2.2](../../recipes/wan22_dmd/README.md#explicit-replicated-local-b1-contract)
and [HunyuanVideo](../../recipes/hunyuan_video_dmd/README.md#explicit-replicated-local-b1-contract)
recipes. These are separate source/test routes, not video GPU replica
acceptance from the FLUX/Qwen receipts. LTX does not expose distributed ownership.

**Planning is a separate lane:** measured `auto`/saved `plan` acceptance for
common-flow families uses CPU1 FP32 `SYNTHETIC_TEST_ONLY` fixtures. Unqualified
pretrained whole-role copies reject before cloning; CUDA and distributed trials
remain refused. Direct reference/manual execution does not enter that copy
boundary. A selected-plan tensor budget does not cover construction, trial
copies or host RSS; all roles materialize before immutable sharing. See
[flow execution and capacity](training.md#modern-native-flow-qualification-and-capacity).

For DMD2, an actual scoped UNet checkpoint capability has real effects and
same-source reference/manual/auto/saved-plan + resumed-cycle parity on the
qualified CPU FP32 stack. A resident single-device CUDA FP32 reference source
path now includes native tensor/RNG/Adam ownership, host-state restore and
current-G export/reload placement. The seven-test A100 acceptance above covers
actual random-tiny reference CUDA lifecycle; **pretrained capacity, quality and
performance remain NOT_RUN**. Optional CUDA tests skip honestly on CPU-only
Torch. CUDA auto/plan/owner-selected trials and
whole-owner copying fail closed until budgeted resident snapshot/restore covers
frozen roles, optimizer, comparison and CPU staging. These singleton tests do
not qualify the separate native distributed owners or full pretrained gates. Other
physical options are available only when their actual runtime predicates and
measured effects pass. Visible-module Program backends are not advertised as
working on opaque native owners. The explicit
[native-owner role backend](native_owner_backends.md) is a separate exception:
actual DMD2 SD1.5/SDXL CPU2/Gloo has complete canonical per-rank checkpoint and exact
new-process continuation evidence, without duplicating native gradient
reduction. This `ddp-fp32-v1` backend supports persistent reference/direct-manual
FP32 only, not cloned auto/plan trials, GAN, mixed precision or FSDP2. Its CUDA/NCCL
source path has not been GPU-qualified. Existing single-device defaults are
unchanged. The same guide documents independently bound Generator/Fake
checkpoint effects (`component-v1`) on the qualified CPU1 stack.
See [model families](model_families.md),
[capabilities](optimization_capabilities.md) and their dedicated CI lanes.

Official Self-Forcing also has a bounded **exact random tiny CPU profile**:
explicit native checkpoint-ON reference, with only noncausal Fake checkpoint
ON/OFF admitted through the same manual/measured-auto/saved-plan owner protocol.
The causal Generator setting remains immutable algorithm identity; its ON/OFF
paths are not equivalent. Real checkpoint/recompute and direct-execution counts,
three-cycle independent native Trainer parity and fresh-process manual restart
are CPU evidence only. The legacy OFF reference is unchanged and offers no
checkpoint alternative. This singleton checkpoint-policy portfolio remains
CPU-only; CUDA trial copying/auto/plan remain rejected pending capacity-bounded
snapshot support. The explicit native FSDP2 paths below are separate, not
qualification of that trial portfolio.

The separate official Self-Forcing `release-after-commit-v1` model profile
relinquishes consumed rollout KV/cache references after native backward and
Adam/EMA commit. Default behavior is unchanged. Actual checkpoint-OFF/ON
three-cycle and fresh-process restart tests cover CPU native loss, gradients,
optimizer/EMA/RNG/input state. The policy has actual release counters, but its
byte count is not allocator-free memory and its new public CUDA path is
unqualified. See [the scoped owner guide](native_owner_backends.md).

## Integrated official optimization profiles

These source-integrated owners use the common Program, persistent native
cycle and complete-cycle/same-topology checkpoint lifecycle, without a
historical run directory or calibration plan:

- **SD1.5 official DMD2 reference:** `recipes/dmd2_sd15/training.fsdp2.json`
  selects `fsdp2-native-sd15-v1`, actual native block/root FSDP2 across
  A100/world16, G1/C10/EMA0 and the native mixed-precision exceptions with FP32
  masters/Adam. It does not enable the Resident/DDP bundle. The
  [reference guide](dmd2_fsdp2_reference.md) records source-lock/ABI and real
  CPU FSDP2 shard/Adam checks; public world16 training/restart/decoded-output
  GPU qualification remains pending. SDXL FSDP2 is not admitted.
- **SD1.5 optimized:** `recipes/dmd2_sd15/training.optimized.json` selects the
  source-bound Resident/DDP mixed-precision profile, with its actual fused
  Adam, activation retention, classifier packing and cast scopes.
- **Official Self-Forcing:** `training.fsdp2-reference.json` and
  `training.optimized.json` under `recipes/self_forcing_official/` select native
  FSDP2 reference and complete R6 optimization respectively, with BF16 compute,
  FP32 masters/reduction, A100/world16, G1/C5/EMA0 and explicit launch policy.
  They preserve causal rollout/KV/native update ownership and are not W4 LiveWan.

The [official chooser and launch/resume guide](official_optimized_training.md)
links each configuration and its historical measurements. Its
[validation appendix](../status/official_optimized_framework.md) separates
source/helper/CPU integration from historical native GPU outcomes, which retain
their **REVIEW** numerical status. The new public distributed optimized
training/restart/export paths still require GPU qualification; they do not
inherit the seven tiny single-device CUDA tests. Clone-based automatic planning
is not enabled for these distributed owners. Existing single-device FP32
defaults and the W1–W6 contracts remain unchanged.
