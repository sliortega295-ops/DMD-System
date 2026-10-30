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
acceptance. See GPU acceptance (source checkout: `docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md`).

| Entry | Data | Execution | Continuous progress | Save/resume | Generate/export | Held-out validation | This integration's verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Public Program factories | Generic raw media + explicit encoder, or prepared native feature tree | Local and distributed reference/manual; existing measured auto/saved-plan path with admitted DDP/FSDP2/HSDP | Persistent models/wrappers/optimizer, versioned one-cycle Program and explicit rank-strided cursor | Complete cycle, same topology; all-rank atomic local-shard checkpoint | Same-topology distributed checkpoint `evaluate` and full-weight `export`; local-only `generate --weights` on new prepared inputs with source/layout-bound loading; PNGs require an explicit model decoder | Optional periodic local-only `validate_batch` diagnostics on explicit held-out records; no inferred quality gate | Real local and multi-process CPU autograd, fresh-process continuation, local-shard restart, distributed outputs and installed-wheel new-input inference |
| W1 Wan | Original rank-qualified trace or explicit fresh cohort stream; separate original-source CPU reference | Native FSDP/Resident/saved plan; complete incumbent stream with Graph/cast/fusion | Fresh native stream: 2..100 distinct cohorts; fixed benchmark remains two-cycle | Complete cycle/same topology; actual named Adam/scheduler/EMA/RNG and data cursor. Existing world16 new-process continuation | [Native current G + Wan2.1 VAE](../../recipes/wan_ar/NATIVE_OUTPUT.md), independent new-prompt decoded video; CPU latent diagnostic remains separate | Not exposed by native binding | Complete-incumbent world16 execution/restart and prior baseline/Resident 50-cycle training reused. Existing trained G4 portable export generated 33×384×640 decoded frames on one A100 through the public CLI; no new training or quality claim |
| W2 Helios No-GAN | Original encoded assets/request and fresh cohort stream; explicit G2/C10/EMA10 warm state | Direct FSDP2 reference or explicit node-HSDP SafeBundle stream; fixed preset/plan separate | Reference/SafeBundle: 2..100 fresh cohorts | Complete cycle/same topology; original G/C Adam, scheduler, SparseEMA, source cursor and inner/outer RNG | [Native current unmerged G + Wan2.1 VAE](../../recipes/helios_nogan/NATIVE_OUTPUT.md), independent new-prompt decoded video | Not exposed by native binding | Prior baseline/HSDP 50-cycle records retained at their original mechanism scope. Current SafeBundle lifecycle and native output chain have source/CPU checks; [rank-local amplified-history continuation](../../recipes/helios_nogan/NATIVE_HISTORY.md) binds an aligned whole-block FSDP2 phase. Attempt 30 stopped before C1 because offline policy was installed before the pinned FA2 dispatch import. The import-order repair passes focused CPU checks and both target Torch 2.10 environments; the failed attempt is retained and the repaired source still needs its own native admission. |
| W3 Helios GAN | Fresh original GAN history/primary/secondary/R1 inputs with explicit systems bootstrap; independent opt-in [native-history ABI](../contracts/helios_gan_native_history_v1.md) | Full original world16 FSDP/C2048/window5 FA3/down40 stream; N70 OFF; CPU Program reference separate | Full stream: finite prepared G1/C5/EMA1 cycles, including 100; fixed preset/plan: two cycles | Complete native G/C/GAN-head Adam, scheduler, EMA, rank RNG, physical/curriculum clocks and fresh stream cursor | [Native current unmerged G + Wan2.1 VAE](../../recipes/helios_gan/NATIVE_OUTPUT.md), independent new-prompt decoded video | Not exposed by native binding | Existing baseline 50-cycle record retained. Current N70-OFF lifecycle, native-history dispatch and tiny native output chain have source/CPU checks; full-size capacity and changed execution/restore endpoints remain separate |
| W4 Self-Forcing / LiveWan | C51 fresh native continuous-sigma Teacher inputs; distinct original reference and CPU feature schemas | Full C51 8G+8Score dual-wave owner; original world16 reference; separate actual-original CPU Program | Full C51 stream: 2..100 G1/C5/EMA0; original reference: 2..64; fixed benchmark: two cycles | C51 stream: drained two-pool complete cycle, both role Adam states, clocks, cursors and RNG; reconstruct transient staging/cache | [Native C51 current full Student export](livewan_native_output.md), independent original four-step streamer + embedded Wan VAE, CPU FP32; CPU Program output remains separate | No automatic quality gate | Original baseline/full 100-cycle training and held-out outputs reused. The bounded in-place C51 Adam restore now passes 50/50 target Torch 2.10 CPU cases on both nodes. A fresh same-source C1 checkpoint and C2 restore are still required to qualify the GPU recovery endpoint. |
| W5 full-sequence H3 | LightX2V native AV assets; Grid1000 explicitly approximate. Separate fresh cached-condition inventory | Existing selected FSDP2/cache policy; separate explicit original reference config, with optional optimized paths OFF | Selected plan: two cycles. Original reference: 2..64 finite fresh-data G1/F5 cycles | Original reference only: complete cycle/same topology, real two-LoRA Adam/schedulers, versions, deterministic sampler cursor and RNG; EMA0 | [Independent converted Grid current-Student output](h3_grid_inference.md): original-reference native checkpoint export and original four-step packed AV and optional original VAE/BigVGAN media via exact patch/stereo ABI, source/CPU-tested | Not exposed by native binding | Actual original Grid loop/data/Adam on tiny CPU construction fixtures, real Gloo failure and restart tests. Pretrained runtime/capacity/numerics NOT_RUN; optimized backward FAIL unchanged |
| W6 original H3 AR | Native rank-qualified previous/current AV; separate cached-AV CPU reference | Original ZeRO-3 reference/manual-AP; frozen two-update benchmark remains separate | Native fresh stream: 2..100 G1/F1/EMA0 cycles, eight rollout steps; CPU reference has its own finite inventory | Native actual-engine save/load, both LoRAs/inactive head/FP32 master Adam, native clocks, rank RNG and cursor; same topology/horizon | [Native current Student export](h3_ar_native_inference.md), exact forward storage and original eight-step new-condition AV latents; [separate actual AV codecs](h3_audio_video_decode.md) | Not exposed by native binding | Source/CPU native restore/export protocol, actual original independent model and codecs; native GPU output/continuation and pretrained quality remain separate; existing completed runs are reused |

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

## Native continuity and output

Frozen two-cycle presets remain reproduction tools. Continuous training uses
separately versioned fresh-stream configs, not a larger `cycles` value on a
benchmark config. The [native guide](native_workloads.md#continuous-native-training)
lists each executable recipe and its exact owner. W1/W2 extend to 100 fresh
cohorts; W3 full GAN, W4 full C51 and W6 native ZeRO-3 now have source/CPU
complete-cycle continuation and checkpoint integration. Their CUDA validation
is a separate next step, not inferred from the CPU tests. W5's original Grid1000
reference remains separate from the unresolved optimized backward path.

Current-G native video export is available for W1/W2/W3 after their own
complete checkpoint restore. Ordinary training does not consume `trainer.export`;
use the explicit export command and codec policy. Original GAN heads and
optimizer state are restored but excluded from portable inference weights.

For an existing trained Generator saved outside the complete-checkpoint format,
`distillctl import-weights` provides a separate initialization-only path:
[W1 complete LoRA + frozen base](wan_weights_initialization.md) and
[W4 complete Student](livewan_weights_initialization.md). Both use the existing
public publisher, strict inference factory and codec. Actual full-size trained
weights have been published through this path without another training run;
the original training source, counters and numerical status remain attached.
The new export starts at cycle zero and is not a resumable training checkpoint.
For W1, its existing full-size artifact has also passed independent single-A100
new-prompt generation through the same public command. Only the standalone
device changed from CPU to CUDA; weights, precision, sampler and prepared
input identities remained fixed. The output does not require the old base,
adapter or VAE directories. This check does not substitute for the separate
all-rank checkpoint-export endpoint or trained-model quality evaluation.

W6 now has an output-only native restore/export bridge and an independent
original-model consumer. Original H3 video/audio codec execution is available
through the separate `decode` command. [Original raw H3 AV preparation](h3_audio_video_encode.md) now encodes real previous/current frame directories and stereo WAV with supplied original text features, feeding the existing cached-input CPU reference without a new training loop. **W5 has its own source/CPU original-reference checkpoint export and an explicit original codec ABI**, including patch inversion and stereo ordering. Its four-step sampler is retained; W6 is not used as a replacement.
Existing completed runs of **50 or more cycles are reused**, with their original
source, workload and numerical scope. A new framework revision does not reset
that evidence or require another long run. Changed save/restore/output endpoints
receive bounded integration checks instead. Both official optimized workflows
already completed 100 cycles with a midpoint fresh-process restart.

The separately versioned [W1 CPU reference](../../recipes/wan_ar/CPU_REFERENCE.md)
uses the actual original model and P0 Full5 actions, not W3 with GAN disabled.
It admits the complete source trace but executes rank0 B1 on CPU; actual
world16 gradient reduction is not being qualified. Its exported current
unmerged G can generate explicit native latents from new prompt tensors
without old training payloads. The original native W1 recipe remains separate.

The separate [W1 complete-incumbent stream preset](../../recipes/wan_ar/INCUMBENT_STREAM.md)
now binds that original native session, its complete Graph/cast/fusion bundle,
fresh 2..100-cohort input owner and common all-rank checkpoint path. CPU scalar
autograd/Adam/EMA and fresh-process restart tests exercise the actual dispatcher
and controller lifecycle with explicitly fake CUDA capture drivers. The later
actual world16 continuation (source checkout: `docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md#resident-stream-follow-up-actual-world16-continuation`)
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

W4's [finite original reference input owner](livewan_inputs.md) remains
independently usable. Full C51 continuation instead uses its distinct
[dual-pool stream recipe](../../recipes/self_forcing/C51_CONTINUATION.md),
with actual two-wave consumption, drain and checkpoint lifecycle. These source/
CPU protocol tests do not replace the forthcoming native world16 recovery run.
The [native current-Student output bridge](livewan_native_output.md) restores
that same full C51 checkpoint and publishes only primary-owned Generator/codec
weights, with no Generator allocation on Score ranks. Its independent original
four-step streamer and embedded VAE currently execute CPU FP32; actual original
tiny models, checkpoint-owner edges and real Gloo role partitioning have CPU
tests. Native CUDA/world16 output and pretrained quality remain separate gates.
The original reference input receipt (source checkout: `artifacts/framework/20261021/original_livewan_inputs_cpu.json`)
and reference state receipt (source checkout: `artifacts/framework/20261021/original_livewan_reference_cpu.json`)
retain their original scope; they are not relabelled as C51 evidence.

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
exact scope and failures (source checkout: `docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md`).

| Integration | Actual CPU reference | Data/output and continuation | Not established |
| --- | --- | --- | --- |
| SD1.5 official DMD2 | Original SDUniModel/SDGuidance, native G1/C10 clock; actual UNet/CLIP/VAE | Prompt-only + independent guidance/real/denoising streams as selected; actual raw VAE intake; exact complete-state new-process recovery; new-prompt current-G image output. Separate native optimized world16 training/restart/export and new-prompt PNG acceptance below | Pretrained quality and settings outside the qualified optimized runtime; FSDP2 reference qualification remains separate |
| SDXL official DMD2 | Original dual CLIP/added conditions, four-step and explicitly initialized single-step399; real unmerged LoRA | Independent native stream counts; actual VAE/CLIP, safe full-state resume and current-G images; separate CPU/Gloo native role-DDP continuation | Pretrained CUDA capacity/quality, CUDA native role-DDP qualification, AMP/FSDP2; SD1.5 optimized profile does not apply |
| Official Self-Forcing | Actual CausalWan/Wan/DMD classes, native 21-latent-frame block3 random-exit/KV/EMA rules | Actual native T5 preparation or exact embeddings; independent G/C streams; exact continuation; current-G full-step native inference/VAE. Separate native optimized FSDP2 world16 100-cycle/new-process restart acceptance below; reference path remains separate | >21-latent-frame decode/reencode training, I2V, uncached flex, full-native decoded output, pretrained quality |
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
qualified FLUX/Qwen CUDA reference (source checkout: `artifacts/framework/20261024/modern_image_cuda_acceptance.json`)
also generates new-prompt PNGs from exported current-Student weights after
retiring training assets. Wan2.2/Hunyuan CUDA qualification (source checkout: `artifacts/framework/20261024/native_video_cuda_acceptance.json`)
likewise exercises actual native video roles/codecs and export-only new-prompt
RGB frames, with exact current-Student CPU relocation. None of these tiny-class
receipts establishes pretrained capacity/quality or changes a frozen workload
contract; CUDA planning remains separately unqualified.

The five common-flow families above (LTX LoRA, FLUX, Qwen-Image, Wan2.2
and HunyuanVideo) additionally have an explicit **single-CUDA manual immutable
sharing** path through the existing Program runtime. The
separate native acceptance receipt (source checkout: `artifacts/framework/20261024/native_flow_cuda_immutable_sharing.json`)
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
replica acceptance (source checkout: `artifacts/framework/20261024/native_flow_image_replicated_acceptance.json`)
contains actual native Gloo2 and A100 NCCL2 evidence: an independent three-cycle
gradient-mean oracle, full-rank fresh-process recovery, current-Student export
and new-prompt PNG output. This is not FSDP/DDP-forward wrapping, AMP, CUDA
planning or pretrained-capacity/quality evidence. A
separate world16 follow-up (source checkout: `artifacts/framework/20261025/native_flow_image_world16_acceptance.json`)
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
Torch. A separate [budgeted resident lifecycle](resident_owner_training.md)
connects the exact tiny-native portfolio to public direct/auto/saved-plan
training and resume without whole-owner copies. It requires a qualified
preconstruction provider and explicit host/device budgets; the original
no-budget CUDA trial refusal is unchanged. The [public lifecycle A100 acceptance](../status/FRAMEWORK_PUBLIC_RESIDENT_ACCEPTANCE_20261028.md)
covers three-cycle auto execution, direct fresh-process resume and current-G
decode for SD1.5, SDXL and official Self-Forcing, plus the shared SD1.5 saved-plan
restart path. These singleton tests do
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
three-cycle independent native Trainer parity are covered by CPU evidence.
The public budgeted portfolio also has the A100 continuation/restart/output
acceptance linked above. The legacy OFF reference is unchanged and offers no
checkpoint alternative. This singleton checkpoint-policy portfolio remains
available through the explicit budgeted resident lifecycle as well; unbudgeted
CUDA trial copying/auto/plan remain rejected. The explicit native FSDP2 paths below are separate, not
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
  The actual full-native public runtime completed 100 cycles on 16 A100 GPUs,
  including a durable cycle 50 checkpoint and fresh-process continuation to 100.
  This exact-source runtime check uses deterministic random initialization and
  synthetic prepared records, not a pretrained-quality or new timing result.
- **Official Self-Forcing:** `training.fsdp2-reference.json` and
  `training.optimized.json` under `recipes/self_forcing_official/` select native
  FSDP2 reference and complete R6 optimization respectively, with BF16 compute,
  FP32 masters/reduction, A100/world16, G1/C5/EMA0 and explicit launch policy.
  They preserve causal rollout/KV/native update ownership and are not W4 LiveWan.

The [official chooser and launch/resume guide](official_optimized_training.md)
links each configuration and its historical measurements. Its
[validation appendix](../status/official_optimized_framework.md) separates
source/helper/CPU integration from historical native GPU outcomes, which retain
their **REVIEW** numerical status. The same appendix records the actual public
SD1.5 optimized and official Self-Forcing 100-cycle runs, both with durable
cycle 50 checkpoints and fresh-process continuation to cycle 100.
Both completed same-source native world16 cycle-100 Student export and
independent new-prompt generation: DMD2 produced a 512 × 512 image and official
Self-Forcing an 81-frame, 368 × 640 video. The
[output receipts](../status/official_optimized_framework.md#public-native-lifecycle-validation-2026-10-29)
record tensor/PNG integrity and both owner cleanups. These results are not
inherited from the seven tiny single-device CUDA tests. Clone-based automatic
planning is not enabled for these distributed owners. Existing single-device
FP32 defaults and the W1–W6 contracts remain unchanged.
