# Capability and verification matrix

The public lifecycle and original-native integrations are different entry
adapters to the existing runtime. This table describes implementation support,
not throughput or pretrained-model quality.

| Entry | Data | Execution | Continuous progress | Save/resume | Generate/export | Held-out validation | This integration's verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Public Program factories | Generic raw media + explicit encoder, or prepared native feature tree | Local and distributed reference/manual; existing measured auto/saved-plan path with admitted DDP/FSDP2/HSDP | Persistent models/wrappers/optimizer, versioned one-cycle Program and explicit rank-strided cursor | Complete cycle, same topology; all-rank atomic local-shard checkpoint | Same-topology distributed checkpoint `evaluate` and full-weight `export`; local-only `generate --weights` on new prepared inputs with source/layout-bound loading; PNGs require an explicit model decoder | Optional periodic local-only `validate_batch` diagnostics on explicit held-out records; no inferred quality gate | Real local and multi-process CPU autograd, fresh-process continuation, local-shard restart, distributed outputs and installed-wheel new-input inference |
| W1 Wan | Original rank-qualified trace/fresh stream; separate complete-source-trace CPU reference | Native FSDP/Resident/saved plan; separate CPU1 rank0 per-sample original Full5 Program | Native stream: 2..64 fresh cohorts; CPU reference: finite original cohorts within fixed scheduler horizon | Native stream: all-rank atomic shards. CPU reference: actual original full mixed model, named Adam/LR/EMA, clocks, source cursor and inner/outer RNG through common complete-cycle checkpoint | CPU reference: current full unmerged G and new-prompt original native latent diagnostics; no decoder | Not exposed by either W1 binding | Native source/contract and scalar engine-state restart; separate actual original tiny BF16-base/FP32-LoRA CPU model/update/fresh-process restart/export. Pretrained/world16 numerical/CUDA/quality NOT_RUN |
| W2 Helios No-GAN | Original encoded assets/request or fresh cohort stream; separate original full-geometry CPU input owner | Selected plan; direct FSDP2/SDPA reference; separate CPU1 rank0 per-sample original warm No-GAN Program | Selected: two cycles. References: 2..64 fresh cohorts from explicit real G2/C10/EMA10 warm state | Complete cycle/same topology; original named G/C Adam, LR, SparseEMA, clocks, source cursor and inner/outer RNG. CPU Program stages all owners before restore | CPU reference: current full unmerged G export and new-prompt original native latent diagnostics; no decoder | Not exposed by native binding | Actual original full tiny mixed-BF16/FP32 model, LoRA256, original warm updates, fresh-process exact restart and independent current-G output; separate Gloo/native admission tests. Pretrained/world16 numerical/CUDA/quality NOT_RUN |
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
`distillctl workloads describe --workload Wn --hardware a100|h100`. This
framework integration does not launch, remeasure or revalidate those GPUs.
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

These extend registration, not the immutable W1–W6 identities. All use the
same `distillctl`, Program, checkpoint and export interfaces.

| Integration | Actual CPU reference | Data/output and continuation | Not established |
| --- | --- | --- | --- |
| SD1.5 official DMD2 | Original SDUniModel/SDGuidance, native G1/C10 clock; actual UNet/CLIP/VAE | Prompt-only + independent guidance/real/denoising streams as selected; actual raw VAE intake; exact complete-state new-process recovery; new-prompt current-G image output | Full production GAN training, pretrained quality, GPU/AMP/distributed |
| SDXL official DMD2 | Original dual CLIP/added conditions, four-step and explicitly initialized single-step399; real unmerged LoRA | Independent native stream counts; actual VAE/CLIP, safe full-state resume and current-G images | Pretrained quality, GPU/AMP/distributed |
| Official Self-Forcing | Actual CausalWan/Wan/DMD classes, native 21-latent-frame block3 random-exit/KV/EMA rules | Actual native T5 preparation or exact embeddings; independent G/C streams; exact continuation; current-G full-step native inference/VAE | >21-latent-frame decode/reencode training, I2V, uncached flex, AMP/FSDP/CUDA, pretrained quality |
| LTX-2.5 | Actual bounded native AV Transformer, Gemma4/connectors and native AV codecs; generic explicit G1/F5 flow DMD, full or separate LoRA | Actual raw media/prompts/exact native features; original-format tiny safetensors import; clock-aware exact recovery; current-G new-input AV decode | Full monolithic 22B pipeline, native architectures outside the documented CPU bounds, GPU/distributed/quality/performance |

For DMD2 only, an actual scoped UNet checkpoint capability has real effects and
same-source reference/manual/auto/saved-plan + resumed-cycle parity. Other
physical options are available only when their actual runtime predicates and
measured effects pass. Visible-module Program backends are not advertised as
working on opaque native owners. See [model families](model_families.md),
[capabilities](optimization_capabilities.md) and their dedicated CI lanes.
