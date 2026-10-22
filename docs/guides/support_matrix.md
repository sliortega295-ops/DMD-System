# Capability and verification matrix

The public lifecycle and original-native integrations are different entry
adapters to the existing runtime. This table describes implementation support,
not throughput or pretrained-model quality.

| Entry | Data | Execution | Continuous progress | Save/resume | Generate/export | Held-out validation | This integration's verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Public Program factories | Generic raw media + explicit encoder, or prepared native feature tree | Local and distributed reference/manual; existing measured auto/saved-plan path with admitted DDP/FSDP2/HSDP | Persistent models/wrappers/optimizer, versioned one-cycle Program and explicit rank-strided cursor | Complete cycle, same topology; all-rank atomic local-shard checkpoint | Same-topology distributed checkpoint `evaluate` and full-weight `export`; local-only `generate --weights` on new prepared inputs with source/layout-bound loading; PNGs require an explicit model decoder | Optional periodic local-only `validate_batch` diagnostics on explicit held-out records; no inferred quality gate | Real local and multi-process CPU autograd, fresh-process continuation, local-shard restart, distributed outputs and installed-wheel new-input inference |
| W1 Wan | Original rank-qualified trace or explicit fresh cohort stream | Original FSDP/Resident capability and saved plan | 2..64 fresh cohorts with explicit reference/manual/plan; literal incumbent bundle remains fixed-trace/two-cycle | Fresh stream only, complete cycle/same topology: original engine weights, named Adam, scheduler, EMA, clocks, input cursor and owned RNG; all-rank atomic shards | Native upstream tools, not unified callbacks | Not exposed by native binding | CPU source/contract/lowering and real Adam/engine-state restart on scalar fixtures; pretrained/GPU restart NOT_RUN |
| W2 Helios No-GAN | Original encoded assets/request, or explicit fresh cohort stream | Existing source-bound selected plan; separate direct FSDP2/SDPA reference path | Selected plan: two cycles. Direct reference: 2..64 fresh cohorts from an explicitly supplied original Step10 G2/C10/EMA10 checkpoint | Direct reference stream only: complete cycle/same topology, original G/C Adam, schedulers, EMA, versions, cursor and RNG | Native upstream tools, not unified callbacks | Not exposed by native binding | Actual licensed input/update/state helpers on tiny CPU DTensor owners; real two-process Gloo public-to-P0 admission. Pretrained CUDA/FSDP restart NOT_RUN |
| W3 Helios GAN | Original GAN/ODE/text assets/request | Actual selected native plan | Existing two-cycle native API | Not exposed by native binding | Native upstream tools, not unified callbacks | Not exposed by native binding | CPU source/contract/lowering |
| W4 Self-Forcing | Original C51 continuous-sigma inputs; separate original-format finite teacher cohorts | Selected C51 role-pool plan; separate world16 FSDP2 G/Teacher plus replicated Critic reference | Selected: two cycles. Explicit reference: 2..64 fresh G1/C5/EMA0 cycles, genuine zero-step initialization | Reference only: actual G/C trainables and named Adam, input cursor, versions, per-rank RNG; all-rank atomic complete-cycle/same-topology. C51 physical pipeline restart unavailable | Native upstream tools, not unified callbacks | Not exposed by native binding | Original input/LoRA/Adam/CPU DTensor, actual meta constructors and Gloo failure tests; outer-loop dispatch fixture is not pretrained/CUDA proof |
| W5 full-sequence H3 | LightX2V native AV assets; Grid1000 explicitly approximate. Separate fresh cached-condition inventory | Existing selected FSDP2/cache policy; separate explicit original reference config, with optional optimized paths OFF | Selected plan: two cycles. Original reference: 2..64 finite fresh-data G1/F5 cycles | Original reference only: complete cycle/same topology, real two-LoRA Adam/schedulers, versions, deterministic sampler cursor and RNG; EMA0 | Native upstream tools, not unified callbacks | Not exposed by native binding | Actual original Grid loop/data/Adam on tiny CPU construction fixtures, real Gloo failure and restart tests. Pretrained runtime/capacity/numerics NOT_RUN; optimized backward FAIL unchanged |
| W6 original H3 AR | Original DiffSynth previous/current AV, rank-qualified locks | Original ZeRO-3 reference/manual/selected plan | Existing two-update API | Not exposed by native binding | Native upstream tools, not unified callbacks | Not exposed by native binding | CPU source/contract/lowering |

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
not a change to its bounded C51 worker. W3/W6 still retain their bounded
native lifecycle here. The archived Helios full trainer's additional
save/validation hooks are preserved, not asserted to be exposed by this adapter.

The separate [full-original Helios model provider](native_workloads.md#explicit-full-original-helios-model-reference)
has tiny FP32 CPU full-class dense-SDPA flow/GAN/PEFT/Adam and model-serialization
tests. It reuses the unchanged original transformer in a private namespace and
rejects unsupported NaViT/varlen execution. That model/backend proof does not
enable W3's public continuous training/restart, or establish pretrained quality,
CUDA/FSDP/world16 correctness or performance; the W3 table entry remains bounded.

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
