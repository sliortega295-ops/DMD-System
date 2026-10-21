# Capability and verification matrix

The public lifecycle and original-native integrations are different entry
adapters to the existing runtime. This table describes implementation support,
not throughput or pretrained-model quality.

| Entry | Data | Execution | Continuous progress | Save/resume | Generate/export | This integration's verification |
| --- | --- | --- | --- | --- | --- | --- |
| Public Program factories | Generic raw media + explicit encoder, or prepared native feature tree | Local and distributed reference/manual; existing measured auto/saved-plan path with admitted DDP/FSDP2/HSDP | Persistent models/wrappers/optimizer, versioned one-cycle Program and explicit rank-strided cursor | Complete cycle, same topology; all-rank atomic local-shard checkpoint | Actual model callbacks; default named-module weight export | Real local and multi-process CPU autograd, fresh-process continuation and local-shard restart |
| W1 Wan | Original rank-qualified trace or explicit fresh cohort stream | Original FSDP/Resident capability and saved plan | Existing 2..64 stream/policy range; manual/incumbent restrictions retained | Not exposed by native binding; final state is evidence | Native upstream tools, not unified callbacks | CPU source/contract/lowering |
| W2 Helios No-GAN | Original encoded assets/request | Actual selected source-bound native plan, including admitted HSDP | Existing two-cycle native API | Not exposed by native binding | Native upstream tools | CPU source/contract/lowering |
| W3 Helios GAN | Original GAN/ODE/text assets/request | Actual selected native plan | Existing two-cycle native API | Not exposed by native binding | Native upstream tools | CPU source/contract/lowering |
| W4 Self-Forcing | Original C51 continuous-sigma inputs | Actual selected C51 role-pool native plan | Existing two-cycle native API | Not exposed by native binding | Native upstream tools | CPU source/contract/lowering |
| W5 full-sequence H3 | LightX2V native AV assets; Grid1000 explicitly approximate | Actual selected FSDP2/cache policy | Existing two-cycle selected-plan API | Upstream mixin already saves LoRA/optimizer/RNG; exact data-cursor restart is not exposed here | Native upstream tools | CPU source/contract/lowering |
| W6 original H3 AR | Original DiffSynth previous/current AV, rank-qualified locks | Original ZeRO-3 reference/manual/selected plan | Existing two-update API | Not exposed by native binding | Native upstream tools | CPU source/contract/lowering |

The archived Helios reference trainer already has checkpoint and validation
hooks. Their existence does not mean the bounded planner worker exposes a
complete restart API. Likewise W5's native checkpoint mixin is preserved; its
presence is not misrepresented as missing or as a universal exact data restart.

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
