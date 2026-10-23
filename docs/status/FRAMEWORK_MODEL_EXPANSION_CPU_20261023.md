# Unified framework expansion — CPU integration, 2026-10-23

This is an implementation and CPU qualification receipt, **not a performance
table, pretrained-quality result or new GPU validation**. W1–W6 retain their
distinct source, input, precision, hardware and numerical-evidence boundaries.
In particular, official Self-Forcing is not an alias for W4 LiveWan, and W5
full-sequence/Grid1000 is not W6 autoregressive H3.

## Delivered user path

All integrations use the existing `distillctl`, `ModelComponents`, Program,
training lifecycle and admitted execution interfaces:

1. Discover models, algorithms, strategies and capabilities without importing
   Torch, initializing CUDA or downloading weights.
2. Prepare actual model-native inputs offline, or admit explicitly declared
   preencoded conditions. Prompt-only and independent native input streams are
   supported. Missing conditions and incompatible tokenizers/VAEs fail closed.
3. Run real gradient updates with persistent roles, optimizers, original native
   clocks, scheduler/EMA rules and data progress.
4. Publish complete-cycle checkpoints only after all required shards complete.
   Fresh-process same-topology continuation restores state, RNG, stream positions
   and versions; initial weights alone are not a restart checkpoint.
5. Export current unmerged Generator/Student plus actual inference components,
   then generate on new prepared inputs without the old training artifacts.
   Image, video, audio, AV and explicit latent outputs require the corresponding
   model-native encoder/decoder, not a generic shape-based replacement.

## Actual new native providers

| Integration | Native implementation and numerical contract | CPU evidence |
| --- | --- | --- |
| SD1.5 / DMD2 | Pinned original `SDUniModel`/`SDGuidance`, actual UNet/CLIP/VAE, single-step and original optional GAN configuration, G1/C10 | Real loss/gradient/Adam/RNG/source-trainer oracle; raw RGB native-VAE preparation; exact new-process recovery and current-G images |
| SDXL / DMD2 | Same original owner; actual dual CLIP, pooled/size conditions, four-step or explicitly initialized single-step399; unmerged LoRA | Same native oracle/recovery/export tests, independent denoising streams, both tokenizer/VAE bindings and strict original generator-weight mapping |
| Official Self-Forcing | Actual CausalWan/Wan, block3 random-gradient-exit rollout, KV state, independent G/C samples and original EMA clock | Native trainer oracle; actual local T5 prompt preparation; real updates, exact recovery and full-step native VAE video |
| LTX-2.5 | Actual AV LTX Transformer, native Gemma4/connectors, AV codecs; generic explicit four-step G1/F5 flow-DMD policy, full or separate A/B LoRA | Native small architectures and original safetensors import; real AV updates, exact full/LoRA recovery and new-input AV generation |

LTX and an independent non-LTX model execute the **same** generic flow-DMD
algorithm. No LTX model-name condition is added to the generic searcher or
training loop. The SD and Self-Forcing algorithms preserve their different
native mathematics rather than being relabeled as that generic flow policy.

The original DMD2 generator key space is preserved in the common export. The
[documented mapping](../../recipes/dmd2_sd15/README.md#official-generator-weight-mapping)
is tested for updated SD1.5, SDXL and SDXL-LoRA weights. It deliberately starts
new optimizer/native clocks; it is not complete training recovery.

## Verification and reproducibility

The [machine-readable receipt](../../artifacts/framework/20261023/model_expansion_cpu.json)
contains final suite counts, JUnit hashes, source identity and evidence scope.
Dedicated native provider lanes must have **zero skips**, not merely return a
successful exit status. Generic and legacy lanes may skip absent optional
providers; those skips are reported separately and never qualify a provider.

The tested model-specific environments are separate:

| Lane | Principal numerical stack |
| --- | --- |
| Common / original helper | Torch2.10 CPU, original Diffusers0.34 helper lane |
| SD DMD2 / official Self-Forcing | Torch2.10 CPU, Diffusers0.37.1, Transformers4.51.3, PEFT0.17.1 |
| LiveWan public original | Torch2.10 CPU, Diffusers0.35.2, Transformers4.51.3, PEFT0.17.1 |
| LTX-2.5 | Torch/TorchAudio2.10 CPU, TorchVision0.25 CPU, Transformers5.14.1, native unmerged LoRA; Python3.12 qualification |

The isolated installed-wheel check executes real media preparation, gradients,
fresh-process continuation, measured automatic selection, saved-plan execution
and new-input generation outside the checkout. It verifies imported numerical
package versions/origins rather than silently using another base environment.
The lightweight wheel check discovers all six workloads and model/algorithm/
strategy/capability catalogs without Torch. No test installs into or upgrades
an existing experiment environment.

The existing six-workload source/hardware maps and five frozen legacy profiles
remain intact. Source/CPU integration checks are not new world16 validation.
See the [complete support matrix](../guides/support_matrix.md) for each
original recipe's actual lifecycle/output support and limitations.

## Optimization and plan execution

The existing visible-module reference/manual/auto/saved-plan backends remain
the physical execution owners. A new source-bound native-owner seam makes
actual SD Generator/Fake UNet checkpointing measurable and executable by the
same planner. Real request/recompute counters, flag restoration, paired native
output/gradient/state/RNG checks and first-postrestart admission are tested.
The optimizer-containing native macrocycle is never replayed.

This proves functional selection and actual saved-policy execution under a
measured CPU tracked-storage constraint, **not CUDA memory savings or speedup**.
Opaque owners do not acquire FSDP/AMP/Graph support simply by listing those
capabilities. Existing model-specific GPU mechanisms retain their native
admission checks; unsupported combinations raise errors.

## Review corrections included

- SD prepared manifests and every row are bound to the actual tokenizer set
  and, when applicable, VAE weights/configuration/scaling. Supplied latents
  require an explicit compatibility assertion and are never falsely labeled as
  encoder output. Train, evaluate and portable generation enforce the binding.
- LiveWan runtime inventory enumerates actual live parameters, buffers, RoPE
  tensors, gradients and Adam moments rather than cloning checkpoint state.
- Generic CPU CI selects explicit files before pytest, excluding conflicting
  provider environments correctly. Dedicated provider CI executes real source
  tests and rejects missing proofs, failures, empty receipts and skips.

## Remaining validation boundaries

- New providers are qualified at **small random actual upstream architectures
  on CPU1/FP32**. This is not full-size pretrained training or quality evidence.
- Full production DMD2 GAN training, CUDA/AMP/distributed execution of the new
  owners, larger official Self-Forcing contexts/I2V, and LTX's complete 22B
  monolithic pipeline remain unqualified or explicitly unsupported.
- Complete-cycle, same-source/topology restart is supported. Arbitrary internal
  interruption or cross-topology recovery is not promised.
- Encoders/decoders and trusted plugins execute Python supplied by the user;
  source hashes are integrity checks, not a sandbox or proof of latent quality.
- DMD2 and LTX source/licenses remain explicit external optional dependencies.
  Model/data licenses are separate; no pretrained weights are bundled/fetched.
- No A100/H100 experiment, performance search, new kernel tuning, main-branch
  modification or publication is part of this receipt.

## Post-install validation and rollback

After choosing the documented separate environment, run its zero-skip CPU
provider lane, then the recipe's model-native admission before a real training
deployment. Healthy signals are complete checkpoints, consistent native clocks,
source/preprocessing identities and real optimization counters. A missing
shard, changed input binding, unsupported layout or failed paired gate must
stop before publication/update; restore the last complete checkpoint and
return to the supported `reference` policy rather than removing the guard.
Real CUDA/distributed correctness, capacity and convergence require their own
operator-owned qualification and are not implied by this CPU release.
