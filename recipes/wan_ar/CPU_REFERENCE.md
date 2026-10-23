# W1 original No-GAN CPU reference

This opt-in reference connects the **original W1 full model and Full5 update**
to the common Program training lifecycle. It is not the W3 GAN trainer with a
flag disabled, a scalar stand-in, a new loss, or a CUDA backend.

## Contract and scope

- The original `prepare_parameterized_wan_cycle` owns dispatch. Its original P0
  helpers perform one G update, five distinct post-update G rollouts and five C
  updates. EMA follows the supplied original task (the reference task uses EMA5).
- The complete original Helios-class model used by W1 retains its native mixed
  BF16 base inventory, original FP32-only modules, FP32 LoRA and actual Adam,
  schedulers and sparse EMA. All three Generator pyramid-patch adapters remain
  trainable. No merged weights, CUDA Graph or optimized replay cache is used.
- Execution is **CPU1, source rank0, B1 per-sample**. The input adapter still
  validates the complete original world16 trace controls and payloads before
  preparing rank0. It does not run world16 reduction or establish global-B16
  numerical equivalence. The native world16 recipe is unchanged.
- Initialization is genuinely G0/C0/EMA0 with empty Adam state. Finite cohorts
  advance their original source steps without trace replay. The fixed native
  scheduler horizon must accommodate every requested cycle.
- Missing conditioning, pyramid weights, source identities, incompatible
  precision, unsupported branches and repeated source cohorts fail explicitly.
  There is no random missing-weight fallback.

Use a source checkout with the separate `.[train,helios-reference]` environment
described in [installation](../../docs/guides/installation.md). This full-model
Diffusers0.37.1 route is distinct from the native-Wan0.35 qualification lane.
Tiny random assets used by tests are explicitly labeled fixtures, not a
pretrained model, media encoder or quality result.

See the [source-scoped CPU acceptance](../../docs/status/WAN_ORIGINAL_CPU_ACCEPTANCE_20261023.md)
for actual model, restart, projection and new-input checks and their limits.

## Prepare supplied original inputs

Start from genuine original captured operation traces. The
[input template](inputs.cpu-reference.template.json) names an ordered finite
cohort inventory and its original input configuration. Each cohort must have
the complete source-qualified rank16 trace and distinct actual source content;
changing a sample name is not fresh data.

The [asset template](assets.cpu-reference.template.json) pins complete local
Generator/Critic architecture and safetensors shards, original arguments,
negative conditioning, input configuration and explicit CPU capacity limits.
Fill these with actual identities; the placeholder files intentionally cannot
be executed. Never convert FP32 fixture weights and describe them as an
admitted native mixed-precision checkpoint.

```sh
distillctl data prepare-wan-reference --sources /YOUR/W1/inputs.json \
  --output /YOUR/W1/prepared
distillctl data validate /YOUR/W1/prepared/manifest.json --sources
```

This command only serializes and validates existing inputs. It neither samples
replacement histories nor changes the original feature producer, noise or loss.

## Train and restore

Fill [`training.cpu-reference.json`](training.cpu-reference.json) with your
asset/prepared paths and a new output directory. Keep the finite sampler,
`execution.adapter: program`, `mode: reference` and CPU device explicit.

```sh
distillctl config validate /YOUR/W1/training.json
distillctl train /YOUR/W1/training.json
distillctl train /YOUR/W1/training.json \
  --resume /YOUR/W1/run/checkpoints/cycle-000002 \
  --cycles 3 --output /YOUR/W1/resumed
```

The common complete-cycle checkpoint owns actual native trainables, named
Adam state, LR schedulers, EMA, clocks, consumed-source chain, native RNG, outer
RNG and the data cursor. Resume is same-source/same-topology at a complete
boundary, not arbitrary-step migration. A partial numerical update poisons
the whole owner; it cannot be retried in place. Restore a preceding complete
checkpoint into a freshly admitted owner instead.

## Export current G and generate with new inputs

```sh
distillctl export /YOUR/W1/training.json \
  --checkpoint /YOUR/W1/resumed/checkpoints/cycle-000003 \
  --output /YOUR/W1/exported
```

The export contains the **current full frozen base plus unmerged G LoRA**, not
EMA, Critic, Adam or a training checkpoint. It uses a separate source/runtime-
and layout-bound inference factory. Projection must not change the live
training model's flags, adapters, parameters, RoPE, RNG or next update.

Prepare a genuinely new `prompt_embeds` tensor with the admitted BF16 BTD
geometry using the ordinary preencoded import. Fill the
[inference configuration](inference.cpu-reference.template.json) from the
export's exact factory/parameters and this new prepared manifest, then run:

```sh
distillctl generate /YOUR/W1/inference.json --weights /YOUR/W1/exported \
  --prepared /YOUR/W1/new-inputs/manifest.json --output-kind latents \
  --output /YOUR/W1/new-latents
```

Generation calls the original native `run_generator` with its explicit
zero-history inference conditions and admitted schedule. Only the exported
model, matching source/runtime and new inputs are needed; old training inputs,
bases, checkpoints and optimizer owners are not. Output is an FP32 BCTHW latent
diagnostic: the original zero-history inference branch allocates its history
in FP32. This is distinct from the unchanged mixed BF16/FP32 model inventory;
the output is not coerced to fit a training-rollout dtype. Without a real VAE decoder, default video output refuses rather
than relabeling latents as media. Pretrained quality, world16 parity, CUDA and
performance remain outside this CPU reference's acceptance.
