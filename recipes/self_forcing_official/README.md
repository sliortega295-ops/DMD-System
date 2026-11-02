# Official Self-Forcing (not LiveWan)

Train the actual upstream causal Wan Generator and noncausal Fake Score with
native random-exit rollout, DMD losses, full trainable parameters and cached
block attention. This is **official Self-Forcing**, separate from
[W4 LiveWan](../self_forcing/README.md). Single-device CPU/`cuda:N` uses FP32 MATH
SDPA; explicit world16 FSDP2 profiles use BF16 compute with FP32 masters/reduction
and native FlashAttention-2.

## Source, environment and assets

Use a separate `train,self-forcing-official` environment; see
[installation](../../docs/guides/installation.md). External
[guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) is pinned at
`33593df3e81fa3ec10239271dd2c100facac6de1`, Apache-2.0. The optional helper fetches
the hash-locked source closure and license, not weights:

```sh
python recipes/self_forcing_official/fetch_source.py /PROJECT/dependencies/Self-Forcing
```

Set `model.parameters.source_root` and supply an explicit
`distillschedule.official_self_forcing_assets.v1` manifest. Its `generator`,
`real_score`, `fake_score` and `vae` entries contain actual native constructor
`config` plus `weights: {"path":"...","sha256":"..."}`. Pin the matching
`negative_prompt_embeds` in native `1,L,C` layout. The fixture below is a readable
complete example. Set `BUNDLE` to your asset directory and `RUN` to a separate
project-local run directory. Inventory files separately from native admission:

```sh
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
```

Explicit Hub acquisition uses the shared [pinned prepare/materialize
workflow](../../docs/guides/model_families.md#prepare-model-assets). A downloaded
source snapshot does not construct the native role manifest or `ode_init.pt`
initialization: supply the strict native states/configs and preserve asset terms.
The official role architectures are:

| Role | Native class | Blocks / width |
|---|---|---|
| Generator | `CausalWanModel` | 30 / 1536 (Wan1.3B) |
| Teacher / Real Score | `WanModel` | 40 / 5120 (Wan14B) |
| Fake Score | `WanModel` | 30 / 1536 (Wan1.3B) |

World16 constructors check these shapes on meta tensors before loading. Supply
FP32 role states in default `torch.save` ZIP format; the distributed loader maps
read-only pages, then copies to independent parameters before sharding. The
single-device loader also accepts its existing legacy format.

## Native training contract

Defaults come from the pinned merge of `default_config.yaml` and
`self_forcing_dmd.yaml`: score and Generator scheduler shift 5, `ts_schedule=false`,
G LR `2e-6`, C LR `4e-7`, real CFG3/fake CFG0, G1/C5, EMA .99 starting at step 200,
block size 3 and 21 latent frames. Single-device defaults explicitly use FP32 and
checkpointing off. T2V uses B1, BFCHW/16-channel/21-frame latents and whole frame
blocks. The random tiny fixture changes architecture/spatial size, LRs, EMA
settings and conditioning provenance explicitly, not the rollout/update equations.

- Native `DMD.generator_loss`, detached double-precision target MSE, original
  CFG/normalizer, flow fake-score loss and scheduler.
- Random-exit cached causal training: prefix no-grad/final grad, native
  context-noise draws and KV/RoPE/end-index writes. Critic samples a **new**
  self-rollout using its **own** prompt, not a cached Generator sample.
- One common macrocycle is G1/C5: G at steps divisible by five (including zero),
  C every step. Independent streams consume one `generator` and five `critic`
  prompt records; an explicit ratio change changes the required stream counts.
- Native AdamW ordering and `EMA_FSDP` arithmetic: EMA initializes after the step
  reaches 200, and subsequent G commits update it. No LR scheduler is invented.
- Native FP64 target arithmetic and complex128 RoPE remain intact; single-device
  EMA shadows, noncapturable Adam step metadata and RNG retain host ownership.

## Prepare prompts or native embeddings

[inputs.prompts.template.json](inputs.prompts.template.json) supplies the native
T5 constructor, pinned local weights/tokenizer, text length and raw prompts.
Preparation calls original `WanTextEncoder.forward`, `T5Encoder` and
`HuggingfaceTokenizer` with local files only:

```sh
distillctl data prepare /PROJECT/inputs.json --output /PROJECT/prepared \
  --preparer distillschedule.data.self_forcing_inputs:prepare_inputs
```

The template declares native BF16 UMT5-XXL storage/output widened exactly to FP32:
`"precision":{"weights":"bfloat16","output":"float32"}`. For an FP32 checkpoint,
select `"weights":"float32"`; strict loading requires the declared dtype rather
than recasting the checkpoint. `encode_text` uses the same declaration for the
matching negative prompt. This is a preparation precision choice, separate from
training precision. The complete 5.68B-parameter BF16 encoder has prepared new
raw prompts through this public CPU path.

[inputs.preencoded.template.json](inputs.preencoded.template.json) instead pins
exact native FP32 `B,L,C` embedding files by unique record ID and SHA256.
Training/inference consume this prepared boundary without implicit tokenization,
scaling or conversion. Source, tokenizer/encoder and precision provenance remain
bound to the prepared rows.

## Train, save, resume and export

Fill the source/assets and independent Generator/Critic stream paths in
[training.json](training.json), then select the owned `trainer.device`.
For FSDP2, use [training.fsdp2-reference.json](training.fsdp2-reference.json);
[training.optimized.json](training.optimized.json) enables phase-weight/text-KV
retention, Critic checkpoint-result reuse and native kernel/CFG specializations.
The [shared launch and optimization guide](../../docs/guides/official_optimized_training.md)
contains the exact world16 environment, `flash_attn` requirement, mechanisms
and performance results. Both FSDP2 profiles use B1/global16, 368×640/81 output
frames, G1/C5/EMA0 and `NCCL_CROSS_NIC=1`.

```sh
# Single-device example; retain the world16 launcher for FSDP2 commands.
distillctl train /PROJECT/train.json --cycles 1 --output /PROJECT/partial
distillctl train /PROJECT/train.json --cycles 3 \
  --resume /PROJECT/partial/checkpoints/cycle-000001 --output /PROJECT/resumed
distillctl export /PROJECT/train.json \
  --checkpoint /PROJECT/resumed/checkpoints/cycle-000003 --output /PROJECT/current-G
```

Use returned checkpoint paths. Training resume requires the same device/topology
and complete G/C cycle; ephemeral rollout caches rebuild from persistent state.
The [common lifecycle](../../docs/guides/training.md) covers checkpoint publication.
For a small random actual CausalWanModel/WanModel/WanVAE_ starter:

```sh
python recipes/self_forcing_official/create_cpu_fixture.py /PROJECT/random-native-fixture \
  --source-root /PROJECT/dependencies/Self-Forcing --cycles 3
distillctl train /PROJECT/random-native-fixture/training.json
```

## Explicit checkpoint-ON reference and Fake-only physical policy

`model.parameters.settings.gradient_checkpointing: true` selects the native
Generator/Fake checkpoint-ON reference; the default is `false`. Generator ON/OFF
are different source policies: causal cross-attention-cache arguments and the
gradient path differ, so a plan cannot switch them or mix their checkpoints.

For the exact random tiny 21x4x4/G1C5/block3 ON reference on the qualified one-thread CPU1
FP32 Torch2.10/Diffusers0.37.1/Transformers4.51.3/einops0.8.1 stack, only
**noncausal Fake checkpointing** is a physical option. Set `execution` to:

```json
{"adapter":"program","mode":"manual","implementation":{
  "backend":"torch-callable-resident-v1","share_immutable":false,"checkpoint_bindings":[],
  "owner_execution":{"native_cycle":{"fake_gradient_checkpointing":false}}}}
```

Explicit `true` measures Fake ON against the same Generator-ON reference.
`auto` with `memory_limit_bytes`, `max_candidates:2`, `samples:2`,
`max_program_runs:5` uses the common measured planner; reuse
`planning/selected-plan.json` through `mode:plan`. Native execution counters and
paired reference admission verify the option, including the first resumed update.
Flags/hooks restore after normal exit or errors. [Budgeted resident
execution](../../docs/guides/resident_owner_training.md) defines separate
snapshot/restore admission for its exact tiny CUDA profile.

The independent [consumed-KV template](training.consumed-kv.json) sets
`native_cache_lifetime: "release-after-commit-v1"`. It releases owner rollout-cache
references after completed backward/clip and Adam/EMA commit, not during attention
or checkpoint replay; [policy details](../../docs/guides/native_owner_backends.md#official-self-forcing-release-consumed-rollout-kv)
explain counters and restart. Relinquished reference bytes are not freed-HBM measurements.

## Generate new prompts

Export contains **current Generator + native VAE**, not EMA/Teacher/Fake or
optimizer state. Use the export's exact factory/parameters and a new prepared
prompt manifest with the [common inference-request builder](../../docs/guides/training.md#generate-and-export):

```sh
distillctl generate /PROJECT/inference.json --weights /PROJECT/current-G \
  --prepared /PROJECT/new-prepared/manifest.json --output /PROJECT/generated --output-kind video
```

This runs all exported steps of the original `CausalInferencePipeline` and native
VAE decode, not a random training exit. Old training assets/checkpoints/prompts
are unnecessary; the external source and compatible runtime remain required.
