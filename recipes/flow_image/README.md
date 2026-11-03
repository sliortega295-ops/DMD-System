# Native image flow-DMD: shared reference and replicated lifecycle

Train FLUX.2 klein Base or original Qwen-Image using native Transformer, text and
VAE classes, unmerged Student/Fake LoRA and the common No-GAN `dmd` algorithm.
Both families support local B1, FP32 or BF16 autocast, complete-cycle recovery,
current-Student image export, immutable sharing and equal-B1 replication.

| Family | Native components and source |
| --- | --- |
| `flux2-klein-base` | `Flux2Transformer2DModel`, `AutoencoderKLFlux2`, Qwen3 text; [pinned source and conditioning](../flux_dmd/README.md) |
| `qwen-image` | `QwenImageTransformer2DModel`, `AutoencoderKLQwenImage`, Qwen2.5-VL text; [pinned source and conditioning](../qwen_image_dmd/README.md) |

Both model sources are Apache-2.0; preserve model, tokenizer and dependency
notices. Their native prediction/packing/conditioning contracts remain distinct.

## Environment and model files

Use an isolated `train,modern-flow` environment with the exact pins in
[environment.json](environment.json), independently of DMD2/Self-Forcing/Helios.
`PY` denotes its Python; `RUN` is a new project-local or scratch directory.
CPU uses `trainer.device: cpu`; CUDA uses the owned `cuda:N` device and a
compatible Torch 2.10 build. Preparation runs on CPU. Ordinary training and
model discovery use local files and do not download weights.

A local bundle needs native `transformer/`, `vae/`, `text_encoder/` and
`tokenizer/` folders with complete safe-format checkpoints/configs. Inventory
and verify the source **before** creating the recipe manifest:

```sh
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
"$PY" recipes/flow_image/lock_local_assets.py "$BUNDLE" --family flux2-klein-base \
  --cpu-max-parameters "$EXPLICIT_PER_COMPONENT_CPU_BUDGET" --text-max-length 512
```

For Hub files, select the family's full pinned revision and use the [explicit
prepare/materialize workflow](../../docs/guides/model_families.md#prepare-model-assets)
instead of local inventory. Then run the same native locker. For Qwen, select
`--family qwen-image` and an explicit supported text width.
`--student`, `--fake`, `--teacher` can select different subdirectories inside
the bundle; their native architectures must match.

The default `strict_fp32` requires FP32 floating storage. Add
`--dtype-policy bf16_or_fp32_to_fp32` to admit BF16 storage converted to FP32
owned state; this is separate from runtime BF16 autocast. Native loaders check
safe shards, tied-parameter ownership, keys, shapes and finite values while
retaining original config bytes. Parameter budgets bound constructors, not total
RAM for all roles, optimizer/EMA and loading temporaries.

## Prepare native inputs

Both family templates use finite sampling by default: one record per complete
single-device cycle. Before preparation, supply at least three distinct records
for the three-cycle lifecycle below; the raw manifest shows just one record.
To intentionally reuse records instead, set `data.sampling` to
`{"kind":"epochs","shuffle":false}` in the training config **before the first run**
and keep it unchanged on resume.

```sh
distillctl data prepare "$RUN/inputs.json" --output "$RUN/prepared" \
  --preparer distillschedule.data.flow_image_inputs:prepare_inputs
```

For example, raw RGB input uses:

```json
{
  "schema_version":"distillschedule.flow_image_inputs.v1",
  "family":"flux2-klein-base", "assets":"/absolute/bundle/assets.json", "mode":"raw",
  "records":[{"id":"sample-0","prompt":"a red bird","image":{
    "path":"input.png","sha256":"REPLACE_WITH_ACTUAL_IMAGE_SHA256"}}]
}
```

Images retain exact RGB geometry divisible by native VAE scale times 2. Encoding
uses posterior mode and native normalization/packing, without implicit resizing,
color conversion or padding. In `mode: prompts`, replace `image` with
`latent_shape: [1,C,H,W]` in unpatched VAE BCHW units; it defines noise geometry.
In `mode: preencoded`, records contain `id` and
`features: {"path":"row.pt","sha256":"..."}` with complete tensor-only rows.
All modes bind source, codec, geometry, dtype, masks and genuine encoded
empty-prompt conditions. FLUX retains native fixed-length Qwen3 features;
Qwen uses crop34 and fixed right padding with the native boolean mask.
Overlength text is not silently truncated.

## Training policy and single-device execution

Copy the family's [FLUX template](../flux_dmd/training.template.json) or
[Qwen template](../qwen_image_dmd/training.template.json), fill asset/data/output
paths and choose existing native Linear LoRA targets. LoRA keeps frozen bases
and separate trainable A/B; full-parameter image training is outside this recipe.
Identical initial bases with zero LoRA B can produce zero first-G surrogate
before Fake diverges.

The configured No-GAN policy is `z=(1-sigma)*x0+sigma*epsilon`,
`v=epsilon-x0`, `x0=z-sigma*v`, with no output negation. Four Euler steps
`[1,.75,.5,.25,0]`, G1/F5, teacher/fake CFG=1, AdamW (`betas=.9,.999`,
`eps=1e-8`, no weight decay), per-actual-update exponential LR clocks and
optional EMA are explicit in the recipe. This is common flow DMD, not official
DMD2/Self-Forcing or the model vendors' distillation algorithms.

`model.parameters.compute_precision` selects `fp32` (default) or `bf16-mixed`.
Native Transformer autocast preserves model normalization/rotary precision;
master weights, Adam/EMA, flow/sampling math and text/VAE preparation remain
FP32. Checkpoints and inference exports retain the setting.

Reference mode needs no optimization. For exact frozen-base sharing, replace
`execution` with:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

Equal frozen Teacher/media/LoRA bases may alias; trainables remain independent.
All roles initially materialize. This direct policy also works on single-device
CUDA without trial clones; [capacity and policy details](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing)
cover supported selection. CPU fixtures additionally support measured
planning; CUDA uses reference or direct sharing.

## Explicit replicated training, not model sharding

Each rank owns complete Student/Fake/Teacher models and consumes one distinct
record per complete cycle (local B1). The global batch is
`world_size = NNODES * GPUS_PER_NODE` records per cycle, not global B1.
Gradients are averaged **before** clipping and one Adam commit.
Rank/world-specific noise leaves the single-rank seed stream unchanged. Copy
a reference recipe, retain its model/algorithm, and set:

```json
{
  "data":{"prepared":"/absolute/prepared/manifest.json","sharding":"rank_strided"},
  "execution":{"mode":"reference"},
  "trainer":{"cycles":3,"seed":17,"device":"cuda",
    "distributed":{"backend":"nccl","timeout_seconds":180},"output_dir":"/absolute/run"}
}
```

For finite sampling, prepare at least `world_size * cycles` distinct records;
incomplete cohorts are not padded or duplicated. `cuda` resolves to each
process's local rank. Choose the same positive `NNODES` and `GPUS_PER_NODE`
counts on all nodes for your allocated, compatible NVIDIA GPUs; each selected
GPU must fit a complete replica. On each node, set `NODE_RANK` from `0` through
`NNODES-1`, with shared files and the same rendezvous address/port:

```sh
"$PY" -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --module distillschedule train "$RUN/train.json" --cycles 1 --output "$RUN/partial"
```

Repeat the same launcher for resumed training and collective export below.
They require the same topology and all checkpoint shards; portable generation
uses a separate single-device inference owner. CPU/Gloo uses `device: cpu`,
`backend: gloo`. To share immutable state inside each replica, use:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-replicated-v1","reduction":"post_backward","bucket_cap_mb":0,"local":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}}
```

This is post-backward replication, not DDP-forward overlap or model sharding.
The same pattern applies to the [Wan](../wan22_dmd/README.md) and
[Hunyuan](../hunyuan_video_dmd/README.md) local-B1 recipes.

## Train, resume, export and generate

```sh
distillctl train "$RUN/train.json" --cycles 1 --output "$RUN/partial"
distillctl train "$RUN/train.json" --cycles 3 \
  --resume "$RUN/partial/checkpoints/cycle-000001" --output "$RUN/resumed"
distillctl export "$RUN/train.json" \
  --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/export"
```

Use the exact returned checkpoint paths. The export contains current unmerged
Student and native VAE/text/tokenizer codecs. Prepare new prompts/raw RGB with
the [export-only preparer](../../docs/guides/data.md#export_inputs), then create an
inference request using the [common builder](../../docs/guides/training.md#generate-and-export).
It uses the export's factory/parameters and independent inference seed, not the
training assets or factory:

```sh
distillctl generate "$RUN/inference.json" --weights "$RUN/export" \
  --prepared "$RUN/new-prepared/manifest.json" --output "$RUN/generated" --output-kind image
```

Output is native VAE-decoded RGB PNG; `--output-kind latents` selects explicit
native normalized latent diagnostics. To exercise the lifecycle with small
random actual-class assets instead of pretrained files:

```sh
"$PY" recipes/flow_image/create_cpu_fixture.py "$RUN/fixture" \
  --family flux2-klein-base --cycles 3
# Alternative family: qwen-image. The helper writes fixture/train.json.
distillctl train "$RUN/fixture/train.json"
```
