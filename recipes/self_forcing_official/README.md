# Official Self-Forcing (not LiveWan)

Opt-in external [Self-Forcing](https://github.com/guandeh17/Self-Forcing) at
`33593df3e81fa3ec10239271dd2c100facac6de1`, Apache-2.0. Source and weights are
not vendored. `fetch_source.py OUTPUT` fetches the small hash-locked source
closure and license only. It never downloads model weights or installs anything.

`create_cpu_fixture.py OUTPUT --source-root SOURCE` creates RANDOM tiny assets
of the actual upstream CausalWanModel, WanModel, and WanVAE_ classes and a
common Program configuration. This proves numerical/lifecycle plumbing, not
pretrained quality or CUDA performance. Its explicit randomly drawn native
embedding assets are NOT described as prompt encodings.

Factory defaults are derived from the ORIGINAL pinned merge of
`configs/default_config.yaml` and `configs/self_forcing_dmd.yaml`, not just
constructor fallbacks. The official recipe uses score `timestep_shift=5`,
generator wrapper scheduler shift5, `ts_schedule=false`, G LR `2e-6`, C LR
`4e-7`, real CFG3/fake CFG0, G1/C5, EMA.99 starting at200, block3 and21 latent
frames. Its actual teacher is Wan14B, generator/fake Wan1.3B with explicit
`ode_init.pt` initialization; public assets must supply the matching native
configs and strict local states. This CPU owner does not silently manufacture
those pretrained states or download them. Documented CPU-only exceptions are
FP32 instead of mixed precision, native gradient checkpointing disabled, no
FSDP/global batch64 claim, and explicit local prepared conditioning/weights.
The RANDOM tiny fixture additionally changes architecture/spatial size, both
LRs, EMA start/decay, and negative-conditioning provenance for test speed;
its temporal random-exit/CFG/flow/update equations remain original.

## Native contract and public lifecycle

- Actual `DMD.generator_loss`, detached double-precision target MSE, original
  CFG/normalizer, original flow fake-score loss and scheduler.
- Native random-exit cached causal training, prefix no-grad / final grad,
  context-noise RNG draws and KV/RoPE/end-index writes. Critic samples a NEW
  self-rollout and consumes its OWN prompt record, never a cached G sample.
- Each atomic common Program call is one G1/C5 macrocycle: G at native steps
  divisible by five (including zero), C on every native step. Settings can
  explicitly change the ratio. There are six distinct prompt records per
  default cycle, split into `generator` (one) and `critic` (five) streams.
- Native AdamW ordering and original EMA_FSDP arithmetic. EMA initializes
  AFTER incrementing to `ema_start_step` (default 200); only subsequent G
  commits update an existing EMA. No invented LR scheduler is added.
- Common `distillctl train`, checkpoint/resume, eval, export and generate.
  Export is CURRENT G + native VAE only, not EMA, teacher, fake or optimizers.
  Generation runs every exported step of the actual CausalInferencePipeline,
  not the training random exit. Native VAE decoding produces the video.

The factory references are `distillschedule.training.self_forcing:build_model`
and `distillschedule.training.self_forcing:build_self_forcing` (catalog names
are `self-forcing-wan` / `self-forcing` when registered by the host integration).
The generated JSON uses explicit factories so it is independently runnable.

## Explicit local inputs/assets

The `distillschedule.official_self_forcing_assets.v1` manifest contains four
roles (`generator`, `real_score`, `fake_score`, `vae`), each with the ACTUAL
native class `config` and a `weights` object `{ "path": ..., "sha256": ... }`.
It also requires a pinned `negative_prompt_embeds` file in native `1,L,C`
layout. Weights are loaded strictly from these files, never implicit hub paths.
All roles, source closure, local bridge files and numerical dependency versions
enter restart identity. The fixture is a complete readable example.

`distillschedule.data.self_forcing_inputs.prepare_inputs` accepts a manifest
with `schema_version: distillschedule.official_self_forcing_inputs.v1`:

- `mode: preencoded`: records contain unique `id` and pinned `embedding`
  `{path, sha256}` files. They contain EXACT native FP32 `B,L,C` prompt embeddings;
  no scaling, tokenization, dtype conversion or synthesized conditioning occurs.
- `mode: prompts`: records contain unique `id` and raw `prompt` strings (or
  string batches). The manifest must additionally contain `encoder` with
  `source_root`, actual native `T5Encoder` constructor `config`, pinned local
  `weights`, local `tokenizer` directory, and `text_len`. Preparation invokes
  the ORIGINAL WanTextEncoder.forward, ORIGINAL T5Encoder layers and ORIGINAL
  HuggingfaceTokenizer with `local_files_only=True`. Its only forward port is
  the explicit CPU device property. Use the same `encode_text` API to produce
  and pin the matching negative-prompt embedding for the model asset manifest.
  Optional `tokenizer_use_fast` is forwarded to that ORIGINAL tokenizer API
  (default true, as upstream). The tiny actual SentencePiece/T5 qualification
  uses explicit false, avoiding protobuf conversion of a slow-only tokenizer
  without inventing an encoder or installing another dependency.

Raw prompt text becomes model conditioning during explicit data preparation;
training and exported inference accept the exact prepared embedding boundary.
New prompts can be prepared separately and used by exported inference without
the training dataset or training model assets.

## Qualification boundary

CPU1/FP32, batch1 only; no autocast or distributed layout. Native T2V, minimum 21 latent
frames, BFCHW layout, 16 latent channels; whole native frame blocks. Longer-frame
training, I2V, independent first frame, uncached causal/flex training and native
gradient-checkpointed cache execution are explicitly rejected, not emulated.

CPU compatibility ports are private, source-locked and explicit: constructor
asset injection, true-singleton broadcast, FP32 math SDPA attention (native
layers/cache body unchanged), geometry-derived allocations, and exactly six
training-boundary `.cuda()` lookups changed to `noise.device`. Uncached causal
execution is rejected rather than quietly removing its causal mask. No global
Torch/CUDA/module monkeypatches are installed. GPU and pretrained media quality
remain `NOT_RUN` regardless of tiny CPU test results.

## Unified runnable entry

Use the separate `.[train,self-forcing-official]` environment (add `test` only
for development). The public training entry is the same as every other model:

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python recipes/self_forcing_official/fetch_source.py /PROJECT/dependencies/Self-Forcing
python recipes/self_forcing_official/create_cpu_fixture.py /PROJECT/random-native-fixture \
  --source-root /PROJECT/dependencies/Self-Forcing --cycles 3

distillctl train /PROJECT/random-native-fixture/training.json --cycles 1 \
  --output /PROJECT/partial
distillctl train /PROJECT/random-native-fixture/training.json --cycles 3 \
  --resume /PROJECT/partial/checkpoints/cycle-000001 --output /PROJECT/resumed
distillctl export /PROJECT/random-native-fixture/training.json \
  --checkpoint /PROJECT/resumed/checkpoints/cycle-000003 --output /PROJECT/current-G
```

This helper creates explicitly random original-class assets/embeddings, **not
pretrained quality data**. For real raw prompts, fill
`inputs.prompts.template.json` with the actual local native T5 architecture,
checkpoint SHA and tokenizer directory, then use
`distillctl data prepare ... --preparer distillschedule.data.self_forcing_inputs:prepare_inputs`.
The template's T5 dimensions describe the pinned native UMT5-XXL constructor,
not a tiny substitute; the bounded test builds a separately labeled small T5.
`inputs.preencoded.template.json` supplies exact already-native embeddings.
The static `training.json` selects the actual registered model/workflow and
independent G1/C5 streams; fill its explicit paths before use.

For exported inference, use the exact `model.factory` and `model.parameters`
from `export.json` in the inference request, a newly prepared input manifest
and explicit inference seed. `distillctl generate ... --weights /PROJECT/current-G
--prepared NEW/manifest.json --output-kind video --output NEW/generated` runs
the native full-step causal inference + actual VAE. This is a model-only
projection; old training assets, optimizer or prepared training prompts are
not required. The [common guide](../../docs/guides/training.md#generate-and-export)
explains checkpoint versus inference identity.
