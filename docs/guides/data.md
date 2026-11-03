# Offline media and native feature data

Preparation uses actual local media and an explicit trusted native encoder,
never downloads models, invents conditions or converts another family's latents.
Use `train`, or `media` beside compatible Torch. Pillow handles frames;
video containers require existing ffmpeg/ffprobe. Both probe and decode enforce
the same self-contained MOV/MP4/Matroska/WebM/AVI/MPEG/TS/FLV/Ogg/GIF allowlist,
not HLS/concat playlists or external segments.

```sh
distillctl data prepare raw/manifest.json --output prepared
distillctl data inspect prepared/manifest.json
distillctl data validate prepared/manifest.json
distillctl data validate prepared/manifest.json --sources
```

`inspect` checks manifest/file hashes without Torch/encoder/tensor deserialization
and reports `HASHES_ONLY_TENSORS_NOT_LOADED`. `validate` safe-loads every feature
with CPU `weights_only=True`, checking tensor shape/dtype/tree; no unsafe pickle
fallback. `--sources` additionally checks original media/documents/encoder code.
Without it, prepared tensors survive original-media relocation/archive.

## Raw manifest

JSON/YAML supports deterministic includes and `${ENV:NAME}`. All local paths,
even included values, resolve from the **top-level** manifest directory.

```json
{
  "schema_version": "distillschedule.data.raw.v1",
  "encoder": {
    "factory": "my_model_encoder.py:encode",
    "parameters": {
      "checkpoint": "/persistent/models/native-vae",
      "checkpoint_sha256": "supply-your-own-checkpoint-identity",
      "native_format": "my-model-native-v1"
    }
  },
  "metadata": {"native_format": "my-model-native-v1"},
  "records": [
    {
      "id": "clip-0001",
      "video": "clips/0001.mp4",
      "audio": "audio/0001.wav",
      "text": {"path": "captions/0001.txt"},
      "metadata": {"split": "train"}
    },
    {
      "id": "clip-0002",
      "video": {"path": "frames/0002"},
      "text": {"value": "An explicitly supplied caption."}
    }
  ]
}
```

IDs are unique/nonempty; each record has image/video/audio/text. Metadata is
finite JSON. Unknown fields and URLs reject. Media references are path strings
or exactly `{path}`; text is explicitly `{path}` or `{value}`, never guessed.
The decoder supplies CPU RGB uint8 HWC/THWC, float32 audio
`{samples:[samples,channels],sample_rate:int}`, exact UTF-8 text and original
ID/metadata. Generic Pillow RGB conversion does not resize; native preparers
may require original RGB instead. Text-only records need no fabricated media.

Frame directories are nonrecursive sorted regular `.png/.jpg/.jpeg/.webp/.bmp/.ppm`
files with equal dimensions; symlinks reject. File video uses `v:0`, no
resampling/autorotation. Audio is uncompressed8/16/24/32-bit PCM WAV normalized
to float32, preserving rate/channels. No implicit trim/pad/resize/recycle or
alternate audio decoder. Direct decoding is available:

```python
from distillschedule.data.decoding import decode_record

record = decode_record(
    {"id": "clip-0001", "video": "clips/0001.mp4", "text": {"path": "captions/0001.txt"}},
    base_dir="/persistent/dataset/raw",
)
```

## Model-specific encoder contract

Factories are trusted `module:callable` or relative `.py:callable` plugins.
Supply identity-checked local assets; a missing encoder/asset is an error.

```python
def encode(record: dict, parameters: dict) -> dict:
    # Load only explicit local, identity-checked native assets. Run your actual
    # VAE/text/audio encoders and any model-required preprocessing here.
    return {
        "video_latents": native_video_encoder(record["video"]),
        "conditioning": native_text_encoder(record["text"]),
    }
```

Return a nonempty string-keyed tensor-tree dict (nested dict/list/tuple);
top-level keys become direct feeds, no extra wrapper. Dense nonquantized real
tensors detach to CPU without cast/pad/reshape. Batch, scale, masks and keys
remain native responsibilities; empty containers/opaque Python objects reject.
Finite-JSON parameters copy per call. Provenance binds entrypoint source and
canonical parameter SHA, **not** arbitrary transitive dependencies/assets:
the plugin must pin/check those itself. Plugins execute user code, not a sandbox.
Raw/includes/media/frame/text/inline-value identities and entrypoint bytes are
rechecked before publication; source drift fails.

## Import existing native features

Use genuine native tensor trees, with explicit file SHA and optional exact
recursive `features.tensors` assertion. Import safe-loads and republishes original
shapes/dtypes; it records `preencoded`, never an encoder-execution claim.

```json
{
  "schema_version": "distillschedule.data.preencoded.v1",
  "metadata": {"native_format": "my-model-native-v1"},
  "records": [
    {
      "id": "clip-0001",
      "features": {
        "path": "features/0001.pt",
        "sha256": "replace-with-the-64-character-file-sha256"
      }
    }
  ]
}
```

```sh
distillctl data import preencoded.json --output prepared-native
```

## Prepared data and traversal

`distillschedule.data.prepared.v1` has status `complete`, ordered records,
metadata/provenance, dataset-relative feature paths/SHA and recursive schemas:

```json
{
  "kind": "dict",
  "items": {
    "video_latents": {"kind": "tensor", "shape": [1, 4, 2, 4, 4], "dtype": "torch.float32"}
  }
}
```

Lists/tuples use `{kind:list|tuple,items:[...]}`. Scalar `[]` and zero-sized
dimensions are valid; negative/boolean/noninteger dimensions, escaped paths,
duplicate IDs, malformed hashes/metadata, missing files and drift reject.

```python
from distillschedule.data import PreparedDataset, prepare

manifest = prepare("/persistent/dataset/raw/manifest.json", "/persistent/dataset/prepared")
dataset = PreparedDataset(manifest)
for index in range(len(dataset)):
    sample_id = dataset.sample_id(index)
    feeds = dataset.sample(index)  # fresh CPU tensor tree; no recycling
identity = dataset.identity
```

Indices are integer0..len-1; negative/boolean/exhausted access rejects. Order is
manifest order; identity binds exact manifest and actual file bytes, rechecked
on access. Epoch/shuffle/batch/rank ownership are explicit trainer policies,
not dataset reuse: see [Program sampling](training.md#explicit-program-data-sampling).
Evaluation/generation indexes still name physical records; native inputs do not
accept that generic policy.

Publication uses exclusive sibling lock/staging, validates every tensor and the
complete manifest, then atomically renames without overwriting. Ordinary errors
remove staging/lock. A killed publisher may leave `.OUTPUT.*`/`.OUTPUT.lock`:
establish that it is dead before cleanup; staging is never a valid dataset.
Hashes/structure are not model compatibility, VAE parity or training quality.

## Original native input preparation

| Workload | Existing producer and original ownership |
| --- | --- |
| W1 | `workloads native prepare-stream`: original rank features/cohort seeds/capture; [Wan stream](../workloads/original_wan_fresh_training.md) |
| W2 | `data prepare-helios-stream --sources ... --training-config ... --output ...`: five steps/world16 cohort, original dataset/history/noise; [reference inputs](../../recipes/helios_nogan/REFERENCE_TRAINING.md) |
| W3 CPU | `data prepare-helios-gan-reference`: original history19, five primary/secondary rows per G1/C5; [recipe](../../recipes/helios_gan/REFERENCE_TRAINING_CPU.md) |
| W4 | [Original TeacherLatents](#livewan_inputs): native InputLoader and80 records/cycle; reference and full C51 use separately versioned lifecycle/input contracts |
| W5 | `data prepare-h3-grid --native-config ... --output ...`: inventory of genuine encoded JSONL; [original Grid](../../recipes/h3-grid-original-training.md) |
| W6 CPU | `data prepare-h3-cached-reference`: original flat/split-packed AV parser, tensor-only equivalence and streamed source loading; [recipe](../../recipes/h3_ar/CPU_REFERENCE.md) |

Distributed native manifests are not generic PreparedDataset. W3/W6 CPU factories
use that common container **with native content/provenance**, not replacement
math. Preparation invents no DMD noise/timestep/rollout. Portable original Helios
G accepts imported new `prompt_embeds`, not an unused five-row training cycle
or a fake text encoder. Missing/unsupported native input rejects, never resamples.

## Images, prompt-only data and model-native preparers

```sh
distillctl data prepare inputs.json --output prepared \
  --preparer distillschedule.data.sd_inputs:prepare_inputs
# Alternatives: distillschedule.data.self_forcing_inputs:prepare_inputs
# or distillschedule.data.ltx_inputs:prepare_inputs
```

SD uses actual CLIP/local VAE; official Self-Forcing actual T5/tokenizer; LTX
actual Gemma/AV VAE/token transforms, not the tiny statistics encoder.
SD schema-v2 binds variant/relative tokenizer hashes to every row and model,
plus actual VAE config/file hashes for image latents. Same-shaped incompatible
bundles reject before train/evaluate/export-generate. Byte-identical relocated
assets stay compatible; portable generation does not reopen old base/preparer
paths. Preencoded SD latents require explicit caller assertion
`preencoded_vae_compatibility:PREENCODED_LATENTS_MATCH_LOCAL_VAE`, not a claim
that encoding ran.

`data.streams` declares independent prepared manifests/`samples_per_cycle` and
optional epoch policy. Each saves its own sampler/position and independent
sampling seed without consuming training RNG. DMD2 G/guidance/denoising/real-image
and official Self-Forcing G/C must not become one shared advancing iterator.
Use [model recipes](model_families.md) for exact counts. For modern-flow and
LTX export-native encoding, use the export-only section below.

<a id="export_inputs"></a>
<a id="export_inputs--adapter-extension-contract"></a>
## New inputs from portable weights, without training assets

`data prepare --preparer` can construct new inputs from a trusted complete
portable export, restoring its real encoders/tokenizers/codecs without training
assets. Supported: FLUX.2 klein/Qwen-Image BCHW, Wan2.2 TI2V-5B **T2V** and
HunyuanVideo-1.5 BCFHW, and LTX-2.5's separate AV `geometry` contract.
Preparation is CPU inference-only; native runtime/precision/geometry still bind.

```json
{
  "schema_version": "distillschedule.export_inputs.v1",
  "weights": "export",
  "mode": "prompts",
  "records": [
    {"id": "new-001", "prompt": "a violet kite", "latent_shape": [1, 4, 4, 4]}
  ]
}
```

```sh
distillctl data prepare new-inputs.json --output new-prepared \
  --preparer distillschedule.data.export_inputs:prepare_inputs
distillctl data validate new-prepared/manifest.json --sources
```

```sh
distillctl generate inference.json --weights export \
  --prepared new-prepared/manifest.json --output generated --output-kind image
# For Wan/Hunyuan, use --output-kind video.
# For LTX-2.5, use --output-kind audio_video.
```

```text
encode_input(state, record, {mode, source_dir, model_metadata})
    -> {features: native CPU tensor tree,
        metadata: nonempty finite JSON object, sources: raw source identities}
```

The JSON shape `[1,4,4,4]` is only the tiny image fixture; tiny Wan and Hunyuan
examples are `[1,4,2,2,2]` and `[1,4,2,4,4]`. Supply the exported model's actual
geometry. Zero prompt-mode latents are explicit noise templates; text must be
explicit, including empty strings. For `mode: raw`, replace `latent_shape` with
exactly one `image: {path,sha256}` or `video: {path,sha256}`. Local paths resolve
against the manifest. Frame hashes use `data.manifest.source_identity(path,
frames=True)`, not directory metadata. Require raw RGB and native compression
divisibility: no resize, crop, resampling, color conversion or truncation.
Pillow handles frames; video containers require both ffmpeg and ffprobe.

Export inventory/completion, factory/runtime/metadata/layout/safe tensors and
callback are admitted before streaming atomic non-overwriting publication.
Wrong shapes/conditions, duplicate IDs, changed media/weights, absent capability
or inconsistent metadata refuse; staging/lock are removed on failure. Python,
NumPy and Torch CPU RNG are restored. Source provenance names the export,
encoder, prompts and raw inventories, not a removed training bundle. After
relocation prepare again; `--sources` needs the recorded paths, ordinary loading
only needs prepared hashes. Explicit `.py` factories must remain available.

The optional `ModelComponents.encode_input` callback above owns exact fields,
native tokenization/codec, features, metadata and real source identities.
Metadata must be nonempty finite JSON and agree across rows; `_export_inputs`
is reserved. It may not mutate state/inputs/files; arbitrary trusted-plugin
external effects cannot be rolled back. New families add capabilities/proofs,
not model-name branches in the preparer. Generate with the exact exported
factory/parameters and a separate [inference request](training.md#generate-and-export).
Use `image`, `video`, or `audio_video` as appropriate: PNG, PNG frames/THWC uint8,
or LTX RGB/PCM WAV, not an implicit MP4 encoder.

<a id="helios_feature_epochs"></a>
## Train Helios from a finite feature pool

For W2/W3, `distillschedule.helios_record_epochs.v1` reuses an immutable pool
of at least 80 distinct native `vae_latent`, `prompt_embed`, `attention_mask`
records. Each complete cycle still has 80 different records; later cycles have
new logical draw identities and original operation-qualified history/noise.
Copies, aliases or reserialized equal content do not enlarge the pool.

```json
{
  "schema_version": "distillschedule.helios_record_pool.v1",
  "seed": 42,
  "cycles": 100,
  "evidence_kind": "ENCODED_FEATURES_UNVERIFIED_QUALITY",
  "records": [
    {"id": "clip-000", "path": "/absolute/data/clip-000.pt", "sha256": "<file SHA256>"}
  ]
}
```

```sh
distillctl data prepare-helios-stream \
  --input-policy distillschedule.helios_record_epochs.v1 \
  --sources /absolute/project/pool.json \
  --training-config /absolute/project/inputs.reference.yaml \
  --output /absolute/project/prepared
```

Supply the full inventory, not the illustrative single row. Use W2's
[inputs settings](../../recipes/helios_nogan/inputs.reference.yaml); W3 substitutes
`prepare-helios-gan-stream` and its unchanged [GAN config](../../recipes/helios_gan/NATIVE_STREAM.md).
Without `--input-policy` the original globally disjoint single-pass rule applies.
Set `data.prepared` to `stream.json` and the training target to its exact horizon.
For a 50/100 boundary use checkpoint/stop at50, then a new process/output with the
same source/assets/topology/stream and no stop setting. Never relabel a changed
order as the old checkpoint. [Exact draw/cursor contract](../contracts/helios_native.md#original_helios_record_epochs_v1).

<a id="livewan_inputs"></a>
## Original LiveWan finite fresh inputs

Use a clean original LiveWan checkout at
`1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`; preparation checks committed
dataset/generator/prompt/license bytes and imports the original dataset in an
isolated namespace. No download or global path rewrite. Legacy cohorts require
exactly80×N genuine archives for1..64 cohorts, with the prompt bundle **outside**
the archive directory. Restricted tensor loading precedes native dataset reads,
with no unsafe fallback. Archives contain only `latents`, `prompt_idx`, `seed`,
`size`, `frames`, `steps`, `cfg`: finite FP16 `[16,21,46,80]`,81frames,
`640*368`,40 Teacher steps, integer seed, positive finite CFG. Prompts have
FP16 `pos[P,512,4096]`, `neg[1,512,4096]`, nonempty text and negative prompt.
A shared prompt index does not make equal latent records distinct.

```bash
distillctl data prepare-livewan-inputs \
  --data-root /ABSOLUTE/PATH/teacher --prompts /ABSOLUTE/PATH/prompts.pt \
  --source-root /ABSOLUTE/PATH/qualified-livewan \
  --cycles 2 --seed 20260830 \
  --output /ABSOLUTE/PATH/prepared/livewan-inputs.json
```

```python
from distillschedule.data.livewan_inputs import prepare_inputs

prepare_inputs(
    "/ABSOLUTE/PATH/teacher", "/ABSOLUTE/PATH/prompts.pt",
    "/ABSOLUTE/PATH/prepared/livewan-inputs.json",
    source_root="/ABSOLUTE/PATH/qualified-livewan",
    cycles=2, seed=20260830,
)
```

```python
import torch
from distillschedule.runtime.livewan_crossscale_worker import InputLoader
from distillschedule.training.livewan_inputs import KIND, NativeTeacherCursor

loader = InputLoader(torch=torch, input_kind=KIND,
                     prepared="/ABSOLUTE/PATH/prepared/livewan-inputs.json")
cursor = NativeTeacherCursor(loader, rank=0, world_size=16)
for _ in range(5):
    row, (latent, context, prompt_idx) = next(cursor)
data_state = cursor.state_dict()  # complete logical G1/C5 data boundary only

restored = NativeTeacherCursor(loader, rank=0, world_size=16)
restored.load_state_dict(data_state)
assert restored.peek() == cursor.peek()
```

```bash
distillctl data prepare-livewan-c51-inputs \
  --data-root /ABSOLUTE/PATH/teacher --prompts /ABSOLUTE/PATH/prompts.pt \
  --source-root /ABSOLUTE/PATH/qualified-livewan \
  --cycles 100 --seed 31415 \
  --input-policy distillschedule.livewan_teacher_epochs.v1 \
  --output /ABSOLUTE/PATH/prepared/livewan-epochs.json
```

Publication is exclusive, atomic and fsynced. Paths/archive/semantic tensor
hashes reject aliases, copies and reserialized equal latents; no repair/resize.
The real `InputLoader` calls original `TeacherLatents.__getitem__`, checking
canonical files on every read. Keep admitted bytes immutable; this is not a
sandbox against concurrent writers. Fixtures explicitly use
`CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED`, never inferred pretrained provenance.

Each logical rank consumes five B1 slots; world16 owns80 distinct samples per
G1/C5/EMA0 cycle at368×640×81, continuous shifted-uniform sigma. The cursor
advances after successful native consumption, saves only complete five-step
boundaries, binds source/input/prompt/order/seed/rank/topology and stops at
exhaustion. Operation-keyed streams/seek do not consume global training RNG and
are not upstream sequential-RNG continuation. Cursor-only save does **not** drain
C51 transport or reconstruct its shadows. `model_bundle_state` preserves actual
resident G/C Parameter/Adam/hooks but is likewise only a state view. The separate
[reference](../../recipes/self_forcing/REFERENCE_TRAINING.md) and
[full C51 lifecycle](../../recipes/self_forcing/C51_CONTINUATION.md) own training.

C51 epochs explicitly select `distillschedule.livewan_teacher_epochs.v1` with
`prepare-livewan-c51-inputs`, retaining `original_livewan_c51_teacher_stream.v1`
and `self_forcing.c51_stream.v1`. The Python equivalent is
`prepare_c51_inputs(...,cycles=100,seed=31415,input_policy=EPOCH_POLICY)` from
`distillschedule.data.livewan_inputs`. At least80 genuine records can supply100
cycles: one seeded permutation repeats, each cycle still has80 unique records,
and draw/epoch/cohort IDs and six operation/control streams never reset. Receipts
disclose reuse; unchanged policy/horizon/order/seed are checkpoint identity.
C51 drains before next-cycle precompute and preserves dual-wave/optimizer/RNG
state; input admission alone is not GPU/restart/quality proof. See the
[input contracts](../contracts/livewan_native.md) and [current-G output](training.md#livewan_native_output).

<a id="wan_weights_initialization"></a>
## W1 Generator weights initialization (not resume)

`distillctl import-weights initialization.json --output portable` performs zero
training cycles and publishes the standard artifact plus `portable.inference.json`.
It does not enter a native worker or load optimizer/EMA/Teacher/RNG/cursor/state.
`checkpoint_cycle:0` is a new initialization boundary, not50 old cycles relabelled
as current-source continuation; normal resume/export retains complete-state gates.

```
distillctl generate portable.inference.json --weights portable \
  --prepared NEW/prepared/manifest.json --output-kind video --output NEW-output
```

The request has exactly these fields; each PIN is
`{path:ABSOLUTE_CANONICAL_LOCAL_FILE,bytes:POSITIVE_INT,sha256:LOWERCASE_64_HEX}`:

| Field | Required value/ownership |
| --- | --- |
| `schema_version`, `kind` | `distillschedule.wan_generator_weights_initialization.v1`, `GENERATOR_WEIGHTS_INITIALIZATION_ONLY` |
| `source` | Exact `wan_generator_import.source_identity()` importer/current inference-factory hashes |
| `parameters` | Complete existing `wan_native_inference.v2` constructor/PEFT targets/exclusions/rank, BF16-base/FP32-unmerged-LoRA, task/rollout/fresh-noise/runtime/source/codec/capacity; no device override, CPU/single owner |
| `base` | `{config:PIN,weights:[PIN,...],indexes:[PIN,...],pyramid:POLICY,key_policy:KEY_POLICY}`;≤32 safe shards/one index, exact receipt `identity.assets.model_files` basename→SHA inventory. Original defaults + `TRANSFORMER_ADDITIONAL_KWARGS` must resolve exactly to `parameters.architecture`; unknown config fields reject |
| `trainables` | `{payload:PIN,parameter_names:[EXACT_ORDERED_NAMES,...]}` from native rank0 `state_evidence.generator` final-trainables safetensors only; every live trainable must match name/order/shape/dense FP32/finite before any copy |
| `codec` | Native `{config:PIN,weights:[PIN,...],provenance:{status,origin,license}}`; real strict AutoencoderKLWan precision/full-state/normalization |
| `origin` | `{contract:distillschedule.original_wan_rank0_final_trainables.v1,receipt:PIN,source_head:ORIGINAL_40_HEX,source_fingerprint:ORIGINAL_64_HEX}`; original production task, finite contiguous cycle/version receipts, writer0/count/path/bytes/SHA and frozen-base inventory must agree |

Trusted producer-format receipts are integrity pins, not an authenticity or
private-commit allowlist. Requests/config/receipts/files are bounded and rehashed;
symlinks/duplicate JSON keys/nonfinite JSON/extra or missing fields reject.
Base is exact bare original model state, finite BF16/FP32 converted only by
original mixed precision with finite results. No partial load, suffix mapping,
trainable key rewrite, optimizer/checkpoint-envelope parser or EMA selection.

`KEY_POLICY` is `{kind:EXACT_NATIVE_STATE_KEYS}` or
`{kind:ORIGINAL_WAN_FROM_PRETRAINED_KEY_REMAP_V1,source_sha256:ORIGINAL_TRANSFORMER_SHA}`.
The latter reproduces only original `scale_shift_table`→`norm_out.scale_shift_table`,
requires the raw alias, rejects collisions and never remaps FP32 LoRA. Raw shard
index ownership is checked before this policy.

`POLICY` is `{kind:COMPLETE_SAFE_BASE}` or
`{kind:SOURCE_SEEDED_WAN_MISSING_PATCHES,seed:20260819}`. Only the six original
patch_short/mid/long weight/bias keys may be absent. Original deterministic
materialization preserves FP32-meta construction: loaded keys follow mixed
dtype, missing patches reset in FP32 then cast BF16. Original nonpersistent
RoPE construction is reused; no other random base fallback.

Old source/receipt/version counters stay in `identity.weights_initialization`;
current factory/constructor/dependencies stay in model metadata. Original
comparison/quality strings copy verbatim to `origin_numerical_comparison` and
`origin_quality_validation`; new artifact quality remains NOT_RUN.
CPU/native-class tests cover strict restore/new-input/original rollout/real-VAE
wiring, not trained fixture quality, full-size fit or GPU acceptance. Required
upstream tests use `R217_REQUIRE_UPSTREAM=1` in an existing environment and
refuse missing dependencies/skips; no install/download or new training is needed.

<a id="livewan_weights_initialization"></a>
## Import an existing Self-Forcing Student

Import an existing assembled **complete FP32** original LiveWan Student,
without repeating training. A full framework checkpoint instead uses
[native export](training.md#livewan_native_output). Both emit portable weights/manifest/
`_SUCCESS` and companion config, not resumable state.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 distillctl import-weights \
  /PROJECT/initialization.json --output /PROJECT/current-student

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 distillctl generate \
  /PROJECT/current-student.inference.json --weights /PROJECT/current-student \
  --prepared /PROJECT/new-inputs/manifest.json --output-kind video \
  --output /PROJECT/new-video
```

| Exact request field | Contract |
| --- | --- |
| `schema_version`, `kind` | `distillschedule.livewan_generator_weights_initialization.v1`, `GENERATOR_WEIGHTS_INITIALIZATION_ONLY` |
| `source`, `parameters` | `livewan_generator_import.source_identity()`; complete `livewan_native_current_generator.v1` inference architecture/geometry/runtime/codec/capacity, no device override |
| `student`, `codec` | `{payload:PIN}`; `{config:PIN,weights:[PIN,...],provenance:{status,origin,license}}` |
| `origin` | `{contract,assembly,selected_summary,reference_summary,numerical_comparison,model_source}`; contract `distillschedule.original_livewan_assembled_student.v1`, other fields pin original receipts and `wan/modules/model.py` |

PIN is `{path:/absolute/canonical/file,bytes:N,sha256:64_lowercase_hex}`. Supply
real local codec pins/license using [the output template](../../recipes/self_forcing/export.native.template.json).
Original payload keys are exactly `model`, `step`, `generator_updates`,
`source_head`, `input_binding`, `scope`; every original full parameter name must
have contiguous finite FP32 shape. No base fill, replica averaging, EMA, merge
or guessed prefix mapping. Source/payload checks precede non-overwriting atomic
publication; safe readers and explicit full-model/codec/KV/attention/copy budgets
remain binding.

Provenance/counters/comparisons remain in `identity.weights_initialization`;
new cycle0 is not old-source training resume. Optimizer/scheduler/EMA/Teacher/
Critic/training RNG/data are not loaded, and caller RNG is preserved. Missing
nonpersistent `freqs` is reconstructed by the exact original `rope_params(1024,...)`
expression, recorded and exported for strict later restoration.

New inputs are exactly detached finite CPU FP32 `context[1,512,4096]` and clean
`world[1,16,N,46,80]`, N=3/6/9. Original four-step FewStepStreamer generates
remaining blocks and real Wan2.1 VAE decodes81RGB frames at368×640, without
implicit longer-history crop/training-row reuse. `--output-kind latents` is
explicit latent-only output, not decoded media.

<a id="h3_audio_video_encode"></a>
## Prepare original H3 previous/current audio-video data

Supply actual RGB frame directories, stereo PCM WAV, explicit previous chunk
and original text features. Existing preparation calls original H3 encoders and
cached parser, not a second DMD trainer:

```sh
distillctl data prepare /project/h3/raw-inputs.json \
  --preparer distillschedule.data.h3_raw_inputs:prepare_inputs \
  --output /project/h3/prepared-raw
```

```text
current / previous:
  video: {path: frame_directory, sha256: frame_inventory_sha256}
  audio: {path: absolute_wav_path, bytes: actual_bytes, sha256: actual_sha256}
  fps: 24
prompt: {path: absolute_safetensors_path, bytes: actual_bytes, sha256: actual_sha256}
```

```text
prompt_embeds:    FP32 [text_tokens, 5120]
text_token_tags: int64 [text_tokens], all 1
```

Use `prepared-raw/prepared/manifest.json` with [CPU reference](../../recipes/h3_ar/CPU_REFERENCE.md)
and its observed asset geometry. Native ZeRO-3 may reuse admitted `.pt` files
under `native-caches/` but retains its own rank/source/shape gates.

Pin source/terms and local safetensors in
[encoder template](../../recipes/h3_ar/encoder.native.template.json), using the
same original source closure/`h3-av-codec` lane as [decoding](training.md#h3_audio_video_decode).
No Student/optimizer/download/CUDA context. Complete or exact encoder-only state
is admitted via meta headers; materialize only video `encoder`/`quant_conv` and
audio `encoder`/`pre_block`/`mean_proj`, not decoders/variance projection. Original
production VAE architecture is fixed; small original classes are marked fixtures.
Storage defaults FP32; original BF16 requires explicit `bfloat16_to_float32`.
Offline compute is CPU FP32 with explicit parameter/working-set/attention/reserve
budgets.

The [raw template](../../recipes/h3_ar/raw.native.template.json) requires unique
IDs, both chunks and safe original text. Video paths resolve from the manifest;
WAV/text/encoder pins are canonical absolute paths. Frames sort by filename and
use `data.manifest.source_identity(path,frames=True)` inventory hashing. Text is
exact FP32 `prompt_embeds[text_tokens,5120]`, int64 all-one
`text_token_tags[text_tokens]`, genuine matching encoder output, not generated
by the AV VAE. Bucket differently encoded geometries explicitly.

| Native input/operation | Exact behavior |
| --- | --- |
| Video |24FPS,17*n+5 RGB frames,H/W divisible32; no resize/interpolation/removal/spatial pad. Extract container frames without resampling |
| Audio |Stereo32kHz PCM WAV; no average/trim/gain/resample |
| Input conversion |THWC uint8→BCTHW FP32[0,1]; waveform TC FP32[-1,1]. Original methods own normalization |
| `encode_video` |Repeat final frames to17 multiple internally, discard original3tail tokens, posterior mean/no sampling |
| `encode_audio` |Right-pad to800sample multiple, posterior mean/no sampling; record both native pad counts |

Current/previous may differ in duration. The adapter preserves actual lengths,
not AV synchronization/cropping: supply aligned recordings yourself.

| Output | Meaning |
| --- | --- |
| `encoded/manifest.json` |Original codec outputs and raw/text/source/weight pins |
| `native-caches/*.pt`, `original-cached-inputs.json` |Safe original-parser tensors/manifest |
| `prepared/manifest.json` |Published original-parser dataset |
| `encoding.json`, `_SUCCESS` |Observed shapes/normalization/pad/provenance and complete outer publication |

Outputs never overwrite. Missing prompt/history/shards/wrong geometry fail;
intermediates stay diagnostic and no outer `_SUCCESS` publishes. Actual original
CPU codec/parser/small gradient/Adam proof is not pretrained/GPU/performance
qualification and does not change W5's backward hold or W5/W6 distinction.
