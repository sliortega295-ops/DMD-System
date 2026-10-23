# Offline media and native feature data

Data preparation is separate from training. It decodes **actual local media**,
calls an explicitly supplied model-specific encoder, and publishes an ordered,
hash-bound tensor dataset. There is no random/synthetic fallback, automatic
model download, implicit latent resize, or conversion between model families.

Install the `train` extra for Torch and the media dependencies, or install the
`media` extra alongside an existing compatible Torch environment. Frame folders
need Pillow; video files need existing `ffmpeg` and `ffprobe` executables on
`PATH`. The data package never installs these executables or fetches assets.
File decoding accepts self-contained MOV/MP4, Matroska/WebM, AVI, MPEG/TS,
FLV, Ogg and GIF containers, not HLS/concat playlists or external segment
references. Both probing and decoding enforce the same format/protocol
allowlist so a hashed file cannot silently introduce unhashed media inputs.

```sh
distillctl data prepare raw/manifest.json --output prepared
distillctl data inspect prepared/manifest.json
distillctl data validate prepared/manifest.json
distillctl data validate prepared/manifest.json --sources
```

`prepare` executes trusted user Python code. `inspect` does not import Torch,
execute an encoder, or deserialize tensor files; it checks the manifest and
feature-file hashes and reports `HASHES_ONLY_TENSORS_NOT_LOADED`. `validate`
additionally loads every feature with `torch.load(map_location="cpu",
weights_only=True)` and checks its tensor tree, shapes, and dtypes. It never
falls back to unrestricted pickle loading. `--sources` also checks original
documents, media, and encoder source; without it, a prepared dataset remains
usable after its original media has been archived or moved.

## Raw manifest

JSON and YAML use the public configuration loader, including deterministic
`include` composition and `${ENV:NAME}` expansion. Local paths are relative to
the **top-level manifest's directory**, including paths introduced by includes.

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

Every record requires a unique, nonempty `id` and at least one of `video`,
`audio`, `image`, or `text`. The `metadata` objects are optional finite JSON. Unknown
record fields are rejected rather than guessed. Video/audio references are a
local path string or an object containing only `path`. Text is explicitly
`{"path": "..."}` or `{"value": "..."}`; a bare string is not guessed to be
either a filename or caption. URLs are rejected, not downloaded.

The generic decoder supplies the encoder with:

| Record field | Decoded value |
| --- | --- |
| `image` | CPU `torch.uint8` RGB tensor `[height, width, 3]`; generic Pillow RGB conversion, no resize |
| `video` | CPU `torch.uint8` RGB tensor `[frames, height, width, 3]` |
| `audio` | `{"samples": CPU float32 [samples, channels], "sample_rate": int}` |
| `text` | Exact UTF-8 string, retaining whitespace |
| `id`, `metadata` | Original ID and optional record metadata |

A video path can be an actual video file or a frame directory. Directories
are nonrecursive and traverse supported regular image files in lexicographic
filename order (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.ppm`); use zero-padded
filenames. Symlink frame entries are rejected. All frames must have the same
dimensions. File-video decoding selects stream `v:0`, with no autorotation or
frame resampling. No decoder trims, pads, resizes, or recycles frames. Audio
supports uncompressed 8/16/24/32-bit PCM WAV, normalized to float32 without
changing its sample rate or channels. Other codecs require an explicitly
prepared supported local source; there is no silent decoder substitution.

The public decoder can also be used directly:

```python
from distillschedule.data.decoding import decode_record

record = decode_record(
    {"id": "clip-0001", "video": "clips/0001.mp4", "text": {"path": "captions/0001.txt"}},
    base_dir="/persistent/dataset/raw",
)
```

## Model-specific encoder contract

`encoder.factory` is a trusted `package.module:callable` or
`relative_file.py:callable`, loaded through the same plugin mechanism as
training factories. It is mandatory: missing native assets or a missing encoder
fail preparation; pixels, noise, or resized unrelated latents are never
substituted as a supposedly compatible model input.

```python
def encode(record: dict, parameters: dict) -> dict:
    # Load only explicit local, identity-checked native assets. Run your actual
    # VAE/text/audio encoders and any model-required preprocessing here.
    return {
        "video_latents": native_video_encoder(record["video"]),
        "conditioning": native_text_encoder(record["text"]),
    }
```

Output is a **nonempty dict of tensor trees**: Tensor leaves in nonempty
string-keyed dictionaries, lists, or tuples. Top-level keys become direct
training feeds; there is no extra `features` wrapper. Dense, non-quantized real
tensors are detached and moved to CPU for storage. Shape and dtype are never
cast, padded, or reshaped by the dataset subsystem. Native batch dimensions,
latent scales, masks, and conditioning keys remain the encoder/model contract.
Empty containers and arbitrary Python objects are rejected.

Encoder parameters are finite JSON, copied for each call. Preparation records
the configured entrypoint source SHA256 and the canonical parameter SHA256.
This does **not** recursively fingerprint imported dependencies or files named
inside arbitrary parameter values: pin/check native checkpoint, tokenizer, and
dependency identities in the plugin and supply their identities explicitly in
parameters/metadata. Trusted plugins can execute arbitrary Python; `prepare`
is not a sandbox. The framework itself performs no asset discovery or download.

Raw manifest/include files and every referenced video, WAV, text file, supported
frame file, or inline text value are hash-bound. Sources and entrypoint code are
rechecked before publication, so mid-preparation drift fails closed.

## Import existing native features

For features already in the correct model-native layout, save a dict of tensor
trees in each `.pt` file and declare its byte SHA256:

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

Each file is safe-loaded and validated, then republished in the same tensor
tree, native shapes, and dtypes. No VAE is called and no latent conversion is
attempted. An optional `features.tensors` field can assert the exact recursive
shape/dtype schema described below; a false assertion fails the import.
Import records provenance as `preencoded` and never claims an encoder ran.

## Prepared data and traversal

A new output contains `manifest.json` and numbered `features/*.pt`. The
`distillschedule.data.prepared.v1` manifest has status `complete`, ordered
records, provenance, and dataset metadata. Each feature reference contains a
dataset-relative path, file SHA256, and recursive `tensors` metadata:

```json
{
  "kind": "dict",
  "items": {
    "video_latents": {"kind": "tensor", "shape": [1, 4, 2, 4, 4], "dtype": "torch.float32"}
  }
}
```

Lists and tuples use `{"kind": "list"|"tuple", "items": [...]}`. Scalar tensor
shape `[]` and zero-sized tensor dimensions are valid; negative, boolean, or
noninteger dimensions are not. Prepared feature references cannot escape the
dataset directory. Duplicate IDs, malformed hashes/metadata, missing files,
and byte drift are rejected.

```python
from distillschedule.data import PreparedDataset, prepare

manifest = prepare("/persistent/dataset/raw/manifest.json", "/persistent/dataset/prepared")
dataset = PreparedDataset(manifest)
for index in range(len(dataset)):
    sample_id = dataset.sample_id(index)
    feeds = dataset.sample(index)  # fresh CPU tensor tree; no recycling
identity = dataset.identity
```

Order is exactly manifest order. Indexes must be integers from `0` through
`len(dataset)-1`; negative indexes, exhaustion, and booleans fail. Epochs,
shuffle, distributed sharding, batching, and sample reuse are explicit trainer
policies, not hidden dataset behavior. `identity` binds the exact manifest bytes
and actual feature-file content; opened readers recheck hashes on access.

The Program trainer defaults to finite manifest-order exhaustion. Its optional
[`data.sampling` policy](training.md#explicit-program-data-sampling) explicitly
enables full epochs, with optional deterministic seed-bound shuffling and fixed
rank-strided cohorts. This does not change the dataset API: inspection,
`sample(index)` and evaluation/generation indexes still refer to physical
manifest records. Original-native input formats do not accept this policy.

Publication uses an exclusive sibling lock and a temporary sibling directory.
All tensors and the complete manifest are validated before the directory is
atomically renamed into place. Existing output directories are never
overwritten; choose a new output for changed sources or encoder parameters.
Ordinary failure removes staging data and the lock. A forcibly killed process
can leave a hidden `.OUTPUT.*` staging directory or `.OUTPUT.lock`; inspect it
and establish that its publisher is no longer running before manual cleanup.
No incomplete staging directory is a valid prepared dataset.

Validation proves local bytes and tensor structure, not model compatibility,
native VAE numerical parity, training convergence, or GPU performance. Those
require the corresponding model/training evidence.

## Original native input preparation

Native workloads keep their own input semantics; their manifests are not the
generic `PreparedDataset` format. These commands inventory or prepare actual
operator-supplied native features rather than applying the tiny media encoder:

| Workload | Producer | Input and boundary |
| --- | --- | --- |
| W1 | `distillctl workloads native prepare-stream` | Original per-rank Wan features and explicit cohort seeds; original capture/data helpers |
| W2 | `distillctl data prepare-helios-stream --sources ... --training-config ... --output ...` | Five actual source steps per world16 cohort, original dataset/history/noise helpers; CPU preparation, no model initialization |
| W4 native teacher cohorts | [`data.livewan_inputs.prepare_inputs`](livewan_inputs.md) | Source-qualified original TeacherLatents, 80 distinct records per cohort and actual worker InputLoader; explicit original reference training/restart supported, fresh C51 pipeline restart unavailable |
| W5 | `distillctl data prepare-h3-grid --native-config ... --output ...` | Content inventory of already encoded original JSONL conditions; no latent generation or model import |

Use the [Wan stream guide](../workloads/original_wan_fresh_training.md),
[W2 reference inputs](../../recipes/helios_nogan/REFERENCE_TRAINING.md) and
[W5 original inputs](../../recipes/h3-grid-original-training.md) for exact
schemas and runtime admission. Supplied records must be genuine and satisfy
their original shapes/precision/RNG/sample contracts. Missing assets and
unsupported branches reject; they are not resampled or replaced with random
tensors. Preparation alone does not establish model compatibility, quality,
CUDA capacity or numerical equivalence.

## Actual-original CPU reference preparation

The separate W3/W6 CPU Program factories use the **common PreparedDataset
container**, with native content and producer-specific metadata. They do not
replace the distributed native input ABIs above:

- [W3 history19](../../recipes/helios_gan/REFERENCE_TRAINING_CPU.md):
  `data prepare-helios-gan-reference` calls the actual original producer;
  five primary/secondary rows form one G1/C5 cycle. No DMD noise/timesteps or
  generated rollouts are synthesized by preparation.
- [W6 cached AV](../../recipes/h3_ar/CPU_REFERENCE.md):
  `data prepare-h3-cached-reference` validates the original flat/split-packed
  previous/current AV cache and proves exact parser equivalence of its
  tensor-only representation. Preparation streams one source at a time.
- [Portable original Helios G](../../recipes/helios_gan/PORTABLE_INFERENCE_CPU.md):
  `data import` accepts new native `prompt_embeds` with exact declared geometry;
  it needs no unused five-row training cycle and supplies no fake text encoder.

These are opt-in local CPU references. Sources, shapes, precision and evidence
labels remain explicit, and the production recipes never invoke the statistics
encoder from the tiny media example.


## Images, prompt-only data and model-native preparers

The raw v1 record also accepts `image: "picture.png"` or
`image: {"path": "picture.png"}`. Generic image decoding uses Pillow RGB and
returns HWC uint8 without spatial resizing. Model-native preparers may impose
stricter color/geometry requirements: SD's real-image VAE path requires exact
single-frame RGB and does not silently convert, crop or resize.
A text-only record is valid; no video or fabricated media is mandatory.

Use the same CLI with the model's explicit preparer:

```sh
distillctl data prepare inputs.json --output prepared \
  --preparer distillschedule.data.sd_inputs:prepare_inputs
# Alternatives: distillschedule.data.self_forcing_inputs:prepare_inputs
# or distillschedule.data.ltx_inputs:prepare_inputs
```

Native manifests carry their own versioned schemas. SD owns CLIP tokenization
and optional actual local VAE encoding; official Self-Forcing owns its actual
local T5/tokenizer; LTX owns its actual Gemma/AV VAE/token transforms. These are
not interchangeable with the tiny example's statistics encoder. Source,
checkpoint and conditioning identities are adapter-owned and bound to prepared
provenance. The pipeline never fills missing conditions with random features.
SD input schema v2 binds the actual variant and relative tokenizer file hashes
to every prepared tensor row and to model admission; image latents additionally
bind actual VAE file/configuration hashes. Same-shaped incompatible bundles are
rejected before updates and generation, including external checkpoint-evaluation
inputs and new inputs to inference-only exports. Byte-identical relocated
tokenizers/VAE assets remain compatible; export generation never reopens the
original preparation/base-model paths. Supplied preencoded SD latents require
`preencoded_vae_compatibility: PREENCODED_LATENTS_MATCH_LOCAL_VAE`, a caller's
explicit assertion against the pinned VAE, not an encoder-execution claim.

`data.streams` in a training request declares independent prepared manifests,
`samples_per_cycle` and optional per-stream epoch
sampling. Each stream has its own saved sampler/position and independently
derived sampling seed, without consuming the training RNG. This represents
DMD2 Generator/guidance/denoising/real-image consumption and official
Self-Forcing's separate G/C prompts; the algorithm admits the exact required
counts. Do not turn independent streams into one shared advancing iterator.
See [model families](model_families.md) for exact recipes.

For FLUX/Qwen-Image/Wan2.2/Hunyuan inference exports with native encoding
capabilities, [export-only new-input preparation](export_inputs.md) uses the
same public plugin entrypoint without original assets/checkpoints. Unsupported
adapters refuse rather than falling back to their asset-based preparer.
