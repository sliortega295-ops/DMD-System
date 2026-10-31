# Decode original H3 video and stereo audio

`distillctl decode` turns prepared native latent tensors into media without
loading Student, Teacher, Fake Score or training state. It uses the same
`ModelComponents`, prepared-data validation and PNG/WAV artifact writer as the
training/generation interface. This is useful after a latent-only inference
export or when decoded media belongs in a separate CPU process.

The H3 codec invokes the original `MiniMaxH3VideoVAE.decode_video` and
`MiniMaxH3AudioVAE.decode_audio` methods. Original mean/std, temporal overlap,
tiling and audio synthesis are retained. Latents are not clamped. No sampling,
training, resizing, cropping or audio/video resampling occurs in this command.

## Prepare assets

Use an existing compatible Torch2.10 CPU environment. The optional
`h3-av-codec` dependency group contains the decoder's dependencies; it does not
install a training/backend stack or fetch a model. Avoid changing a validated
training environment just to decode media.

1. Supply an existing trusted original H3 source tree. The exact three source
   files/hashes are listed in
   [decoder.native.template.json](../../recipes/h3_ar/decoder.native.template.json).
   Source and model weights are external assets, not redistributed by this
   adapter. Record the real source origin and applicable terms, then explicitly
   accept them. A hash is not a license or permission grant.
2. For the default factory, supply local safetensor shards in the canonical
   original DiffSynth VAE state-dict format (`WarpedTensor` keys end in `.weight`).
   A complete native VAE or its exact decoder-only state is accepted. The loader
   admits every header and reads only `post_quant_conv`/`decoder` for video and
   `dec_in_proj`/`decoder` for audio. Missing or unknown keys fail. It never
   constructs the unused concrete encoders. Native BF16 storage may be
   explicitly converted to FP32 using `storage_dtype: bfloat16_to_float32`;
   runtime decoding stays FP32. No implicit storage conversion is applied.
3. Copy the template as your codec manifest. Fill actual paths, bytes and
   SHA256s. Set exact latent/output shapes, video FPS and 32kHz audio. Video
   frame count follows the original temporal planner, not a guessed multiplier.
   Template geometry is an example, not an automatic change to W6's workload.
   Full native video decoding has about 2.42B parameters, plus about 64.94M
   audio parameters: give it explicit construction/working-set/host-reserve
   capacity. A smaller Student does not make this decoder smaller.

### Original FL2VA weight files: explicit import, not automatic fallback

Original FL2VA safetensors use a different parameter namespace from the
canonical DiffSynth codec classes. Do not point raw files at the default
`build_decoder` factory or manually ignore its strict-load errors. The separate
`build_fl2va_decoder` factory imports them through the exact associated official
DiffSynth converters, without rewriting or copying the weight payloads:

- Video register tokens and block scales are relabeled to `.weight`. The one
  officially excluded `decoder.mask_token` is checked for its original shape
  and storage dtype; the original decoder appends zeros and never owns this
  parameter. No other unknown/missing keys are ignored.
- Audio's three `pre_block.attn` bias keys are relabeled to `.weight`, not
  dropped. All original full-state headers are admitted even though only
  decoder-owned tensors are materialized.

First fill a normal `decoder.native.template.json` codec manifest with your
actual original source, architecture, geometry, raw weight pins and budgets.
Then fill [decoder.fl2va.template.json](../../recipes/h3_ar/decoder.fl2va.template.json)
with its actual manifest pin and an operator-supplied compatible converter
source tree. The two converter files and their model-config association are
source-pinned separately from the three codec definitions. The template names
the supported original FL2VA source version; another revision/format is not
silently accepted. Supply the actual origin/license terms and explicitly
accept them. Neither source nor pretrained weights are fetched or bundled.

In the same public `decode.json` shown below, change **only** the decoder block:

```json
{
  "factory": "distillschedule.training.h3_av_codec:build_fl2va_decoder",
  "parameters": {"manifest": {
    "path": "/project/h3-output/fl2va-import.json",
    "bytes": 1234,
    "sha256": "REPLACE_WITH_ACTUAL_IMPORT_MANIFEST_SHA256"
  }}
}
```

Run the same `distillctl decode` command. Raw/canonical format is explicit:
there is no format detection, alternate decoder, precision fallback, crop or
reduced production architecture. BF16 storage requires the base manifest's
existing explicit `bfloat16_to_float32` policy. Complete full-native or exact
decoder-only **raw** inventories are required; mixed raw/canonical names fail.
The import receipt reports actual per-role key/rename/exclusion counts and
converter/source identities, not campaign-specific counts or checkpoint IDs.

An optional bounded preflight is available before materializing weights:

```python
from distillschedule.training.h3_av_weight_import import inspect_headers
report = inspect_headers({"manifest": {
    "path": "/project/h3-output/fl2va-import.json",
    "bytes": actual_import_bytes,
    "sha256": actual_import_sha256,
}})
```

Run that in the same explicit CPU/world1 Torch2.10 environment. It checks
source, original meta layouts, safe headers and geometry/budget arithmetic.
Its `HEADERS_ONLY_NOT_WEIGHT_HASH_LOAD_OR_DECODE` result is **not** a full
weight-payload SHA, finite-weight, host-headroom, strict-load or media proof.
Actual decode retains all those original gates and restores caller RNG. The
default v1 decoder and separate encoder/grid consumers remain unchanged; raw
import is opted into only through this new factory.

## Prepare latent inputs

A prepared record contains FP32 CPU `video_latents` in BCTHW (B=1,C=24) and
`audio_latents` in SCT (stereo S=2,C=32). Both shapes must match the codec
manifest. Use a new sample ID and actual feature hashes through the standard
preencoded import. For an existing latent generation artifact, extraction is:

```python
from pathlib import Path
import json
import torch
from safetensors.torch import save_file
from distillschedule.workloads.best_known.registry import file_sha256

root = Path('/project/h3-output')
values = torch.load(root / 'latents/generation.pt', map_location='cpu', weights_only=True)
latents = {name: values['generated'][name].contiguous()
           for name in ('video_latents', 'audio_latents')}
save_file(latents, root / 'native-latents.safetensors')
(root / 'input.json').write_text(json.dumps({
    'schema_version': 'distillschedule.data.preencoded.v1',
    'records': [{'id': 'new-h3-output', 'features': {
        'path': 'native-latents.safetensors',
        'sha256': file_sha256(root / 'native-latents.safetensors')}}],
    'metadata': {'purpose': 'original-H3-decode-only'}}))
```

```sh
distillctl data import /project/h3-output/input.json --output /project/h3-output/prepared
```

No real examples or latent features are synthesized to fill missing inputs.

## Decode with the same command family

Create `decode.json` next to your prepared directory. `data.prepared` and a
local Python decoder reference resolve relative to that config, not the shell
working directory. The codec manifest itself is an explicit pinned local file:

```json
{
  "schema_version": "distillschedule.decode_prepared.v1",
  "decoder": {
    "factory": "distillschedule.training.h3_av_codec:build_decoder",
    "parameters": {"manifest": {
      "path": "/project/h3-output/codec.json",
      "bytes": 1234,
      "sha256": "REPLACE_WITH_ACTUAL_CODEC_MANIFEST_SHA256"
    }}
  },
  "data": {"prepared": "prepared/manifest.json", "sample_index": 0},
  "output_kind": "audio_video"
}
```

Replace `bytes` and SHA256 with actual values; they are not example acceptance
values. Then run outside the training launcher:

```sh
distillctl decode /project/h3-output/decode.json --output /project/h3-output/decoded
```

The result includes `frames/*.png`, `audio.wav` (stereo 32kHz PCM16),
`generation.pt`, `generation.json` and `_SUCCESS`. It preserves full native
frame/sample lengths rather than trimming one stream to fit the other. Media
muxing is a separate step. The command restores caller RNG and closes adapter
resources on success or error; it does not create an optimizer or run a model
training cycle.

## Scope

The explicit adapter is CPU FP32, one process, the pinned original H3 codec.
The separate random fixture shrinks only the actual video Transformer's
constructor dimensions and keeps the full native BigVGAN audio decoder; it is
not a pretrained/quality result. This consumer is not W5/W6 native Student
export, ZeRO-3 checkpoint reconstruction, AV encoding or a W5 latent-ABI
qualification. Those integrations remain separately tracked. Other models may
supply their own existing `ModelComponents.decode` factory without adding a
model-name case to the command or the planner.

For the independent input side, see [raw H3 AV encoding](h3_audio_video_encode.md).
The encoder loads its own parameter subtrees and does not need these decoder
weights; decoder-only weights cannot serve as an encoder checkpoint.
