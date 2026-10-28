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
2. Supply local safetensor shards in the original native VAE state-dict format.
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
