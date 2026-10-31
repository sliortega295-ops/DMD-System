# Prepare original H3 previous/current audio-video data

The common data command now accepts raw **RGB frame directories + stereo PCM
WAV**, with an explicit previous chunk and the original preencoded text features.
It calls the original H3 video/audio encoders and then the existing cached-input
parser. The resulting prepared dataset enters the existing H3 reference trainer;
there is no new DMD algorithm or second training loop.

```sh
distillctl data prepare /project/h3/raw-inputs.json \
  --preparer distillschedule.data.h3_raw_inputs:prepare_inputs \
  --output /project/h3/prepared-raw
```

Use `/project/h3/prepared-raw/prepared/manifest.json` as `data.prepared` in the
[H3 CPU reference configuration](../../recipes/h3_ar/CPU_REFERENCE.md). Match
its model asset manifest's geometry to the observed encoded shapes. The native
ZeRO-3 stream keeps its own rank/shape/source admission: encoding a file does not
change that distributed workload contract. Its input preparation may reuse the
actual `.pt` files under `native-caches/`, after normal native admission.

## Encoder assets

Copy [encoder.native.template.json](../../recipes/h3_ar/encoder.native.template.json)
and fill the source origin/terms and exact local safetensor pins. Use the same
original source closure and existing `h3-av-codec` dependency environment as the
[decoder](h3_audio_video_decode.md). No Student, optimizer, model download or CUDA
context is needed for offline encoding.

The encoder manifest is independent of the decoder manifest. It accepts complete
original VAE state or exact encoder-only state. On the meta device it inspects
all headers, then loads only:

- video: `encoder` and `quant_conv`;
- audio: `encoder`, `pre_block` and `mean_proj`.

Unused decoders and the unused audio variance projection are not materialized.
Production architecture is fixed to the original VAE. A separately labeled small
original-class fixture exists for CPU tests, not as a smaller production codec.
FP32 storage is the default; original BF16 weights require the explicit
`bfloat16_to_float32` storage option. Offline computation remains CPU FP32.
Set parameter/working-set/attention budgets and a host reserve for your machine.

## Raw input manifest

[raw.native.template.json](../../recipes/h3_ar/raw.native.template.json) shows the
full schema. Each record has a unique ID, **actual** `current` and `previous`
chunks, and a safe original text-feature file:

```text
current / previous:
  video: {path: frame_directory, sha256: frame_inventory_sha256}
  audio: {path: absolute_wav_path, bytes: actual_bytes, sha256: actual_sha256}
  fps: 24
prompt: {path: absolute_safetensors_path, bytes: actual_bytes, sha256: actual_sha256}
```

Video paths resolve relative to the manifest; pinned WAV/text/encoder files use
canonical absolute paths. Obtain the frame-directory hash with the existing
`distillschedule.data.manifest.source_identity(path, frames=True)`. Frames are
read in sorted filename order. The text file contains exactly:

```text
prompt_embeds:    FP32 [text_tokens, 5120]
text_token_tags: int64 [text_tokens], all 1
```

Use genuine outputs from the matching text encoder. The audio/video VAE does not
encode text or invent missing history. The current and previous chunk can have
different declared lengths; samples in one prepared reference dataset must have
the same resulting geometry. Bucket other shapes explicitly.

### Geometry and numerical handling

- Video: 24 FPS, `17*n+5` RGB frames; height/width divisible by 32. No resize,
  interpolation, frame removal or spatial padding. Compressed videos should be
  extracted to explicit frames without resampling before this first adapter.
- Audio: stereo 32kHz PCM WAV. No channel averaging, trim, gain or resampling.
- Pixels are converted from THWC uint8 to BCTHW FP32 in `[0,1]`; waveform is TC
  FP32 in `[-1,1]`. The **original** methods own pixel/latent normalization.
- `encode_video` repeats the final frames to a multiple of 17 internally, then
  discards its original three tail tokens. `encode_audio` pads the right side to
  a multiple of 800 samples. Both return posterior **means**, without sampling.
  Padding counts are recorded per chunk, not hidden as input resampling.
- These two native methods do not synchronize audio/video duration for the user.
  Supply correctly aligned recordings; the adapter preserves their actual
  lengths rather than cropping one modality to fit the other.

## Outputs and provenance

`prepared-raw/` contains:

| Path | Contents |
| --- | --- |
| `encoded/manifest.json` | Actual original codec outputs with all raw media, text, source and weight identities |
| `native-caches/*.pt` | Safe tensor-only previous/current inputs in the original parser's layout |
| `original-cached-inputs.json` | Original cached-input manifest, not a new training contract |
| `prepared/manifest.json` | Published original-parser dataset, ready for reference training |
| `encoding.json`, `_SUCCESS` | Observed shapes, normalization, native padding and complete provenance links |

Output directories are never overwritten. Incomplete failures retain their
intermediate evidence and do not publish the outer `_SUCCESS`. Missing prompt,
previous chunk, codec shards or incorrect media geometry fail explicitly.

Source and Torch2.10 CPU acceptance uses actual original encoders, PNG/WAV input,
the original parser and a real small H3 gradient/Adam update. This adds no new
pretrained quality, native CUDA-capacity or performance result. W5's existing
backward numerical hold is unchanged; W5 Grid and W6 AR remain different flows.

## Continuous integration

The `original-codec-contracts` CPU job checks public schemas, safe headers,
storage conversion, import-only discovery and refusal behavior on Python3.10/3.12.
It uses the existing no-skip receipt gate and needs no private source or weights.
Actual original VAE execution and raw-media-to-update qualification have separate
operator-asset CPU evidence; that job does not claim to run those private assets.
