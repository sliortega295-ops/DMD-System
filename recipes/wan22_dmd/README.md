# Wan2.2 TI2V-5B: text-to-video flow DMD

This recipe uses the native `WanTransformer3DModel`, residual `AutoencoderKLWan`,
`UMT5EncoderModel` and T5 tokenizer through the common DMD-System lifecycle.
Student, Teacher and Fake Score use the same model family. Training uses unmerged
LoRA, four Euler steps, G1/F5 updates, local batch one and CFG=1.

## Environment and assets

Install the separate `modern-flow` dependency group using
[environment.json](environment.json). Model/config revisions and Apache-2.0
attribution are recorded in [source.json](source.json).

Place the native `transformer/`, `vae/`, `text_encoder/` and `tokenizer/`
directories in one local bundle, then inventory the files:

```sh
python recipes/wan22_dmd/lock_local_assets.py /ABSOLUTE/local-wan22 \
  --cpu-max-parameters YOUR_PER_COMPONENT_LIMIT
```

The limit bounds each native constructor before weight loading; choose it for
available memory. Student/Fake/Teacher can select distinct subdirectories with
`--student`, `--fake` and `--teacher`, retaining a common architecture. The loader
checks file hashes, safe shards, shapes and finite values. For BF16-stored
checkpoints, add `--dtype-policy bf16_or_fp32_to_fp32`; this explicitly converts
stored weights to the FP32 owned training state.

## Prepare inputs

Use the [raw](inputs.raw.template.json), [prompt](inputs.prompts.template.json)
or [preencoded](inputs.preencoded.template.json) template. RGB frame directories
use Pillow; container video decoding also needs `ffmpeg` and `ffprobe` on PATH.
Keep the original resolution and frame rate: `Fraw=(Flatent-1)*4+1`, with height
and width divisible by32. The native spatial16 VAE and spatial2 Transformer
patches operate on BCFHW latents; do not patchify RGB twice.

Offline preparation uses the native posterior mode, VAE mean/std normalization,
226-row UMT5 text features, native zero padding and genuine empty-prompt features.
Prompt-only records specify latent geometry as a noise template. Preencoded
records retain all context, masks, geometry and encoder-binding fields. Sample
IDs, original source hashes and data cursors survive training and recovery.

```sh
distillctl data prepare /ABSOLUTE/inputs.json --output /ABSOLUTE/prepared \
  --preparer distillschedule.data.wan22_inputs:prepare_inputs
```

## Train, resume and export

Copy [training.lora.json](training.lora.json) and set the asset, prepared-data and
output paths. Select the owned device with `trainer.device` (`cpu` or `cuda:N`).
The default native compute is FP32. For BF16 compute, add
`"compute_precision":"bf16-mixed"` to `model.parameters`: native Transformer
calls use BF16 autocast, while master weights, Adam, EMA, codecs and flow/sampling
math stay FP32. Checkpoints, saved plans and inference exports bind this setting.

```sh
distillctl train /ABSOLUTE/train.json --cycles 2 --output /ABSOLUTE/run
distillctl train /ABSOLUTE/train.json --cycles 3 \
  --resume /ABSOLUTE/COMPLETE_CHECKPOINT --output /ABSOLUTE/resumed
distillctl export /ABSOLUTE/train.json --checkpoint /ABSOLUTE/FINAL_CHECKPOINT \
  --output /ABSOLUTE/export
```

Use the checkpoint path returned by `train`. Complete-cycle recovery retains
parameters, Adam, schedulers, EMA, RNG, data progress and role/update clocks.
The native velocity is `v=epsilon-x0`; the Transformer consumes `1000*sigma`
expanded over patches, and the four-step Euler rollout keeps its original sign.

## Execution strategies

Use `{"mode":"reference"}` for the reference implementation. The existing
frozen-state sharing capability uses this single-device policy:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

For explicitly launched equal-rank B1 replicas, set
`data.sharding="rank_strided"`, a matching `trainer.distributed.backend`
(`gloo` on CPU or `nccl` on CUDA), and each process's actual local device.
Reference mode uses the shared post-backward reducer. Its manual sharing variant is:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-replicated-v1","reduction":"post_backward","bucket_cap_mb":0,"local":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}}
```

Gradients are averaged before clipping and one Adam commit; each rank consumes
independent records and RNG. Sharing aliases only immutable bases after role
construction, retaining disjoint Student/Fake trainables. See
[execution capabilities](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing)
for the actual policy/capacity conditions.

## Generate with the current Student

The portable export contains the current Student, VAE, UMT5 and tokenizer.
Use its returned factory/parameters in an inference config, prepare a new prompt
with the [export-only preparer](../../docs/guides/export_inputs.md), then run:

```sh
distillctl generate /ABSOLUTE/inference.json --weights /ABSOLUTE/export \
  --prepared /ABSOLUTE/new-prepared/manifest.json --output /ABSOLUTE/generated --output-kind video
```

Output contains decoded THWC uint8 RGB and a PNG frame sequence. Portable inference
can run on CPU independently of the original training bundle and checkpoints.

## Tests and provenance

```sh
DMD_MODERN_FLOW_CPU=1 python recipes/wan22_dmd/create_cpu_fixture.py /ABSOLUTE/NEW-fixture
DMD_MODERN_FLOW_CPU=1 python -m pytest -q tests/test_training_wan22_cpu.py tests/test_training_flow_precision.py
DMD_MODERN_FLOW_CPU=1 DMD_NATIVE_FLOW_REPLICATED=1 python -m pytest -q tests/test_training_flow_video_replicated.py
```

The fixtures use actual upstream small model classes. Tests compare native
forward/gradients, text/VAE/Euler behavior, reference/sharing updates, saved plans,
fresh-process recovery and new-prompt output. The Gloo2 oracle independently
averages sequential B1 gradients. Historical native FP32 CUDA acceptance is at
`artifacts/framework/20261024/native_video_cuda_acceptance.json`; CPU fixture
results are separate from pretrained-model GPU measurements.

Wan model/config sources, Diffusers and Transformers retain Apache-2.0 attribution;
keep tokenizer, ftfy and sentencepiece notices. No pretrained weights are bundled.
