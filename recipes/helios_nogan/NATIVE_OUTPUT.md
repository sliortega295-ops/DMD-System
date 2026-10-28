# Native checkpoint → current Student → new-prompt video

Use this with the native [reference](REFERENCE_TRAINING.md) or
[SafeBundle stream](SAFE_STREAM.md) recipe. The CPU-reference diagnostic remains
separate. The [output contract](../../docs/contracts/HELIOS_NATIVE_OUTPUT_V1.md)
defines the checkpoint, precision and output semantics.

1. Keep the complete checkpoint's source, runtime, original task/assets, input
   stream, topology and reference/SafeBundle strategy. Do not relabel an
   older-source checkpoint or substitute the frozen two-cycle preset.
2. Copy your training config. Add the object in
   [export.native-policy.template.json](export.native-policy.template.json) as
   `trainer.export`. Fill actual local Wan2.1 VAE config/safetensor paths, exact
   sizes/hashes, origin and license. Match `prompt_shape` to your offline T5
   features. Budgets are explicit admission ceilings, not measured memory.
3. Under the same existing two-node world16 launcher, on every rank run:

```sh
distillctl export /project/native-with-output-policy.json \
  --checkpoint /project/run/checkpoints/cycle-000100 \
  --output /project/outputs/current-student
```

Export restores complete state and does no training update or capture. Terminal
cycle checkpoints are allowed for output. Remove `trainer.export` before
returning to ordinary training. No launcher, model download or implicit
offload is created by this command.

4. Prepare a new `distillschedule.data.preencoded.v1` manifest containing only
   declared BF16 `prompt_embeds`, real feature hashes and new sample IDs. In an
   independent process outside torchrun:

```sh
distillctl data import /project/new-prompts/inputs.json --output /project/new-prompts/prepared
distillctl generate /project/outputs/current-student.inference.json \
  --weights /project/outputs/current-student \
  --prepared /project/new-prompts/prepared/manifest.json \
  --output-kind video --output /project/outputs/new-video
```

The emitted config defaults to CPU. A single CUDA inference device can be
selected with `trainer.device`, provided the full unsharded G, VAE and working
set plus reserve fit; sharded training fit does not imply inference fit. The
same declared frames/resolution are used, without silent downscaling.

The export contains current unmerged Student and codec weights, so the second
process does not need the old training/base/VAE directories. Success produces
`frames/*.png`, `generation.pt`, `generation.json` and `_SUCCESS`, with native
std/mean decoding and no latent clamp. Video-container encoding is a separate
consumer step. This path has source/CPU proof; pretrained GPU output is a later
acceptance item.
