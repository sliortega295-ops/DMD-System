# SD1.5 DMD2: native FSDP2 reference

Use `recipes/dmd2_sd15/training.fsdp2.json` for the SD1.5 reference that shards
Generator and the native guidance tree (Teacher, Fake Score and classifier)
across 16 A100 GPUs. Use `training.optimized.json` for the current optimized
Resident/DDP combination. Both use `distillctl train`, the same upstream DMD2
model/loss classes and the common continuous-training/checkpoint lifecycle.

## Train and resume

Fill in the pinned DMD2 source, local SD1.5 bundle and prepared prompt/real-image
streams in the recipe. No assets are downloaded implicitly. Launch with the
existing distributed environment (`Torch 2.10`, `diffusers 0.37.1`,
`transformers 4.51.3`, `peft 0.17.1`):

```bash
torchrun <your-16-rank-launch-options> --no-python \
  distillctl train /project/training.fsdp2.json
```

For continuation, set `trainer.cycles` to the final cycle number and
`trainer.resume_from` to a completed checkpoint, then launch on the same
topology. A checkpoint includes each rank's trainable/master shards, Adam
moments, scheduler clocks, native RNG and data-stream positions. Publication
requires all rank shards. Cached execution weights are reconstructed; they are
not saved as persistent replicas.

Model and asset construction is local until every rank passes admission.
The native preparation stage then shards the roles and binds the existing,
still-empty Adam parameter groups to their new shards before any update.
Optimizer and scheduler objects, group order and learning-rate clocks stay
unchanged. A rank-local load failure cannot enter another rank's sharding
collective.

`distillctl export` materializes full inference Generator weights collectively
on every rank, then only rank 0 publishes the portable image model. Training
checkpoints and inference exports are different formats. Checkpoint generation
also runs on all ranks; do not invoke a sharded Generator from rank 0 alone.

## Contract

- SD1.5, 512×512, B1/rank, global B16; G1/C10/EMA0 per cycle.
- Native precision: FP32 Generator/Fake masters and Adam, BF16 gradient compute
  and Teacher, FP16 VAE, FP32 text encoder/classifier heads; no outer AMP.
- Block-level and root FSDP2 with `reshard_after_forward=True`; no CPU offload.
- Original native update/RNG/sample order and world-scaled scheduler clocks.
- The classifier's original direct `.forward(...)` becomes a module call to
  execute FSDP hooks. The numerical function and its arguments are unchanged;
  the reversible source transform is scoped to one owner.
- No fused Adam, classifier packing, activation-retention or cast-cache bundle
  is enabled. Automatic calibration/clone-based trials and SDXL FSDP are not
  admitted by this versioned backend.

## Validation status

The placement and classifier ABI are exact imported definitions from the
measured FSDP2 baseline. CPU tests cover real FSDP2 local shards and Adam state,
source checks and native classifier math/hooks. Public world16 training,
restart and decoded-output qualification is a separate pending GPU check;
CPU tests do not supply new timing results.
