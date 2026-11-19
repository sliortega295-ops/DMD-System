# Helios No-GAN SafeBundle continuation

Use [`training.safe-stream.json`](training.safe-stream.json) to opt into the
native SafeBundle with complete-cycle checkpoints and restart. See the
[execution contract](../../docs/contracts/helios_native.md#helios_nogan_safe_stream_v1)
for the stream and checkpoint formats.

Prepare 2..100 fresh original No-GAN33/222/B1 cohorts using the unchanged
[reference input preparation](REFERENCE_TRAINING.md) workflow. Use the
[`assets.reference.template.json`](assets.reference.template.json) manifest for
local model, precast, Step10 G2/C10/EMA10 and negative-conditioning assets.
For export and decoded video, see [Student output](NATIVE_OUTPUT.md).

Copy [`training.safe-stream.json`](training.safe-stream.json),
[`task.json`](task.json) and your `assets.reference.json` into your project.
Fill the paths, match `trainer.cycles` to the prepared cohort count and choose a
new output directory outside the checkout. Run this preset directly, without
a saved plan.

```sh
distillctl config validate /absolute/project/training.safe-stream.json
distillctl train /absolute/project/training.safe-stream.json --dry-run
```

These commands check the recipe and input metadata on CPU. Before training, set
the pinned local FA2 mapping:
`LOCAL_KERNELS=kernels-community/flash-attn2=/absolute/local/path`. Leave
allocator and optimization/benchmark overrides unset.

Run `distillctl train /absolute/project/training.safe-stream.json` on each rank
of your two-node, eight-ranks-per-node torchrun allocation. The native P0 owner
runs the model, loss and Adam/LR/EMA updates. Check per-cycle receipts for the
SafeBundle counters, window size of five, version census and drained rollouts.

Resume only a published incomplete-target complete-cycle checkpoint, using
the exact same source, assets, stream, horizon, policy and native 2x8 node-HSDP
layout, with a new output:

```sh
distillctl train /absolute/project/training.safe-stream.json \
  --resume /absolute/project/prior-run/checkpoints/cycle-000002 \
  --output /absolute/project/new-resume-run
```

## Stop at a durable checkpoint without changing the stream

For a prepared 100-cycle stream, keep `trainer.cycles: 100` and set
`trainer.checkpoint_every: 50` plus `trainer.stop_after_cycle: 50`. The first
process saves a complete all-rank checkpoint and exits after cycle 50. Both the
reference and SafeBundle use the G1/C5/EMA5 cycle boundary.

Start a new process with the same source/assets/full prepared stream and a new
output directory. Set `trainer.resume_from` to `checkpoints/cycle-000050`, then
**remove `trainer.stop_after_cycle`** (or set it to a later checkpoint boundary,
such as 100). Never change `cycles` to 50 or shorten/reprepare the stream. An
explicit stop at or before the restored cycle is rejected before model build.

The stop must be an integer inside the unchanged horizon with a positive
checkpoint cadence. An intermediate stop must be divisible by that cadence;
the final horizon is always a valid save boundary. Rank-disagreeing stop
controls, missing/corrupt shards and incomplete checkpoint publications fail
closed. Without the stop field, the existing full-horizon behavior is unchanged.
Public export discards this training-only stop control and restores the explicit
export checkpoint without running training updates.

A graceful early stop reports `STOPPED_*_AT_CHECKPOINT`. The
`training_segment` record gives the configured horizon, starting cycle,
requested stop, last completed cycle and number of cycles executed.

Integration tests cover native helpers and checkpoint state on CPU/Gloo.
