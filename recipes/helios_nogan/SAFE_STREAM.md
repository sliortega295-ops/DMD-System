# Native SafeBundle fresh stream (research, GPU NOT_RUN)

This separate opt-in connects the complete W2 SafeBundle to the **original
native** finite stream owner. It is not the CPU reference, not a widened old
two-cycle plan, and not a stable/recommended training recipe. Read the
[versioned contract](../../docs/contracts/helios_nogan_safe_stream_v1.md).

Prepare 2..100 fresh original No-GAN33/222/B1 cohorts using the unchanged
[reference input preparation](REFERENCE_TRAINING.md) workflow. Supply your own
explicit local model/precast/Step10 G2-C10-EMA10/negative asset manifest using
[`assets.reference.template.json`](assets.reference.template.json). Cold start,
arbitrary epochs, GAN, other shapes and decoded outputs are not supported here.

Copy [`training.safe-stream.json`](training.safe-stream.json),
[`task.json`](task.json) and your `assets.reference.json` into your project.
Replace every placeholder, match `trainer.cycles` to the prepared cohort count,
and choose a new output outside the checkout. This strategy does not require a
saved plan. The existing `training.json` remains the frozen two-cycle preset;
`training.reference.json` remains explicit unoptimized reference.

```sh
distillctl config validate /absolute/project/training.safe-stream.json
distillctl train /absolute/project/training.safe-stream.json --dry-run
```

Those commands validate CPU metadata/binding, **not GPU dependencies or parity**.
Before actual native execution, explicitly set your existing pinned local FA2
mapping (`LOCAL_KERNELS=kernels-community/flash-attn2=/absolute/local/path`).
No kernel is installed/downloaded; no historical/private asset defaults are
used. Missing/wrong extension or imported callable refuses. Do not set allocator
or inherited optimization/benchmark overrides.

Actual execution, when separately authorized and qualified, enters
`distillctl train /absolute/project/training.safe-stream.json` on each of the
existing two nodes x eight fresh torchrun ranks. The operator owns allocation
and rendezvous; this API neither reserves nodes nor launches SSH. Original P0
owns models, kernels, Adam/LR/EMA and loss. Per-cycle receipts must show the
full SafeBundle's executed effects, window5, native version census and drained
rollouts. Capacity, strict numerical and long-run quality gates remain open.

Resume only a published incomplete-target complete-cycle checkpoint, using
the exact same source, assets, stream, horizon, policy and native 2x8 node-HSDP
layout, with a new output:

```sh
distillctl train /absolute/project/training.safe-stream.json \
  --resume /absolute/project/prior-run/checkpoints/cycle-000002 \
  --output /absolute/project/new-resume-run
```

Tiny original-helper and Gloo state tests do not establish native GPU binary,
kernel numerical, sustained capacity or pretrained training quality acceptance.
No speedup or all-six optimized lifecycle completion is claimed.
