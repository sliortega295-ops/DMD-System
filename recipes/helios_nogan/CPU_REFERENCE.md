# W2 original warm No-GAN CPU reference

This separately named Program adapter runs the complete original Helios model,
the original No-GAN P0 cycle, and the common training/checkpoint/output lifecycle.
It is not W1 with a new label, W3 with GAN disabled, or a new numerical loop.
The existing native FSDP2 and saved-plan recipes remain unchanged.

## Exact scope

- Original 384×640, 33 frames, three pyramid budgets `[2,2,2]` and LoRA256.
  The mixed BF16 base, original FP32-only modules and FP32 adapters remain
  distinct. All three Generator history-patch adapters are trainable.
- Warm initialization requires actual G2/C10/EMA10 masters, nonempty named Adam
  moments, schedulers and sparse EMA. Clip1, original learning rates, constant
  LR and EMA decay/update-after policy are checked, not supplied by a toy loop.
- Each subsequent complete cycle executes G1/C5/EMA5 through the original
  `_Observer`/`_prepare_cycle` and P0 helpers. Five post-G B1 rollouts use distinct
  original entries. One nonreplayable Program operation owns this whole cycle.
- Physical execution is **CPU1, source rank0, B1 per-sample**. All sixteen
  source-rank controls are admitted; the other fifteen samples and distributed
  gradient reduction are **not executed** by this diagnostic adapter.
- Inputs are a finite stream of 2–100 fresh original cohorts. Reusing bytes with
  new names, task steps or seeds is not accepted as fresh data. No history,
  missing weight, conditioning or latent-resolution fallback is synthesized.

Install the separate `.[train,helios-reference]` dependency group in a compatible
environment; see [installation](../../docs/guides/installation.md). Actual
minimum-version CPU tests use Torch2.10, Diffusers0.37.1 and PEFT0.17.1. The
native `.34` helper and `.35` Wan lanes are separate environments.

## Supply the warm state and original inputs

Fill the [public asset envelope](assets.cpu-reference.template.json), the
[warm materialization envelope](bootstrap.cpu-reference.template.json) and
[input envelope](inputs.cpu-reference.template.json) with real absolute local
paths and content pins. Placeholders intentionally fail admission. Complete
original Args and the original capture configuration must agree; do not use an
unrelated scheduler horizon or an arbitrary scalar checkpoint.

The warm envelope references the original six-role native asset manifest and
LogicalCheckpointPackage v3, complete original safe base shards, and explicit
provenance. Its two accepted public provenance kinds are separate:

1. `OPERATOR_SUPPLIED_ORIGINAL_W2_STEP10`: an explicitly pinned operator-supplied
   original warm package. Its identity is admitted, not silently certified as
   a newly observed historical training run.
2. `ACTUAL_ORIGINAL_W2_CPU_TEST_WARM_NOT_HISTORICAL_STEP10`: the test suite really
   executes two original CPU No-GAN cycles from cold owners, using source
   cohorts beginning at10 and15, then serializes G2/C10/EMA10. Full input,
   action, forward/commit and payload receipts bind this fixture. It is **not**
   evidence for historical step0 inputs, a pretrained run or world16 reduction.

The actual-CPU warm receipt uses schema
`distillschedule.helios_nogan_actual_cpu_warm_receipt.v2`. It binds both executed
frozen base configurations and every safe weight shard in addition to the warm
trainables and optimizer state. A same-layout replacement base, an old v1
receipt, or a changed source is rejected before constructing or restoring the
training owners. Rebuild the actual fixture for its new source identity; do not
edit an old receipt to make it match.

The separate `STATE_READER_FIXTURE_ONLY` kind tests materialization from declared
synthetic gradients. It is rejected by the public training owner and cannot be
used to claim an original DMD update or warm training trajectory.

Use the existing native cohort producer to prepare original feature inputs,
then import that stream without replacing its sampler or noise:

```sh
distillctl data prepare-helios-nogan-reference --sources /YOUR/W2/inputs.json \
  --output /YOUR/W2/prepared
distillctl data validate /YOUR/W2/prepared/manifest.json --sources
```

The input adapter preserves step-qualified text/mask identities, all source
controls and payload hashes, and checks distinct source bytes. Each prepared
record owns one complete five-entry cycle, not one Critic update.

## Train and recover a complete boundary

Fill [the training template](training.cpu-reference.json), then use the existing
public commands:

```sh
distillctl config validate /YOUR/W2/training.json
distillctl train /YOUR/W2/training.json
distillctl train /YOUR/W2/training.json \
  --resume /YOUR/W2/run/checkpoints/cycle-000001 \
  --cycles 3 --output /YOUR/W2/resumed
```

Outer cycle0 is the genuine warm boundary G2/C10/EMA10; after outer cycle3 the
native clocks are G5/C25/EMA25. Checkpoints preserve named parameters, Adam
moments/options/clocks, LR, EMA, source cursor and native/outer RNG. They require
a completed public cycle and the same source, inputs, runtime and topology.
Warm outer0 snapshots can recover a failed first cycle in the state-owner API;
that does not publish a fictitious completed public cycle.

A partial G, Teacher, C or restore failure poisons the whole owner. It may not
retry an update in place. All-role validation precedes restore copies; rebuild
or restore a preceding complete state before continuing. Transient attention
caches are not durable optimizer state.

## Export the restored current G and use new inputs

```sh
distillctl export /YOUR/W2/training.json \
  --checkpoint /YOUR/W2/resumed/checkpoints/cycle-000003 \
  --output /YOUR/W2/exported
```

This uses the actual current **full base plus unmerged Generator LoRA**, not
Critic, Adam, a delta-only archive or EMA substitution. Projection is a pure
view: flags, buffers, modes, parameters, RNG and the next native update must not
change. Under the frozen EMA update-after750 policy, early EMA shadows equal
current G numerically; object/inventory selection, not an invented numerical
difference, distinguishes this export.

Prepare new BF16 `prompt_embeds` of shape `[1,512,4096]` with the ordinary
preencoded importer. Fill the [inference template](inference.cpu-reference.template.json)
using the export's exact inference factory/parameters, then run:

```sh
distillctl generate /YOUR/W2/inference.json --weights /YOUR/W2/exported \
  --prepared /YOUR/W2/NEW-inputs/manifest.json --output-kind latents \
  --output /YOUR/W2/new-latents
```

The independent factory constructs only the original G and strictly loads every
exported key before calling original `run_generator`. It needs matching source
and runtime plus the new prompt, not old bases, logical packages, training
inputs, Adam or checkpoint directories. Its explicit native zero-history
conditioning is an inference choice, not a claim of equality to training
history. Output is finite FP32 BCTHW `[1,16,9,48,80]`, as returned by the native
zero-history branch; it is not coerced to BF16 or relabeled as decoded video.
No VAE, raw-media encoder, pretrained quality, CUDA/FSDP parity or performance
claim is provided by this CPU reference.
