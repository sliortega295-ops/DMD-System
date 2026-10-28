# Direct W2 reference training (version 1)

This separate opt-in route uses **your content-pinned original Step10
G2/C10/EMA10 package** and newly prepared native feature cohorts. It does not
need the historical benchmark's SSH node manifest, HF cache, flash-attention
asset, or original trace bundle. No checkpoint is supplied, downloaded, or
manufactured. This is not cold-start Step0 or an arbitrary long-run optimizer.

## Supply the actual assets

Copy [`assets.reference.template.json`](assets.reference.template.json) to your
project as `assets.reference.json`. Replace every path and SHA256 with actual
files. The independently versioned `distillschedule.helios_reference_assets.v1`
contract has exactly six native roles:

- Generator root (`transformer_ode/config.json`) and its BF16 pre-cast package.
- Score root (`transformer/config.json`) and its BF16 pre-cast package.
- Original `distillschedule.logical_checkpoint.v3` bootstrap, including FP32
  master/Adam moments, original scheduler state and sparse EMA.
- Original negative-prompt asset (official text, BF16 `[1,512,4096]` embedding).

Every role pins its actual identity file. The logical role also explicitly
pins `ema/sparse_ema_manifest.json` (or `model_ema/sparse_ema_manifest.json`,
matching the original loader's first-existing-directory rule). This binds
the EMA decay/update settings and shard hashes even when the original logical
manifest does not itself hash the EMA manifest. The bootstrap must retain
global step 10, G2/C10 parameter and optimizer versions, matching scheduler
epochs and EMA10. Its embedded manifest hash can be genuinely new; it is not
compared with a historical experiment hash or relabelled to match fresh inputs.

The existing pre-cast utility, logical format and loaders remain authoritative.
Before model construction, all ranks verify actual pre-cast shard bytes,
source config/index, logical master/moment payloads, EMA shards, negative
conditions and local input payloads. Identity-file-only CPU admission is not a
pretrained-model or numerical correctness certificate.

## Prepare explicit fresh inputs on CPU

Use [`sources.reference.template.json`](sources.reference.template.json) as an
inventory template. Each cohort has an explicit seed and five lists of sixteen
**distinct actual pre-encoded feature files** (one per source step/rank). There
are 80 source files per cohort. Supply at least 2 and at most 100 cohorts; never
repeat a feature set or substitute an old captured trace. Each file contains
native `vae_latent`, `prompt_embed` and `attention_mask` data. Use your original
feature encoder/data preparation; this command does not turn raw video into
model features or synthesize training data.

[`inputs.reference.yaml`](inputs.reference.yaml) contains only the original
input-capture settings, without historical model/dataset paths. Input settings
are explicit and source-bound. The currently supported branch is normal,
globally synchronized No-GAN 33-frame/222, world16/B1. Unsupported amplified or
rank-local stage/history branches fail closed; they are not resampled, padded,
or silently replaced. Illustrative template seeds do not promise that your
source data samples a supported branch.

```sh
distillctl data prepare-helios-stream \
  --sources /absolute/project/sources.reference.json \
  --training-config /absolute/project/inputs.reference.yaml \
  --output /absolute/project/prepared
```

Preparation uses the original dataset/history/noise/scheduler helpers. Cohorts
start at steps 10, 15, 20, ...; trace IDs and RNG keys remain step/rank qualified.
The first **newly prepared** cohort supplies startup controls only under this
explicit reference lifecycle, never under legacy benchmark/canonical loading.

## Validate, run, resume

Copy [`training.reference.json`](training.reference.json), the unchanged
[`task.json`](task.json), and your asset manifest into your own project.
Set the prepared `stream.json`, matching cohort count, checkpoint cadence and
a **new** output directory outside the source checkout.

```sh
distillctl config validate /absolute/project/training.reference.json
distillctl train /absolute/project/training.reference.json --dry-run
```

Actual training is entered on every existing torchrun rank: exactly two real
nodes of eight CUDA ranks each, with matching rank metadata and NCCL/Gloo
availability. The operator owns allocation/rendezvous and invokes
`distillctl train /absolute/project/training.reference.json` on each rank using
their normal launcher. No SSH launcher or node reservation is run by this
route. Enter on fresh ranks, not an embedding application's initialized group.
The public wrapper owns the default NCCL and full-world Gloo metadata groups;
original P0 explicitly adopts only that exact group and does not destroy it.
The wrapper collectively admits source/config/assets/stream files and imports,
then P0 collectively admits startup and immutable payloads before model build.

Original P0 still owns model construction, BF16/FP32 policy, SDPA, original
G/C AdamW and schedulers, loss, gradient clipping and EMA recurrence. Each
complete cohort executes the existing G1/C5/EMA5 preparation/dispatch helpers.
No measured winner, selected plan or optimized implementation is inherited.

Resume only an incomplete stream at a published complete-cycle checkpoint:

```sh
distillctl train /absolute/project/training.reference.json \
  --resume /absolute/project/prior-run/checkpoints/cycle-000001 \
  --output /absolute/project/new-resume-run
```

Resume requires identical source, stream, immutable asset pins/verified bytes,
topology and reference policy. It restores trainables, actual Adam/schedulers,
EMA, clocks and per-rank runtime RNG. Wrong G/C/EMA boundaries and foreign
identities reject before owned state is changed. This finite Step10 bootstrap
route does not implement changing the data/horizon on resume, arbitrary Step0
training, or selected-plan execution. Public export restores the explicit
checkpoint without training updates; generation uses the exported Student.

## Stop at a durable checkpoint without changing the stream

For a prepared 100-cycle stream, keep `trainer.cycles: 100` and set
`trainer.checkpoint_every: 50` plus `trainer.stop_after_cycle: 50`. The first
process executes only cycles 1–50 and returns after the complete all-rank
checkpoint is durable. It does not execute cycles 51–100 and replay them later.
The boundary remains G1/C5/EMA5 for both explicit reference and SafeBundle.

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

A graceful early stop has a distinct `STOPPED_*_AT_CHECKPOINT` status, not the
complete-run status. `training_segment` records the full horizon, start,
requested stop, completed cycle and actual executed count. Neither an early
segment nor a resumed segment is formal performance or 100-cycle GPU evidence;
native GPU reliability/quality acceptance remains **NOT_RUN**.

## Evidence boundary

The tests cover real original producer/consumer/update/checkpoint helpers with
tiny CPU DTensor owners, plus actual public-to-P0 admission and two-process
Gloo fault propagation. The CPU test substitutes only topology/CUDA resource
surfaces and stops at an explicitly synthetic model boundary. It is **not**
world16/FSDP/CUDA/NCCL/pretrained quality, capacity or performance evidence.
Actual cluster validation remains **NOT_RUN**. Original fixed benchmark,
canonical plan/source and numerical/capacity checks are not relaxed.
