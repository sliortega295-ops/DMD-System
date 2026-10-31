# Train Helios from a finite feature pool

W2 No-GAN and W3 GAN can reuse an immutable dataset across epochs through
`distillschedule.helios_record_epochs.v1`. Each complete cycle still consumes
80 distinct records; a later cycle may revisit those records with a new logical
draw identity and the original operation-qualified history/noise sampling.
You do not need 8,000 different source files to run 100 cycles.

## Prepare the pool

Encode your source videos using the model's existing encoder. Each `.pt` or
`.safetensors` record must contain native `vae_latent`, `prompt_embed` and
`attention_mask` tensors at the recipe's original geometry. Pin at least 80
different records in a JSON document with this structure:

```json
{
  "schema_version": "distillschedule.helios_record_pool.v1",
  "seed": 42,
  "cycles": 100,
  "evidence_kind": "ENCODED_FEATURES_UNVERIFIED_QUALITY",
  "records": [
    {"id": "clip-000", "path": "/absolute/data/clip-000.pt", "sha256": "<file SHA256>"}
  ]
}
```

The single record above illustrates the fields, not a runnable one-record
dataset. Supply the complete inventory; copied files, aliases and identical
native content do not count as different records. Preparation keeps source
files unchanged and checks their bytes and shapes before native capture.

For W2, use the existing input settings from
[`inputs.reference.yaml`](../../recipes/helios_nogan/inputs.reference.yaml):

```sh
distillctl data prepare-helios-stream \
  --input-policy distillschedule.helios_record_epochs.v1 \
  --sources /absolute/project/pool.json \
  --training-config /absolute/project/inputs.reference.yaml \
  --output /absolute/project/prepared
```

For W3, use `prepare-helios-gan-stream` instead and the unchanged native GAN
configuration from the [GAN stream guide](../../recipes/helios_gan/NATIVE_STREAM.md).
Model assets, update rules, precision and supported branch restrictions remain
those of the selected recipe. Without `--input-policy`, the original explicit
single-pass cohort policy remains in effect.

## Train and resume

Point the existing reference or optimized training configuration to the new
`stream.json`, keeping `trainer.cycles` equal to the prepared horizon. For a
50/100 boundary, set `checkpoint_every` and `stop_after_cycle` to 50. After the
complete-cycle checkpoint is published, start a new process with the same
configuration, source, topology, assets and stream; remove `stop_after_cycle`
and resume that checkpoint into a new output directory.

The receipt records record count, logical draw count, epoch policy hashes and
the next draw cursor. Dataset order or policy changes require a newly prepared
stream, not relabeling an old checkpoint. The detailed format is in the
[versioned contract](../contracts/original_helios_record_epochs_v1.md).
