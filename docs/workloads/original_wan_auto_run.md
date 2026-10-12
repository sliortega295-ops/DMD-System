# Original Wan: new task inputs to existing optimization strategies

This entry continues PR #33 on the original Wan1.3B / Helios-pyramid no-GAN
protocol. It does not use the newer single-device Euler test protocol and does
not invent kernels. The old five canonical profiles remain unchanged.

## What now happens automatically

Given a valid `OriginalWanTask`, matching model initialization assets and new
pre-encoded features, one command per preallocated node validates source geometry,
prepares native history and sampled inputs, writes an immutable rank16 trace,
creates the existing seven-arm strategy campaign, calibrates admitted original
backends, selects a plan and runs native confirmation. No manual trace hash or
sequence of calibration/assembly/plan commands is needed.

Candidates are FSDP control or Resident + async128/Tree, chunks 512/1024/2048 and
eligible no-grad LoRA epilogue. FSDP remains chunk512/non-fused. All original
Generator, Score, Critic and EMA functions are retained.

## Inputs

`--task`: existing original-Wan task JSON. Resolution must be a multiple of 64,
frames 4k+1, G1/Fn 1..64, LoRA rank must match the initialization, and three stage
horizons are explicit. Actual exits remain sampled: eight total horizon positions
do not imply eight model calls on every rollout. Hardware remains 2x8 GPUs,
physical B1/global16. These are task requirements, not parameters changed to
obtain speedups.

`--sources`: see `source_features.template.json`. Sixteen distinct local `.pt`
or safetensors files, one per global rank, plus seed and cohort. Required keys:
`vae_latent` of shape `[sections,16,latent_frames,H/8,W/8]`, `prompt_embed` of
shape `[tokens,4096]`, and explicit binary `attention_mask` of shape `[tokens]`.
Text and mask can have a leading B1. Torch files use safe `weights_only` loading.
No VAE encoding, raw-video resizing, missing-mask reconstruction, old-trace
resizing or replacement-clip sampling is performed. New resolution requires
features actually encoded at that resolution.

`--model-assets`: `model_root`, `bootstrap`, `negative_prompt`, as in
`model_assets.template.json`. Existing original asset validators compute the
hashes. Changing rank needs a matching explicit initialization; trained adapters
are not silently reset or converted.

`--training-config`: the original template's input-preparation settings. Only
implemented history dropout/corruption and timestep sampling settings can differ;
optimizer, loss and scheduler changes that the worker would ignore are rejected.
Shape, frequency, rank and stage horizons come from the task, not inactive YAML
fields. GT-history does not apply the non-GT saturation augmentation.

## Native branch boundary

The existing measured world16 backend admits the normal globally synchronized
pyramid schedule. The upstream all-zero GT-history branch samples amplified exits
independently per rank. Capture refuses that branch without resampling, changing
dropout or relabeling it as the normal path. Empty initial history and T2V history
dropout can therefore fail admission. Do not search for a passing seed and present
it as the original data distribution. Supporting this branch requires a separate
collective-safe contract; it is not fixed by dropping validation.

The explicit operation-qualified CPU/Python RNG streams define a new paired
experiment, not byte-identical continuation of an old CUDA RNG stream. Comparison
arms consume the same captured inputs.

## One command per node

Both already allocated nodes must see identical source and shared absolute data
and output paths. Run on each node, changing only `--node-rank`:

```bash
python scripts/distillctl.py workloads native auto-run \
  --task task.json --sources source-features.json --model-assets model-assets.json \
  --training-config configs/distillschedule/templates/helios_stage3_384x640.yaml \
  --output /shared/new-wan-task \
  --node-rank 0 --master-addr NODE0_HOST --master-port 29401 --execute
# Second node: --node-rank 1, same host, port, input and output paths.
```

Without `--execute`, inputs are validated but no campaign/GPU is started. The
default budget is seven component arms plus one selected-plan confirmation,
eight distributed launches total. `--without-fusion --max-launches 5` requests
four non-fused arms plus confirmation. An insufficient budget is rejected before
capture, not used to truncate candidates silently.

`--resume` retains campaign journals and budgets. Unknown interrupted native
launches are not automatically repeated. Failed input preparation is preserved;
no automatic resampling or trial-counter reset. Readiness is atomic. There is
no SSH, scheduler submission or allocation of new GPUs.

Separate `capture-inputs`, `prepare-task`, `prepare-campaign`, `calibrate`,
`assemble`, `plan` and `run` commands remain available. Input capture/preparation
uses the same flags above without rendezvous arguments, and requires `--execute`
to write new captured inputs.

## Evidence

CPU tests call the actual native dataset/history/scheduler functions and replay
captured schedules through the original sampler with an injected model boundary.
Controller tests inject workers. Neither is a pretrained-model, GPU-collective,
quality or speedup result. Full-model GPU validation remains NOT_RUN. Numerical
REVIEW remains REVIEW, not PASS.

Campaign cycles are paired fixed-cohort diagnostic runs, not streaming fresh
long-training data. Helios14B, H3, LiveWan, GAN, arbitrary GPU pool layouts and
automatic shared-backbone choices are not parameterized by this entry.
