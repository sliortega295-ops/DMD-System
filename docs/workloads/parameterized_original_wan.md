# Parameterized ORIGINAL Wan optimization bridge

This opt-in layer joins PR29's event-frontier search to the **existing** Wan1.3B
Helios-pyramid no-GAN implementation. It does not substitute PR30's new
single-device Euler/Diffusers training protocol. The five canonical profiles
and the `workloads search --strategy model-based` interface are unchanged.

## Implemented vs validated

The original task schema routes through `workloads native`. It generates
world16 FSDP or Resident/async128/Tree candidates, modulation chunks 512/1024/2048,
and the existing Generator no-grad LoRA epilogue; FSDP's previously supported
control remains chunk512/epilogue-off. All choices use the original constructors,
optimizer configuration, loss, score calculation, sampler and EMA helper.

CPU tests cover capability generation, same-backend plan selection, capacity
rejection, explicit dispatch through original helper interfaces, and actual
modulation settings on all 30 block objects. A test doubles as an original
model-forward boundary to check shape/rank/stage-horizon propagation. These
are **not** full-model or CUDA tests. Multi-GPU execution, CUDA kernel correctness,
full-model numerical parity, speed and training quality remain NOT_RUN here.

## Configurable contract

`OriginalWanTask` supports resolution on the three-stage VAE/patch lattice
(height and width multiples of 64), frames of the form 4k+1, G1/Fn (1..64),
LoRA rank (1..256, matching the bootstrap), original update-boundary source
steps, per-stage denoising horizons, and declared EMA cadence.

The active pyramid sampler replaces the ordinary denoising list with each
stage scheduler's timesteps. Therefore **stage2_inference_steps** changes work;
the ordinary list is retained at its original value, not falsely advertised
as adaptable. Random exits still come from the declared trace: a horizon of
four per stage is not a claim that every rollout executes twelve calls.

Still fixed: Wan1.3B 30-block architecture, world16/two nodes of eight, physical
B1/global-B16, original precision/history semantics, three pyramid stages and
no-GAN loss. This does not parameterize Helios14B, H3, LiveWan, arbitrary pool
splits, backbone-sharing decisions, CPU offload or other kernels.

## Source data requirement

Changed tasks need **newly captured original-protocol traces and matching
bootstrap tensors**. `wan_trace.load_original_trace` verifies all sixteen ranks,
geometry, native branches, required tensors and per-stage call inventories.
Legacy Full5 traces are accepted only for the exact original default task.
There is no cropping, truncation, invented data, or relabeling of old traces.

This implementation consumes traces; it does **not** yet automate original
training-data capture or provide streaming fresh-cohort training. Worker cycles
replay the matched fixed cohort for calibration/diagnosis. Do not present that
as long training or as a new-task zero-adaptation demonstration.

New trace contract payloads include `distillschedule_original_wan_task` equal to
the complete task JSON, plus matching `resolution`, `stage_spatial_shapes`,
`full_latent_shape`, `dtype=bfloat16`, and `dfake_gen_update_ratio`. This binding
must be written during original capture, not added afterward to recycled data.

## Unified entry points

Inspect candidates without a model or GPU:

```bash
python scripts/distillctl.py workloads native validate \
  --task configs/distillschedule/original_wan/full5_reference.json
```

Edit a copy of `assets.template.json` to reference the existing local original
model, bootstrap, trace bundle and official negative-prompt embedding. Replace
both all-zero digest placeholders with the actual immutable asset hashes.

Generate a budgeted campaign in a **new shared directory**:

```bash
python scripts/distillctl.py workloads native prepare-campaign \
  --task configs/distillschedule/original_wan/full5_reference.json \
  --assets /shared/wan-assets.json --output /shared/wan-campaign
```

This prepares seven component-calibration arms and one selected-plan confirmation
within the default eight-launch budget. `--without-fusion --max-launches 5`
prepares the four reference-kernel arms plus confirmation. No GPU is launched
by preparation. This is a new explicit campaign, not a reset of older budgets.

Run the same prepared campaign on each existing node, changing only node-rank:

```bash
python scripts/distillctl.py workloads native campaign \
  --task /shared/wan-campaign/task.json \
  --campaign /shared/wan-campaign/campaign.json \
  --node-rank 0 --master-addr NODE0_HOST --master-port 29401 --execute
# On node 1, use --node-rank 1 and the same host, port and shared files.
```

The controller launches the same native CLI under torchrun, applies NCCL/kernel
settings before initialization, aggregates successful rank receipts, searches
observed implementations and memory-feasible orders, then runs the selected
plan. Failed arms remain failed and consume budget. Unknown interrupted
launches are not silently restarted. Node-local journals and per-cycle native
receipts are durable. `--resume` only skips completed, identity-matched work;
it never resets launch counts. Process timeout cleanup targets only the
controller's own child process group.

Individual `calibrate`, `assemble`, `plan`, and `run` subcommands remain available.
`plan --calibration` accepts one or more completed collection JSON files; source,
task, physical work, assets and costs must match. Missing costs are returned as
uncalibrated; no historical speedup constant or model-name winner is substituted.
Synthetic cost fixtures may exercise CPU selection but cannot launch a GPU run.

## Verification and performance scope

Worker checks actual backend reduction, per-block chunk settings, fusion
installation/invocations and trace-derived sampler call counts. Final trainable,
Adam and EMA states use the original evidence writers; the campaign compares
them with the matching FSDP control using declared tolerances. Out-of-tolerance
results stay REVIEW. PASS here still does not establish long-run video quality.

Calibration contains per-action synchronization and is diagnostic only. The
confirmation is a two-cycle fixed-trace run with one warm-up cycle; this layer
does not report a speedup using calibration durations as a formal baseline.

## Next blocking items

1. GPU-smoke the real original backend and compare its numerical state to the
   matched FSDP control, beginning with the original supported shape.
2. Integrate task-bound trace capture/data preparation before claiming that a
   changed resolution/frequency requires no manual input adaptation.
3. Extend the same capability/binding interface to the other four protocols;
   do not replace them with Euler or advertise unsupported model layouts.
