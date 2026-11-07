# Train original native and official workflows

The native adapter lowers public requests to existing original-model owners.
It does not replace algorithms, choose a historical winner or allocate devices.
Use a clean source checkout and the recipe's separate dependency lane. Original
benchmark presets remain two-cycle research bindings; W5's optimized backward
numerical FAIL remains unresolved. Fresh-stream lifecycles are separate.

## Start with an explicit request

Each recipe contains a portable `training.json` and its original task/request
binding. Paths are relative to that configuration, not the current directory.
Replace every `REPLACE_WITH_*` value with your own qualified input. Unfilled
placeholders and missing files produce errors before execution. No weights,
datasets, source snapshots, or credentials are downloaded by these recipes.

```bash
distillctl doctor --config recipes/helios_nogan/training.json
distillctl train recipes/helios_nogan/training.json --dry-run
# Only after admission, allocation and operator authorization:
distillctl train recipes/helios_nogan/training.json
```

The Python equivalents are:

```python
from distillschedule.training import load_training_config
from distillschedule.training.native import describe_native, validate_native, run_native

config = load_training_config("recipes/helios_nogan/training.json")
describe_native(config)              # catalog/capabilities only; opens no assets
validate_native(config)              # actual CPU native admission, no CUDA/model construction
run_native(config, dry_run=True)     # validated lowering, no process or output
run_native(config)                   # existing runtime; not an exit-code -> parity claim
```

Do not run the last line on an unallocated cluster. Real execution requires a
clean committed checkout. `describe_native` and CPU admission are not evidence
that a 1.3B/14B model trained, that CUDA is available, or that quality/parity
passed. A source/layout change requires new source-bound admission and costs.

## Original mathematical and execution boundaries

| Recipe | Original algorithm | Real API | Admitted cycle count |
| --- | --- | --- | --- |
| [W1](../../recipes/wan_ar/training.json) | Wan hierarchical AR, No-GAN, default G1/C5/EMA5 | `strategies.wan_cli.main` -> `wan_worker.run_worker` | 2..100 fresh cohorts with a compatible saved policy or explicit manual/reference strategy; literal incumbent bundle remains fixed-trace/two-cycle |
| [W2 optimized](../../recipes/helios_nogan/training.json) / [direct reference](../../recipes/helios_nogan/training.reference.json) | Helios14B hierarchical AR, G1/C5/EMA5 | Preset/plan: `search.execution.run_plan`; direct reference: original P0 setup and preparation/update helpers | Preset/plan: 2. Direct reference: 2..100 fresh cohorts, explicit original Step10 G2/C10/EMA10 bootstrap |
| [W3](../../recipes/helios_gan/training.json) | Helios14B DMD + GAN-G/GAN-D, G1/C5/EMA1 | `search.execution.run_plan` | Exactly the selected native contract's 2 |
| [W4 optimized](../../recipes/self_forcing/training.json) / [original reference](../../recipes/self_forcing/REFERENCE_TRAINING.md) | Self-Forcing, continuous sigma, G1/C5/EMA0 | Preset/plan: `search.execution.run_plan`; reference: original baseline ModelBundle and update helper | Preset/plan: 2. Original reference: 2..64 finite teacher cohorts, explicit genuine zero-step assets |
| [W5 optimized](../../recipes/h3_fullseq/training.json) / [original reference](../../recipes/h3-fullseq-original-training.md) | Full-sequence AV continuous H3, 4-step, G1/F5, LoRA128 | Preset/plan: `search.execution.run_plan`; original reference: `entrypoint` and original continuous trainer | Preset/plan: 2. Original reference: 2..64 finite cached-condition cycles |
| [W6](../../recipes/h3_ar/training.json) | Original DiffSynth previous+current AV, continuous sigma, 8-step, G1/F1, LoRA64, ZeRO-3 | `strategies.h3_ar_cli.main` -> `h3_ar_worker.run_worker` | Exactly 2 native updates |

The table above names the original benchmark configs, not the separately
versioned continuous-training configs below. Raising `cycles` on a two-update
benchmark is rejected rather than silently truncated. Shapes, objective,
precision and update counts stay owned by each original task. W5 and W6 both use continuous sigma, with separate full-sequence and AR owners.

## Continuous native training

Use these explicit recipes for persistent training and complete-cycle restart.
They reuse the original native owners; the frozen benchmark configs are not
modified. Direct training does not need a saved plan.

| Workload | Executable recipe and owner | Admitted cycles | Checkpoint / recipe guide |
| --- | --- | --- | --- |
| W1 | [incumbent stream](../../recipes/wan_ar/training.incumbent-stream.json); `wan_worker` → persistent `wan_session` | 2..100 fresh cohorts | Named G/C Adam, scheduler, EMA and RNG; [INCUMBENT_STREAM.md](../../recipes/wan_ar/INCUMBENT_STREAM.md) |
| W2 | [SafeBundle stream](../../recipes/helios_nogan/training.safe-stream.json); `helios_worker` | 2..100 fresh cohorts | Original native state and warm clocks; [SAFE_STREAM.md](../../recipes/helios_nogan/SAFE_STREAM.md) |
| W3 | [full GAN stream](../../recipes/helios_gan/training.stream.json); `helios_gan_training` | finite prepared stream, including 100 cycles | Full G/C/GAN-head Adam and EMA; [NATIVE_STREAM.md](../../recipes/helios_gan/NATIVE_STREAM.md) |
| W4 | [full C51 stream](../../recipes/self_forcing/training.c51-stream.json); `livewan_c51_training` | 2..100 fresh cohorts | Two-pool drain, role state and wave cursors; [C51_CONTINUATION.md](../../recipes/self_forcing/C51_CONTINUATION.md) |
| W5 | [original continuous reference](../../recipes/h3_fullseq/training.original-reference.json); `h3_fullseq_training` | 2..64 finite cached-condition cycles | Both LoRAs, original Adam/schedulers, sampler and RNG; [h3-fullseq-original-training.md](../../recipes/h3-fullseq-original-training.md) |
| W6 | [native AR stream](../../recipes/h3_ar/training.native-stream.json); `h3_ar_training` | 2..100 fresh cycles | Original ZeRO-3 model/master Adam and native clocks; [NATIVE_CONTINUATION.md](../../recipes/h3_ar/NATIVE_CONTINUATION.md) |

W2 also has a direct-reference fresh stream. W4's original-reference input owner
is not full C51. W3's explicit
[native-history policy](../contracts/helios_native.md#helios_gan_native_history_v1)
keeps full GAN and original detached-rollout persistence; full-size capacity and
restart require their own native evidence.

### Native current-Student output

| Workload | Output path and unchanged computation |
| --- | --- |
| W1/W2/W3 | [W1](../../recipes/wan_ar/NATIVE_OUTPUT.md), [W2](../../recipes/helios_nogan/NATIVE_OUTPUT.md), [W3](../../recipes/helios_gan/NATIVE_OUTPUT.md): native complete restore, current unmerged G, real Wan2.1 VAE; W3 keeps GAN restore and N70 OFF |
| W4 | [Full C51](training.md#livewan_native_output): both role pools restored, only current G exported; original four-step streamer |
| W5 | [Full-sequence output](training.md#h3_fullseq_inference): current LoRA plus four frozen AdaLN entries and converted-model four-step sampler |
| W6 | [AR output](training.md#h3_ar_native_inference): original eight-step previous/current AV sampler; [new raw AV inputs](data.md#h3_audio_video_encode) use original encoders/text features |

Exports are inference weights, not resumable checkpoints. Reuse recorded runs
of50+ complete cycles, including W1 continuation and LiveWan100-cycle records;
do not repeat long training to check a changed endpoint. Source/CPU output or
restart coverage does not complete a missing native GPU endpoint.

## Choose reference, manual or saved execution

- W1 [reference](../../recipes/wan_ar/training.reference.json) supplies the
  [original FSDP strategy](../../recipes/wan_ar/reference_strategy.json), lowering
  to `benchmark-manual --implementation ... --cycles 2`.
- W6 [reference](../../recipes/h3_ar/training.reference.json) selects `zero3-native`;
  AP is only explicit `zero3-ap` or a measured selected plan.
- W2 [reference](../../recipes/helios_nogan/REFERENCE_TRAINING.md), W4
  [zero-step reference](../../recipes/self_forcing/REFERENCE_TRAINING.md) and W5
  [original-config reference](../../recipes/h3-fullseq-original-training.md) do not
  inherit a historical optimized winner.
- W2-W5 presets use manual mode. For resolved reference candidates and measured
  controls use [native calibration](../runbooks/native_planner_workflow.md).
  W4 C51 requires `c51_full_v1`; C34 is never a fallback.
- Plan mode consumes the saved selection unchanged: no search/rebind/run-ID
  rewrite. Use the native replay/rebind workflow first. W2-W5 recompile node
  commands and validate candidate/profile/semantics/request/assets; W1/W6 verify
  source/cost/order/version/capacity. CPU fixture costs cannot launch GPU work.

W1/W6 commands execute inside operator-owned world16 torchrun ranks on two
8-GPU nodes. W2-W5 use their native dispatcher/node manifest. No rendezvous,
visibility, allocator or ZeRO fallback is invented by the public training API.
See [Wan planning](../runbooks/native_planner_workflow.md#wan_compute_state_planning_v1) and
[W6 planning](#original_h3_ar_native).

## Required inputs and portable configuration

| Field | Required content |
| --- | --- |
| `model.workload` | W1-W6; not a best-profile ID or an arbitrary model factory |
| `execution.task` | Original Wan/H3-AR task JSON (W1/W6), `OriginalHeliosRequest` (W2/W3), `LiveWanRequest` (W4), or `H3Request` (W5) |
| `execution.assets` | Exact native asset JSON; selected W2-W5 must be byte-equivalent as JSON to the saved plan's `asset_config`. W2/W5 direct references use their separately documented asset schemas, not that dispatcher manifest |
| `execution.plan` | Selected source-bound native plan for plan mode; omitted for explicit reference/manual mode |
| `execution.preset` | Explicit versioned manual preset, no saved plan or implementation override; frozen benchmark is two-cycle, fresh-stream presets use the explicit limits above |
| `execution.implementation` | Explicit Wan strategy JSON path or W6 `zero3-native` / `zero3-ap` enum; reference/manual only |
| `execution.capacity_policy` | W6's existing common host/GPU capacity-policy JSON (equal to saved plan in plan mode); W4 direct reference requires its explicit versioned common capacity policy |
| `execution.transport` | Optional existing SHA-bound reviewed R9 deployment for its supported W5 subspace; never a generic shell-command transport |
| `execution.expected_head` | Optional exact source HEAD assertion; saved-plan source checks are always mandatory |
| `trainer.output_dir` | Explicit project output directory; existing native ownership/non-overwrite rules remain active |
| `trainer.cycles` | Actual supported native count, checked against selected contract; not a timing normalization |
| `trainer.checkpoint_every` | Explicit fresh-input lifecycles support complete-cycle saves, including full GAN/C51/native AR; frozen benchmark bindings require `0` |

Preset assets must match that preset; plan assets must equal saved `asset_config`
as JSON. Native asset paths follow their original API, usually absolute, not
generic request-path rules. Use original Wan assets (source checkout:
`configs/distillschedule/original_wan/assets.template.json`), W6
`workloads native prepare-assets` and the linked recipes. Actual source/runtime,
weights/trace/shape, topology and capacity are checked before model construction.

| Lifecycle | Required prepared input and seed/state |
| --- | --- |
| W1 fresh | `original_wan_cohort_stream.v1`, exact distinct cohort count; compatible selected/manual/reference policy, not historical timing. Fixed replay instead omits prepared/kind and explicitly sets `replay_fixed_trace:true` |
| W2 reference | `original_helios_cohort_stream.v1`, five rank/step feature sets per cohort; used model/precast/bootstrap/negative/EMA assets and warm G2/C10/EMA10, not cold Step0 |
| W3 | Full GAN/native-history stream and its own checkpoint identity; never W2 relabeling |
| W4 reference | `original_livewan_teacher_cohorts.v1`, `algorithm.name: original_self_forcing_reference.v1`, reference mode, actual zero-step G0/C0/EMA0 assets, exact finite horizon/seed/capacity |
| W5 reference | `original_h3_fullseq_latent_dataset.v1`, genuine cached conditions, resolved native config/source/dependencies/order and exact original seed; continuous sigma objective |
| W6 | Versioned per-rank cache/parser/source metadata, including rank6 audio94. Cache roots are configurable; original filenames, checksums and tensor metadata stay bound to the input contract |

Native streams are not generic PreparedDataset/statistics-encoder manifests.
Unrecognized prepared data or RNG overrides reject, not get ignored. Only the
explicit W4/W5 bindings use their separately supplied original seed; otherwise
source/bootstrap/stream owns RNG. Reprepare source-bound inputs for changed
geometry or RNG; do not rehash captured old traces.

## Actual-original CPU references and model provider

| Reference | Owner and lifecycle |
| --- | --- |
| [W1 CPU Full5](../../recipes/wan_ar/CPU_REFERENCE.md) | Original parameterized cycle/P0, mixed BF16/FP32 and FP32 LoRA, cold-zero Adam/EMA, atomic nonreplayable Program call; common checkpoint/current-G export; CPU1 source-rank0 is not global-B16 reduction parity |
| [W2 warm No-GAN](../../recipes/helios_nogan/CPU_REFERENCE.md) | Full constructor, original mixed precision/LoRA256 and real G2/C10/EMA10; original updates and common restart/current-G output, not CUDA/world16 qualification |
| [W3 GAN CPU](../../recipes/helios_gan/REFERENCE_TRAINING_CPU.md) | Full original transformer/PEFT/history19, actual G1/C5/GAN/EMA curriculum; `data prepare-helios-gan-reference`, train/resume and [unmerged-G output](../../recipes/helios_gan/PORTABLE_INFERENCE_CPU.md) |
| [W6 cached AV CPU](../../recipes/h3_ar/CPU_REFERENCE.md) | Licensed cached-only original50-block/8-step/G1F1/LoRA64 trainer/Adam, flat or split parser; `data prepare-h3-cached-reference`, train/resume and [Student output](../../recipes/h3_ar/PORTABLE_INFERENCE_CPU.md) |

These FP32/CPU1 references require genuine pinned local assets, immutable finite
inputs and explicit configs; unsupported distributed/auto/manual/plan/epoch
modes reject. Deterministic CPU fixtures exercise these interfaces.
The cached-only derivative retains per-file origins/AST/license,
not a claim to redistribute a full DiffSynth install or silently stub codecs.

The separate full Helios model provider imports the unchanged model:

```python
from distillschedule.adapters.helios.reference_model import (
    load_reference_model_class, reference_model_identity,
)

identity = reference_model_identity(backend="torch-sdpa")  # No Torch/model import.
HeliosTransformer3DModel = load_reference_model_class(backend="torch-sdpa")
```

Use `[train,helios-reference]` (Diffusers0.37.1), not the separate .34 test or
.35 native-Wan lane. Both APIs recheck all83 frozen files, manifest/backend bytes
and private imports. Four private modules compile admitted source directly,
never timestamp-valid/unchecked bytecode; the scoped importer is removed and
failed new namespaces cleaned for retry. Only explicit `torch-sdpa` uses the
original dense fallback; NaViT/varlen refuses. Identity discovery imports no
Torch; class loading constructs no model, optimizer or download. Native CUDA
`helios.*` imports remain separate.

## Complete-cycle recovery and evidence

```json
{"trainer": {"cycles": 6, "checkpoint_every": 2, "output_dir": "training-run"}}
```

For W1 merge those fields into the full six-cohort request. Resume from
`training-run/checkpoints/cycle-000002` into a new output with the same full
stream/target/source/bootstrap/runtime/layout. Use shared persistent storage and
explicit torchrun `GROUP_RANK`; W4/W5/W2 fixed-horizon owners retain their own
restrictions, not a general target-extension promise.

The original owner saves trainables/buffers, named Adam/groups/moments,
schedulers, EMA/counter, native clocks, absolute cursor and per-rank
Python/NumPy/Torch/owned-device RNG only at drained zero-gradient boundaries.
All rank files are hashed before manifest/`_SUCCESS`; frozen bases rebind by
hash and temporary caches rebuild. State I/O stays outside native compute
windows. Final-state fingerprints or `wan_worker.save_final_state` are evidence,
not a complete resumable checkpoint.

Use each recipe's source, shape, update and capacity checks.
[Performance measurements](performance_validation.md) retain their original timing scope.

<a id="native_presets"></a>
## Direct native optimization presets

For example, the `execution` section of [W3](../../recipes/helios_gan/training.json):

```json
{
  "adapter": "native",
  "mode": "manual",
  "preset": "helios_gan.fa3_down40_nomerge.v1",
  "task": "task.json",
  "assets": "qualified-native-assets.json"
}
```

Replace asset placeholders and choose a new output directory outside the source
checkout, then inspect and launch on the allocated devices:

```bash
distillctl doctor --config /project/recipe/training.json
distillctl train /project/recipe/training.json --dry-run
distillctl train /project/recipe/training.json
```

W1/W6 use world16 torchrun across two eight-GPU nodes. W2–W5 use the native
two-node dispatcher and the supplied node manifest.

### Preset mapping

| Workload / preset | Main mechanisms | Continuation |
| --- | --- | --- |
| W1 `wan_ar.incumbent21a.v1` | Resident, chunk2048, LoRA/norm fusion, G rollout/cast and C gradient graphs; retain4 disabled | [Incumbent stream](../../recipes/wan_ar/INCUMBENT_STREAM.md) |
| W2 `helios_nogan.safe_ed8.v1` | Node-HSDP, chunk512/window5, SafeBundle normalization/LoRA/replay reuse; warm No-GAN bootstrap | [Safe stream](../../recipes/helios_nogan/SAFE_STREAM.md) |
| W3 `helios_gan.fa3_down40_nomerge.v1` | World16 FSDP, chunk2048/window5, SM80 FA3, down40 replay, separate GAN heads; LoRA unmerged | [GAN stream](../../recipes/helios_gan/NATIVE_STREAM.md) |
| W4 `self_forcing.c51.v1` | Full C51 8G+8Score pipeline with C34 prerequisites | [C51 continuation](../../recipes/self_forcing/C51_CONTINUATION.md) |
| W5 `h3_fullseq.cpu_staged_trim.v2` | continuous H3 CPU-staged trimmed FA3 replay, depth2/N128 | [Original-reference lifecycle](../../recipes/h3-fullseq-original-training.md) |
| W6 `h3_ar.zero3_ap.v1` | ZeRO-3, selected-cache/unit prefetch, CPU-staged replay, fusion and communication overlap | [Native AR continuation](../../recipes/h3_ar/NATIVE_CONTINUATION.md) |

The original six presets require exactly two native cycles,
`checkpoint_every: 0`, no resume and the original fixed-input contract. They
provide quick comparisons and benchmark reproduction. For long training and
checkpoint recovery, choose the corresponding continuation entry above.

Use only manual+preset, not `plan`, `implementation` or `transport` together.
`training.native_presets` owns typed factories/current-source binding; reference
and saved-plan selection remain explicit. Original worker numeric/state/capacity
checks still apply.

<a id="official_optimized_training"></a>
## Official Self-Forcing and DMD2 optimized training

The complete public profiles share `distillctl` train/checkpoint/resume metrics.
FSDP2 is the primary baseline for both. On16×A100-40GB/globalB16, the original
unprofiled C2 window (C1 warmup), not a new long-run average, reports:

| Workflow | FSDP2 seconds | Optimized seconds | Speedup |
| --- | ---: | ---: | ---: |
| DMD2 SD1.5 GAN512×512, G1/C10 | 7.514 | 3.475 | 2.162× |
| Official Self-Forcing368×640×81, G1/C5 | 50.739 | 30.959 | 1.639× |

Auxiliary DMD2 Resident/DDP is6.114→3.475s(1.759×). Both optimized public paths
also completed100cycles, a cycle50 checkpoint/fresh-process continuation,
same-source world16 Student export and independent new-prompt image/video;
output tensor/PNG and both-node cleanup checks passed. Original fixture/data
scope, later longer-cycle records and exact receipt identities remain in the
[validation appendix](performance_validation.md#official-dmd2-and-self-forcing) and
[performance guide](performance_validation.md), not reinterpreted here.

| Workflow | Explicit reference | Complete optimized |
| --- | --- | --- |
| SD1.5 | `training.fsdp2.json`; single-device FP32 `training.gan.json`; supplemental FP32 `training.ddp-fp32.json` | [DMD2](../../recipes/dmd2_sd15/README.md) `training.optimized.json` |
| Official Self-Forcing | `training.json` / native `training.fsdp2-reference.json` | [Self-Forcing](../../recipes/self_forcing_official/README.md) `training.optimized.json` |

Fill owned source/assets/input/output paths and use separate environments.
Self-Forcing retains upstream FA2/compatible Triton; encode prompts offline,
not T5 on workers. The distinct DMD2 native reference is below.

```bash
pip install -e '.[dmd2]'
# Or, in the Self-Forcing environment:
pip install -e '.[self-forcing-official]'
```

```bash
distillctl config validate /path/to/training.optimized.json
distillctl doctor --config /path/to/training.optimized.json
distillctl capabilities show official-dmd2-resident
distillctl capabilities show official-self-forcing-fsdp2
```

```bash
# DMD2: set these before torchrun initializes the NCCL group.
export NCCL_ALGO=allreduce:Tree
export NCCL_MIN_CTAS=16

torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m distillschedule train /path/to/dmd2.training.optimized.json
```

```bash
# Set this before torchrun initializes the NCCL group, for either FSDP2 recipe.
export NCCL_CROSS_NIC=1

torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m distillschedule train /path/to/self-forcing.training.optimized.json
```

```bash
# Use the same torchrun topology/environment as the initial invocation.
# Replace the final `train ...` arguments with:
distillctl train /path/to/training.optimized.json \
  --resume /path/to/run/checkpoints/cycle-000100 \
  --cycles 200 --output /path/to/resumed-run
```

Launch on each of two8-GPU nodes with one resolved config. DMD2 requires the
explicit Tree/min16 settings above **before** group initialization. Self-Forcing
FSDP2 reference/R6 instead requires `NCCL_CROSS_NIC=1`; do not inherit DMD2's
settings. The library validates, never mutates, that environment.

`trainer.cycles` is the final target; `checkpoint_every` sets cadence. Resume
on the same topology. Native role/Adam/RNG/data/clocks persist, while KV/replay
caches rebuild. The complete profiles, not independently searched toggles, bind:

- `dmd2-sd15-resident-tree-v1`: resident ready-bucket DDP, fused AdamW with FP32
  masters/moments, retained Fake epsilon, packed classifier trunks with original
  separate B1 heads/losses, update-version-scoped BF16 casts. No-grad G/heads
  remain FP32, gradient/Teacher BF16 and VAE FP16.
- `self-forcing-wan-fsdp2-r6-v1`: native FSDP2 phase/text-KV retention, Critic
  attention/FFN-down/projection replay reuse, Score-before-KV release and G
  backward/commit release, host integer positions, context hoisting/fusions and
  paired conditional/unconditional Teacher block gathers.

Direct Program execution calls the native owner; it requires no old benchmark
script/run directory/plan. Official Self-Forcing is not W4 LiveWan.

<a id="dmd2_fsdp2_reference"></a>
## SD1.5 DMD2: native FSDP2 reference

Use `recipes/dmd2_sd15/training.fsdp2.json` for the SD1.5 reference that shards
Generator and the native guidance tree (Teacher, Fake Score and classifier)
across 16 A100 GPUs. Use `training.optimized.json` for the current optimized
Resident/DDP combination. Both use `distillctl train`, the same upstream DMD2
model/loss classes and the common continuous-training/checkpoint lifecycle.

### Train and resume

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

### Contract

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

The baseline's exact placement/classifier ABI and source checks are retained.
Use [current public validation](performance_validation.md#official-dmd2-and-self-forcing), rather
than treating a CPU FSDP/Adam fixture as new timing or pretrained GPU evidence.

<a id="original_h3_ar_native"></a>
## W6 task-native planner workflow

Use the [versioned H3 contracts](../contracts/h3_native.md) and one clean source
for CPU/I/O admission, calibration, save and execution. The external coordinator
owns allocation, two-node torchrun/UUID/rendezvous, leases, cancellation, resource
sampling and cleanup; output belongs in remote persistent project storage.

```bash
python scripts/distillctl.py workloads native list-adapters
python scripts/distillctl.py workloads native validate \
  --task configs/distillschedule/original_h3_ar/w6_reference.json
```

```bash
python scripts/distillctl.py workloads native prepare-assets \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --native-manifest /REMOTE/RUN/configs/native-manifest.json \
  --checkpoint-audit /REMOTE/RUN/artifacts/checkpoint-audit.json \
  --output /REMOTE/RUN/configs/assets.json
```

```bash
python scripts/distillctl.py workloads native calibrate \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --assets /REMOTE/RUN/configs/assets.json \
  --capacity-policy /REMOTE/RUN/configs/capacity.json \
  --implementation zero3-ap --output /REMOTE/RUN/calibration-ap --execute
```

```bash
python scripts/distillctl.py workloads native assemble \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --rank-receipts /REMOTE/RUN/calibration-ap/rank_*.json \
  --output /REMOTE/RUN/assembled-ap
python scripts/distillctl.py workloads native plan \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --calibration /REMOTE/RUN/assembled-native/collection.json /REMOTE/RUN/assembled-ap/collection.json \
  --output /REMOTE/RUN/artifacts/selected-plan.json
```

```bash
python scripts/distillctl.py workloads native run \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --assets /REMOTE/RUN/configs/assets.json \
  --capacity-policy /REMOTE/RUN/configs/capacity.json \
  --plan /REMOTE/RUN/artifacts/selected-plan.json \
  --output /REMOTE/RUN/selected-execution --execute
```

1. `prepare-assets` reads the accepted **two-cycle** manifest to locate pinned
   source/helpers/cache/libraries, never to execute its old command. Checkpoint
   payloads hash once; an unchanged stat-bound audit is reused. Keep assets/audit
   immutable and regenerate fresh owner startup/UUID receipts if hosts change.
2. Freeze scalar-v1 capacity's four explicit byte fields or the
   [v2 node map](../contracts/h3_native.md#original_h3_ar_capacity_v2). With no
   `memory.peak`, require each node's exact fresh finite hard limit, maintained
   reserve0 and unchanged OOM counters; sampled current cannot prove a smaller
   limit/positive reserve. The64GiB startup headroom is independent, not sustained.
3. Serially calibrate `zero3-native` and `zero3-ap` under owned2×8 torchrun.
   Set `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`; before Torch import, set both
   supplied allocator variables to exactly `expandable_segments:True` for native
   or `expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:4`
   for AP. Keep Torch2.10.0+cu128/native offline/USE_TF/USE_FLAX/NCCL policy;
   WORLD_SIZE16/LOCAL_WORLD_SIZE8 and eight fresh receipt UUIDs in original order.
4. Assemble all16 rank receipts per arm, then plan from both collections. Missing,
   mixed or edited receipts reject. The common search may select either arm;
   historical255.670644s is not a cost on this source.
5. Read `selected.implementation`, install its allocator environment, execute
   saved plan; compare an explicit `benchmark-manual --implementation zero3-ap`
   arm at identical task/source/assets/capacity/environment/two-cycle boundary.
   Validate all16 native census/Adam/versions/cache epochs/backend witnesses,
   resource/OOM/cleanup and matched rank inputs/state/numerics before comparing
   max16C2. Search/exit0 does not prove≤3% matching, parity or speedup.

No default capacity numbers are supplied. Do not import GAN's1GiB reserve or
turn H3's64GiB entry guard into a maintained floor; historical provenance is in
the contract. Driver reserve checks are sampled, not continuous-extrema proof.
