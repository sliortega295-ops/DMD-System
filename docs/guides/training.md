# Train, save, resume and generate

Install `.[train]` in an existing compatible environment. For JSON-only
inspection `pip install .` is enough; YAML includes need `.[config]`. Optional
native runtimes keep their pinned upstream dependencies. `distillctl doctor`
reports installed versions without importing Torch or constructing a model.

The [tiny media tutorial](../../examples/framework/README.md) gives executable
offline commands with actual files and real autograd. It is a randomly
initialized fixture, not a pretrained model or a quality experiment.
For a native model, choose its [family and recipe](model_families.md#choose-a-compatible-integration),
then the execution path below. The same save/resume/export/generate commands
apply; the [support matrix](support_matrix.md) records each path's evidence.

## The five configuration sections

| Section | User responsibility |
| --- | --- |
| `model` | An explicit `module:factory` or local `.py:factory` plus parameters; loading checkpoint, dtypes, trainables and role models belongs here |
| `data` | A prepared manifest containing the exact feeds needed by the algorithm; original-native recipes use their documented asset format |
| `algorithm` | A factory describing one cycle's actual math using existing hooks or Program calls; sample policy, loss, optimizer and EMA meanings are explicit |
| `execution` | `reference`, admitted `manual`, source-bound `plan`, or measured `auto`; optional implementation/order and bounded search settings |
| `trainer` | Target complete cycles, seed, device, output directory, checkpoint frequency and optional restart checkpoint |

Configuration schema is `distillschedule.training.v1`. JSON and YAML reuse the
existing include composition and `${ENV:NAME}` expansion. File paths are
relative to the request file. Plugin Python is trusted user code and executes
only during the requested preparation/training/output operation, not doctor.

```sh
distillctl config validate train.json
distillctl config show train.json
distillctl doctor --config train.json
distillctl train train.json --dry-run
distillctl train train.json
```

By default, the Program adapter requires enough prepared records for the
requested total cycles. It reads manifest order without implicit recycling,
resizing or sample substitution. Algorithms may use all modalities and nested
native feature trees in a record. They retain responsibility for distinguishing
Student samples, Fake samples and previous/current history.

## Explicit Program data sampling

To train beyond one pass over prepared records, opt in explicitly in the normal
five-section request:

```json
{"data": {"prepared":"prepared/manifest.json",
          "sampling":{"kind":"epochs", "shuffle":true}}}
```

`data.sampling` accepts exactly `kind` (`finite` or `epochs`) and an explicit
boolean `shuffle`. Omitting the whole field retains finite, unshuffled manifest
order and does not insert a default into configuration identity. Explicit
`finite` requires `shuffle:false`; it still rejects targets exceeding available
records. Original-native recipes reject this Program-only option; their six
input contracts and original sampling semantics are unchanged.

`epochs` traverses every physical record exactly once per full epoch. With
`shuffle:false`, each epoch is manifest order. With `shuffle:true`, the stable
`sha256-sort-v1` algorithm orders record indexes by a versioned hash of
`trainer.seed`, the zero-based epoch and manifest index, breaking ties by index.
It consumes no Python, NumPy, Torch or CUDA training RNG; only the current
epoch's permutation is cached. It never changes prepared records, shapes,
batching, model noise generation or algorithm update semantics.

Distributed training uses the same fixed rank-strided cohorts over that order.
In epoch mode, dataset length must be divisible by the **actual world size**;
there is no padding, dropped tail or incomplete rank cohort. A target may end
partway through its final epoch after any complete cohort. The checkpoint cursor
remains total consumed records, exactly `completed_cycles × world_size`, not a
wrapped index. The policy, seed, dataset and framework algorithm identity plus
that cursor reconstruct the exact next records in a fresh process; no separate
mutable sampler state is saved. Changed policy, seed or dataset cannot resume
the old checkpoint. Summary receipts identify the policy and ordering algorithm.

`PreparedDataset.sample(index)` and `build_cycle` continue to accept physical
manifest indexes. In particular, generation/evaluation `--sample-index` is never
interpreted as a shuffled position or epoch cursor.

## Explicit or automatic strategy

The default execution is `{"mode":"reference"}`. For a manual Program policy,
provide the actual native capability, for example:

```json
{"mode":"manual", "implementation": {
  "backend":"torch-callable-resident-v1",
  "share_immutable":false,
  "checkpoint_bindings":["student_forward"]
}}
```

The algorithm must mark that binding replay-safe. The same native admission
rejects unsafe checkpointing or invalid immutable sharing. A measured automatic
request may use:

```json
{"mode":"auto", "memory_limit_bytes":100000000,
 "max_candidates":3, "samples":1, "max_program_runs":8}
```

This calibration performs extra trial updates on isolated state. Actual
training then starts from the original state using the selected implementation.
The initial full-state gate remains mandatory. The selected plan, effective
policy, operation order and update versions are written beside the run.
Timing is diagnostic training wall time, not a new performance result.

For replay from the same initial model and prepared input (including the
sampling policy and seed that select its first record), use
`{"mode":"plan","plan":"path/to/selected-plan.json"}`. Changed model,
input, source or calibration hashes reject through the original planner.

### Choose an official native execution path

The native owners preserve the original algorithm inside the common Program
and cycle lifecycle. Select a recipe explicitly; direct training needs no
calibration run or historical saved plan.

| Workflow | Reference choices | Direct optimized choice |
| --- | --- | --- |
| SD1.5 official DMD2 | [Single-device FP32](../../recipes/dmd2_sd15/README.md); [native FSDP2](dmd2_fsdp2_reference.md) via `training.fsdp2.json`; separate FP32 native role-DDP recipe | `recipes/dmd2_sd15/training.optimized.json`: native mixed-precision Resident/DDP bundle |
| Official Self-Forcing | [Single-device FP32](../../recipes/self_forcing_official/README.md); native FSDP2 via `training.fsdp2-reference.json` | `recipes/self_forcing_official/training.optimized.json`: native FSDP2 with the complete versioned optimization profile |

The FSDP2 and optimized recipes bind their declared A100/world16, precision,
source, input and launch contracts. Follow the [official launch/resume guide](official_optimized_training.md)
for these native owners, not the generic visible-module backend options below.
SDXL retains its own FP32 single-device/native role-DDP scope; it does not
inherit SD1.5 FSDP2 or the SD1.5 optimization profile. Public distributed GPU
qualification and historical native measurements are listed separately in the
[support matrix](support_matrix.md#integrated-official-optimization-profiles).

### Modern native flow qualification and capacity

FLUX.2-klein-base, Qwen-Image, Wan2.2 TI2V-5B T2V and HunyuanVideo1.5 T2V
retain their native model/text/VAE classes and source-bound local assets.
Together with LTX LoRA, they use the existing common flow-DMD runtime:

- **One CPU or CUDA device, FP32:** persistent `reference`, or exact direct
  `manual` [immutable sharing](optimization_capabilities.md#direct-single-cuda-immutable-sharing)
  with `share_immutable:true` and no checkpoint bindings. This does not clone a
  CUDA owner or change the loss/update schedule.
- **Explicit replicas:** FLUX/Qwen use [equal-B1 post-backward reduction](../../recipes/flow_image/README.md#explicit-replicated-training-not-model-sharding),
  reference or exact manual sharing, on CPU/Gloo or CUDA/NCCL. Each rank owns
  complete models; this is not model sharding or DDP-forward wrapping. The
  video adapters' separate source routes and qualification limits are in the matrix.
- **Measured `auto` / saved `plan`:** CPU1 FP32 synthetic-fixture trials only,
  not CUDA planning or distributed trials. These use real gradients,
  G1/F5 Adam/scheduler/EMA, fresh text records and new-process restart through
  the same planner and lifecycle.

The [support matrix](support_matrix.md#additional-model-and-workflow-integrations)
links actual tiny CUDA reference/manual and image-replica receipts separately
from pending full-pretrained, capacity and quality gates. These flow recipes
are not official DMD2 or Self-Forcing reproductions.

The family Role clone boundary explicitly refuses
`LOCAL_PRETRAINED_UNVERIFIED` (or missing/unqualified) provenance before copying
the native role graph. This covers both public auto/saved-plan training and
direct common-ABI calibration/paired execution. CPU/single-CUDA reference and manual
persistent non-clone execution remain available with the declared local asset
and parameter-budget checks. `SYNTHETIC_TEST_ONLY` is a trusted caller label,
**not** a verified payload-size/origin check or a memory-safety guarantee. Do not
relabel pretrained weights to bypass this qualification refusal. Existing
official CUDA whole-owner clone refusals remain in force.

`memory_limit_bytes` bounds the **selected plan's tracked tensor storage** and
measured retained/workspace terms. It is not a host RSS, model-construction,
CPU staging, concurrent reference/selected snapshot, or whole-role clone budget.
All image role, VAE and text parameter budgets are now meta-checked before any
concrete component is constructed; Wan/Hunyuan retain their analogous
all-component preflight. Parameter counts still do not bound RSS or staging.
Large/full-pretrained clone trials are unsupported, not implicitly admitted by
supplying a large selected-plan limit. No new offload backend is provided.

Inspect the actual inventory rather than assuming a physical option exists.
These G1/F5 fixtures currently admit two candidates: private and immutable-base
sharing. No replay-safe binding is declared, so checkpoint choices are not
invented. With one sample each, calibration performs three program runs, then
the mandatory selected-plan reference gate performs two more. The latter's
selected execution is the first formal training cycle. Thus three formal auto
cycles require seven program runs total, not eight; a saved-plan run needs four.
`max_program_runs` bounds calibration only, not the subsequent paired admission
or formal cycles. Selection may legally choose private storage; there is no
speedup or optimality claim. The original loss/state/source/resource gates are
unchanged.

## Complete-cycle restart

```sh
distillctl train train.json --resume run/checkpoints/cycle-000004 \
  --cycles 6 --output resumed
```

`cycles` is the total desired completed count, not six additional cycles. A
fresh output directory preserves the earlier run. The checkpoint carries:

- All declared role modules and their trainability/training mode.
- Optimizers and explicitly declared stateful schedulers, EMA or history.
- Python/NumPy/Torch RNG, initialized CUDA RNG if relevant, logical versions,
  completed cycle and exact next prepared-data cursor.
- Configuration, data, plugin, runtime and policy identities.

Only complete zero-gradient cycle boundaries are checkpointed. `state.pt`, its
manifest and `_SUCCESS` are atomically published; missing or altered shards
reject. The same topology, runtime, data and mathematical configuration are
required. Output path, save frequency and total target cycles may change.
Transient replay caches are reconstructed by the runtime.

For distributed Program training, every rank writes a local tensor shard;
DTensor metadata binds global shape/stride, mesh rank order and placements.
No pickled process groups or implicit parameter all-gather are used for restart.
The completion marker is published only after **all** rank files have been
written and hashed on the shared checkpoint filesystem. A missing rank,
changed optimizer ownership, topology or shard layout rejects restoration.

Native distributed checkpoint capabilities are documented per recipe; a
sampled fingerprint or final tensor-evidence file is not a restart checkpoint.

## Distributed Program training

The same algorithm and lifecycle run through the existing `ExecutionBackend`.
There is no model-name dispatcher and no second distributed solver. A recipe
must declare `ReplicaSemantics` (the DMD hook builder accepts
`replica_batch_size`); rank-local history in otherwise replicated state is not
silently admitted. Each rank's local record may itself contain a model-defined
batch. The objective remains the declared equal-rank mean.

Set these fields explicitly for a CPU functional run:

```json
{
  "data": {"prepared":"prepared/manifest.json", "sharding":"rank_strided"},
  "trainer": {"cycles":4, "seed":7, "device":"cpu", "output_dir":"run",
              "distributed":{"backend":"gloo", "timeout_seconds":180}}
}
```

These are fragments of the normal five-section request, not a separate config
format. By default, supply at least `cycles × world_size` records; at global
cursor `k`, rank `r` consumes manifest record `k+r`, then the cursor advances by
`world_size`. Explicit epoch sampling instead resolves that logical position
through the policy described above. No padding or dropping an incomplete rank
cohort is permitted, and reuse requires the explicit epoch opt-in.

Before any model factory, restart branch or output creation, ranks must agree
on the total target cycles, checkpoint frequency, resolved shared output root
and whether to resume from the same completed manifest. A missing manifest or
invalid `_SUCCESS` on any rank rejects collectively. Explicit checkpoint mount
aliases are allowed when their manifest bytes and completion hash agree.
These controls may change between runs, but must match across ranks within one
invocation; they do not change the checkpoint's mathematical identity.

```sh
torchrun --standalone --nproc-per-node=2 -m distillschedule train train.json
torchrun --standalone --nproc-per-node=2 -m distillschedule train train.json \
  --resume run/checkpoints/cycle-000002 --cycles 6 --output resumed
```

Reference mode uses replicated post-backward mean; measured `auto` and saved
`plan` modes use the real admitted candidate, including DDP, FSDP2 and HSDP
where the current backend/mesh permits them. A manual option uses the existing
`torch-callable-replicated-v1` schema with its local policy, reduction and bucket
size. Models, optimizers, DDP wrappers, FSDP layout and hooks are installed once,
not rebuilt at every cycle. Resume installs the **saved layout** before loading
its local shards, then continues the saved policy without re-calibration.
These are generic Program capabilities, not every model's backend menu. The
modern-flow restrictions above and official native-owner recipes still apply;
for example, FLUX/Qwen replication does not admit auto/FSDP/HSDP candidates.

The group is owned by the caller when already initialized. Otherwise an
explicit `trainer.distributed` request initializes only the resources supplied
by torchrun. This API does not reserve GPUs, acquire nodes or launch SSH jobs.
CPU/Gloo restart is functionally tested; CUDA/NCCL validation is separate and
not implied by those tests.

## Run artifacts

`resolved-config.json` and `identity.json` identify the request. `metrics.jsonl`
records cycle, actual sample ID and physical `sample_index`, zero-based `epoch`
and global `epoch_offset`, total consumed-record `cursor`, scalar outputs,
finite state, actual operations, logical versions and policy hash. `summary.json`
records completion, sampling policy/algorithm and the latest
complete checkpoint. An interrupted/failed run retains `failure.json` and its
last completed checkpoint. Failed partial updates are never counted complete.
Distributed rank metrics are in `ranks/rank-NNNNN/metrics.jsonl`; the shared
summary records all actual completed ranks rather than predicting rank counts.

Model-owned atomic operations can expose their already-computed host losses
with the optional [metric collector](extensions.md#model-and-media-adapter).
The original Helios GAN CPU reference records one Generator loss, each of five
ordered Critic losses and their mean. The cached-H3 CPU reference records its
actual DMD, GAN and Critic losses. These are not new losses, quality metrics or
numerical-equivalence claims. Collection occurs after the diagnostic workload
timer and does not rerun a model or coerce new tensor values to the host.

## Optional periodic held-out validation

Generation via `evaluate` is **not** a validation metric. A Program model may
separately implement `validate_batch(state, feeds, parameters)`, returning a
nonempty mapping of named, finite scalar metrics. Opt in with both fields:

```json
{
  "data": {"prepared":"prepared/manifest.json",
           "validation_prepared":"heldout/manifest.json"},
  "trainer": {"cycles":6, "seed":7, "device":"cpu", "output_dir":"run",
              "validation":{"every_cycles":2, "samples":2, "parameters":{}}}
}
```

This fragment belongs in the normal five-section request. `samples` counts
fixed prepared **records per rank**, not batches inferred from tensor shapes;
the adapter owns each record's internal batch and metric definition. There must
be at least `samples × world_size` records; this initial validation capability
is **single-process only**. Any distributed validation request rejects before
model construction, wrapper installation, training updates or output creation.
DDP/FSDP2/HSDP training without validation is unchanged; no distributed or GPU
validation correctness is claimed. Original-native recipes do not accept this
new option. Missing pairs, unknown options, invalid counts and non-JSON or
nonfinite parameter maps reject rather than silently disabling validation.

At complete cycles divisible by `every_cycles`, validation rereads the same
first `samples` records from its explicit manifest, independent of the training
cursor. Use a genuinely disjoint held-out dataset: the framework cannot infer
semantic overlap between prepared media. There is no forced final validation
outside that interval and no validation at initialization. Validation data,
callback source and all settings bind checkpoint identity; changed inputs or
settings cannot silently resume an earlier checkpoint.

The callback runs under no-grad and evaluation mode, using a deterministic
per-record validation stream. Exact nested module modes and Python/NumPy/Torch
and the initialized, owned CUDA device's RNG state are restored, even on error.
Declared model/optimizer/scheduler/EMA/history state and gradient contents and
ownership are guarded using the existing runtime signatures. Mutation stops
training; **only modes and RNG are restored**, not corrupted model or optimizer
state. A failed callback does not publish a new complete checkpoint. This is a
trusted-plugin contract, not a sandbox for hostile Python or hidden globals.

`validation.jsonl` records rank, cycle, fixed sample IDs and equal-record mean
metrics. `validation-summary.json` and the training summary preserve the last
event and actual ranks, with process-local run counts. Metrics are explicitly
**adapter-defined held-out diagnostics, not a quality gate**: no pass threshold,
early stopping, best-model selection, quality label or hidden metric is inferred.
`quality_validation` remains `NOT_RUN`. There are no validation reads/callbacks
when disabled, and enabling a read-only callback does not change training math,
state, RNG or data progress. See the [stateful workflow tutorial](../../examples/framework/README.md#another-workflow-history-ema-and-schedulers)
for a real held-out metric and fresh-process restart example.

## Generate and export

```sh
distillctl evaluate train.json --checkpoint run/checkpoints/cycle-000004 --output generated
distillctl export train.json --checkpoint run/checkpoints/cycle-000004 --output exported
```

`evaluate` reads a **training checkpoint** with its original training request
and prepared dataset. All original source/configuration/data/runtime/topology,
module/optimizer/layout and shard-hash admission gates still apply. It installs
the saved persistent Program policy before restoring local checkpoint shards;
it does not run another training update or recalibrate the plan.

For a distributed checkpoint, launch `evaluate` or `export` with the same
torchrun topology and explicit distributed request used by training. Every
rank participates in necessary FSDP/HSDP forwards and full-weight gathers.
`--sample-index N` selects the evaluation cohort beginning at N: rank r reads
N+r, with no padding/recycling. Each rank writes under
`generated/ranks/rank-NNNNN/`; primary publishes the shared `generation.json`
and `_SUCCESS` only after all rank outputs succeed. Export has one shared
artifact, written by primary only. Unsupported custom distributed exporters
reject rather than guessing rank ownership.
Generation parameter maps are admitted as finite canonical JSON and must agree
on every rank before any model/forward is constructed for evaluation; differing
denoising step counts cannot silently enter different collective sequences.

`export` creates `distillschedule.inference_export.v1`: named, full module state
dictionaries in `weights.pt`, plus `export.json` binding every file hash, the
exported factory reference/source hash, effective non-device factory arguments
(including the supplied default seed), model metadata and module state layout.
The directory is atomically published after the complete manifest and its
hash-bound `_SUCCESS` marker are written. It is explicitly labeled
**INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT**: no optimizer, scheduler, data
cursor or RNG restart state is exported. Do not pass it to `train --resume`.

### Build a separate inference request

Program adapters may export an **inference projection**, not their training
factory. Official DMD2 SD1.5/SDXL, official Self-Forcing and modern flow models
do this: their exported factory constructs only the inference components.
Consequently, `generate train.json --weights exported` is not a portable
recipe and rejects when its model factory or parameters differ from the export.
Keep the original request for `evaluate` and `export`; create `inference.json`
for `generate` instead. This uses the existing training-request schema, not a
new configuration type.

First prepare genuinely new inputs with your model recipe's preparer. The
[export-only preparer](export_inputs.md) is available only for its explicitly
admitted modern flow adapters, **not** SD or official Self-Forcing. Once
`new-prepared/manifest.json` is ready, run the following in your installed model
environment. The four shell arguments are paths relative to the current working
directory (absolute paths also work); relative fields inside `train.json`,
including a local algorithm factory, resolve relative to that original file.

```sh
python - train.json exported new-prepared/manifest.json inference.json <<'PY'
import json
import sys
from pathlib import Path

from distillschedule.training.config import TrainingConfig, load_training_config
from distillschedule.training.inference import read_export

train_path, weights_path, prepared_path, inference_path = (
    Path(arg).expanduser().resolve() for arg in sys.argv[1:]
)
training = load_training_config(train_path)
_, manifest = read_export(weights_path)
request = {
    "schema_version": "distillschedule.training.v1",
    "model": {key: manifest["model"][key] for key in ("factory", "parameters")},
    "algorithm": dict(training.algorithm),
    "data": {"prepared": str(prepared_path)},
    "execution": {"adapter": "program", "mode": "reference"},
    "trainer": {
        "cycles": 1, "seed": 909, "device": "cpu",
        "output_dir": str(inference_path.parent / "unused-inference-run"),
    },
}
inference = TrainingConfig.from_dict(request, source=inference_path)
inference_path.write_text(json.dumps(inference.to_dict(), indent=2) + "\n")
PY
distillctl generate inference.json --weights exported \
  --prepared new-prepared/manifest.json --sample-index 0 \
  --output new-generated --output-kind image
```

The Python step validates the export's completeness and hashes with `read_export`
and validates the new request with `TrainingConfig`; it does not load weights,
import/build the model or algorithm, read the old training data or run training.
Copy **only** the exported `model.factory` and exact `model.parameters`, not
the whole export `model` object (`source`, `metadata` and `modules` are not
request fields), and do not retain the old registered `model.name`. Keep every
exported parameter, including its model-initialization seed and explicit external
source paths, unchanged. Do not add old base `model_id`/asset paths, Teacher/Fake
or EMA state. Required matching external source/dependencies remain necessary.

The algorithm section is retained as schema metadata only; generation never
calls its factory. The new request deliberately drops old streams, validation,
distributed/resume settings, manual overrides and saved plans. Its one cycle
and unused trainer output directory satisfy the schema but are not executed.
The independent inference seed is `trainer.seed: 909`; it does not replace the
exported model seed. Choose an explicitly supported single inference device if
changing the CPU default, with capacity for the complete inference model.
Run `generate` as a **single process**, not under torchrun or with leftover
distributed-launch environment variables. Use `--output-kind image` for SD and
image flow models, `video` for official Self-Forcing, or `audio_video` for an
adapter that actually supplies both modalities (for example, admitted LTX).

`generate` is single-process inference from that artifact and a **new explicit
prepared manifest**. It does not open the original training dataset, run or
checkpoint, and does not import/build the algorithm just to recover a model.
The declared model factory must still be available with matching source,
non-device arguments, metadata and module layout. The factory may naturally
construct optimizers, but their training state is neither needed nor restored.
Byte-identical local `.py` factories may be explicitly relocated by updating
the current request's path while keeping the same callable symbol; their hash
and module layout must still match. Package/module references remain exact.
This inference-only relocation does **not** relax checkpoint-resume identity.
The artifact is not a bundle of Python source or pretrained dependencies;
trusted external dependency identities remain the adapter's responsibility.
Completeness, schema, path bounds and every file hash are checked before model
imports/construction or output creation. Changed source, incompatible weights,
symlink/path escapes and incomplete exports reject.

Both generation paths put modules in evaluation mode and invoke the adapter's
actual `generate(state, feeds, parameters)` and `decode(value, parameters)`
under no-grad. `trainer.seed` is reset immediately before inference, independent
of factory initialization or RNG consumed assembling a training workload.
Receipts record this `inference_seed` separately from the exported model's seed,
alongside the actual canonical `generation_parameters` passed to the callbacks.

Portable `generate` defaults to decoded video, but explicitly supports
`--output-kind image|video|audio|audio_video|latents`. The adapter must actually
supply the requested output; no guessed latent reshape is a decoder.

| Output kind | Actual decoded contract | Files |
| --- | --- | --- |
| image | `image`: nonempty HWC uint8 RGB/RGBA | `image.png` and safe tensor artifact |
| video | `video`: nonempty THWC uint8 RGB/RGBA | exact diagnostic PNG frames and safe artifact |
| audio | `audio`: `{samples: finite floating TC in [-1,1], sample_rate: positive int}` | PCM16 WAV at the declared native rate and safe artifact |
| audio_video | both the audio and video contracts above | frame directory + WAV; not an implicitly muxed movie |
| latents | named finite floating tensors and each exact layout/shape/dtype in `latent_schema` | explicitly labeled native diagnostic, not media |

Audio samples are explicitly converted to FP32 for PCM serialization (including
BF16 input), clamped to [-1,1] and quantized to PCM16; no rate or channel count
is guessed or resampled. Original safe tensor output is also retained. Video
and image pixels preserve the decoder's declared dimensions. Missing generation
or requested decoding support rejects before output publication. Legacy
checkpoint `evaluate` may keep non-media safe tensor results and report
`video.status=NOT_AVAILABLE`; malformed explicitly supplied media still fails.

Both `evaluate` and `generate` accept `--parameters path/to/finite.json`, an
explicit JSON object passed to the model's native generation/decoder. Duplicate
keys, nonfinite values and wrong root types reject. Parameter meaning and any
pinned codec assets are admitted by that model; the CLI never infers them.
For example, DMD2 explicitly selects `--output-kind image`, LTX selects
`--output-kind audio_video`, and a decoder-free H3 route uses `latents`.

Decoded outputs remain diagnostic. Pretrained quality, FVD and VBench are
**NOT_RUN** in this framework integration.

Custom formats require paired `export` and explicit `load_export` callbacks;
see [the extension contract](extensions.md). The framework never infers a loader
from a filename or treats custom bytes as a resumable checkpoint.

### Inference-only projections and native latent diagnostics

Adapters with atomic training state can project the restored **current** model
into a separately source-bound inference factory, through the optional
`InferenceProjection` hook. Portable generation uses the full unmerged
exported weights, not optimizer/checkpoint state or original training data.
The [Helios CPU example](../../recipes/helios_gan/PORTABLE_INFERENCE_CPU.md)
uses actual original native rollout on new prompt tensors in a new process.

For a model without an admitted real media decoder, explicitly use
`distillctl generate ... --output-kind latents`. The artifact records named
finite tensors, exact layout/shape/dtype and
`LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY`; it writes no pretend RGB frames.
The default remains `--output-kind video` and still requires a real decoder.
Exported weights never become a resumable training checkpoint.
