# Train, save, resume and generate

Install `.[train]` in an existing compatible environment. For JSON-only
inspection `pip install .` is enough; YAML includes need `.[config]`. Optional
native runtimes keep their pinned upstream dependencies. `distillctl doctor`
reports installed versions without importing Torch or constructing a model.

The [tiny media tutorial](../../examples/framework/README.md) gives executable
offline commands with actual files and real autograd. It is a randomly
initialized fixture, not a pretrained model or a quality experiment.

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
distillctl generate train.json --weights exported \
  --prepared new-prepared/manifest.json --sample-index 0 --output new-generated
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
original factory reference/source hash, effective non-device factory arguments
(including the supplied default seed), model metadata and module state layout.
The directory is atomically published after the complete manifest and its
hash-bound `_SUCCESS` marker are written. It is explicitly labeled
**INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT**: no optimizer, scheduler, data
cursor or RNG restart state is exported. Do not pass it to `train --resume`.

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

Portable `generate` requires the decoded-video capability: the decoder returns
`{"video": tensor, ...}` with nonempty THWC uint8 RGB/RGBA video (or generation
already returns this decoded contract). `generation.pt` retains the
generated/decoded values; PNGs in
`frames/` preserve the decoder's exact pixels and dimensions. There is no latent
resize, guessed normalization or substitute quality renderer. Missing generation
or compatible decoding support rejects before output creation. For compatibility
with non-video models, legacy `evaluate` may instead save latent/audio/other safe
outputs to `generation.pt` without PNGs and explicitly reports
`video.status=NOT_AVAILABLE`. Malformed explicitly supplied video still rejects.
Video outputs are
diagnostic frames, not evidence of useful pretrained quality: FVD, VBench and
other quality validation remain **NOT_RUN**.

Custom formats require paired `export` and explicit `load_export` callbacks;
see [the extension contract](extensions.md). The framework never infers a loader
from a filename or treats custom bytes as a resumable checkpoint.
