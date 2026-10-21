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

The Program adapter requires enough prepared records for the requested total
cycles. It reads the declared order without implicit recycling, resizing or
sample substitution. Algorithms may use all modalities and nested native
feature trees in a record. They retain responsibility for distinguishing
Student samples, Fake samples and previous/current history.

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

For replay from the same initial model and prepared input, use
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
format. Supply at least `cycles × world_size` records; at global cursor `k`,
rank `r` consumes record `k+r`, then the cursor advances by `world_size`. No
padding, recycling or dropping an incomplete rank cohort is permitted.

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
records cycle, sample, scalar outputs, finite state, actual operations, logical
versions and policy hash. `summary.json` records completion and the latest
complete checkpoint. An interrupted/failed run retains `failure.json` and its
last completed checkpoint. Failed partial updates are never counted complete.
Distributed rank metrics are in `ranks/rank-NNNNN/metrics.jsonl`; the shared
summary records all actual completed ranks rather than predicting rank counts.

## Generate and export

```sh
distillctl evaluate train.json --checkpoint run/checkpoints/cycle-000004 --output generated
distillctl export train.json --checkpoint run/checkpoints/cycle-000004 --output exported
```

Evaluation invokes the adapter's actual `generate(state, feeds, parameters)` and
optional `decode(value, parameters)` under no-grad and writes `generation.pt`
plus metadata. This operation generates a sample; it does not automatically
calculate FVD, VBench or another quality metric. An adapter may implement a
custom export callback; otherwise the framework exports named module state
dictionaries. `export.json` explicitly labels inference weights as
**not a resumable training checkpoint**. Missing generation support rejects.
