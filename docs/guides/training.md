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

Generic checkpoints currently own one process/device. Native distributed
checkpoint capabilities are documented per recipe; a sampled fingerprint or a
final tensor-evidence file is not advertised as a complete restart checkpoint.

## Run artifacts

`resolved-config.json` and `identity.json` identify the request. `metrics.jsonl`
records cycle, sample, scalar outputs, finite state, actual operations, logical
versions and policy hash. `summary.json` records completion and the latest
complete checkpoint. An interrupted/failed run retains `failure.json` and its
last completed checkpoint. Failed partial updates are never counted complete.

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
