# Train, save, resume and generate

Choose a [recipe](../../recipes/README.md), install its [environment](installation.md)
and [prepare inputs](data.md). The [CPU tutorial](../../examples/framework/README.md)
walks through the complete lifecycle with a small real model.

```sh
distillctl config validate train.json
distillctl doctor --config train.json
distillctl train train.json --dry-run
distillctl train train.json
```

## The five configuration sections

| Section | Contents |
| --- | --- |
| `model` | Registered name or `module:factory`, checkpoint, precision, trainable roles and adapters |
| `data` | Prepared inputs or independently sampled streams |
| `algorithm` | Distillation flow, rollout, losses, updates and EMA |
| `execution` | Reference, manual strategy, saved plan or measured automatic selection |
| `trainer` | Complete cycles, seed, device, distributed group, output and checkpoint settings |

The schema is `distillschedule.training.v1`. JSON/YAML support includes and
`${ENV:NAME}` expansion. Relative paths resolve against the request file.
Python plugins execute as trusted user code during the selected operation.

## Explicit Program data sampling

Default sampling consumes prepared records in order. To train over multiple
epochs, configure:

```json
{"data":{"prepared":"prepared/manifest.json",
         "sampling":{"kind":"epochs","shuffle":true}}}
```

`finite` with `shuffle:false` selects one finite traversal; `epochs` repeats
complete epochs. Deterministic `sha256-sort-v1` shuffling uses the seed and
epoch without consuming training RNG. The saved cursor identifies the exact
next records after recovery. With rank-strided inputs, each rank reads its own
record; dataset size must be divisible by world size. Native workload recipes
retain their original sampling logic.

## Explicit or automatic strategy

Direct training defaults to `{"mode":"reference"}`. Choose an applicable
implementation explicitly to use existing optimizations:

```json
{"mode":"manual","implementation":{
  "backend":"torch-callable-resident-v1",
  "share_immutable":false,
  "checkpoint_bindings":["student_forward"]
}}
```

The algorithm declares replay-safe bindings and immutable roles. The execution
backend validates the actual combination. For the six original workloads use
[versioned native presets](native_presets.md); for official DMD2/Self-Forcing use
[their optimized recipes](official_optimized_training.md).

Automatic calibration and search are optional:

```json
{"mode":"auto","memory_limit_bytes":100000000,
 "max_candidates":3,"samples":1,"max_program_runs":8}
```

Trials run on isolated state, then training starts from the original state with
the selected strategy. CPU search shares its remaining budget across unvisited
candidates. Saved plans bind actual backend, operation order and state policy:

```json
{"mode":"plan","plan":"planning/selected-plan.json"}
```

Use the saved plan with its matching inputs, initial state and calibration.
For an external project, the [optimization harness and Skill](optimizer_skill.md)
reuse the project's own training and checking commands.

## Complete-cycle restart

By default, checkpoints are saved **every 30 minutes at the next complete cycle
boundary**, and at normal completion. The last **three** complete checkpoints
are retained. Configure seconds and retention under `trainer`:

```json
{"trainer":{"cycles":100,"output_dir":"run","seed":7,"device":"cuda",
            "checkpoint_interval_seconds":1800,"checkpoint_keep_last":3}}
```

For fixed-cycle saving, set `checkpoint_every: 10` instead. An explicit
`checkpoint_every: 0` disables saving; `checkpoint_keep_last: 0` keeps all
completed publications. Time-based saving agrees collectively across ranks.

```sh
distillctl train train.json --resume run/checkpoints/cycle-000050 \
  --cycles 100 --output resumed
```

`cycles` is the target total, not the number of additional cycles. Use a new
output directory. Checkpoints include role parameters, gradients at the
complete-cycle boundary, optimizer/scheduler/EMA, RNG, data positions, logical
versions and the effective strategy. Transient caches are rebuilt.

The common lifecycle supports same-topology recovery across compatible minor
framework fixes. Versioned restore contracts separate source provenance from
state compatibility; model/plugin, algorithm, data, precision, layout and update
contracts remain bound to the checkpoint. New output paths, save settings and
target cycle counts can be chosen on recovery.

Distributed checkpoints contain local tensor shards and mesh/placement metadata.
Only after all rank files are complete and hashed are the manifest and
`_SUCCESS` published. Retention removes only this run's complete checkpoints.
Each native recipe documents its corresponding continuation entry.

## Distributed Program training

Use the same training request under `torchrun`. Declare the backend and
rank-strided data ownership:

```json
{
  "data":{"prepared":"prepared/manifest.json","sharding":"rank_strided"},
  "trainer":{"cycles":4,"seed":7,"device":"cpu","output_dir":"run",
             "distributed":{"backend":"gloo","timeout_seconds":180}}
}
```

```sh
torchrun --standalone --nproc-per-node=2 -m distillschedule train train.json
```

For CUDA use `device: "cuda"` and `backend: "nccl"`. The launcher's local rank
selects the device. [Execution capabilities](optimization_capabilities.md) cover
Resident, replica, FSDP2, HSDP and role-pool choices. The algorithm declares its
`ReplicaSemantics`; replicas preserve the declared equal-rank objective.

Ranks agree on target cycles, checkpoint policy, output root and resume
manifest before model construction. Each rank consumes its own input; the
cursor advances by world size. Parameter topology and optimizer ownership are
restored before reading local checkpoint tensors.

## Run artifacts

| File | Contents |
| --- | --- |
| `resolved-config.json`, `identity.json` | Effective request and source/data identity |
| `effective-policy.json` | Strategy actually installed |
| `metrics.jsonl` | Cycle, samples, data progress, losses, versions, finite checks and runtime receipt |
| `summary.json` | Completed cycles/ranks, latest checkpoint and timing totals |
| `checkpoints/` | Atomically published training states |
| `failure.json` | Failure and last complete checkpoint |

Distributed metrics live under `ranks/rank-NNNNN/`. Existing `wall_s` measures
Program execution, including initial plan admission. Metrics must be serialized,
appended and closed on every rank before checkpoint publication, retention or
effective-policy publication. Rows therefore use `prepublication_wall_s` instead
of the former checkpoint-inclusive `cycle_wall_s`: this window and row `timings_s`
cover the completed prepublication stages, not subsequent logging, checkpoint or
policy publication. Summary `cycles_wall_s` sums full completed cycles, including
logging, checkpoint publication/retention and policy publication; summary
`timings_s` retains the full costs of those stages separately. `job_wall_s` includes
setup and all cycles through final rank collection.
Historical native benchmark timers keep their own reported windows.

## Optional periodic held-out validation

A model can provide `validate_batch` for read-only adapter metrics. Configure
both a prepared validation dataset and its cycle cadence:

```json
{
  "data":{"prepared":"prepared/manifest.json",
          "validation_prepared":"heldout/manifest.json"},
  "trainer":{"cycles":6,"seed":7,"device":"cpu","output_dir":"run",
             "validation":{"every_cycles":2,"samples":2,"parameters":{}}}
}
```

Validation runs on fixed held-out records, preserving training RNG, data cursor
and model state. Results go to `validation.jsonl` and `validation-summary.json`.
See [adapter callbacks](extensions.md#model-and-media-adapter) for the interface.

## Generate and export

Evaluate the current Student or export it from a training checkpoint:

```sh
distillctl evaluate train.json --checkpoint run/checkpoints/cycle-000050 --output generated
distillctl export train.json --checkpoint run/checkpoints/cycle-000050 --output exported
```

Use the checkpoint's topology for distributed evaluation/export. Every rank
participates in necessary forwards and gathers. Exported inference weights have
their own manifest and do not carry optimizer or data-progress state.

### Build a separate inference request

An inference projection contains the current Student and the components needed
for generation. It can run independently of the training checkpoint. Prepare
new conditions using the selected model's data preparer or the
[export-input preparer](export_inputs.md), then construct a request from the
export's model factory and exact parameters:

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

The inference seed is independent of model initialization. Use `--output-kind`
`image`, `video`, `audio_video` or `latents` as appropriate for the adapter.
Generation uses the new prepared inputs without rerunning the training algorithm.
See [native output recipes](native_workloads.md) for model-specific codecs and
current-Student formats.
