# W5: original Grid1000 upstream training (v1)

This is an **explicit reference lifecycle**, not an optimized planner recipe.
It calls the existing `grid_entrypoint` prepare/build/load/set-model/set-data
sequence and its existing setup and Student1/Fake5 loop. It does not substitute
a benchmark trace or implement another loss/update algorithm.

**Evidence boundary:** CPU source, finite-input, real original loss/loop and
checkpoint mechanics only. Pretrained CUDA/FSDP capacity and cross-reference
numerical qualification remain **NOT_RUN**. The known optimized W5 sampled
backward numerical **FAIL remains FAIL**. This route cannot select, time,
calibrate, export, or bless that candidate. This **reference** owner supports
[current-Student export, original four-step generation and AV decoding](../docs/guides/h3_grid_inference.md).

## Explicit inputs and original config

Supply a fully resolved original JSON configuration; this binding never fills
one in or overrides it. Its target iterations, seed, save cadence and output
directory must exactly match the public request. The complete original config
hash (including its original target/scheduler horizon) remains fixed on resume.

The admitted original protocol is approximate Grid1000, four rollout model
evaluations, fixed end step 3, LoRA128, B1/world16, Student1/Fake5 and EMA0,
BF16 transformer/original FP32 latent math, native per-block checkpointing,
original constant LR/no warmup, and original FA3. Optional fusions, split2,
replay caches and LoRA scale/epilogue elisions must be explicitly OFF. The
existing rollout-reorder switch must be explicit with group size five.
`training.a100.activation_policy` must explicitly select either existing
`none` or existing `checkpoint_boundary_cpu`, with explicit pin-memory choice
and detailed events false. CPU boundary staging is not labeled no-offload and
does not itself prove the model fits. Original capacity/runtime gates still run.

The first data scope is deliberately narrow:

* `data` has only `train`, with exactly `name=latent_dataset`, one absolute
  JSONL `data_path`, `batch_size=1`, `dataset_repeat=1`, `shuffle=true`,
  `drop_last=true`, `num_workers=0`, and an explicit boolean `pin_memory`.
* Each JSONL row contains exactly `id` and `condition_path`. These are genuine
  cached prompt-condition records used by the original H3 `LatentDataset`,
  not six fixed rank traces wrapped as a dataset. Paths resolve relative to the
  metadata file. No implicit `negative_condition.pt`, LMDB, raw media, dataset
  repeats, random transforms, max-sample truncation or worker processes.
* At least **96 × target cycles** distinct records, divisible by 16. Distinct
  IDs, payload paths and payload bytes are required. Before model construction,
  the actual upstream dataset loads every cached condition and checks original
  text-only shape/dtype/finite gates and distinct condition tensor identities.
  Renaming a copied fixed trace cannot qualify it as a fresh row. This is not a
  claim about the external provenance of supplied raw data.
* The actual `DistributedSampler` retains its original seed zero, epoch zero,
  rank and ordering. Enough unique rows mean no epoch rollover or padding.
  Each complete cycle consumes exactly six rank-local records. Fresh iterator
  construction retains upstream's RNG draw; resumed iterator construction and
  seek preserve the restored Python/NumPy/Torch/current-device CUDA RNG.

Inventory already supplied data through the common data command (it neither
downloads nor generates model/data payloads):

```sh
distillctl data prepare-h3-grid \
  --native-config /srv/project/configs/original-grid.json \
  --output /srv/project/configs/grid-inputs.json
```

The same producer is available through Python:

```python
from distillschedule.training.h3_grid_inputs import prepare_inputs
prepare_inputs("/srv/project/configs/original-grid.json",
               "/srv/project/configs/grid-inputs.json")
```

## Source, asset and dependency contract

`execution.assets` is a JSON object with exactly these fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | `distillschedule.original_h3_grid_assets.v1` |
| `lightx2v` | `{path, head}`; clean standalone checkout, exact `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be` |
| `diffusers` | `{path, head}`; clean standalone checkout, exact `9284607295a09f759aadd65ed08f48b35feea6d9` |
| `native_config` | `{path, sha256}` of the supplied fully resolved original JSON |
| `immutable_files` | Absolute-file-path to SHA256 map from `immutable_files(config, environment)`; every used converted-model config/index/shard, Grid manifest/BF16 binary, and original local FA3 kernel file |
| `dependency_versions` | Explicit versions for every distribution in `h3_grid_training.DEPENDENCIES`; Diffusers must be `0.40.0.dev0` |
| `environment` | The exact `environment_snapshot(os.environ)` intended on every rank; no changes are applied by this binding |

`immutable_files` and `environment_snapshot` are public helpers in
`distillschedule.training.h3_grid_training`. Prepare pins in the **actual
qualified original runtime**, not from the CPU test dependency overlay.
Runtime package versions and Diffusers/LightX2V import origins are checked
collectively before model construction. No dependency is supplied by a private
path in the implementation. Operators provide their already-qualified sources,
converted checkpoint, original Grid table and compiled FA3 kernel.

Only `H3_ADALN_GRID_MANIFEST` (exactly the supplied config path) and
`H3_FLASH_ATTN3_LOCAL_PATH` are admitted `H3_*` variables. Inherited planner,
benchmark, optimization or `DISTILLSCHEDULE_*` variables are rejected, not
silently unset. Other recorded CUDA/NCCL/Torch/HF/import-path environment values
must match. This reference does not inherit an incumbent profile's environment.

## Public request and execution

Start with [training.original-reference.json](h3_fullseq/training.original-reference.json)
and fill the input inventory, original request, asset and output paths. It is
separate from the frozen optimized `training.json`. Its fields correspond to:

```json
{
  "schema_version": "distillschedule.training.v1",
  "model": {"workload": "W5"},
  "algorithm": {"name": "original_h3_fullseq_grid1000"},
  "data": {
    "input_kind": "original_h3_grid_latent_dataset.v1",
    "prepared": "grid-inputs.json"
  },
  "execution": {
    "adapter": "native", "mode": "reference",
    "task": "h3-request.json", "assets": "grid-assets.json"
  },
  "trainer": {
    "cycles": 3, "seed": 20260817, "checkpoint_every": 1,
    "output_dir": "/srv/project/runs/original-grid"
  }
}
```

`h3-request.json` is an explicit existing `H3Request` (schema, height, width,
frames) matching the original config. There is no selected plan, implementation
override, transport or automatic optimizer selection in this lifecycle.

`distillctl config validate REQUEST.json` checks the public schema and
`distillctl doctor --config REQUEST.json` describes capability. Full lazy CPU
admission is `distillctl train REQUEST.json --dry-run`, or the Python
`validate_native(load_training_config(path))` API. These commands do not
initialize Torch/CUDA, load models, reserve devices, start nodes or qualify
native capacity/numerics. Full admission hashes the supplied files; large
checkpoint/table inventories may take time.

Enter `distillctl train REQUEST.json` on **every rank of an operator-owned,
clean-source, two-node/eight-ranks-per-node torchrun launch**. The binding starts
no cluster launcher. It establishes a Gloo metadata plane before rank-local
source/config/input/dependency checks; model setup and tensor collectives retain
the original NCCL/topology owner. Runtime matched-compute/Grid/capacity checks
remain active. A rank-local admission failure is shared before model use.

## Exact restart and limits

Checkpoints are canonical `training_checkpoint.v2` all-rank atomic directories
under `output_dir/checkpoints/cycle-NNNNNN`, with an additional versioned
original-Grid identity/policy. They store both actual named role LoRAs,
optimizer ownership/hyperparameters/moments/update clocks, actual constant
schedulers, score counter, sampler epoch/rank/next cursor and all native RNG.
The common writer requires all rank shards/hashes and a zero-gradient complete
cycle before publishing the manifest and `_SUCCESS`.

Set `trainer.resume_from` to a complete checkpoint and keep the **same** original
config, target/horizon, cadence, output, inputs, source, dependencies, assets,
environment and topology. Every shard/progress field is admitted before state
mutation; incompatible Adam/scheduler/role/clock state is rejected collectively.
Do not lower the target to manufacture an interruption. To stop durably at
cycle 25 of a 50-cycle run, keep public `cycles: 50`, native
`max_train_iters: 50`, and the full 50-cycle input inventory unchanged; set
public `checkpoint_every: 25` and native `save_every_iters: 25`, then add public
`stop_after_cycle: 25`. This execution-only control never enters the original
native config or changes its scheduler horizon. All ranks must agree on the
stop before construction. The original loop returns only after the complete
G1/F5/EMA0 checkpoint and per-rank progress receipts are published.

An explicit stop requires positive checkpoint cadence and must be either a
cadence boundary or the unchanged terminal horizon (which forces its final
checkpoint even off cadence). Without the optional control, the original
training length and checkpoint cadence are unchanged. For continuation, remove
`stop_after_cycle` or move it to a later valid boundary, set `resume_from` to
the saved directory, and retain the same original config and full horizon.
A stop at or behind the restored cycle is rejected before model construction.
`STOPPED_AT_CHECKPOINT` is partial training, not a completed horizon or a
numerical qualification; the rank receipt records the full horizon and actual
segment separately. Output-only export discards this training stop control.

Historical `h3-a100-shared-backbone-v1`, cursorless `trainer_state.pt`, partial
directories, benchmark receipts and completed-target checkpoints are not exact
resume inputs. Changing target/horizon or resharding still requires a separate
contract. For inference export, restore a complete reference checkpoint through
the [output-only owner](../docs/guides/h3_grid_inference.md#export-an-original-reference-native-checkpoint);
a terminal checkpoint is legal for export, not for training beyond its horizon.
AdaLN/attention/layout caches and
diagnostic counters are rebuildable runtime state, not optimizer checkpoints.

CPU tests may explicitly use `DMD_H3_TEST_UPSTREAM` to locate the real pinned
upstream checkout and `DMD_H3_TEST_DIFFUSERS_SOURCE` for the exact source-only
Diffusers checkout. Full CPU admission fixtures do not import or install that
Diffusers source; identity-only model/table/kernel files are not native assets.
Their tiny differentiable DiT/cache/FSDP boundary substitutes
do not replace original rollout, quantized sigma, loss, backward, clipping,
Adam or scheduler math. CPU/Gloo tests qualify mechanics only, not native
world16 pretrained equivalence, CUDA memory or performance.
