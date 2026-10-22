# Train through the six original native integrations

`execution.adapter: native` lowers a public training configuration to the
existing original-model runtime. It does **not** replace those algorithms with
the small Program example, infer a historical winner, invent costs, reserve
GPUs, or change native numerical gates. For the extensible model/algorithm
factory lifecycle, use the Program training adapter instead.

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
| [W1](../../recipes/wan_ar/training.json) | Wan hierarchical AR, No-GAN, default G1/C5/EMA5 | `strategies.wan_cli.main` -> `wan_worker.run_worker` | 2..64 fresh cohorts with a compatible saved policy or explicit manual/reference strategy; literal incumbent bundle remains fixed-trace/two-cycle |
| [W2 selected](../../recipes/helios_nogan/training.json) / [direct reference](../../recipes/helios_nogan/training.reference.json) | Helios14B hierarchical AR, G1/C5/EMA5 | Selected: `search.execution.run_plan`; direct reference: original P0 setup and preparation/update helpers | Selected: 2. Direct reference: 2..64 fresh cohorts, explicit original Step10 G2/C10/EMA10 bootstrap |
| [W3](../../recipes/helios_gan/training.json) | Helios14B DMD + GAN-G/GAN-D, G1/C5/EMA1 | `search.execution.run_plan` | Exactly the selected native contract's 2 |
| [W4](../../recipes/self_forcing/training.json) | C51 Self-Forcing, continuous sigma, G1/C5/EMA0 | `search.execution.run_plan` | Exactly the selected native contract's 2 |
| [W5 selected](../../recipes/h3_fullseq/training.json) / [original reference](../../recipes/h3-grid-original-training.md) | Full-sequence AV Grid1000, 4-step, G1/F5, LoRA128 | Selected: `search.execution.run_plan`; original reference: `grid_entrypoint` and original Grid trainer | Selected: 2. Original reference: 2..64 finite cached-condition cycles |
| [W6](../../recipes/h3_ar/training.json) | Original DiffSynth previous+current AV, continuous sigma, 8-step, G1/F1, LoRA64, ZeRO-3 | `strategies.h3_ar_cli.main` -> `h3_ar_worker.run_worker` | Exactly 2 native updates |

These are integrations of original native owners, **not** six arbitrary
long-training engines. Fresh-stream support is explicit for W1/W2/W5;
W3/W4/W6 retain bounded native APIs. A request for 100 cycles on a two-update
runtime is rejected, never silently executed as two. Unsupported shapes or protocol
fields reject through the original typed task/request validators. W5 is not a
continuous-sigma variant, and W6 is never dispatched through W5 Grid1000.

## Reference before planning; no hidden winner

W1 and W6 have real direct-reference entrypoints:

* [W1 reference configuration](../../recipes/wan_ar/training.reference.json)
  explicitly supplies [the original FSDP strategy](../../recipes/wan_ar/reference_strategy.json).
  It lowers to `benchmark-manual --implementation ... --cycles 2`.
* [W6 reference configuration](../../recipes/h3_ar/training.reference.json)
  explicitly supplies `zero3-native`; it lowers to `benchmark-manual` with the
  exact native capacity policy. `zero3-ap` is available only as an explicit
  manual implementation or as a selected measured plan.

These commands execute **inside** an already configured world16 torchrun rank
environment on each of two eight-GPU nodes. The training binding does not
invent rendezvous settings, change CUDA visibility, switch allocator policy,
fall back to ZeRO-2, or launch nodes for them. Follow the original
[Wan runbook](../runbooks/wan_compute_state_planning_v1.md) and
[W6 runbook](../runbooks/original_h3_ar_native.md).

W2 also has a [direct reference recipe](../../recipes/helios_nogan/REFERENCE_TRAINING.md)
without a historical trace or node manifest; W5 has an [original-config reference
lifecycle](../../recipes/h3-grid-original-training.md). Neither route inherits a
measured winner. Both require their own qualified original assets and data,
and execute inside an operator-owned world16 allocation, not a shell launcher.

The other W2-W5 bindings accept saved-plan mode only. To run their corresponding
reference first, explicitly resolve the intended generated reference candidate
through the existing [native calibration workflow](../runbooks/native_planner_workflow.md),
then provide its source-bound resolved receipt for W2/W3/W5. W4's public C51
recipe accepts the full C51 selection only; run its matched control through
the existing calibration workflow, not by relabeling C34 as C51. An optimized
candidate is never substituted when a reference or measurement is unavailable.

Plan mode consumes `execution.plan` **as saved**. It does not run a search,
choose a best-known profile, rebind a run ID, or overwrite an earlier run. Make
any fresh run ID using the existing native replay/rebind workflow **before**
providing the plan here. W2-W5 rederive source/runtime node commands with the
original compiler, verify registered candidate/profile/semantic/request/asset
bindings, and dispatch the same object to `run_plan`. W1/W6 use their original
selected-plan validators, including hashed costs, source, operation order,
versions and measured capacity. A CPU-fixture plan cannot launch a GPU worker.

W4 requires an explicit `c51_full_v1` selected implementation; the old C34
compatibility profile is not a fallback. The saved selection remains the source
of mesh, retention, precision and kernel policy for every workload.

## Required inputs and portable configuration

| Field | Required content |
| --- | --- |
| `model.workload` | W1-W6; not a best-profile ID or an arbitrary model factory |
| `execution.task` | Original Wan/H3-AR task JSON (W1/W6), `OriginalHeliosRequest` (W2/W3), `LiveWanRequest` (W4), or `H3Request` (W5) |
| `execution.assets` | Exact native asset JSON; selected W2-W5 must be byte-equivalent as JSON to the saved plan's `asset_config`. W2/W5 direct references use their separately documented asset schemas, not that dispatcher manifest |
| `execution.plan` | Selected source-bound native plan for plan mode; omitted for explicit reference/manual mode |
| `execution.implementation` | Explicit Wan strategy JSON path or W6 `zero3-native` / `zero3-ap` enum; reference/manual only |
| `execution.capacity_policy` | W6's existing common host/GPU capacity-policy JSON; must equal the saved plan in plan mode |
| `execution.transport` | Optional existing SHA-bound reviewed R9 deployment for its supported W5 subspace; never a generic shell-command transport |
| `execution.expected_head` | Optional exact source HEAD assertion; saved-plan source checks are always mandatory |
| `trainer.output_dir` | Explicit project output directory; existing native ownership/non-overwrite rules remain active |
| `trainer.cycles` | Actual supported native count, checked against selected contract; not a timing normalization |
| `trainer.checkpoint_every` | W1/W2/W5 explicit fresh-input lifecycles: supported at complete cycles; all selected benchmark bindings still require `0` |

For W2-W5, extract `asset_config` from the plan you actually compiled; do not
edit a path in a saved receipt to make it fit another machine. Asset locations
inside a native manifest follow its original API (usually explicit absolute
paths), not the training-config path rules. Existing original asset preparation
remains authoritative: [Wan asset template](../../configs/distillschedule/original_wan/assets.template.json),
W6 `workloads native prepare-assets`, and the workload runbooks. Assets carry
real SHA256 identities, not placeholder zero hashes or a claim that a directory
exists. Physical source/runtime, upstream revision, full tensor/trace, allocated
hardware and live capacity gates still execute on the original worker before
model construction; CPU doctor reports that deferred scope explicitly.

W1 fresh training uses `data.prepared` **only** with
`data.input_kind: original_wan_cohort_stream.v1`: the manifest prepared by the
original `workloads native prepare-stream` API. The number of complete,
distinct RNG-qualified cohorts must match `trainer.cycles`; the selected
compatible policy is reused, not its old trace's time/capacity prediction.
This is not the generic PreparedDataset manifest. To deliberately replay a
fixed trace, omit `prepared` and `input_kind` and explicitly set
`data.replay_fixed_trace: true`; that is replay evidence, not fresh-data training.

W2 direct training requires `data.input_kind: original_helios_cohort_stream.v1`
and its prepared `stream.json`. The source inventory contains five real
step/rank-qualified feature sets per cohort, not a recycled benchmark trace.
Its [portable asset template](../../recipes/helios_nogan/assets.reference.template.json)
pins only used model/pre-cast/bootstrap/negative roles, including the actual
EMA manifest. The original Step10 optimizer/EMA boundary remains mandatory;
cold-start Step0 and arbitrary checkpoint versions are not supported.

W5 original training requires `data.input_kind: original_h3_grid_latent_dataset.v1`
and an inventory of genuine original cached conditions. The fully resolved
native config, source/dependency identities, target horizon and data order are
explicit. `trainer.seed` must equal the supplied original config. This is still
the Grid1000 **approximate** objective, not continuous sigma.

Other W2-W6 data remains qualified inside native assets; generic prepared data
is rejected rather than ignored. Except for W5's explicit original-config
binding, native RNG is bound by the source/bootstrap/prepared trace and
`trainer.seed` overrides reject. Reprepare inputs and the matching policy to
change RNG or geometry. Native feature inventories are not the generic
PreparedDataset format and do not use the tiny statistics encoder.

W6's v2 rank-input contract additionally pins original per-rank cache paths,
parser/source hashes and shape metadata, including rank 6's native audio94.
The public template has no personal/site paths, but **does not make arbitrary
W6 cache relocation valid**. A relocation needs a new reviewed versioned input
contract and matched admission; do not crop, pad or rewrite historical locks.

## Lifecycle and evidence are not interchangeable

W1 fresh cohort streams can opt into periodic complete-cycle restart:

```json
{"trainer": {"cycles": 6, "checkpoint_every": 2, "output_dir": "training-run"}}
```

Merge those trainer fields into a complete native training configuration, with
`data.prepared` pointing to the six-cohort native stream. Reference/manual
strategies use that same original worker without requiring calibration; saved
plans retain their source/strategy admission. At restart, keep the **same full
stream and target count**, set `trainer.resume_from` (or CLI `--resume`) to
`training-run/checkpoints/cycle-000002`, and use a new output directory.
Do not recapture inputs or change the source, bootstrap, runtime or layout.
Use a shared persistent checkpoint filesystem and explicit torchrun `GROUP_RANK`.

The restart state is a view of the original engines, not a substitute trainer:
optimizer-owned trainable weights/buffers, named Adam groups and moments,
schedulers, EMA shadows/counter, engine clocks, absolute cohort cursor and
per-rank Python/NumPy/Torch/owned-device CUDA RNG. Frozen weights are rebound by
their original asset hashes; temporary rollout/attention caches are rebuilt.
Only zero-gradient complete cycles publish a manifest after **every rank's
shard** has been written and hashed. State I/O is separately recorded outside
the original compute timer. A failed write is not a valid checkpoint.
CPU tests exercise original dispatch/cursor logic and actual engine/Adam
restart on scalar fixtures; real pretrained Wan/CUDA restart remains NOT_RUN.

W2's direct reference and W5's original-config reference also expose atomic
complete-cycle restart over actual original owners. Their guides specify
distinct fixed-horizon/data/topology restrictions; do not apply W1's CLI
target-extension example to them. Both start metadata admission before model
construction and reject a borrowed application process group. Tests exercise
real original helpers, persistent CPU optimizer owners and real Gloo control
collectives; topology/CUDA/model-construction fixtures are explicitly synthetic,
not world16 GPU evidence.

W3/W4/W6 and selected benchmark bindings do not yet expose complete restart
through this front door. Native generation, decoding and portable inference
export also remain unavailable here. Unsupported requests fail instead of
silently being ignored.
W1's existing final writer saves trainable/Adam/EMA **evidence** automatically
outside native timers (`wan_worker.save_final_state`); it is not a complete RNG/
distributed restart checkpoint. W2-W6 sampled state/fingerprint JSON is not a
checkpoint. The Program adapter supplies the extensible durable lifecycle;
native support must be added per original algorithm before claiming it.

Historical W3 finalization failure/REVIEW, W4 REVIEW and W5 sampled-backward
FAIL boundaries remain unchanged. Existing correctness receipt bindings and
native gates are not waived by this front door. Read the
[six-workload correction](../status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md).
CPU binding tests and dry-runs create **no new large-model GPU validation,
quality result or speedup claim**. Cluster validation remains **NOT_RUN**.
