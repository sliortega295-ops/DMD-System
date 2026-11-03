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
[versioned native presets](native_workloads.md#native_presets); for official DMD2/Self-Forcing use
[their optimized recipes](native_workloads.md#official_optimized_training).

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
For an external project, the [optimization harness and Skill](optimization_capabilities.md#optimizer_skill)
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
[export-input preparer](data.md#export_inputs), then construct a request from the
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

<a id="livewan_native_output"></a>
## Current Student export from the full C51 pipeline

Full C51 output restores a complete original two-pool checkpoint; it is not
CPU Program training or official Self-Forcing. Add `trainer.export` from
[the template](../../recipes/self_forcing/export.native.template.json), retaining
source/assets/input/seed/horizon/cadence. Pin the complete FP32 Wan2.1
AutoencoderKLWan config/shards by path/bytes/SHA and real origin/license.
For an assembled full Student without training state, use
[weights initialization](data.md#livewan_weights_initialization), not resume.

```sh
torchrun <your original world16 launch arguments> -m distillschedule export \
  /PROJECT/training-with-export.json \
  --checkpoint /PROJECT/train/checkpoints/cycle-000100 \
  --output /PROJECT/outputs/current-student
```

```sh
distillctl data import /PROJECT/new-inputs.json --output /PROJECT/prepared-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 distillctl generate /PROJECT/outputs/current-student.inference.json \
  --weights /PROJECT/outputs/current-student \
  --prepared /PROJECT/prepared-new/manifest.json \
  --output /PROJECT/generated --output-kind video
```

All original world16 role/Adam/data/RNG shards restore, including Score ranks;
output exits before streamer/KV/text projection/transport/lookahead/update.
Terminal checkpoints are export-only. Rank0 stages its complete resident current
Student and codec; Score ranks only join metadata/error checks, never acquire G.
Parameters/gradients/modes/clocks/RNG remain unchanged; exact nonpersistent
complex RoPE is exported. Sharded/DTensor primary state rejects.

The artifact, companion `.inference.json` and `.native-owner` receipts are
separate from resumable state. New preencoded inputs contain exactly finite
CPU FP32 `context[1,512,4096]` and clean `world[1,16,N,46,80]`, N=3/6/9.
No trace reuse, negative prompt, synthetic history or implicit prefix crop.
The original four-step FewStepStreamer keeps block3/renoise/clean-t0 KV commit,
cache22, no noisy context/graphs and an independent RNG stream. Prefix+suffix
produce21 normalized latents →81RGB frames at368×640; `latents` skips decoding.
The consumer is explicit CPU FP32/MATH, not bitwise CUDA-training equivalence.
Budgets include full G/codec/complex-RoPE/KV/attention and host/cgroup reserve;
its context-local no-grad inference capacity scope does not widen training limits.
Tiny original-class/Gloo proofs do not qualify production world16 CUDA/quality.

<a id="h3_grid_inference"></a>
<a id="h3_grid_inference--export-an-original-reference-native-checkpoint"></a>
## W5: current Grid1000 Student output

W5 uses converted Diffusers MiniMax-H3, LightX2V's original **four** evaluations,
video/audio flow shifts6/3, LoRA128 and new text conditions. It never uses W6's
eight-step/LoRA64/previous-current sampler. Training deleted frozen AdaLN
projections, so an ordinary remaining checkpoint is incomplete for inference.
Export keeps full remaining frozen state, current unmerged Student, exactly four
BF16 Grid-mmap rollout-table entries and native-dtype nonpersistent RoPE. Do not
randomly rebuild projections, merge LoRA, requantize sigma or use continuous
AdaLN. The original first-step duplicate table rows are addressed, not altered.

```text
prompt_embeds:   BF16 CPU [1, text_tokens, 5120]
text_token_tags: int64 CPU [text_tokens], every tag 1
```

```json
{
  "decoder_manifest": {
    "path": "/PROJECT/h3-codec.json",
    "bytes": 12345,
    "sha256": "REPLACE_WITH_ACTUAL_SHA256"
  }
}
```

```sh
distillctl generate /PROJECT/current-grid-student.inference.json \
  --weights /PROJECT/current-grid-student \
  --prepared /PROJECT/new-text/manifest.json \
  --parameters /PROJECT/media-parameters.json \
  --output-kind audio_video --output /PROJECT/generated-av
```

```sh
distillctl export native-export.json --checkpoint /project/checkpoints/cycle-000064 \
  --output /project/current-grid-student
```

`h3_grid_inference.current_student_view` is read-only and preserves DTensors,
not an admission/all-gather. `export_components` projects an explicitly
consolidated CPU snapshot; `build_inference_model` strictly restores the real
converted class/PEFT, all buffers and dtypes after host-capacity admission.
Supply pinned source/terms/assets and Torch2.10/Diffusers0.40.0.dev0, no model
fallback/download. Discovery imports no Torch.

New imported conditions contain exactly CPU BF16 `prompt_embeds[1,text_tokens,5120]`
and int64 `text_token_tags[text_tokens]`, every tag1. Config binds independent
seed/geometry/layout/source/four-step schedule; wrong/previous-chunk inputs reject.
Original noise order, FP32 latents and data-ward velocity remain unchanged.
Latent output is packed BLD `video_tokens`/`audio_tokens`.

For media, the pinned decoder manifest must match original Grid geometry:
video `[1,24,T,H/16,W/16]`, stereo `[2,32,A]`, FPS24/32kHz,
`T=(frames-5)/17*5+2` for17*n+5 frames, `A=round(frames/24*40)`.
Original `unpatchify_video` uses1×2×2 patches; `unpack_audio` is stereo
channel-major. Conversion does not change values/normalization; original VAE
applies mean/std once and its temporal planner checks frames. No crop/resample.
`decoder_manifest` changes decoding only. `distillctl decode` can use the same
bridge with explicit `manifest`, `grid_source`, `geometry` and safe packed inputs.

Native export supports only the original-reference owner, **not** the held
optimized candidate. Add [trainer.export](../../recipes/h3_fullseq/export.native.template.json)
to its unchanged config/source/assets/horizon/seed/cadence/input inventory and
enter on every already-launched world16 rank. Both roles/Adam/schedulers/clocks/
sampler/RNG restore; output validates cursor without loader iteration or next
sample and performs zero updates. Terminal is export-only. Only output receipts
go to a fresh `.native-owner`, never identity-bearing config edits.

All DTensor owners materialize, only primary retains full CPU state. Check exact
table/LoRA/layout/source and host/cgroup/largest-tensor GPU reserves first.
Unused CUDA allocator release is output-only/outside training timers. A failed
companion config invalidates completion but preserves diagnostic payload.
Random original-class/public-CLI/Gloo tests are implementation proof, not
pretrained quality, world16 CUDA or the100-cycle reliability gate.

<a id="h3_ar_native_inference"></a>
## Independent H3 Student inference

The independent W6 consumer includes the full frozen50-block base and current
**unmerged** Student forward-storage weights, never Fake/discriminator/optimizer
or old caches. It uses pinned original DiT and DMD2 (import-path-only derivative),
Torch2.10 CPU/one thread with `h3-cached-reference`, dense attention and original
eight-step continuous schedule under BF16 autocast. BF16/FP32 state remains per
key; FP32 Adam masters never substitute for current BF16 LoRA. No quantization,
merge, alternate sampler or implicit distributed-CUDA fallback.

```sh
distillctl generate inference.json --weights /project/current-student \
  --prepared /project/new-conditions/manifest.json \
  --output /project/new-output --output-kind latents
```

```sh
distillctl export native-export.json --checkpoint /project/checkpoints/cycle-000100 \
  --output /project/current-student
```

Use new [cached previous/current AV](../../recipes/h3_ar/CPU_REFERENCE.md) or
[raw original encoding](data.md#h3_audio_video_encode): prompt, clean previous AV and
packing must match artifact geometry. Current latents are templates for fresh
noise, not replayed training values/RNG. Latent output names `video_latents` and
`audio_latents` include layout/`_SUCCESS`; decode with the original codec below.

For native checkpoint export add
[the explicit template](../../recipes/h3_ar/export.native.template.json) to the
unchanged fresh-stream config: same source, seed, horizon, cadence, assets,
strategy and capacity. Every already-launched world16 rank restores full native
state/FP32-master Adam/RNG. Zero updates occur; terminal is export-only. The
saved actual rank input is resolved without advancing cursor/asking for101.
ZeRO-3 gathers one layer at a time, only primary retains CPU base/Student;
Fake/discriminator are excluded. Strict config/layout/dtypes and output reserves
remain bound. Artifact/companion config enable independent new-input generation.
Meta-model/physical/cgroup checks precede allocation with explicit parameter,
attention, working-set and reserve budgets: do not shrink shape/precision to fit.
Source/CPU proofs are not native CUDA/world16/pretrained output qualification;
W5 Grid1000 is a different binding.

<a id="h3_audio_video_decode"></a>
## Decode original H3 video and stereo audio

`distillctl decode` runs original `MiniMaxH3VideoVAE.decode_video` and
`MiniMaxH3AudioVAE.decode_audio` without Student/Teacher/Fake/training state.
Mean/std, overlap/tiling and synthesis remain original; no latent clamp,
sampling, crop, resize or AV duration resampling. Use Torch2.10 CPU FP32,
one process and the separate `h3-av-codec` lane, not a training-stack change.

Pin the three original source files/terms in
[the decoder template](../../recipes/h3_ar/decoder.native.template.json) and
complete local safetensors or exact decoder-only state. Headers admit only
video `post_quant_conv`/`decoder` and audio `dec_in_proj`/`decoder`; unused
encoders are not materialized. Unknown/missing keys reject. Native BF16 storage
requires explicit `storage_dtype:bfloat16_to_float32`, never silent conversion.
Video is about2.42B parameters and audio64.94M: declare construction/working-set/
host-reserve limits and exact output shapes/FPS/32kHz; the original temporal
planner, not a guessed multiplier, determines frames. Hashes are not licenses.

Prepared inputs are finite CPU FP32 `video_latents` BCTHW(B1,C24) and
`audio_latents` SCT(stereo2,C32), exactly matching the codec manifest. Extract
actual outputs, import and decode with these unchanged examples:

```python
from pathlib import Path
import json
import torch
from safetensors.torch import save_file
from distillschedule.workloads.best_known.registry import file_sha256

root = Path('/project/h3-output')
values = torch.load(root / 'latents/generation.pt', map_location='cpu', weights_only=True)
latents = {name: values['generated'][name].contiguous()
           for name in ('video_latents', 'audio_latents')}
save_file(latents, root / 'native-latents.safetensors')
(root / 'input.json').write_text(json.dumps({
    'schema_version': 'distillschedule.data.preencoded.v1',
    'records': [{'id': 'new-h3-output', 'features': {
        'path': 'native-latents.safetensors',
        'sha256': file_sha256(root / 'native-latents.safetensors')}}],
    'metadata': {'purpose': 'original-H3-decode-only'}}))
```

```sh
distillctl data import /project/h3-output/input.json --output /project/h3-output/prepared
```

```json
{
  "schema_version": "distillschedule.decode_prepared.v1",
  "decoder": {
    "factory": "distillschedule.training.h3_av_codec:build_decoder",
    "parameters": {"manifest": {
      "path": "/project/h3-output/codec.json",
      "bytes": 1234,
      "sha256": "REPLACE_WITH_ACTUAL_CODEC_MANIFEST_SHA256"
    }}
  },
  "data": {"prepared": "prepared/manifest.json", "sample_index": 0},
  "output_kind": "audio_video"
}
```

```sh
distillctl decode /project/h3-output/decode.json --output /project/h3-output/decoded
```

Replace template bytes/SHA with actual pins; config-relative paths resolve from
that config. Output contains PNG frames, stereo32kHz PCM16 WAV,
`generation.pt/json` and `_SUCCESS`, preserving complete independent AV lengths;
muxing is a consumer step. RNG restores and resources close on success/error,
without optimizer/update. This codec is not Student/ZeRO export or an encoder;
[raw encoding](data.md#h3_audio_video_encode) loads different original subtrees.
The random codec fixture is not pretrained/quality/native-GPU proof. Other
models supply existing `ModelComponents.decode`, not planner model-name branches.
