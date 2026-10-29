# Official Self-Forcing (not LiveWan)

Opt-in external [Self-Forcing](https://github.com/guandeh17/Self-Forcing) at
`33593df3e81fa3ec10239271dd2c100facac6de1`, Apache-2.0. Source and weights are
not vendored. `fetch_source.py OUTPUT` fetches the small hash-locked source
closure and license only. It never downloads model weights or installs anything.

`create_cpu_fixture.py OUTPUT --source-root SOURCE` creates RANDOM tiny assets
of the actual upstream CausalWanModel, WanModel, and WanVAE_ classes and a
common Program configuration. This proves numerical/lifecycle plumbing, not
pretrained quality or CUDA performance. Its explicit randomly drawn native
embedding assets are NOT described as prompt encodings.

Factory defaults are derived from the ORIGINAL pinned merge of
`configs/default_config.yaml` and `configs/self_forcing_dmd.yaml`, not just
constructor fallbacks. The official recipe uses score `timestep_shift=5`,
generator wrapper scheduler shift5, `ts_schedule=false`, G LR `2e-6`, C LR
`4e-7`, real CFG3/fake CFG0, G1/C5, EMA.99 starting at200, block3 and21 latent
frames. Its actual teacher is Wan14B, generator/fake Wan1.3B with explicit
`ode_init.pt` initialization; public assets must supply the matching native
configs and strict local states. This owner does not silently manufacture
those pretrained states or download them. Documented CPU-only exceptions are
FP32 instead of mixed precision, native gradient checkpointing disabled by default, no
FSDP/global batch64 claim, and explicit local prepared conditioning/weights.
The RANDOM tiny fixture additionally changes architecture/spatial size, both
LRs, EMA start/decay, and negative-conditioning provenance for test speed;
its temporal random-exit/CFG/flow/update equations remain original.

## Native contract and public lifecycle

- Actual `DMD.generator_loss`, detached double-precision target MSE, original
  CFG/normalizer, original flow fake-score loss and scheduler.
- Native random-exit cached causal training, prefix no-grad / final grad,
  context-noise RNG draws and KV/RoPE/end-index writes. Critic samples a NEW
  self-rollout and consumes its OWN prompt record, never a cached G sample.
- Each atomic common Program call is one G1/C5 macrocycle: G at native steps
  divisible by five (including zero), C on every native step. Settings can
  explicitly change the ratio. There are six distinct prompt records per
  default cycle, split into `generator` (one) and `critic` (five) streams.
- Native AdamW ordering and original EMA_FSDP arithmetic. EMA initializes
  AFTER incrementing to `ema_start_step` (default 200); only subsequent G
  commits update an existing EMA. No invented LR scheduler is added.
- Common `distillctl train`, checkpoint/resume, eval, export and generate.
  Export is CURRENT G + native VAE only, not EMA, teacher, fake or optimizers.
  Generation runs every exported step of the actual CausalInferencePipeline,
  not the training random exit. Native VAE decoding produces the video.

The factory references are `distillschedule.training.self_forcing:build_model`
and `distillschedule.training.self_forcing:build_self_forcing` (catalog names
are `self-forcing-wan` / `self-forcing` when registered by the host integration).
The generated JSON uses explicit factories so it is independently runnable.

## Explicit local inputs/assets

The `distillschedule.official_self_forcing_assets.v1` manifest contains four
roles (`generator`, `real_score`, `fake_score`, `vae`), each with the ACTUAL
native class `config` and a `weights` object `{ "path": ..., "sha256": ... }`.
It also requires a pinned `negative_prompt_embeds` file in native `1,L,C`
layout. Weights are loaded strictly from these files, never implicit hub paths.
All roles, source closure, local bridge files and numerical dependency versions
enter restart identity. The fixture is a complete readable example.

`distillschedule.data.self_forcing_inputs.prepare_inputs` accepts a manifest
with `schema_version: distillschedule.official_self_forcing_inputs.v1`:

- `mode: preencoded`: records contain unique `id` and pinned `embedding`
  `{path, sha256}` files. They contain EXACT native FP32 `B,L,C` prompt embeddings;
  no scaling, tokenization, dtype conversion or synthesized conditioning occurs.
- `mode: prompts`: records contain unique `id` and raw `prompt` strings (or
  string batches). The manifest must additionally contain `encoder` with
  `source_root`, actual native `T5Encoder` constructor `config`, pinned local
  `weights`, local `tokenizer` directory, and `text_len`. Preparation invokes
  the ORIGINAL WanTextEncoder.forward, ORIGINAL T5Encoder layers and ORIGINAL
  HuggingfaceTokenizer with `local_files_only=True`. Its only forward port is
  the explicit CPU device property. Use the same `encode_text` API to produce
  and pin the matching negative-prompt embedding for the model asset manifest.
  Optional `tokenizer_use_fast` is forwarded to that ORIGINAL tokenizer API
  (default true, as upstream). The tiny actual SentencePiece/T5 qualification
  uses explicit false, avoiding protobuf conversion of a slow-only tokenizer
  without inventing an encoder or installing another dependency.

Raw prompt text becomes model conditioning during explicit data preparation;
training and exported inference accept the exact prepared embedding boundary.
New prompts can be prepared separately and used by exported inference without
the training dataset or training model assets.

## Qualification boundary

The source path accepts one explicit `cpu` or `cuda:N` device, FP32 MATH SDPA,
batch1 only; no autocast or distributed layout. Actual random-tiny CUDA
oracle, lifecycle and attention tests passed on A100/Torch 2.10 under an
explicit deterministic test policy; the CPU-only lane alone still cannot
establish CUDA execution. Native T2V, exactly 21 latent
frames, BFCHW layout, 16 latent channels; whole native frame blocks. Longer-frame
training, I2V, independent first frame and uncached causal/flex training are
explicitly rejected, not emulated. The checkpoint-ON reference below is an
explicit, separately identified source policy, not a replacement for the default.

Single-device compatibility ports are private, source-locked and explicit: constructor
asset injection, true-singleton broadcast, FP32 math SDPA attention (native
layers/cache body unchanged), geometry-derived allocations, and exactly six
training-boundary `.cuda()` lookups changed to `noise.device`. Uncached causal
execution is rejected rather than quietly removing its causal mask. No global
Torch/CUDA/module monkeypatches are installed. Full pretrained GPU training
and media quality remain `NOT_RUN`; tiny tests are not quality evidence. See
[GPU acceptance scope](../../docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).

Modules, native non-buffer schedules, complex128 RoPE, VAE normalization and
negative/unconditional embeddings are resident before AdamW construction.
Placement does not cast native plain tensor dtypes: upstream FP64 conversion
and target-loss arithmetic remain FP64, and RoPE remains complex128. The
runtime identity records matmul/cuDNN TF32, float32 matmul precision and
determinism policies without changing caller-global settings; live training
and inference reject policy drift. FP32 storage/MATH SDPA does not imply TF32
is disabled, and is not native FA/BF16 qualification. The
original `EMA_FSDP` shadow deliberately stays detached CPU FP32, as do Adam's
noncapturable scalar step and host RNG metadata. CPU construction and updates
never initialize or seed unrelated CUDA generators; a CUDA owner scopes only
its own current device and generator. Offline native T5 preparation stays CPU.

The common Program moves prepared embeddings to the explicitly selected
trainer device. Direct owner/inference calls must already supply exact native
embeddings on that device; wrong-device rows are rejected, never silently
copied. Factory and trainer device identity must agree. Checkpoints restore
only the same training device/topology and complete G/C cycle; scheduler aliases,
native plain tensors, moments, negative context and RNG are validated before
mutation, while ephemeral rollout caches are rebuilt. Only current-G/VAE
inference exports allow an explicit device relocation.

The exact tiny Generator-ON profile also has an opt-in
[budgeted resident lifecycle](../../docs/guides/resident_owner_training.md):
manual Fake checkpoint policy, measured `auto`, saved `plan` and same-device
fresh-process resume use one persistent native owner. Construction and trial
capacities are explicit. Without that admission, CUDA auto/plan/owner-selected
trials are rejected before model allocation; whole-owner CUDA copies remain
forbidden. This portfolio does not qualify pretrained/AMP/BF16/native-FA,
distributed or generation-quality support. Optional actual
tiny CUDA regressions are in `tests/gpu/test_training_self_forcing_lifecycle.py`;
CPU-only Torch reports their deliberate `NOT_RUN` skips.

## Explicit checkpoint-ON reference and Fake-only physical policy

Set `model.parameters.settings.gradient_checkpointing` to `true` to choose
the actual native Generator/Fake checkpoint-ON reference. The default remains
`false`, including the existing `training.json` and fixture generator. **Never
treat Generator ON/OFF as equivalent physical strategies:** the pinned causal
source passes different cross-attention-cache arguments in those branches and
changes the gradient path. The bound setting is algorithm/source identity and
cannot be overridden by a plan or mixed across resumable checkpoints.

The pinned models have the old `_set_gradient_checkpointing(module, value)`
signature, which newer Diffusers enable APIs cannot call. During native model
construction only, a private instance-local enable adapter invokes the original
setter through `Module.apply`; it is removed even on failure. No source file,
global Diffusers/Torch method, native forward, or checkpoint function is patched.

For the actual random tiny fixture's exact architecture, 21x4x4 latent geometry,
G1/C5 block3/official denoising schedule and qualified CPU1 FP32 stack
(Torch2.10, Diffusers0.37.1, Transformers4.51.3, einops0.8.1), the ON reference
offers **only noncausal Fake checkpoint OFF** as a physical alternative. Put
this in the same config's `execution` field after selecting the ON reference:

```json
{
  "adapter": "program",
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": false,
    "checkpoint_bindings": [],
    "owner_execution": {"native_cycle": {"fake_gradient_checkpointing": false}}
  }
}
```

Explicit `fake_gradient_checkpointing: true` is also measurable against that
same ON reference. `mode: auto` with `memory_limit_bytes`, `max_candidates: 2`,
`samples: 2`, `max_program_runs: 5` uses the common measured planner; its saved
`planning/selected-plan.json` is reusable via `mode: plan` and `plan: PATH` for
the same initial bound workload. The winner may remain reference: it is never
preselected by model name. Manual, auto, saved-plan and continued-cycle execution
all use this one owner capability, the same runner and the same reference gate.
After resuming an optimized policy, its first real update is paired again from
the restored state. Calibration never advances live model/RNG/data state.

The scoped option changes only the actual Fake flag and installs read-only
block pre-hooks. Counters identify actual entry from the original native
`custom_forward` called by Torch `checkpoint` versus its backward `recompute_fn`
using inspected code objects. They do not infer requests from the option, or
replay the optimizer-containing macrocall. OFF requires zero checkpoint/recompute
entries **and actual direct grad-enabled Fake block execution**. Hooks and flags
are restored on normal exit and errors; no modes, args, cache policy or RNG
semantics are rewritten.

Tests execute three actual tiny native cycles against the source's own
`Trainer.train`: exact losses/gradients, model/Adam/EMA, rollout/cache traces,
clocks and RNG. Public fresh-process manual restart and measured common planning
are separately checked. This is **tiny CPU functional evidence only**, not
pretrained, GPU capacity, memory savings, quality or speedup evidence. Other
settings/architectures/runtimes expose baseline only; explicit unsupported
overrides fail without silently changing the reference. CUDA copying and
auto/plan/owner-execution trial guards remain unchanged.

## Unified runnable entry

For the independent consumed-cache lifetime policy, copy
[`training.consumed-kv.json`](training.consumed-kv.json) and fill the same local
assets/input paths as `training.json`. Its only model change is
`native_cache_lifetime: "release-after-commit-v1"`; the reference default remains
unchanged. It drops rollout-cache references after each completed native
backward/clip and Adam/EMA commit, never during attention or checkpoint replay.
Real release counters appear in cycle metrics. Bytes count relinquished owner
references, not measured freed HBM. CPU native three-cycle and new-process
restore checks pass; this public policy's CUDA/performance acceptance is not
established. See [policy details](../../docs/guides/native_owner_backends.md#official-self-forcing-release-consumed-rollout-kv).

Use the separate `.[train,self-forcing-official]` environment (add `test` only
for development). The public training entry is the same as every other model:

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python recipes/self_forcing_official/fetch_source.py /PROJECT/dependencies/Self-Forcing
python recipes/self_forcing_official/create_cpu_fixture.py /PROJECT/random-native-fixture \
  --source-root /PROJECT/dependencies/Self-Forcing --cycles 3

distillctl train /PROJECT/random-native-fixture/training.json --cycles 1 \
  --output /PROJECT/partial
distillctl train /PROJECT/random-native-fixture/training.json --cycles 3 \
  --resume /PROJECT/partial/checkpoints/cycle-000001 --output /PROJECT/resumed
distillctl export /PROJECT/random-native-fixture/training.json \
  --checkpoint /PROJECT/resumed/checkpoints/cycle-000003 --output /PROJECT/current-G
```

This helper creates explicitly random original-class assets/embeddings, **not
pretrained quality data**. For real raw prompts, fill
`inputs.prompts.template.json` with the actual local native T5 architecture,
checkpoint SHA and tokenizer directory, then use
`distillctl data prepare ... --preparer distillschedule.data.self_forcing_inputs:prepare_inputs`.
The template's T5 dimensions describe the pinned native UMT5-XXL constructor,
not a tiny substitute; the bounded test builds a separately labeled small T5.
`inputs.preencoded.template.json` supplies exact already-native embeddings.
The static `training.json` selects the actual registered model/workflow and
independent G1/C5 streams; fill its explicit paths before use.

For exported inference, use the exact `model.factory` and `model.parameters`
from `export.json` in the inference request, a newly prepared input manifest
and explicit inference seed. `distillctl generate ... --weights /PROJECT/current-G
--prepared NEW/manifest.json --output-kind video --output NEW/generated` runs
the native full-step causal inference + actual VAE. This is a model-only
projection; old training assets, optimizer or prepared training prompts are
not required. The [common guide](../../docs/guides/training.md#generate-and-export)
explains checkpoint versus inference identity.

## Optimized distributed configuration

Use [`training.optimized.json`](training.optimized.json) for the integrated optimization bundle. See the [launch, mechanisms and baseline/optimized performance guide](../../docs/guides/official_optimized_training.md). This uses the same continuous-training and checkpoint commands as the reference configuration.

Both native world16 FSDP2 configurations use these role architectures:

| Role | Native class | Blocks / width |
|---|---|---|
| Generator | `CausalWanModel` | 30 / 1536 (Wan1.3B) |
| Teacher / Real Score | `WanModel` | 40 / 5120 (Wan14B) |
| Fake Score | `WanModel` | 30 / 1536 (Wan1.3B) |

The public constructor checks these boundaries on meta tensors before loading
role payloads. A 1.3B Teacher is not a substitute for this profile. Supply native
FP32 role checkpoints in the ZIP format written by the default `torch.save`.
The distributed loader maps the read-only source pages, then copies into each
role's independent parameters before sharding; training never mutates the asset
file. This avoids a second private full-Teacher input allocation on every rank.
The ordinary single-device reference still accepts its existing legacy format.
