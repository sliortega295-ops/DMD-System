# Original LiveWan finite fresh inputs

`original_livewan_teacher_cohorts.v1` admits the actual pinned LiveWan
`TeacherLatents` format and routes it through the existing worker's real
`InputLoader`. It is **not** a fresh C51 training command, a model checkpoint,
or proof of pretrained quality. A separate opt-in
[original reference lifecycle](../../recipes/self_forcing/REFERENCE_TRAINING.md)
consumes this legacy input kind for finite complete-cycle training and restart.
For the full optimized two-pool pipeline, use the separate
[C51 continuous training and restart recipe](../../recipes/self_forcing/C51_CONTINUATION.md).
That adapter supports an explicit finite Teacher-pool epoch policy, described
below; the legacy default still requires disjoint records across cycles.
The selected C51 two-cycle binding and historical `FormalTrace` semantics,
including the default historical prefix behavior, are unchanged. A historical
50-cycle reused trace is not fresh input.
The immutable semantics are specified in the
[versioned input contract](../contracts/original_livewan_teacher_cohorts_v1.md).

The separate `training.livewan_state.model_bundle_state` API serializes the
actual resident Generator or Critic `ModelBundle` owner in place. It binds the
original LoRA source, frozen-base receipt, named trainables and fixed native
AdamW policy, preserving Parameter objects, gradient views and hooks. These are
state views, **not** a pipeline checkpoint command: they do not establish input
consumption, complete outstanding transfers, reconstruct shadows or supply
physical cycle/version admission. The distinct reference state factory binds
actual world16 Generator DTensors and replicated Critic owners; it does not
relax this resident factory or supply C51 physical drain/recovery.
The CPU state-owner receipt (source checkout: `artifacts/framework/20261021/original_livewan_state_cpu.json`)
separates tiny original-LoRA/Adam continuation from untested full-model and
distributed behavior.

## Supplied assets and preparation

Use a clean local LiveWan checkout at the exact gitlink
`1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`. Preparation verifies its Git identity,
tracked clean state and exact committed bytes of `wanstreamer/data.py`, the
teacher/prompt generator scripts, and Apache-2.0 license. No upstream source
is vendored or downloaded, and global import paths are not rewritten. The
original dataset file is imported directly under an isolated module name.

Supply a directory containing **exactly 80 × N** original `.pt` records for
1..64 finite complete cohorts, plus the original prompt bundle **outside**
that directory. Each record needs the original `gen_teacher.py` dictionary:
`latents`, `prompt_idx`, `seed`, `size`, `frames`, `steps`, `cfg`. Admission
requires exact float16 `[16,21,46,80]` latents, 81 pixel frames, `640*368`, 40
teacher sampling steps and finite valid metadata. No cropping, padding,
resampling or dtype repair is performed. The original prompt bundle requires
`pos: [P,512,4096]`, `neg: [1,512,4096]` (float16), `prompts` text and
`neg_prompt`. Its file bytes, positive tensor identities and record-to-prompt
bindings are hashed. Multiple genuine samples may share a `prompt_idx`;
their record/dataset identities remain independent of that prompt index.

```bash
distillctl data prepare-livewan-inputs \
  --data-root /ABSOLUTE/PATH/teacher --prompts /ABSOLUTE/PATH/prompts.pt \
  --source-root /ABSOLUTE/PATH/qualified-livewan \
  --cycles 2 --seed 20260830 \
  --output /ABSOLUTE/PATH/prepared/livewan-inputs.json
```

The equivalent Python preparation API is in the data layer:

```python
from distillschedule.data.livewan_inputs import prepare_inputs

prepare_inputs(
    "/ABSOLUTE/PATH/teacher", "/ABSOLUTE/PATH/prompts.pt",
    "/ABSOLUTE/PATH/prepared/livewan-inputs.json",
    source_root="/ABSOLUTE/PATH/qualified-livewan",
    cycles=2, seed=20260830,
)
```

This publishes complete, fsynced manifest bytes atomically, never overwriting
an existing file. Failed admission or serialization leaves no partial final
manifest, so a clean retry can use the same output path. All archives first pass
`torch.load(weights_only=True)` before any original dataset load; unsupported
pickle globals reject without an unrestricted fallback. File and semantic
latent hashes reject missing/changed files, repeated paths, copied archives
and reserialized copies of the same latents. Keep admitted source/archives
immutable while consuming them; file hashes are rechecked at each read.
The original dataset reads only admitted canonical record paths; retargeting a
directory-entry symlink does not redirect its later unrestricted read. This is
not a sandbox against concurrent writers replacing admitted files themselves.
Generation provenance and pretrained quality cannot be inferred from an
archive's format or hashes. For CPU fixture use, set
`evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED"` explicitly.

## Actual native consumer and data cursor

```python
import torch
from distillschedule.runtime.livewan_crossscale_worker import InputLoader
from distillschedule.training.livewan_inputs import KIND, NativeTeacherCursor

loader = InputLoader(torch=torch, input_kind=KIND,
                     prepared="/ABSOLUTE/PATH/prepared/livewan-inputs.json")
cursor = NativeTeacherCursor(loader, rank=0, world_size=16)
for _ in range(5):
    row, (latent, context, prompt_idx) = next(cursor)
data_state = cursor.state_dict()  # complete logical G1/C5 data boundary only

restored = NativeTeacherCursor(loader, rank=0, world_size=16)
restored.load_state_dict(data_state)
assert restored.peek() == cursor.peek()
```

Every next sample calls the real original `TeacherLatents.__getitem__` through
the production `InputLoader`, not a generic prepared-feature substitute. Each
logical rank owns its B1 slot in each of five steps; all 16 ranks together own
80 distinct samples per cohort. New cohorts are disjoint. The frozen contract
is 368×640/81 frames, global B16, G1/C5/EMA0 and continuous shifted-uniform
sigma. Operation keys and six independently derived seed streams follow the
existing native trace convention, but this new stream is **not** continuation
of upstream sequential training RNG. Inventory/permutation, reconstruction
and cursor seeking do not draw from Python or Torch's training RNG.

`state_dict`/`load_state_dict` support the existing training checkpoint
stateful-object serializer. State binds the full source/input/prompt/order/
seed identity, logical rank, world16 topology, complete cycle and next
operation. Exhaustion stops: there is no implicit next epoch or reuse. Cursor
progress advances only after successful native input consumption. This is
logical data ownership, **not** a claim that C51's physical role-pool pipeline
has reached a save-safe boundary.

## Deliberately not implemented here

The legacy input adapter alone does not supply C51 training or restart. The
separate [full C51 adapter](../../recipes/self_forcing/C51_CONTINUATION.md)
connects the existing native cycle owner to the public lifecycle. Its drain
boundary occurs **before** next-cycle precompute and waits for transport,
shadow, optimizer and staged-send effects. It saves Generator trainables,
Critic LoRA, both named Adam states, versions, rank-local RNG and the data
cursor in an atomic all-rank same-topology checkpoint. Frozen bases and
temporary caches are reconstructed; EMA remains zero. The separate
[current-Student export and generation guide](livewan_native_output.md)
covers the native output bridge. Neither route substitutes another training
loop or changes the legacy input contract.

The opt-in CPU tests require `DMD_LIVEWAN_TEST_UPSTREAM` to name the actual
pinned checkout. They use stride-expanded native-shape fixtures with small
storage, the original source dataset and real `InputLoader`; they do not
prove pretrained quality, CUDA/FSDP/transport execution or model restart.

## Explicit finite-pool epochs for the C51 stream

The separately opt-in [finite teacher-pool epoch policy](../contracts/original_livewan_teacher_epochs_v1.md)
supports ordinary dataset reuse through the **existing**
`self_forcing.c51_stream.v1` native lifecycle. It does not change the defaults
described above: without `--input-policy`, C51 v1 still requires exactly
`80 * cycles` distinct records; the original reference input kind remains
unchanged. With this policy, at least 80 real distinct teacher records can
supply 100 finite cycles without inventing 8000 teacher archives.

```bash
distillctl data prepare-livewan-c51-inputs \
  --data-root /ABSOLUTE/PATH/teacher --prompts /ABSOLUTE/PATH/prompts.pt \
  --source-root /ABSOLUTE/PATH/qualified-livewan \
  --cycles 100 --seed 31415 \
  --input-policy distillschedule.livewan_teacher_epochs.v1 \
  --output /ABSOLUTE/PATH/prepared/livewan-epochs.json
```

The Python equivalent is `prepare_c51_inputs(..., cycles=100, seed=31415,
input_policy=EPOCH_POLICY)`, importing `EPOCH_POLICY` from
`distillschedule.data.livewan_inputs`. The manifest keeps the C51 input kind
and adds a separately versioned `input_policy` with exact order and draw count.
Keep `data.input_kind="original_livewan_c51_teacher_stream.v1"` and explicitly
select `execution.preset="self_forcing.c51_stream.v1"` in the existing native
training config. No recipe, asset/capacity gate or trainer defaults change.

The fixed seeded permutation repeats without dropping records. Stable record
identity is distinct from logical draw/epoch/cohort identity. Every complete
cycle still has 80 distinct records; every logical operation has its own
existing rollout/sigma/noise/gradient/control seeds. Input iteration and
restoration never consume training RNG. Runtime receipts explicitly disclose
record reuse rather than labelling reused archives as fresh. Manifest policy,
horizon, order and seed are checkpoint identity: do not regenerate a different
stream for resume. C51's existing complete dual-wave cursor, safe drain,
parameter/optimizer state and cache-version rules remain mandatory.

CPU tests in `tests/test_training_livewan_epoch_inputs.py` use only 80 tiny
native-format fixture archives and the actual original dataset/loader. They
include logical horizon100/cursor50 restore and tiny original-LoRA/Adam
new-process continuation, not real pretrained training or native CUDA quality.
No new training run or speedup is claimed by this input-policy extension.
