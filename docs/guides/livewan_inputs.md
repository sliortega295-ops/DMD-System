# Original LiveWan finite fresh inputs (data consumer only)

`original_livewan_teacher_cohorts.v1` admits the actual pinned LiveWan
`TeacherLatents` format and routes it through the existing worker's real
`InputLoader`. It is **not** a fresh C51 training command, a model checkpoint,
or proof of pretrained quality. Public W4 training explicitly rejects this
input kind until its native complete-cycle lifecycle seam is implemented.
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
physical cycle/version admission. The public fresh-training guard remains.
The [CPU state-owner receipt](../../artifacts/framework/20261021/original_livewan_state_cpu.json)
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

Before fresh public C51 training or restart, the existing native cycle owner
must expose a drain boundary **before** next-cycle precompute. All transport,
shadow, optimizer and staged-send effects must be complete; a barrier alone
does not suffice. Exact generator trainables, critic LoRA, both named Adam
states, versions, rank-local RNG and this data cursor must be saved in an
atomic all-rank same-topology checkpoint. Frozen bases/temporary caches and
teacher/shadow identities must be reconstructed and verified. EMA remains
zero. Native VAE output/export is another separate bridge. This unit neither
lifts worker cycle bounds nor supplies a substitute solver/update loop.

The opt-in CPU tests require `DMD_LIVEWAN_TEST_UPSTREAM` to name the actual
pinned checkout. They use stride-expanded native-shape fixtures with small
storage, the original source dataset and real `InputLoader`; they do not
prove pretrained quality, CUDA/FSDP/transport execution or model restart.
