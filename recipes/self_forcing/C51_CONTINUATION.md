# Explicit full-C51 fresh-stream continuation

**Source / CPU protocol integration only. Native two-node CUDA continuation,
cross-reference numerical comparison, and 100 actual fresh DMD cycles with a
midpoint process restart are NOT_RUN. Numerical status remains REVIEW.**
This is W4 LiveWan, not the separate official Self-Forcing workflow or sequential
reference, and not a new benchmark. Existing recipes/templates remain unchanged.

Use `training.c51-stream.json` as an explicit opt-in request. It enters the same
original C51 worker setup, installers, persistent `ModelBundle`, two streamers,
`_run_pipeline_8x8_continuous` / `_run_pipeline_8x8`, and native optimizers.
There is no reference fallback, synthetic trainer, automatic plan substitution,
dependency installation, cluster reservation, or implicit asset download.

## Inputs and assets

Each cycle consumes **80 distinct original Teacher archives**. By default,
records are also disjoint across cycles: horizon 100 requires 8000 archives.
Alternatively, explicitly enable the finite-pool epoch policy shown below to
reuse at least 80 genuine archives across cycles. It preserves distinct records
within each cycle and fresh operation-keyed RNG, without relabeling archives
as new Teacher samples. The existing restricted-load,
exact source, unique path/archive/latent-content, prompt, full 368x640/81-frame,
global B16 and original four-step/dynamic-world/continuous-sigma gates remain.

```sh
distillctl data prepare-livewan-c51-inputs \
  --data-root /PROJECT/teacher --prompts /PROJECT/prompts.pt \
  --source-root /PROJECT/clean-pinned-livewan --cycles 100 --seed 31415 \
  --input-policy distillschedule.livewan_teacher_epochs.v1 \
  --output /PROJECT/c51-inputs.json
```

Omit `--input-policy` for the original disjoint-record policy. Both paths use
the same native `TeacherLatents` reader and existing training owner. See the
[input policy](../../docs/contracts/original_livewan_teacher_epochs_v1.md) for
the exact permutation, progress and checkpoint identity.

Cached original Teacher data is allowed. Supplied archive metadata does **not**
attest pretrained Teacher quality or upstream sequential RNG equivalence.
The existing six independent operation-keyed seed streams are preserved.
Legacy `prepare-livewan-inputs` remains its own schema and 1..64 contract.

The default asset schema is `distillschedule.livewan_c51_assets.v1`, with exactly:

- `native`: the complete existing `distillschedule.livewan_reference_assets.v1`
  object described in [REFERENCE_TRAINING.md](REFERENCE_TRAINING.md), including
  clean real LiveWan/Wan2.1 gitlinks, complete Student/Critic/Teacher inventories,
  explicit base-or-Stage1 Student, original LoRA32 seed, zero initial versions /
  Adam steps, pinned dependency versions and original precision controls.
  Its reference FA2 execution API is **not** used by C51.
- `wan21_overlay`: `{ "root": "/PROJECT/original-livewan-wan21-overlay",
  "files": { ... } }`. The existing
  `scripts/distillschedule/prepare_livewan_crossscale_source.sh` produces the
  original attention/config overlay without editing either gitlink. The
  `training.livewan_c51_assets.overlay_inventory(native, root)` read-only helper
  verifies its entire imported Python inventory against those real gitlinks and
  returns this object. A clean unpatched Wan checkout is not the C51 backend.
  The patched tree is **not** claimed to be a clean/fabricated Git checkout.
- `kernels`: `fa2` and `fa3`, each `{ "root": "/PROJECT/explicit-kernel-root",
  "files": { ... } }`. `training.livewan_c51_assets.kernel_files()` gives the
  mandatory unchanged C51 artifact hashes. Missing or changed files fail before
  model construction. No private historical run/mount is required.

### Existing local Hugging Face attention providers

For the same native FA2/FA3 kernels supplied as local HF artifacts rather than
a pip `flash-attn` distribution, explicitly select
`distillschedule.livewan_c51_assets.v2`. It keeps the fields above and adds
`attention_provider`. The nested `native.runtime.dependency_versions` records
the other eight original distributions; a bare reference-v1 document still
requires all nine. Do not invent a `flash-attn` version or change the reference
asset contract to select this path.

Use `kernel_files(hf_provider=True)` for the complete seven-file inventory of
each component. Construct the provider descriptor in the intended runtime:

```python
from pathlib import Path
import torch
from distillschedule.runtime.livewan_c51_provider import ABI, SCHEMA
from distillschedule.workloads.best_known.registry import file_sha256

attention_provider = {
    "schema_version": SCHEMA,
    "abi": {**ABI, "torch_extension_sha256": file_sha256(Path(torch._C.__file__))},
}
```

The v2 owner binds the real provider functions, native extensions and ABI to
its effective plan and checkpoint. It uses the original loaders and attention
math, with no dependency installation or kernel substitution. CPU provider
admission does not import the original Wan model package, whose T5 class
default requires an owned CUDA context; native worker admission retains that
import and source check. Record the actual training environment, not a CPU
preparation snapshot, in the training asset document.

Use the same explicit PyTorch-only import controls during asset preparation
and on every worker:

```bash
export USE_TF=0
export USE_FLAX=0
```

These controls are included in the asset environment identity. They prevent
unneeded TensorFlow/Keras/OpenCV imports during model setup from changing the
dynamic-library search path after it has been recorded.

Original Wan `save_pretrained` configs omit several `ignore_for_config`
constructor arguments. The adapter uses their pinned original defaults for
patch shape, attention window and norm flags. Explicit conflicting values,
missing core geometry or changed config/weight hashes still fail.

Set `DISTILLSCHEDULE_FLASH_ATTN_KERNEL` and `DISTILLSCHEDULE_SCORE_FA3_KERNEL`
to those exact explicit roots. All other effective recorded environment,
dependency, dtype/TF32 and backend controls must match the admitted native
asset object. Execution requires the existing validated **2.10.0+cu128** runtime
and the original CUDA/native dependencies; CPU protocol tests on newer Torch do
not qualify it. Model payloads are fully hashed during startup admission and
rechecked once after construction, **not reread every cycle**.

Supply an explicit capacity policy with schema
`distillschedule.livewan_c51_capacity.v1` and the unchanged common
`gpu_metric`, all-rank `memory_limits`, `host_limits`, `gpu_reserve_bytes` and
`host_reserve_bytes` gates. There are no fabricated default budgets or disabled
numerical/capacity checks. Large checkpoint staging must fit the admitted host
reserve. For current-Student export and new-input generation, use the
[native output guide](../../docs/guides/livewan_native_output.md) with a
complete checkpoint from this adapter.

## Horizon 100, stop at 50, fresh-process restore

Enter the public training command on each **already launched** rank using the
repository's native training CLI. The owner requires two physical nodes, eight
ranks per node, exact `RANK/LOCAL_RANK/LOCAL_WORLD_SIZE/GROUP_RANK/WORLD_SIZE`,
and owns fresh NCCL plus Gloo metadata groups. It refuses an embedding group.

The example keeps the admitted input horizon and seed fixed at 100/31415,
publishes `checkpoints/cycle-000050/_SUCCESS`, and stops after cycle 50.
`stop_after_cycle` is a complete checkpoint boundary: **no cycle-51 lookahead
or shadow broadcast is submitted before that publication or after the stop**.
For a new process, keep source/assets/inputs/seed/horizon/checkpoint cadence /
capacity/topology exact, set `resume_from` to that checkpoint, set
`stop_after_cycle` to 100 (or omit it), and choose a new shared output directory.
The stop control/output path are not changes to the admitted training horizon.

G lanes actually load both assigned samples (ten per lane/cycle); Score lanes
consume the corresponding received latents and source-qualified prompts.
Physical G B1 and Critic B2 are deliberately different. Depth-one lookahead
continues outside requested drain boundaries. Two pinned host slots per role
provide real send/H2D-completion backpressure independent of horizon length.
Publication reconciles all 16 actual physical clocks and 80 matched row/record
identities, drains submitted host work, rejects pending shadow/reducer work,
checks original C51 operators plus fresh row-derived calls and finite outputs,
and rechecks live host/HBM/OOM reserve.

Checkpoints contain actual owned G or Critic trainables/buffers/modes, named
materialized Adam, local physical versions, dual-wave history/cursors and
per-rank Python/Torch/CUDA/NumPy RNG. There is no native scheduler in this fixed
original optimizer policy and EMA remains zero. Frozen Teacher/bases are
reconstructed from explicit inventories; streamer/KV/SAC/cast/transport/shadow
scratch is not serialized. All shards/identities/layouts are staged before any
live owner mutation; resume first syncs current Score LoRA into the G shadow.

Receipts distinguish per-segment physical commits, complete logical progress,
actual mechanism counters and current-Student **training-rollout** fingerprints.
They are not a new inference-quality, parity, stability or speedup claim.
