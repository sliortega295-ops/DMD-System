# Explicit full-C51 fresh-stream continuation

**Source / CPU protocol integration only. Native two-node CUDA continuation,
cross-reference numerical comparison, and 100 actual fresh DMD cycles with a
midpoint process restart are NOT_RUN. Numerical status remains REVIEW.**
This is W4 LiveWan, not official Self-Forcing W4-alternative, not the sequential
reference, and not a new benchmark. Existing recipes/templates remain unchanged.

Use `training.c51-stream.json` as an explicit opt-in request. It enters the same
original C51 worker setup, installers, persistent `ModelBundle`, two streamers,
`_run_pipeline_8x8_continuous` / `_run_pipeline_8x8`, and native optimizers.
There is no reference fallback, synthetic trainer, automatic plan substitution,
dependency installation, cluster reservation, or implicit asset download.

## Inputs and assets

Prepare **80 distinct original Teacher archives per cycle**: 8000 for horizon
100, not two old cohorts relabeled with new keys. The existing restricted-load,
exact source, unique path/archive/latent-content, prompt, full 368x640/81-frame,
global B16 and original four-step/dynamic-world/continuous-sigma gates remain.

```sh
distillctl data prepare-livewan-c51-inputs \
  --data-root /PROJECT/teacher --prompts /PROJECT/prompts.pt \
  --source-root /PROJECT/clean-pinned-livewan --cycles 100 --seed 31415 \
  --output /PROJECT/c51-inputs.json
```

Cached original Teacher data is allowed. Supplied archive metadata does **not**
attest pretrained Teacher quality or upstream sequential RNG equivalence.
The existing six independent operation-keyed seed streams are preserved.
Legacy `prepare-livewan-inputs` remains its own schema and 1..64 contract.

The new asset schema is `distillschedule.livewan_c51_assets.v1`, with exactly:

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
reserve. Inference generation/export are not offered by this adapter.

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
