# Original LiveWan finite teacher cohorts, version 1

Input kind: `original_livewan_teacher_cohorts.v1`.
Manifest schema: `distillschedule.original_livewan_teacher_cohorts.v1`.
Cursor schema: `distillschedule.livewan_teacher_cursor.v1`.

This contract describes input ownership only, not a new training algorithm,
C51 checkpoint, model export, or quality measurement. Historical FormalTrace,
C27 source hashes and C51 selected-plan contracts remain unchanged.

## Source and record identity

- Original LiveWan gitlink `1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`, clean
  tracked tree, exact committed dataset/generator/prompt-generator/license bytes.
- Exactly `80 * N` distinct original teacher records for `1 <= N <= 64` finite
  cohorts. Canonical paths, archive bytes and latent tensor content must be
  distinct. Copies, replay, missing records, padding and silent repair reject.
- Each archive has only `latents`, `prompt_idx`, `seed`, `size`, `frames`,
  `steps`, `cfg`: finite FP16 latent `[16,21,46,80]`, 81 frames, `640*368`,
  40 teacher-generation steps, nonnegative integer seed and positive finite CFG.
- The original prompt bundle has FP16 `pos[P,512,4096]`, `neg[1,512,4096]`,
  matching nonempty prompt strings and a negative-prompt string. Its archive
  and positive tensor identities are bound. A record index is not a prompt
  index; independent records may share one prompt.
- Restricted tensor loading precedes original dataset loading, without an
  unrestricted fallback. The actual original TeacherLatents consumes the
  admitted canonical files. Source and archives remain operator-owned and
  immutable; concurrent local filesystem writers are outside this guarantee.
- Only complete manifest bytes publish, with exclusive atomic publication.
  Existing output files and historical evidence are never replaced.

## Logical schedule and RNG

The unchanged workload shape is 368x640, 81 frames, logical world16/globalB16,
G1/C5/EMA0 per complete cycle, continuous shifted-uniform sigma. Each cohort
owns five steps and sixteen logical B1 slots. Cohorts do not share records.
Teacher generation steps in an input archive are not Student rollout steps.

A local RNG seeded by the explicit 63-bit seed determines record order. Keys
are `step=S/slot=R` for a single cohort, otherwise
`cycle=C/step=K/global_step=S/slot=R`. World-frame and RoPE-gap controls and the
six existing operation seed streams are derived from those keys. Construction,
seeking and consumption do not draw from Python/Torch training RNG. This
independent operation-keyed stream is **not** upstream sequential-RNG
continuation; the manifest states that distinction explicitly.

## Cursor and failure behavior

A logical rank cursor advances only after actual original dataset consumption.
Save/restore admits only complete five-step boundaries and binds source,
input/prompt/order/seed identity, rank, world16 topology and next operation.
Foreign identities, incomplete cycles and changed payloads reject before cursor
mutation. Exhaustion stops; there is no implicit next epoch or trace replay.

This logical cursor is not acknowledgement of C51 physical two-wave consumption
or a drained role-pool pipeline. Fresh public W4 training remains rejected until
those native lifecycle owners are connected and qualified. No GPU execution,
pretrained quality or full native model restart follows from data admission.

See the [operator guide](../guides/livewan_inputs.md) for preparation and use.
