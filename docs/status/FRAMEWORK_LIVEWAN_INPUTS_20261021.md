# Original LiveWan input-owner integration

This source/CPU follow-up adds public offline preparation of the actual pinned
TeacherLatents format and its consumption by the existing native InputLoader.
It is not full W4 training, checkpoint recovery, generation quality or timing.
See the [versioned contract](../contracts/original_livewan_teacher_cohorts_v1.md),
[operator guide](../guides/livewan_inputs.md) and
[machine-readable receipt](../../artifacts/framework/20261021/original_livewan_inputs_cpu.json).

## What actually executes

The original licensed LiveWan dataset file at gitlink
`1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5` is imported directly after source and
restricted-archive admission. Real original TeacherLatents reads pass through
the production InputLoader, not a copied dataset or package substitute. Records
retain independent prompt identities; prepared cohorts are finite and disjoint.
The logical world16 rank cursor uses existing complete-cycle state serialization.

The public preparation command is `distillctl data prepare-livewan-inputs`.
It does not download data or models, repair shapes/dtypes, repeat samples or
consume training RNG. Archives in CPU tests have native tensor shapes with
expanded small storage and explicitly declare fixture status. They are not
pretrained Teacher generations or quality evidence.

## Review and observed verification

Behavior commit: `e61eed95894e698b0d9dddff82b5d0122a1b8fda`.
Frozen full-test HEAD: `74d6ccc59e92f80bdf53d5fa02575adbf75c2d2a`.

Nine local review lenses and one independent validator batch completed; all
seven validated findings were fixed. Tests first reproduced the historical
hash-contract regression, consumer-time alias escape and partial output after
write failure. Source admission remains explicit; no canonical hashes or
historical receipts were weakened. The original C27 context method is restored
byte-for-byte, and the new consumer binds original reads to admitted canonical
paths. Manifest publication is atomic and exclusive. Dependency-pin changes
now select the actual-source CI job, which also checks the historical method.

| Source-scoped CPU check | Observed result |
| --- | --- |
| Parent focused original-input/adjacent regression | 138 passed, zero skips |
| Full refactor regression | 2153 passed, 85 existing skips, 20 passing subtests |
| Native Wan architecture/strategy regression | 173 passed, zero skips |
| Framework/data/open-workload/organization | 703 passed, two genuine CUDA-only skips |
| Final changed-Python lint | Seven files, F/E9 pass |
| Workflow and documentation | YAML/Python here-doc parsing and four guide/contract links pass |

These host tests use the existing Torch 2.14 CPU environment, with a separate
native-Wan dependency overlay. Minimum Torch 2.10 hosted qualification is a
separate exact-commit result, not inferred here. The external peer produced an
authentication-shaped no-review; local adversarial review completed. No
cross-model corroboration is claimed.

## Still not delivered by this unit

The data cursor is logical, not acknowledgement of physical C51 two-wave
consumption. Native fresh training still needs an explicit window/drain owner,
model/Adam/version/RNG persistence and verified shadow reconstruction. A barrier
alone is not such a boundary. The public native training adapter therefore
continues to reject this input kind instead of silently replaying a benchmark.

The operator must keep admitted source/archives immutable. Canonical-path
binding fixes alias retargeting, but separate restricted/original file reads do
not sandbox concurrent writers replacing admitted files. No GPU, SSH, model
download, performance experiment, main modification or old-queue resume occurred.
