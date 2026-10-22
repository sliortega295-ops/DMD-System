# Original native lifecycle: W2 and W5

This follow-up supplements the [public core acceptance](FRAMEWORK_CPU_ACCEPTANCE_20261021.md).
It records source/CPU implementation, not pretrained GPU correctness, capacity,
quality or speed. The [machine-readable receipt](../../artifacts/framework/20261021/native_lifecycle_cpu.json)
pins behavioral commits, suite scopes and JUnit hashes. Historical receipts and
`source_import_map.json` are unchanged; original-file updates have explicit
old/new entries in `source_layout.lock.json`.

## Actual original owners, not replacement DMD loops

| Path | Implemented scope | CPU evidence |
| --- | --- | --- |
| W2 direct reference | Explicit original Step10 G2/C10/EMA10 package; fresh step/rank-qualified cohorts; original P0 model setup, preparation, updates, clipping, Adam, scheduler and EMA; complete-cycle same-topology save/resume | Licensed original producer/update/state helpers with real CPU DTensor/Adam owners; interrupted three-cycle continuation matches uninterrupted state and RNG. Actual public-to-P0 entry is tested with real two-process Gloo and explicitly synthetic CUDA/topology/model boundaries |
| W5 original reference | Supplied original config; finite real cached-condition inventory; original Grid1000 setup/data/rollout/loss/update loop; exact deterministic sampler cursor and complete-cycle same-topology save/resume | Real pinned LightX2V and Diffusers source admission, actual original loop/loss/Adam on tiny CPU construction fixtures, real Gloo rank failures and exact interrupted continuation |
| Public native data preparation | Discoverable W2 cohort and W5 condition-inventory commands; original producers remain owners | CLI contracts, actual W5 file inventory, no model/CUDA initialization; compatibility entry remains available |

W2 uses the separate `distillschedule.helios_reference_assets.v1` contract:
only used Generator/Score model/pre-cast roots, original logical bootstrap and
negative conditions. Its EMA manifest and payloads are independently pinned.
Historical node/HF/flash/trace manifests are not prerequisites of this direct
reference route. Cold-start Step0 and arbitrary bootstrap clocks are unsupported.

W5's exact upstream source commits are LightX2V
`d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be` and Diffusers
`9284607295a09f759aadd65ed08f48b35feea6d9`. Reading those sources in CPU tests
does not assert that their native CUDA dependency environment was installed or
qualified. Model/table/kernel payloads remain operator-supplied and content-pinned.
Optional optimized transforms are OFF in the reference lifecycle, and the known
optimized sampled-backward **FAIL remains FAIL**.

## Admission and restart behavior

Both routes own fresh training groups and use a CPU metadata plane before
rank-local source, file and dependency failures can strand peers in model
collectives. Borrowed embedding groups reject without being destroyed. Source,
configuration, data, immutable payloads, layout and checkpoint clocks are bound
before loading state into the original live owners. Missing shards, changed
payloads, foreign identities and wrong cursor/version boundaries reject.

Only complete zero-gradient cycle boundaries publish the common atomic
all-rank checkpoint. Temporary activation/replay caches are rebuilt, not saved
as optimizer state. A finite prepared stream is not a claim of unlimited data
or arbitrary scheduler-horizon changes. Follow each recipe's exact restart
restrictions rather than extrapolating a different workload's options.

## Observed source-scoped checks

| Check | Observed result |
| --- | --- |
| W2 host original-helper/public-startup/legacy regression suite | 329 passed, one genuine CUDA-only skip; 398.79 seconds |
| W2 same-file follow-up after unused test-import cleanup | 22 passed; 122.88 seconds |
| W5 host original-loop/data/state/Gloo suite | 207 passed, zero skips; 46.47 seconds |
| Public native preparation and adjacent CLI regression | 43 passed; 2.90 seconds |
| W2 CPU dependency/source/CI-contract checks | 31 passed, three installed-wheel tests excluded from this focused invocation; full installed-wheel acceptance is separate |

Tests ran on existing Python 3.12/Torch 2.14.0+cpu. Actual original SDPA helper
imports use real pinned `kernels`/Triton dependencies in a disposable CPU overlay;
no substitute packages, kernel compilation or GPU launches. Minimum Torch 2.10
CI separately requires the actual W2 startup and restart tests to execute, not
skip. The dedicated W5 CI reads the exact upstream sources and rejects skipped
original-source tests. A workflow definition is not a completed CI result;
consult the exact PR commit's checks.

## Reviewed follow-up and final CPU regression

The [reviewed CPU receipt](../../artifacts/framework/20261021/native_lifecycle_reviewed_cpu.json)
pins behavior HEAD `dc67c0b02c123c40147fe260fe632f23bccd6ed7`. All seven
independently validated review findings have proof-first regressions and fixes.
New input paths restrict tensor archive deserialization before original owners
read values. W2 rejects repeated sources independently of augmentation seed;
W5 resolves path aliases and exchanges next-sample failures on the metadata
plane before model work. W2 preserves old rank receipts on refusal and releases
copied restart payloads before resumed computation. EMA admission holds only
one full CPU shard at a time. None changes the native training mathematics.

| Final source-scoped check | Observed result |
| --- | --- |
| Refactor regression | 2153 passed, 85 skipped, 20 passing subtests |
| Native Wan architecture/strategy regression | 173 passed, zero skips, using its separate Diffusers 0.35.2/PEFT 0.17.1 overlay |
| Framework, data, open-workload and organization | 662 passed, two genuine CUDA-only skips |
| Changed-source static checks | 24 Python files pass F/E9; checked compilation and 17 shell scripts pass |

Earlier mismatched helper-environment and harness-setup failures remain archived;
no assertions, goldens, tolerances or skips were changed to obtain these outcomes.
Helios helper tests use Diffusers 0.34.0 separately from native Wan's 0.35.2.
Minimum Torch 2.10 CI is a separate exact-commit result, not inferred from these
Torch 2.14 CPU runs. The external peer returned no authenticated review; the
local adversarial fallback completed, and no cross-model corroboration is claimed.

Output admission is an entry-time ownership check, not an atomic reservation
against another job. Each invocation still needs a unique operator-owned output
directory. Restricted input admission does not sandbox concurrent modifications
of the underlying filesystem. Pretrained CUDA/NCCL/FSDP and host-capacity gates
remain unchanged and unqualified by this CPU-only follow-up.

## Remaining production-native boundaries

This section records the original native-adapter scope of this dated receipt.
Later [W3 CPU Program training](../../recipes/helios_gan/REFERENCE_TRAINING_CPU.md)
and [W6 cached-source training](../../recipes/h3_ar/CPU_REFERENCE.md) add separate
local references. The current [capability matrix](../guides/support_matrix.md)
distinguishes those paths; this historical receipt is not rewritten as new GPU evidence.


All six recipes still retain their individual algorithms and actual selected
execution binding. W1/W2/W5 expose their documented opt-in native reference
continuation. The later [W4 reference follow-up](FRAMEWORK_LIVEWAN_REFERENCE_20261021.md)
adds a distinct original baseline lifecycle without widening its C51 worker.
W3/W6 retain bounded selected/native workers, not universal
arbitrary-long trainers. W6 also requires a complete authorized upstream source
distribution; its historical evidence snapshot is not that distribution.
Common native generation/export and those remaining lifecycle adapters are
not implemented by this change. The public extensible Program lifecycle has
its own real tiny-model media/update/resume/generation proof; it must not be
used as evidence that six pretrained models passed native lifecycle acceptance.

See the [support matrix](../guides/support_matrix.md), [W2 guide](../../recipes/helios_nogan/REFERENCE_TRAINING.md)
and [W5 guide](../../recipes/h3-grid-original-training.md). No GPU, SSH,
performance search, main modification or old-queue resume occurred.
