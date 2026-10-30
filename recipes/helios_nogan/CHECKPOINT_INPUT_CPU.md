# W2 native-history checkpoint-input CPU capacity candidate v1

**Explicit research opt-in, not a fallback or a native-capacity GO.** Select
`training.native-history-safe-checkpoint-cpu.json` and supply current local assets
and a freshly prepared native-history stream. The existing native-history,
reference and SafeBundle presets/defaults retain their previous behavior. This
candidate is a lifecycle/capacity extension, not a new kernel or timing result.

The complete original node-HSDP 2x8 / C512 / p0 / window5 / stride0 SafeBundle
remains enabled, including local pinned FA2, original LoRA checkpoint replay
scope and all effective original flags. Incorrect merged-LoRA and historical
attention replay are not enabled. Original training, sampling, gradient chains,
precision, updates, optimizer, EMA and RNG remain authoritative.

## Narrow boundary

Only the **exact first hidden-state input object** saved by each original whole
Generator block's non-reentrant checkpoint is synchronously copied to CPU. It is
restored on the same device and dtype when that checkpoint replays. Auxiliary
patch checkpoints, shared encoder/rope inputs, parameters, optimizer state and
recomputed intermediates are not offloaded. The original checkpoint callable and
its original LoRA wrapper are retained; the unmodified whole-block/FSDP/callable
validator runs before FSDP materialization while original FP32 DTensor optimizer
ownership is visible. Only the admitted wrapper's installation is inside the
materialized phase; the original callable is restored on exit.

The pack/unpack closures retain no original hidden tensor or argument tuple.
Since saved-tensor hooks disable the ordinary saved-variable version check, a
new detached zero-storage sentinel preserves that input's shared version counter;
only the sentinel's data is replaced, never the original tensor's. In-place
mutation through a surviving alias is refused even after the original object is
gone. Copies are synchronous. Repeated unpacks retain the lease until the graph
releases it. Error exits cancel CPU leases and escaped backward fails closed.

## Exact declared host bounds

Original B1 current latent stages are `9 x {12x20, 24x40, 48x80}` with `(1,2,2)`
patching. Unchanged short/mid/long histories add `1920 + 240 + 240 = 2400` tokens.
The admitted BF16 hidden inputs are therefore:

| Stage | Input shape | Bytes for 40 boundaries |
| --- | --- | ---: |
| 0 | `[1, 2940, 5120]` | 1,204,224,000 |
| 1 | `[1, 4560, 5120]` | 1,867,776,000 |
| 2 | `[1, 11040, 5120]` | 4,521,984,000 |

Maximum retained host storage is **7,593,984,000 bytes/rank** and
**60,751,872,000 bytes/node**, partitioned into eight disjoint rank budgets. Select
the included pageable capacity file, or explicitly select the pinned file.
Unselected storage must have a zero budget. No shared node overcommit is allowed.
This bounds retained hidden-input storage, **not all host memory or HBM**.

The extra node budget must fit **in addition to the unchanged 512 GiB startup
reserve before original model construction**. Runtime copy admission retains the
unchanged 64 GiB host reserve; each restore retains the unchanged 512 MiB owned-GPU
reserve. Fresh physical/cgroup-hierarchy observations reuse the existing host
admission helper; these observations are not allocations or reservations against
other processes. Runtime budget overflow, unexpected shapes/dtypes, extra stage
calls, wrong source/mesh/callable/version or leaked boundary storage fail closed.
No retry, smaller input, precision change or alternate strategy is substituted.

## Identity, checkpoints and receipts

The public preset lowers through `helios_worker` into the same original P0 owner.
The parsed capacity policy and independently versioned execution policy enter the
training/checkpoint identity before construction. Complete-cycle save and restore
reuse the existing native state owners and all-rank checkpoint publisher. CPU
leases and caches are **never serialized**; fresh sessions are rebuilt at each G
action, including after restore. Old and new policy checkpoints cannot be mixed.

Each G action records executed checkpoint/copy/restore/release counts, exact
copied/restored bytes, live/peak lease bytes/counts, and source/stream/task/weight
version plus block/shape/dtype/device/input-version metadata. Successful native
receipts require all 120 original block-stage boundaries and no remaining leases.
Copies and releases are always 120. Restores are bound to the original validated
executed `trace_entries[0].stage_schedule`, not a configured exit or a permitted
count range. Let `s` be the last stage in `{1,2}` with a nonzero `denoising_exit`,
or zero if neither has a prefix. Only stages `s..2` remain connected to the sole
final-x0 loss. Each required block input must restore exactly once; each dead
prefix input must record an actual unused release. Stage index, block, shape and
per-boundary restore/release outcomes are checked, including duplicates.

| Original stage exits | Required restores | Restored bytes/rank |
| --- | ---: | ---: |
| `(0,1,1)` | 40 | 4,521,984,000 |
| `(0,1,0)` | 80 | 6,389,760,000 |
| `(0,0,0)` | 120 | 7,593,984,000 |

The host declaration remains the full three-stage bound in every case. Nothing
forces the original exits, reconnects dead graphs, adds losses/backwards or draws
extra training RNG to satisfy a census.
Detached rollout actions install no offloader. Existing original call/update and
SafeBundle execution counters remain separate and mandatory.

## Verification boundary

Local tests use actual Torch non-reentrant checkpointing and Adam, but synthetic
small CPU tensors, not a native HSDP owner. They are
`CPU_SYNTHETIC_TEST_ONLY`: no native capacity, numerical, timing or quality
promotion. Native Torch 2.10 / original 40-block integration / target GPU are
**NOT_RUN** until independently executed and reviewed. Run
`verify_checkpoint_inputs_target_cpu.py` only in an already provisioned target
Torch 2.10 full-stack CPU environment; it refuses missing dependencies instead of
installing them or reporting a skipped PASS. Full pretrained HSDP capacity remains
unproved, even when this component test passes. The historical 82.99s observation
does not apply to this new scope.

That entry checks the production hidden shapes through the original meta-device
preprocessor, then compares the actual small 40-block class with and without CPU
leases for the three prefix patterns above (outputs, all gradients, named Adam,
RNG and lifetimes). This is still CPU component/composition evidence, not the
original full sampler, target GPU or pretrained native HSDP qualification. The
local real CPU FSDP ordering regression explicitly substitutes only its expected
physical topology after confirming unmodified native topology rejection; the
production owner/callable validators and CUDA-only factory refusal remain strict.
