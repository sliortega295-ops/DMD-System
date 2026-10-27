# W6 native finite continuation v1

Opt-in preset: `h3_ar.zero3_ap_stream.v1`. Input kind:
`original_h3_ar_fresh_stream.v1`. This is the actual original native
`H3ARSession` / `DMD2Trainer` / `AlternatingAdamW` / DeepSpeed ZeRO-3 owner,
not the independent cached-AV CPU reference. Its reference choice is explicitly
`zero3-native` through this same owner. No fallback occurs.

## Unchanged mathematics and bounded scope

The original 50-block model, previous56/current56 AV, 768x1344/24fps,
eight-step rollout, G1/F1/EMA0, LoRA64, world16 B1, native BF16 with autocast
cache disabled, optimizer ownership and FP32 masters are unchanged. Native
GAN start remains 500: it must remain inactive throughout this 2..100-cycle
contract. Rank6 retains its original 94-frame audio schema; other ranks retain
their own SHA-bound original parser metadata. No crop, padding, normalization,
fake condition, replacement sampler or changed loss is admitted.

The old immutable rank-input manifest and `native_graph` two-cycle entry retain
their exact guards and reject this manifest. A new finite graph emits the same
whole native update, operation order, census and version transitions.

## Fresh input authority

The manifest binds the original source receipt, rank-input contract, one seed,
an immutable finite horizon, and exactly 16 distinct records per cycle. Records
are explicitly ordered cycle-major, then rank0..15. Each supplies `cycle`,
`rank`, unique `record_id`, absolute cached-record `path` and `sha256`, plus a
nonempty `sources` mapping of named original source/encoding file references
(`path`, `sha256`). All used input bytes are checked. Preparation derives tensor
shape/dtype/scalar metadata from the original deterministic parser; training
checks it again. No model or media is downloaded or encoded by this command.

Every rank validates its next original parser result before collective model
allocation. That same CPU result is consumed by the shared constructor.
Subsequent fresh inputs are prepared before dispatch and committed only after
the actual update/census/clock checks succeed. A failed update is not replayed
or represented as a completed cycle.

## Complete native checkpoint

The common `training.checkpoint` publisher remains authoritative. Native engine
files are sidecars inside the same all-rank, SHA-bound, `_SUCCESS` publication.
Each actual engine calls `save_checkpoint(tag='state', save_latest=False,
exclude_frozen_parameters=True)`. Registered permanent trainables include both
LoRAs and every inactive discriminator parameter. A scoped `requires_grad`
policy includes them while excluding the frozen backbone; `finally` restores
every original flag on both success and failure. No optimizer codec is invented.

The boundary stores trainer/student/fake versions, native `global_steps=2*cycle`,
`active_role`/`train_heads`, named trainability and module modes, per-rank
Python/NumPy/Torch/owned-CUDA RNG and the fresh stream cursor. The manifest binds
current framework/source-layout/native/helper/checkpoint/input identities,
precision, topology, capacity, horizon, seed and effective DeepSpeed config.
Native ordered `_get_zero_param_shapes` partitions and actual writer flags and
file-name getters are recorded, not inferred from current trainability.

Restore requires a pristine new-process native engine, the same topology and
all common/native shard hashes. Wrong identity, missing/corrupt files, named or
ordered layout/config drift refuse before restore mutation. Native model-shard
keys, dtypes, source-version/config and client-state fields are audited first;
`load_module_strict=False` permits only intentionally absent frozen parameters.
The actual `load_checkpoint` restores optimizer states and FP32 masters.
DeepSpeed0.18.4's engine source is pinned to
`ef3b4797809453f4941eaa769c9cd4e397e3b52085131e8916990ea78d4f0f6c`.
Native checkpoints are trusted pickle artifacts: never restore untrusted files.
An I/O/backend failure during native mutation is not advertised as rollback-safe.

Before publication the unchanged epoch helper proves consumption/event drain,
advances only its read-only view and clears result/FIFO/prefetch metadata.
These transient objects and pinned buffers are never checkpoint state. On
resume, actual trainer and optimizer clocks are restored; only `view.origin` and
process-local observer baselines are rebased. An empty new pool never receives
`begin_next_cycle`. First resumed staging is honestly cold, with the frozen
helper's historical origin-based metadata retained separately. Later drained
cycles reuse only process-local pinned storage, never stale results.

## Evidence boundary

Source and focused CPU protocol tests are prerequisites, not native GPU
qualification. CPU tests use explicit `SYNTHETIC_TEST_ONLY` engine/CUDA/pool/
collective substitutes and real licensed native class bodies where possible.
The actual installed DeepSpeed serialization-method test constructs no engine.
New-process world16/model-sized checkpoint, capacity, numerical behavior and
decoded AV remain **NOT_RUN / not implemented**, respectively. Prior native
two-cycle and independent CPU reference evidence is reused, not rerun or
relabelled as continuation proof. No speedup or quality claim is made.
