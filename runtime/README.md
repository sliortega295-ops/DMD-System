# Runtime compatibility layer

`runtime/` contains compatibility mechanisms still referenced by maintained
DMD-System training paths, source-audited adapters or tests. It is **not** a
second public training API. Use `distillctl`, current `recipes/`, and the
`src/distillschedule/` training, Program and physical execution interfaces for
new integrations; see [framework ownership](../docs/architecture/framework.md).

Current module families include:

- **DMD/EMA semantics:** `dmd_boundary.py`, `ema.py`, `sparse_ema.py`,
  `gan_rolephase_official.py` and `gan_score_packing.py`.
- **Role pooling and trace:** `critic_pool_*.py`, `gst_pool_*.py`,
  `operation_trace.py`, `rollout_trace.py`, `static_world16_trace.py`,
  and `trace.py`.
- **State, loading and capacity:** `residency.py`, `immutable_model_backing.py`,
  `precast_checkpoint.py`, `host_admission.py`, `official_baseline.py`,
  and `static_fsdp2_p0_helios.py`.

This is an inventory, not a guarantee that every module is exercised by each
current recipe. Check actual imports and entrypoints before reusing a mechanism.
The pre-2026-08-15 Route-A/Route-B launchers and exclusive offload prototypes
have been [retired](../docs/architecture/legacy_route_retirement.md); do not
resurrect them from Git as defaults. Some smaller source-compatibility helpers
remain at their original paths because audited imports/fixtures still need them.

No historical source checksum or CPU fixture by itself proves current multi-GPU
correctness, memory safety, or end-to-end speedup. Preserve the native checker,
checkpoint/recovery rules and measured timing window of the actual workload.
