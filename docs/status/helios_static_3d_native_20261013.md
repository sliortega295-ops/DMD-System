# Helios No-GAN static TP4/DP2/PP2 binding

The 14B path is explicit opt-in via `--workload helios-nogan` in the existing
native static worker. It keeps 40 native blocks (20/20), original G1/C5/EMA5
order, 16 independent rank-qualified B1 samples, and no DMD-specific
optimization. Native CPU tests do not imply a GPU capacity result.

## Source-specific state/precision

- Restore the original logical FP32 masters on CPU, then partition TP/PP.
- Restore **real** logical Adam first/second moments and the physical
  scheduler steps; do not use Wan's fresh-zero-moment bootstrap.
- Restore sparse EMA with shared read-only mmap pages. Only uniquely owned
  TP slices move to GPU; source shadow dtype and original recurrence remain.
- The reference 14B FSDP2 policy requests BF16 compute with FP32 masters.
  Explicit per-projection BF16 compute views preserve FP32 LoRA masters,
  autograd and PEFT adapter control. Root Linear and Conv3d use their native
  equations. This is precision compatibility, not a speed optimization.
- Wan default remains FP32 LoRA compute. No canonical profiles change.

The new CPU checkpoint fixtures observed expected-red missing-binding
failures, then exercise real nonzero Adam slices, sparse EMA dtype/recurrence,
and BF16 Linear/Conv3d forward/backward with FP32 master gradients.
GPU capacity/timing at the original implementation commit was pending. The
subsequent actual owned launch is recorded below; CPU tests alone are not
being promoted to a capacity PASS.
Numerical trajectory revalidation: **DEFERRED**, not a numerical PASS.

The source step10 checkpoint records provenance trace `bd97ddc9...96f12f`,
whereas the already-validated future execution trace is `654d4901...46d3`.
These are two distinct pinned identities, just as the original FSDP2
launcher admits checkpoint and execution trace separately. The static
worker now requires both, rather than incorrectly equating historical
checkpoint provenance with current inputs. A wrong checkpoint-provenance
SHA still fails closed. No checkpoint or trace payload is edited.

## Actual world16 capacity result

- Run: `helios_nogan_native_static_tp4dp2pp2_c2_20261013_r1`.
- Executed source: `b1749becc6226d4e99095676042d64bd55601b3c`.
- Manifest SHA256:
  `26e29a68d56912cf7e6a9a2faa2a6651c28b35d9e41d383724a5245971e4558f`.
- **BLOCKED_CAPACITY_CUDA_OOM**, cycle 1, Generator gradient rollout,
  PP stage 0. Native B1 latent was `[1,16,9,24,40]`.
- Allocation failed at the original source expression
  `scale_shift_table.unsqueeze(0) + temb.float()` (536 MiB requested).
  Exact transformer block ordinal was not instrumented and is **UNKNOWN**.
- Maximum allocated/reserved: **36.7465 / 37.5488 GiB**; observed minimum
  driver free at failure: **346.375 MiB**.
- No complete cycle or C2 time, and no final rank PASS receipts. The peer
  aborted; it is not independently classified as CUDA OOM.
- Both node owners verified cleanup PASS; cgroup `oom` and `oom_kill`
  deltas were zero on both nodes. No host OOM is claimed.
- GPipe retains eight independent B1 graphs per DP replica. This is a
  lifecycle observation, not a measured byte-level memory decomposition.
- Workload, global batch, native modulation, checkpoint policy, precision,
  update order, and all inputs were left unchanged. No DMD-specific kernel,
  caching, merge, offload, or reduced microbatch count was used to make the
  static baseline fit.

Persistent evidence:
`research_artifacts/DMD-System/main_experiments_20261013/reports/helios_nogan_native_static_3d_capacity_r1.{json,md}`
and the detailed world16 report in the same directory. Failure logs and owner
receipts are preserved on the shared project storage. Under the accepted
main-experiment plan this arm is recorded as a real capacity failure while
other workload bindings are audited; no incomplete time enters the table.
