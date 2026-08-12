# Executed GAN pipeline runtime snapshot (2026-08-12)

This directory is a review snapshot of the **exact model-worker/runtime source**
used by the real world16 `G40/S10/A128` runs.  The active server experiment
kept Helios and the low-memory runtime in a separate pinned DistillSchedule
checkout; the DMD-System launcher, contracts, receipts, and orchestration live
in this repository.

The snapshot is included so that review of the pipeline implementation does
not depend on access to the server checkout.  It is evidence, not a second
runtime entry point: do not silently switch the launcher to this copy without
a new source-parity gate.

## Source identity

- DistillSchedule runtime HEAD:
  `171e3bf38605b790b26341496d714cbf3766fb03`
- Remote execution checkout:
  `/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_pipeline_exec_20260811`
- Local review checkout:
  `/home/lyy/DistillSchedule_gan_runtime_20260810`
- Local and remote SHA-256 values were rechecked after the run and were exact.
- Per-file hashes are in `SOURCE_SHA256SUMS`.

## Included files

- `scripts/distillschedule/probe_gan_pipeline_world16.py`: real 16-rank
  persistent 8G+8Score worker and update/queue boundaries.
- `runtime/frozen_block_streaming.py`: immutable block lifecycle and bounded
  Generator no-grad phase residency.
- `runtime/activation_offload.py`: audited saved-tensor offload primitive.
- `runtime/critic_pool_helios.py`: Helios model construction and rollout seam.
- `helios/utils/utils_helios_post.py`: exact Helios math path used by the worker.

The current measured candidate is intentionally **not presented as the final
pipeline implementation**.  It passes the real capacity/correctness gate, but
its one low-overhead timing window is 824.685006 s and the formal receipt
aggregation failed after model execution.  See
`docs/status/gan_pipeline_rolephase_20260812.md` before modifying it.
