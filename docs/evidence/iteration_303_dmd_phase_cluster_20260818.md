# MiniMax-H3 DMD-System phase-cluster evidence (2026-08-18)

The adjacent archive contains the preserved one-cycle world16 evidence for the
same-HEAD native control and the DistillSchedule phase-cluster candidate,
including summaries, raw boundary-offload receipts, manifests and logs. It does
not contain model weights, prompt tensors, or Python environments.

Contract:
- DMD-System HEAD: `295f98ca3d5828475f4aca5d709e322bcf82c30d`
- LightX2V: `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be`
- 2 nodes x 8 A100-40GB, world16, BF16, 768x1344x124, B1/rank
- fixed `end_step_idx=3`, continuous sigma, `_flash_3_hub`
- exact all-boundary checkpoint CPU staging policy

One-cycle results:

| Variant | Status | Wall | Max allocated | Max reserved |
|---|---|---:|---:|---:|
| DistillSchedule phase-cluster | PASS | 914.102536 s | 28.10 GiB | 38.23 GiB |
| Native control | PASS | 917.993973 s | 27.94 GiB | 38.23 GiB |

Both runs completed 16/16 ranks with 31 application forwards, 6 backward
graphs, 1 Student update, 5 Fake updates, and zero new cgroup OOM events.
The apparent one-run reduction is 3.891437 s (0.4240%), below the project
performance-claim threshold; it is not a formal speedup.

Important parity boundary: Student parameter/optimizer hashes matched 16/16,
but Fake parameter/optimizer hashes matched 0/16 despite equal recorded sample
IDs. The candidate is therefore marked `CROSS_SCHEDULE_STATE_PARITY_FAIL` and
must not be promoted to Full5 timing until latent/noise/sigma/gradient inputs
are operation-keyed and reconciled.
