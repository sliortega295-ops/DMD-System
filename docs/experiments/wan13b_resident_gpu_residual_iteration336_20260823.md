# Iteration 336 — Wan1.3B resident GPU-residual correction

## Status

- **Source/runtime correction:** PASS.
- **World16 workload receipts:** 16/16 PASS.
- **Post-fix memcpy attribution:** PASS with exact bytes/events/durations reconciliation.
- **Formal paired timing:** NOT_RUN.
- **Performance claim:** bounded single-run unprofiled diagnostic only.

## Root cause and fix

The pre-fix launcher requested `chunked_residual_storage=cuda_out_of_place` but pinned a
historical Helios transformer source (`776a26c3...`) that neither accepted the
`residual_storage` argument nor read `_distillschedule_chunked_residual_storage` from a
block. The configuration was therefore a silent no-op and gradient-bearing block sweeps
continued to stage gated residuals through CPU.

Executed correction HEAD: `879816ad0a36ef718ae67dbf2c1159820bd29f4e`.

The fix selects the previously validated GPU-residual runtime
(`5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`) and fails closed
before model construction unless the helper signature, autograd signature, two block
call sites, source path, and source SHA prove that `cuda_out_of_place` is physically
supported. The summarizer requires all 16 ranks/components to report the same capability
and source identity.

## Frozen workload

- 2 nodes x 8 A100-40GB, world16, B1/rank, global batch 16
- resident-clean, rollout B1, Score packing off
- Generator/Critic/EMA updates = 1/5/5; final versions = G3/C15/EMA15
- BF16 frozen backbone; FP32 LoRA; SUM then divide by world-size 16
- same bootstrap, trace bundle/binding, model identity, RNG, and 80 rank-step identities
- no parameter, activation, or optimizer offload; no streaming; no model swap

## World16 execution

| Run | Gate | Status | Wall | Max allocated | Max reserved | Min driver free |
|---|---|---|---:|---:|---:|---:|
| `iter336_correctness_resident_gpu_residual_20260823` | correctness | PASS 16/16 | 36.856060 s | 19.22 GiB | 19.64 GiB | 18.98 GiB |
| `iter336_diag_resident_gpu_residual_20260823` | unprofiled diagnostic | PASS 16/16 | 35.757181 s | 19.23 GiB | 19.70 GiB | 18.89 GiB |
| `iter336_profile_resident_gpu_residual_20260823` | Nsight diagnostic | PASS 16/16 | 50.310064 s | 19.23 GiB | 19.70 GiB | 18.84 GiB |

No new CUDA or cgroup OOM occurred. Both nodes were empty after the capture.

## Exact post-fix transfer evidence

Nsight Systems 2024.6.2.225 captured both nodes. Reports and SQLite exports are non-empty,
all 16 exact range receipts passed, and export/stats logs contain no parser or report
processing errors.

Like-for-like node0/device0 exact-range comparison:

| Direction | Pre-fix events | Post-fix events | Pre-fix bytes | Post-fix bytes | Pre-fix union | Post-fix union |
|---|---:|---:|---:|---:|---:|---:|
| D2H | 997 | 37 | 25.316620 GiB | 37 B | 8.029461 s | 0.000086 s |
| H2D | 9,064 | 244 | 12.755536 GiB | 0.097226 GiB | 1.032566 s | 0.007882 s |

Across all 16 post-fix GPUs:

- D2H = **688 events / 688 bytes total**;
  per-GPU union median = 0.111 ms.
- H2D = **3904 events / 1.555620 GiB total**;
  per-GPU union median = 8.186 ms.
- The three pre-fix residual signatures (33,914,880 / 14,008,320 / 9,031,680 bytes)
  have **0 events / 0 bytes** across all 16 GPUs.
- Memcpy range-crossing events = 0; classified bytes/events reconcile exactly.

The remaining D2H consists only of one-byte control/check copies. Thus the old residual
D2H is physically eliminated rather than merely overlapped or hidden. Remaining H2D does
not contain the old residual/reload size signatures.

## Diagnostic timeline comparison

| Representative GPU0 category | Pre-fix | Post-fix | Delta |
|---|---:|---:|---:|
| exact profile wall | 70.505866 s | 50.267583 s | -20.238283 s |
| non-NCCL kernel union | 25.222761 s | 25.122071 s | -0.100690 s |
| NCCL all-reduce union | 4.806667 s | 2.197168 s | -2.609499 s |
| D2H union | 8.029461 s | 0.000086 s | -8.029375 s |
| H2D union | 1.032566 s | 0.007882 s | -1.024684 s |
| no traced GPU activity | 30.929036 s | 22.458610 s | -8.470426 s |

Compute union is essentially unchanged. The diagnostic wall falls through removed D2H/H2D,
smaller untraced gaps, and faster execution of the same 68 serialized NCCL all-reduce
kernels. The memcpy result proves the residual mechanism; it does not by itself prove the
causal split of the NCCL and untraced-gap reductions.

## Unprofiled timing boundary

- matched pre-fix resident-clean diagnostic: 56.187420 s
- post-fix GPU-residual diagnostic: 35.757181 s
- reduction: 20.430239 s (36.36%); equivalent bounded diagnostic ratio: 1.571x

This is one unprofiled comparison, not a formal three-pair speedup. The profiled wall is
never used as a speed denominator. Numerical cross-arm parity was intentionally not used
as the blocker for this systems diagnostic, so `formal_eligibility` remains false.

## Evidence

Small evidence is under
`artifacts/evidence/wan13b_resident_gpu_residual_iteration336_20260823/`.
Raw `.nsys-rep` and `.sqlite` files remain on the experiment servers and are identified by
SHA/path receipts; they are intentionally not committed because each SQLite is about 1.4 GB.
