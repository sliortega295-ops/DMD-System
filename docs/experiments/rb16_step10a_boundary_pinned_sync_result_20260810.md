# Iteration 303: selective pinned Step10-A checkpoint boundaries

**Status: correctness PASS; bounded performance negative.**

The candidate used the strongest prior legal ZeRO-3 window from Iteration 297 and changed
only three census-proven saved-tensor signatures to synchronous pinned D2H pack and
synchronous H2D unpack. The strict-v4 workload remained DP8, MB1/GAS2/global batch 16,
two selected graphs, two backwards and one `G2 -> G3` update.

## Correctness and admission

The continuous Full5 correctness run passed all 16 ranks, exact identity/state checks,
`G3/C15/EMA15`, 60 generated and 60 consumed payloads, queue flush and zero OOM delta.
Each Generator rank copied 235--240 allowlisted storages, reached 13.891--14.145 GiB peak
pinned boundary storage and ended with zero live CPU bytes. The run used `module_to` for
the later transition because the concurrent Iter299 target-pinned state plus boundary
storage was statically rejected above the cgroup limit.

## Preliminary low-overhead timing

| metric | Iter297 strongest static | pinned boundary | change |
|---|---:|---:|---:|
| Generator phase-sum median | 115.489513 s | 133.831677 s | **+15.88%** |
| strict-v4 baseline | 123.115506 s | 133.831677 s | **+8.70%** |

Pinned wave medians were rollout 24.643/25.254 s and backward 40.361/42.088 s. Node0
cgroup peak was 510,986,096,640 bytes; GPU peak was 31,762 MiB per Generator rank; no OOM.

The candidate therefore fails the preregistered `<110.804 s` gate. Shape/count ownership
was real, but explicit offload did not convert the existing path in-place; it added
synchronous pinned allocation and copy work. Async prefetch is not attempted because the
required synchronous substrate regressed. This family is closed rather than patched
invasively.

## Evidence

- `artifacts/runs/rb16_f5_iter303_boundary_pinned_sync_correctness_20260810_001/`
- correctness SHA `1f636d7e23d1a439ebb828a1fb5bd61baca8a09b177cfdb17790db3b3c2dddcb`
- `artifacts/runs/rb16_step10a_iter303_boundary_pinned_sync_timing_20260810_001/`
- `artifacts/manifests/iteration303_step10a_boundary_pinned_sync_result.json`
