# Iteration 328 — Wan1.3B fair Resident/FSDP2 result

## Status

- **Execution status:** all four fixed-revision validate and correctness jobs completed on 16/16 ranks.
- **Numerical comparison status:** **FAIL** for every candidate against the `resident` reference under the unchanged `DEFAULT_P0_PARITY` contract.
- **Formal paired timing:** **NOT_RUN**, fail-closed as preregistered.
- **Performance/speedup claim:** none.

## Source and compatibility patch

- PR #22 starting HEAD: `839d55039a47977e6fb10bf1df61b869918826a6`
- Executed fixed HEAD: `95492b2a04055e80698d35936f081a1f70dfb0c0`
- Local branch: `codex/wan13b-fair-server-compat-20260822`
- Patch: remove runtime annotations from the DDP communication hook because PyTorch 2.10 checks hook annotations by exact runtime identity while `from __future__ import annotations` turns them into strings.
- Mathematical behavior is unchanged: asynchronous `SUM` all-reduce followed by division by world size 16.
- Focused tests at the fixed HEAD: `9 passed`.
- The compatibility commit is included in the PR #22 result-publication update.

The first `resident-ddp` attempt at the original PR HEAD failed before an update with:

```text
ValueError: Communication hook: bucket annotation should be dist.GradBucket.
```

That failed run is preserved. After the minimal compatibility fix, all four placements were rerun from validate through correctness at the same exact HEAD; results below do not mix revisions.

## Frozen workload

- 2 nodes x 8 A100-40GB, world16, B1/rank, global batch 16
- No-GAN, rollout B1, Score packing off
- Generator/Critic/EMA updates = 1/5/5; final versions = G3/C15/EMA15
- frozen backbone compute BF16; LoRA parameter/compute FP32; reduction FP32
- gradient mean convention = SUM then divide by 16
- no parameter, activation, or optimizer offload; no streaming; no model swap
- identical bootstrap, trace, model identity, RNG, and 80 rank-step rollout identities

All four runtime dtype receipts passed on Generator and Critic blocks 0, 15, and 29, under grad-enabled and no-grad execution:

```text
input = BF16
base weight = BF16
LoRA A/B = FP32
```

## Per-placement execution

The wall values below come from correctness-mode runs and are **diagnostic only**. They are not matched formal timings and must not be used for a speedup claim.

| Placement | Validate | Correctness | Ranks | Final state | Diagnostic wall | Max allocated | Max reserved | Min driver free |
|---|---|---|---:|---|---:|---:|---:|---:|
| resident | PASS | PASS | 16/16 | G3/C15/EMA15 | 56.446 s | 18.433 GiB | 19.332 GiB | 19.245 GiB |
| world16-fsdp | PASS | PASS | 16/16 | G3/C15/EMA15 | 69.881 s | 4.216 GiB | 6.555 GiB | 32.012 GiB |
| resident-ddp | PASS | PASS | 16/16 | G3/C15/EMA15 | 59.204 s | 21.004 GiB | 21.902 GiB | 16.664 GiB |
| resident-fast | PASS | PASS | 16/16 | G3/C15/EMA15 | 58.175 s | 21.005 GiB | 21.902 GiB | 16.664 GiB |

## Strict cross-placement correctness

Thresholds remained unchanged:

```text
mean_relative <= 0.005
normalized_l2 <= 0.005
cosine >= 0.9999
max_abs <= 0.05 + 0.01 * reference_peak_abs
```

| Candidate vs resident | Rollout rows failed | Worst rollout mean-rel | Worst rollout nL2 | Min rollout cosine | G param failures | Worst G nL2 | C param failures | Worst C nL2 | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| world16-fsdp | 80/80 | 0.015008 | 0.015785 | 0.999816 | 50/618 | 0.049128 | 50/612 | 0.020968 | FAIL |
| resident-ddp | 80/80 | 0.015006 | 0.015795 | 0.999817 | 50/618 | 0.022634 | 45/612 | 0.010467 | FAIL |
| resident-fast | 80/80 | 0.015053 | 0.015841 | 0.999816 | 50/618 | 0.021850 | 75/612 | 0.010645 | FAIL |

Optimizer-state failures:

| Candidate vs resident | G Adam failures | Worst G Adam nL2 | C Adam failures | Worst C Adam nL2 |
|---|---:|---:|---:|---:|
| world16-fsdp | 2/1236 | 0.006807 | 17/1224 | 0.011873 |
| resident-ddp | 2/1236 | 0.006474 | 32/1224 | 0.011450 |
| resident-fast | 2/1236 | 0.006888 | 17/1224 | 0.011682 |

## Conclusion boundary

The PR #22 dtype gate proves that the tested Resident and FSDP paths now really execute BF16 frozen-base plus FP32 LoRA parameters. That repair did **not** close the strict numerical gap.

The nearly identical rollout divergence of `resident-ddp` and `world16-fsdp` relative to `resident` is evidence consistent with a remaining difference in DDP/FSDP wrapping, checkpoint replay, bucketed/asynchronous reduction, reduction tree, or gradient clipping order. It is not yet a proven root cause. `resident-fast` additionally changes fused Adam/foreach EMA execution order, but its rollout divergence is already present before those final-state operations.

Per the frozen gate, no warm-up or three-pair unprofiled timing was launched. Thresholds were not relaxed and failed rows were not discarded.

## Resource and cleanup evidence

- CUDA OOM: 0
- New cgroup OOM/oom-kill: 0
- Final node0 counters: `oom=0`, historical `oom_kill=4`
- Final node1 counters: `oom=0`, historical `oom_kill=0`
- Final GPUs: 0 MiB used on all 16 devices; no compute processes
- Local and remote source checkouts: clean at `95492b2a04055e80698d35936f081a1f70dfb0c0`

## Next bounded question

If continued, the smallest safe next diagnostic is a single-update, same-input gradient comparison at three boundaries: before communication, after SUM/divide, and after global clipping. Run it for `resident`, `resident-ddp`, and `world16-fsdp` without timing. This can distinguish forward/recompute drift from collective-order and clipping drift. Do not relax the current numerical contract before that attribution.
