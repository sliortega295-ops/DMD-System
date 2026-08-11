# No-GAN suite status

**Status date:** 2026-08-08
**Current task:** N2 target-graph Generator minimum-offload search
**Accepted capacity result:** bounded capacity PASS; conservative `g4` selected

The frozen N2 graph is 8 Generator ranks on node 0 plus four resident Student
and four resident Teacher ranks on node 1. It replays the exact official
384x640 rank-qualified trace at isolated Step15-A, starts from G3/C11, performs
one global-batch-16 Generator update with ZeRO-1 and CPU optimizer offload, and
changes only the Generator frozen-block streamed set. Shared-file tensor
exchange is correctness transport and is not timing evidence.

## Attempt ledger

| Attempt | Result | Cause | Claim |
|---|---|---|---|
| `n2-g-offload-target-20260808-001-anchor-g40` | **FAIL before workload** | The new Generator executor disabled DeepSpeed BF16 while loading a native BF16 ZeRO-1 checkpoint. DeepSpeed therefore requested nonexistent `zero_pp_rank_*` optimizer shards instead of the present `bf16_zero_pp_rank_*` shards. The peer Score job was cancelled; neither DMD nor Generator training ran. | Runtime configuration failure only; no capacity or timing result |
| `n2-g-offload-target-20260808-002-anchor-g40` | **PASS** | Fresh target-graph anchor after restoring the checkpoint's BF16 execution mode. | Real-14B isolated Step15-A capacity/correctness evidence; not timing |
| `probe-g32/g24/g16/g12/g8/g4` | **PASS** | Frozen profile order, progressively fewer streamed Generator blocks; Student and Teacher stay resident. | `g4=[0,24,1,5]` is the user-selected conservative point |
| `probe-g2` | **PASS, supplemental** | It finished and was summarized before the stop took effect. | Best measured capacity point, but not an exact minimum and not the selected downstream point |
| `probe-g1` | **Not Run** | Candidate file/root existed, but the user stop arrived before either launcher started. | No capacity evidence |
| `probe-g0`, continuous/Full5 validation, formal timing | **Not Run** | Bounded search stopped after the requested `g4` success. | No minimum or performance claim |

The failure produced no new cgroup OOM event. It is preserved in
`artifacts/manifests/n2_target_g_offload_anchor_20260808_001_result.json` and
permits one bounded, non-invasive retry that restores the checkpoint's BF16
precision mode. The failed immutable run ID will not be overwritten or reused.

The official negative-prompt asset passed exact validation with manifest hash
`ca0de6928f83ebd450418d33f8990696763d3228069ec4ac0efb2444694cf78c`
and tensor hash
`470581384766363123713892abd21b6c2bd07a5569c90d2100a4c9aab84e504d`.
Historical Route-A tail-1/tail-3/tail-5/all-40 experiments remain diagnostics
and are not N2 search bounds.

## Bounded result

All completed Attempt-002 candidates used the same source HEAD
`1ac4f02ed0ed6c721aea34647a3b1a61cbb14225`, contract digest
`1d634393bffed554f002d51d9a10d581961a8453534edb1b0f357f7a85a09ba7`,
16 rank-qualified samples, exact trace/x0 identity and one `G3 -> G4` update.
Every completed result has 16/16 rank records, with resident C11 Student and
resident official-base Teacher roles.

| G streamed blocks | Result | G peak alloc/reserved (GiB) | Rank-0 materializations | Rank-0 streamed bytes (GiB) | Rank-0 H2D wall sum (s) |
|---:|:---:|---:|---:|---:|---:|
| 40 | PASS | 29.359 / 29.396 | 560 | 366.533 | 362.814 |
| 32 | PASS | 29.359 / 29.396 | 448 | 293.226 | 278.178 |
| 24 | PASS | 29.359 / 29.396 | 336 | 219.920 | 211.207 |
| 16 | PASS | 29.359 / 29.396 | 224 | 146.613 | 127.860 |
| 12 | PASS | 29.652 / 30.727 | 168 | 109.960 | 95.111 |
| 8 | PASS | 32.278 / 33.352 | 112 | 73.307 | 61.302 |
| **4** | **PASS, selected** | **34.906 / 36.010** | **56** | **36.653** | **19.799** |
| 2 | PASS, supplemental | 36.232 / 37.428 | 28 | 18.327 | 8.654 |

Relative to all-40, selected `g4` reduces rank-0 streamed materializations and
bytes by 90%, while increasing the measured Generator peak from 29.359/29.396
GiB to 34.906/36.010 GiB allocated/reserved. The H2D wall counter is diagnostic
and not an end-to-end speedup measurement. It is emitted by Generator rank 0,
not a cluster-wide copy-engine interval.

The initial `g8` postprocess failure was an IPv6 SSH-gateway failure after both
remote workers had already returned zero. Its preserved 16-rank evidence was
summarized over IPv4 without rerunning the workload. No new cgroup OOM occurred,
and the post-stop audit found no GPU process on either node.

The curated machine-readable result is
`artifacts/manifests/n2_target_g_offload_bounded_search_20260808_002_result.json`.
Because `g1/g0` and continuous transition/Full5 validation are Not Run, neither
`g4` nor `g2` may be called the exact minimum or a final timing baseline.
