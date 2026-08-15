# PR13 FSDP2 Static Full5: source attribution of P2/P3 transfers (Iteration 295)

- **Run ID:** `iteration_295_source_attribution_20260814`
- **Status:** attribution diagnostic complete; no new Full5 and no formal timing launched.
- **Current phase:** exact Nsight SQLite correlation audit, per-rank/per-phase transfer attribution, and bounded report generation.
- **Frozen formal timings:** No-GAN `214.066222628 s`; GAN-on `348.074251748 s`.

## 1. Scope and preflight

The worktree was revalidated before analysis:

- Worktree: `<local-worktree>`
- Branch: `agent/fsdp2-static-full5-baselines-20260814`
- HEAD: `075a47b039ab84ed168242683354901c3697942e`
- `git status`: clean; local HEAD equals origin branch.
- No owned Full5 worker or Nsight extraction process was left running after the read-only audit. The two nodes had no active experiment GPU allocation at preflight.
- The node0 cgroup `oom_kill` counter is historical (`4`); node1 is `0`. No P2/P3 receipt reports a CUDA or cgroup OOM.
- No P0/P1/P2/P3 raw report, SQLite export, `.nsys-rep`, or old figure was overwritten.
- Remote source/runtime staging used by the completed captures: `8dd73a31345f4cf7482a9c36db3566d03cc5f76f`.

Frozen workload: 2 nodes × 8 A100-SXM4-40GB, world16, 384×640, global batch 16, both backbones FSDP2 `FULL_SHARD`, model swap 0, parameter offload false, activation offload false, unchanged trace/RNG/precision/optimizer/call/update/version semantics.

## 2. Raw SQLite availability and schema audit

The P2/P3 raw SQLite exports are still available on the remote staging filesystem and were queried read-only; they were not copied into the local artifact tree because each is 1.8--2.3 GB. Sizes:

| run | node0 SQLite | node1 SQLite |
|---|---:|---:|
| No-GAN P2 | 1,856,864,256 B | 1,855,586,304 B |
| GAN-on P3 | 2,317,414,400 B | 2,316,464,128 B |

Nsight Systems on both nodes is `2023.3.1.92-233133147223v0`. The export is not missing the CUDA runtime/activity layers: it retains the tables and fields needed for the CUDA memcpy → runtime API join. It **does not** populate NVTX range linkage fields (`rangeId`, `endGlobalTid`), so application phase attribution uses same-TID interval containment, not a range-ID guess.

Representative P2 node0 table/field audit (P3 has the same relevant schema; exact per-node counts and joins are in `artifacts/profile/source_attribution/schema_audit/` and `raw_correlation/*_source.json`):

| table | rows | fields and non-null counts | connection result |
|---|---:|---|---|
| `CUPTI_ACTIVITY_KIND_MEMCPY` | 1,716,646 | `start,end,correlationId,globalPid,bytes,copyKind,srcKind,dstKind`: 1,716,646 each | all 1,716,646 rows join a runtime row |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | 16,221,863 | `start,end,correlationId,globalTid,eventClass,nameId`: 16,221,863 each | `cudaMemcpyAsync_v3020`: 1,716,646 calls |
| `NVTX_EVENTS` | 50,502 | `start,globalTid,domainId`: 50,502; `text`: 33,061; `rangeId`: 0; `endGlobalTid`: 0 | 33,049 named intervals available for containment |
| `PROCESSES` | 311 | `globalPid,pid,name`: 311 each | 8 Python worker processes; PID-to-local-rank order inferred |
| `ThreadNames` | 759 | `globalTid,nameId`: 759 each | worker/runtime TIDs available |
| `CUPTI_ACTIVITY_KIND_KERNEL` | 13,000,294 | `start,end,correlationId,globalPid`: 13,000,294 each | retained for compute/NCCL context |
| `CUPTI_ACTIVITY_KIND_MEMSET` | 76,267 | `start,end,correlationId,globalPid,bytes`: 76,267 each | retained separately from memcpy |

Across all four raw exports, exact join counts were:

| capture | memcpy rows | runtime matches | unmatched | activity PID absent from `PROCESSES` | worker PIDs |
|---|---:|---:|---:|---:|---:|
| P2 node0 | 1,716,646 | 1,716,646 | 0 | 0 | 8 |
| P2 node1 | 1,716,634 | 1,716,634 | 0 | 0 | 8 |
| P3 node0 | 1,795,102 | 1,795,102 | 0 | 0 | 8 |
| P3 node1 | 1,795,090 | 1,795,090 | 0 | 0 | 8 |

The join is:

```text
memcpy.correlationId
  -> runtime.correlationId
  -> runtime.globalTid and encoded globalPid
  -> worker PID / inferred rank
  -> shortest named NVTX interval on that same globalTid
```

Runtime correlation is therefore exact. The remaining phase uncertainty is specifically the missing populated NVTX range ID/end-TID relation, not a missing CUDA runtime correlation ID.

## 3. Attribution method and reconciliation

For each `cudaMemcpyAsync` activity row, the extractor joins `(globalPid, correlationId)` to the unique CUDA Runtime API call, then assigns the worker/rank and the innermost same-TID named NVTX interval. No wall-clock overlap was used to invent a source. `sum_duration` counts every event; `union_duration` is interval union and is never added to compute/NCCL to claim GPU utilization.

Every source category sum reconciles the original CUDA memcpy table exactly for bytes, event count, and summed duration. The per-rank/per-phase machine table contains direction (`H2D/D2H/D2D`), bytes, events, summed duration, interval union, and size histogram.

Durable machine-readable outputs:

- `artifacts/profile/source_attribution/source_attribution_summary.json`
- `artifacts/profile/source_attribution/transfer_attribution_source_summary.csv`
- `artifacts/profile/source_attribution/transfer_attribution_source_rows.csv`
- `artifacts/profile/source_attribution/detailed_source_attribution.json`
- `artifacts/profile/source_attribution/detailed_source_attribution.csv`
- `artifacts/profile/source_attribution/reconciliation_checks.csv`
- `artifacts/profile/source_attribution/schema_audit/sqlite_schema_and_join_summary.json`

## 4. Node-level observed transfer totals

Values below are CUDA activity bytes from the P2/P3 raw exports, not logical application counters. Decimal GB/TB are used here; exact bytes are in the CSV/JSON.

| run/node | H2D bytes / events / sum duration | D2H bytes / events / sum duration | D2D bytes / events / sum duration |
|---|---|---|---|
| P2 node0 | 484.469839 GB / 113,432 / 88.143560 s | 966.524928 GB / 10,654 / 370.722147 s | 6.646830 TB / 1,592,560 / 13.841809 s |
| P2 node1 | 484.469839 GB / 113,432 / 69.001433 s | 966.524928 GB / 10,642 / 361.329807 s | 6.646830 TB / 1,592,560 / 14.510039 s |
| P3 node0 | 963.813834 GB / 154,816 / 149.460968 s | 1,922.826245 GB / 19,166 / 744.493466 s | 6.859707 TB / 1,621,120 / 13.913126 s |
| P3 node1 | 963.813834 GB / 154,816 / 111.505436 s | 1,922.826245 GB / 19,154 / 704.547755 s | 6.859707 TB / 1,621,120 / 15.096180 s |

The node-level copy intervals overlap heavily with compute and NCCL. The sums in this table are not wall time and must not be interpreted as utilization.

## 5. D2H source result (the requested question)

### No-GAN P2

Per node, the `966,524,928,414 B` D2H total is exactly:

| source/owner category | bytes | decimal GB | fraction |
|---|---:|---:|---:|
| `FSDP_PRE_FORWARD_ALL_GATHER/*` named framework phase | 483,262,464,382 | 483.262464 | 50.000000% |
| exact `pt_autograd_*` runtime-thread owner, no named NVTX interval | 483,262,464,000 | 483.262464 | 50.000000% |
| outer capture only | 4 | ~0 | ~0% |
| no NVTX containment | 28 | ~0 | ~0% |
| **total** | **966,524,928,414** | **966.524928** | **100%** |

The named framework half is `FSDP_PRE_FORWARD_ALL_GATHER/critic` (361,758,720,040 B) plus `.../generator` (121,503,744,048 B), with only sub-kilobyte rollout/capture residues. The other half is not guessed from timing: all of its runtime calls are owned by `pt_autograd_*` threads. The autograd-thread total differs from the exact `UNKNOWN_NO_NVTX_TID` bytes by only 2,432 B (per node).

### GAN-on P3

Per node, the `1,922,826,245,190 B` D2H total is exactly:

| source/owner category | bytes | decimal GB | fraction |
|---|---:|---:|---:|
| `FSDP_PRE_FORWARD_ALL_GATHER/*` named framework phase | 931,528,708,926 | 931.528709 | 48.445808% |
| exact `pt_autograd_*` runtime-thread owner, no named NVTX interval | 961,413,120,000 | 961.413120 | 50.000000% |
| no named NVTX containment | 26,148,864,231 | 26.148864 | 1.359918% |
| outer capture only | 3,735,552,033 | 3.735552 | 0.194274% |
| **total** | **1,922,826,245,190** | **1,922.826245** | **100%** |

The named framework half is `critic` 810,024,964,584 B plus `generator` 121,503,744,048 B (and 294 B of rollout residue). The exact autograd-thread join again differs from the unlabelled source bytes by only 2,432 B per node.

**Interpretation:** the dominant D2H is proven to be issued from FSDP/autograd execution contexts, but the current data does not distinguish parameter unshard/reshard, activation/saved-tensor movement, or other host staging inside those framework calls. It is therefore **not valid to label this as model offload**. The strongest safe statement is “FSDP framework-phase and PyTorch autograd-engine-owned transfers; operation subsource unresolved.”

## 6. H2D and diagnostics cross-check

- P1→P2 logical input H2D is identical. P2 observed input-labeled H2D is only about 0.657 GB/node; the remaining H2D is likewise mostly exact `pt_autograd_*` owner traffic. Input staging is not the explanation for the 966.5 GB D2H.
- P3 input-labeled H2D is about 0.710 GB/node; P3 observed H2D is 963.814 GB/node, again dominated by autograd-thread owner traffic.
- P2 and P3 are clean profiles with `profile_diagnostics` off. No `DIAG_OUTPUT_SHA_D2H`, `DIAG_LOSS_D2H`, `DIAG_FINGERPRINT_D2H`, or GAN diagnostic NVTX source accounts for the large D2H.
- P2/P3 use CUDA-resident EMA. The explicit `EMA_GPU_UPDATE` source has no D2H contribution; the P1→P2 D2H reduction was exactly 12,514,426,880 B/node, confirming the CPU-EMA contribution was removed.
- No source row supports attributing the remaining D2H to `parameter/model offload` or `activation offload`; both are disabled by the frozen contract.

## 7. Code-path audit

Relevant boundaries on HEAD `075a47b`:

- `scripts/distillschedule/probe_static_fsdp2_full5_world16.py:225-227,355-359,429-431`: trace/input loads and application H2D counters (`INPUT_H2D/*`).
- `.../probe_static_fsdp2_full5_world16.py:233-242,365-374,432-452`: FSDP forward/reshard/backward/reduce/optimizer NVTX boundaries.
- `.../probe_static_fsdp2_full5_world16.py:90-118`: profile-only trainable fingerprint D2H; disabled in clean profile.
- `.../probe_static_fsdp2_full5_world16.py:629-642,718-724`: `profile_diagnostics` and `ema_device` switches; P2/P3 use diagnostics off and CUDA EMA.
- `src/distillschedule/profiling/static_profile_diagnostics.py:91-113`: diagnostic scalar/SHA/fingerprint `.cpu()` scopes; absent from clean capture.
- `src/distillschedule/runtime/fsdp2_static.py:505-531`: CPU EMA emits `EMA_CPU_D2H`; CUDA EMA updates the local shadow on device and does not issue EMA D2H.
- `src/distillschedule/profiling/static_nsys_bounded.py:141-180`: bounded native-order activity stream and union analysis; it does not perform application source attribution.

## 8. Exact, unknown, and NOT_RUN boundaries

**Proven:** four raw SQLite files are available; CUDA activity/runtime correlation is complete; worker PID/rank mapping is inferred and complete; all source-category bytes/events/summed durations reconcile to raw memcpy; P2/P3 D2H owner/category percentages above; GPU EMA D2H removal; clean diagnostics do not account for the bulk.

**Excluded:** old 15-rank data is not used as a world16 conclusion; formal No-GAN/GAN timings are unchanged; no new Full5 or timing was launched; compute/NCCL/copy overlaps are not added.

**Still unknown:** `pt_autograd_*` operation subsource (parameter unshard/reshard vs activation/saved-tensor or other FSDP/PyTorch staging), the 26.149 GB P3 no-containment group, and the small outer-capture residue. `UNKNOWN_NO_NVTX_CONTAINMENT` bytes are exact residuals, not guesses.

**Capture limitation:** NVTX `rangeId`/`endGlobalTid` columns are zero/non-populated in these exports. A future bounded canary would need populated range correlation or explicit CUDA runtime correlation markers at the FSDP/activation boundaries. This iteration does not launch it.

## 9. Deliverables and gate

Figures (diagnostic only):

- `artifacts/profile/source_attribution/figures/p2_source_attributed_timeline.png`
- `artifacts/profile/source_attribution/figures/p3_source_attributed_timeline.png`

The upper panels show same-TID NVTX phase intervals for 16 inferred worker ranks; the lower panels are source/category-stacked observed D2H bytes. They are not formal timing plots. Per-rank/per-phase exact values are in `detailed_source_attribution.csv`.

Schema/join audit and hashes:

- `artifacts/profile/source_attribution/schema_audit/sqlite_schema_and_join_summary.json`
- `artifacts/profile/source_attribution/source_attribution_artifacts_sha256.json`

Local code syntax validation: `python3 -m py_compile` passed for the existing analyzer, diagnostics, FSDP2 runtime, and static probe. Local `pytest` was **NOT_RUN** because this home environment has no `pytest` module; no source code was changed in this iteration. Previously recorded focused remote tests remain historical evidence, not a new test run.

**Gate:** `PASS_DIAGNOSTIC_SOURCE_ATTRIBUTION_COMPLETE_PARTIAL_OPERATION_SUBSOURCE`. Stop here as authorized. No performance optimization, no formal timing, and no new capture should start without a separate authorization.
