# Iteration 301 — MiniMax-H3 Native Nsight Attribution

## Status and claim boundary

- **Run ID:** `iteration_301_minimax_native_nsys_20260818_r2` (diagnostic profile); the first run with the requested base ID is preserved as `FAIL_NSYS_OPTION_INCOMPATIBLE` because Nsight 2023.3.1.92 rejected `--nccl-trace=default`.
- **Analysis status:** `PASS_WITH_PARTIAL_STAGE_MARKER_PROVENANCE`.
- **Purpose:** overlap-aware attribution only; no schedule or training-math change and no new formal timing.
- **Formal Native reference remains frozen at:** **917.9939734640066 s** (917.994 s). The profiled rank-max envelope below (**916.773712021 s**) is not a timing denominator and is not used for speedup.
- **Definition:** `exposed` is the interval not covered by a non-NCCL CUDA compute kernel on the same rank. It is an interval attribution, not causal removable time or a speedup prediction.
- **Denominator:** pooled values are the sum of exact per-rank interval unions over 16 ranks, `D = 14668.344849198 GPU-s`; percentages below use this GPU-second denominator. Rank-max values are the worst-rank timeline proxy and must not be added to pooled values or interpreted as a global wall-clock decomposition.

## Workload identity

| Item | Identity |
|---|---|
| Profiling worktree | `/home/lyy/Helios/DMD-System/worktrees/lightx2v-minimax-selective-boundary-offload-nsys-20260818` |
| Profile workload HEAD | `8dcd5bf87e398afe765a78e517ef3259f4965b83` |
| Analysis-only HEAD | `fd7aef21d24e92f36f9b4aaa5d1d5dc95de8c6df` (stage-envelope repair; not used to run the workload) |
| Selective workload base | `c0c9985cfd9c3716a0fa3e78f810dd637dad60a6` |
| LightX2V | `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be` |
| Hardware | 2 nodes × 8 × NVIDIA A100-SXM4-40GB, driver 535.161.08 |
| Shape / dtype | MiniMax-H3 768×1344, 124 frames, BF16, B1/rank |
| Parallelism | world16, FSDP2 FULL_SHARD, sequence parallel 1 |
| Trainable state | Student/Fake LoRA rank128; Teacher frozen |
| Workload | Student1 + Fake5, fixed `end_step_idx=3`, no GAN/EMA/model swap/parameter offload |
| Boundary policy | `checkpoint_boundary_cpu`, native per-block checkpoints, 50 boundaries/grad-forward; Student 1 grad graph, Fake 5 grad graphs |
| Attention | `_flash_3_hub` |
| Trace/config | config SHA `1e615d1c9a379f177356661e13323cafed91a4a8599cf1ee5e5683f034d8fe22`; prompt cache SHA `a2e479c5c8f8c656f4548d24f01408081e7c2631e07fc2cb89882dec56c5e5dd`; runtime boundary manifest SHA `14ca27dd213427b955a828bbdccec71a699be6ff7a75b051021783b75aa6b6d0` |

The old preflight text contained a superseded boundary-manifest hash (`2f3c...`). The **per-rank runtime boundary receipts and the profile workload use `14ca...`**, so the latter is the authoritative identity for this run; the mismatch is preserved in the manifest rather than silently hidden.

The receipt `source.census` pointers are historical provenance from a prior boundary-census flush (`c576488...`); they are not used as the profile code identity. The profile worker receipts and profile manifest identify the executed workload as `8dcd5bf...`; policy counters and observed transfer sizes are taken from the current run.

## Correctness / capture gates

- 16/16 range receipts: PASS; all range receipts are retained under `raw/range_receipts/`.
- Native application census per rank: **Student 24, Fake 6, Teacher 1 = 31 forwards**; **Student 1, Fake 5 = 6 backwards**; fixed exit `3`: PASS.
- Boundary receipts: Student 50 selected checkpoint calls, Fake 250; 300 D2H + 300 H2D large boundary transfers per rank; live CPU boundary bytes return to zero: PASS.
- No new CUDA OOM/NCCL failure; cgroup `oom_kill` unchanged (node0 historical 4→4, node1 0→0); no owned GPU worker remained after teardown.
- **Stage marker caveat:** ranks 13 and 15 did not emit the Student stage NVTX marker. Their Student window is explicitly inferred as cycle-marker start → first Fake marker (`cycle_marker_to_first_fake_inferred_missing_nvtx`). The 31-call census, copy reconciliation, and stage envelopes are retained, but stage provenance is partial; this is not reported as a perfect marker-complete capture.

## Native cycle decomposition (overlap-aware)

| Category | Union / activity (GPU-s) | Hidden by compute (GPU-s) | Exposed vs compute (GPU-s) | % of pooled denominator |
|---|---:|---:|---:|---:|
| Compute kernel union | 14044.602 | n/a | n/a | 95.748% |
| NCCL (all classes) | 2914.200 | 2556.088 | 358.112 | 19.867% |
| Boundary-offload candidate (large H2D+D2H) | 244.745 | 0.710 | 244.035 | 1.669% |
| NCCL ∪ boundary-offload activity | 3058.340 | n/a | 502.237 | 20.850% |
| No traced GPU activity | 121.079 | UNKNOWN | UNKNOWN | 0.825% |

Worst-rank cycle values (not pooled): rank-max wall **916.774 s**, NCCL union **324.106 s**, NCCL exposed **29.282 s**, boundary-offload union **16.727 s**, boundary-offload exposed **16.673 s**, and movement union exposed **38.180 s**. The pooled unique/shared exposed split is:

- NCCL-only exposed: **258.202 GPU-s** (1.760%); rank-max **22.891 s**.
- Offload-only exposed: **144.126 GPU-s** (0.983%); rank-max **9.983 s**.
- NCCL/offload shared exposed: **99.909 GPU-s** (0.681%); rank-max **7.753 s**.
- Sum = **502.237 GPU-s**, exactly the movement-exposed union.

NCCL hidden fraction is **87.71%** of NCCL union; boundary-offload hidden fraction is only **0.29%** of its candidate union. This says most NCCL overlaps local compute, while the selected boundary copies are almost entirely outside the compute union in this profile.

## NCCL collective breakdown

| Collective | Events | Pooled union (GPU-s) | Exposed vs compute (GPU-s) | Rank-max union (s) | Rank-max exposed (s) |
|---|---:|---:|---:|---:|---:|
| all-gather | 30,912 | 2,024.158 | 236.994 | 224.624 | 19.401 |
| reduce-scatter | 4,992 | 889.189 | 120.264 | 99.430 | 10.306 |
| all-reduce | 124 | 0.853 | 0.853 | 0.078 | 0.078 |
| other/unknown NCCL | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| **total** | **36,028** | **2,914.200** | **358.112** | **324.106** | **29.282** |

Kernel-name classification indicates the majority of NCCL activity is FSDP-style **all-gather** (about 69.44% of NCCL union), followed by **reduce-scatter** (about 30.51%); all-reduce is negligible. This is a kernel classification, not a source-level proof that every all-gather is parameter materialization, but it is consistent with FSDP2 FULL_SHARD forward/backward communication.

## Boundary-offload evidence

The large-copy bucket uses a 64 MiB/event threshold only as a conservative selector; it is not blindly labelled activation offload. Runtime receipts identify the selected operation as `checkpoint_boundary_cpu`:

- Student: 50 boundary scopes / grad forward, **20,283,648,000 B** logical/storage offloaded; 50 D2H + 50 H2D boundary transfers.
- Fake: 250 boundary scopes over 5 grad forwards, **101,410,713,600 B** logical/storage offloaded; 250 D2H + 250 H2D boundary transfers.
- World16 observed candidate: **4,800 H2D + 4,800 D2H events**, each direction **1,947,124,838,400 B = 1,813.401 GiB**; total round-trip candidate **3,626.803 GiB**.
- Observed large-event sizes **405,619,200–405,672,960 B (386.829–386.880 MiB)** match the boundary manifest expected range exactly. This is strong source evidence for selective checkpoint-boundary activation movement, not model/backbone swap.
- All copies, including small copies and D2D, remain separately reported. Total raw H2D/D2H/D2D bytes are **1,814.370 / 1,813.403 / 635.560 GiB**, respectively; copy sum-duration is not wall overhead.

The profile shows these boundary transfers are almost not hidden by non-NCCL compute: exposed candidate **244.035 GPU-s** pooled, versus only **0.710 GPU-s** hidden. That is an overlap statement, not a causal speedup estimate.

## Stage decomposition

Stage values use rank-max stage wall and pooled compute/NCCL percentages; exposed columns are rank-max interval exposure, not additive wall components.

| Stage | Wall rank-max (s) | Compute union | NCCL union | NCCL exposed rank-max (s) | Offload union rank-max (s) | Offload exposed rank-max (s) | Movement exposed rank-max (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student update | 170.189 | 93.45% | 19.08% | 5.357 | 3.288 | 3.279 | 6.805 |
| Fake update 0 | 149.891 | 95.80% | 19.07% | 5.051 | 3.055 | 3.047 | 6.546 |
| Fake update 1 | 149.280 | 96.32% | 19.81% | 4.788 | 3.071 | 3.061 | 6.242 |
| Fake update 2 | 149.182 | 96.41% | 20.08% | 4.657 | 3.091 | 3.081 | 6.142 |
| Fake update 3 | 149.187 | 96.42% | 20.26% | 4.701 | 3.041 | 3.031 | 6.171 |
| Fake update 4 | 149.270 | 96.41% | 21.01% | 4.843 | 3.039 | 3.029 | 6.331 |

## Application DiT call attribution

The generated `dit_breakdown.csv` has 496 rows = 31 application calls × 16 ranks. The following are per-call means over those rows; byte columns are totals for the group across all 16 ranks. Checkpoint recompute is not counted as an application forward.

| Role / grad mode | Calls total | Calls/rank | Mean call wall (s) | Mean compute union (s) | Mean NCCL union (s) | Mean NCCL exposed (s) | H2D total (GiB) | D2H total (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| student / nograd | 368 | 23 | 13.604 | 13.438 | 3.035 | 0.114 | 0.493 | 0.001 |
| student / grad | 16 | 1 | 21.121 | 17.384 | 1.830 | 1.827 | 0.022 | 302.231 |
| fake / grad | 80 | 5 | 20.872 | 17.423 | 1.741 | 1.737 | 0.113 | 1511.170 |
| fake / nograd | 16 | 1 | 13.531 | 13.415 | 2.891 | 0.067 | 0.018 | 0.000 |
| teacher / nograd | 16 | 1 | 15.403 | 14.951 | 3.648 | 0.212 | 0.020 | 0.000 |

Interpretation: each rank has 23 Student no-grad calls plus one Student grad call, five Fake grad calls, and one Teacher no-grad call. The large D2H volume is concentrated in Student grad and Fake grad calls, exactly where boundary checkpoint receipts report CPU staging; no application-level D2H boundary traffic is visible in no-grad Student/Teacher calls.

## Reconciliation and capture limitations

Raw CUDA copy reconciliation is exact (all residuals are zero):

- H2D: 43,730 events, 1,948,165,229,992 B, summed duration 100.852077632 s.
- D2H: 6,962 events, 1,947,126,493,568 B, summed duration 144.102990014 s.
- D2D: 6,864 events, 682,427,807,232 B, summed duration 1.163081197 s.
- Other copy: 0 events / 0 B.

The `no_traced_gpu_activity` union is **121.079 pooled GPU-s / 0.825%** (rank-max 8.357 s). It is **not** proven GPU idle: it may include host launch gaps, CUDA runtime synchronization, barriers, profiler gaps, or activity not represented in the selected export. This run did not add a new CPU-sampling or API-backtrace capture. Therefore the proper label is `UNKNOWN_NO_GPU_ACTIVITY`, not CPU idle.

The first profiling launch failed before workload execution because Nsight 2023.3.1.92 rejected `--nccl-trace=default`; it is preserved and not mixed with the successful run. The successful run used one `nsys` session per node, 8 workers/session, `cuda,nvtx` tracing, `cudaProfilerApi` capture, and no advanced NCCL option.

## Conclusion (diagnostic only)

**Conclusion A+B, with compute still dominant:**

1. **NCCL is substantial in timeline activity but mostly hidden by compute.** The pooled NCCL union is 2,914.200 GPU-s, of which 2,556.088 GPU-s (87.71%) overlaps non-NCCL compute; exposed NCCL is 358.112 GPU-s pooled and 29.282 s on the worst-rank proxy. Kernel names are dominated by all-gather, with reduce-scatter as the main backward class.
2. **Selective boundary offload is a real, separately evidenced exposed movement source.** Its candidate union is 244.745 GPU-s pooled, with 244.035 GPU-s exposed versus compute (rank-max 16.673 s). The exact 386.83–386.88 MiB event sizes and runtime receipts match the 50 Student + 250 Fake boundary scopes.
3. **NCCL/offload overlap matters.** The union-exposed movement is 502.237 pooled GPU-s / 38.180 s worst-rank proxy, not 358.112 + 244.035; the shared exposed portion is 99.909 pooled GPU-s / 7.753 s worst-rank.
4. The evidence does **not** support conclusion C (“movement is all hidden”). It also does not justify converting these interval values into a claimed 38.180 s speedup.
5. The formal 917.994 s Native number remains unchanged. No optimization or new formal timing was run.

## Minimal next question (not executed)

The highest-value bounded follow-up is to test whether the **selective boundary D2H/H2D** can be scheduled in a legal lead window or reduced without changing checkpoint semantics, while keeping the same 31/6 application inventory. Before implementing it, add a capture or runtime counter that proves consumer-ready timing and verifies boundary-copy dependency; do not infer causal savings from the 244.035 GPU-s value alone.

## Artifacts

- Report: `reports/native_nsys_report.md`
- Attribution JSON: `native_nsys_attribution.json`
- Plot: `reports/native_nsys_attribution.png`
- Tables: `stage_breakdown.csv`, `dit_breakdown.csv`, `copy_size_histogram.csv`, `nccl_kernel_breakdown.csv`
- Raw receipt/log tree: `raw/`
- Analyzer source and synthetic test: profiling worktree at `/home/lyy/Helios/DMD-System/worktrees/lightx2v-minimax-selective-boundary-offload-nsys-20260818/`
- Remote raw `.nsys-rep`/SQLite paths and SHA256 values are recorded in `manifest.json`; large raw files were intentionally not copied into the local evidence subtree.
