# Research log — Iteration 301

## Iteration 301 — Native MiniMax-H3 Nsight movement attribution

### Milestone / Gate
PR15/LightX2V Native MiniMax-H3 selective checkpoint-boundary-offload diagnostic. Formal Native one-cycle result remains frozen at 917.9939734640066 s. This iteration is diagnostic-only; no formal timing or schedule optimization is authorized.

### Question
Within the already passing world16 Native Student1 + Fake5 workload, how much FSDP2/NCCL activity and selective checkpoint-boundary movement is exposed against non-NCCL compute, and how much is hidden by compute?

### Hypothesis
Most NCCL timeline activity is overlapped by DiT compute, while the selected CPU boundary checkpoint transfers create a smaller but more exposed movement component. The profile must distinguish interval exposure from causal removable time.

### Frozen workload / allowed changes
- 2 nodes × 8 A100-SXM4-40GB, world16, FSDP2 FULL_SHARD, SP1.
- MiniMax-H3 768×1344, 124 frames, BF16, B1/rank.
- Student/Fake LoRA rank128, frozen Teacher, Student1 + Fake5, fixed `end_step_idx=3`.
- 31 application forwards (Student24/Fake6/Teacher1), 6 backward graphs (Student1/Fake5).
- Existing `checkpoint_boundary_cpu` per-block policy only; no whole-stage `save_on_cpu`, no segment=8, no parameter offload, model swap, GAN, EMA, batching, or schedule change.
- Allowed profiling-only changes: Nsight marker/launcher/analyzer and fail-closed stage-window reconstruction. No training math changes.

### Evidence / commands
- Profile branch workload HEAD: `8dcd5bf87e398afe765a78e517ef3259f4965b83`.
- Analyzer-only HEAD: `fd7aef21d24e92f36f9b4aaa5d1d5dc95de8c6df`.
- Successful Run ID: `iteration_301_minimax_native_nsys_20260818_r2`.
- First attempted Run ID preserved as `iteration_301_minimax_native_nsys_20260818`; failed before workload because Nsight 2023.3.1.92 rejected `--nccl-trace=default`.
- Successful capture: one Nsight session/node, 8 workers/session, `cuda,nvtx`, `cudaProfilerApi`, advanced NCCL option disabled.
- Analyzer command and remote `.nsys-rep`/SQLite paths are in `manifest.json`.

### Correctness / capture results
- 16/16 range receipts PASS.
- Per-rank census PASS: Student24, Fake6, Teacher1; Student1/Fake5 backward; fixed exit=3.
- Runtime boundary receipts PASS: Student 50 and Fake 250 checkpoint-boundary calls; 300 large D2H + 300 large H2D events/rank.
- Receipt `source.census` pointers refer to historical flush head `c576488...`; this is retained as provenance only. The executed profile identity is the current worker HEAD `8dcd5bf...`.
- Raw H2D/D2H/D2D bytes, events, and summed durations reconcile exactly (all residuals zero).
- Large event sizes 405,619,200–405,672,960 B match boundary manifest range.
- Stage marker provenance is partial: ranks 13 and 15 lack Student stage NVTX; analyzer infers their Student window from cycle start to first Fake marker and records this explicitly. No claim of marker-complete stage capture is made.
- No new CUDA OOM/NCCL error; cgroup oom_kill unchanged; owned workers cleaned.

### Measured results
- Diagnostic rank-max cycle envelope: 916.773712021 s; rank-median 916.771355247 s. Not a formal timing result.
- Pooled denominator: 14,668.344849198 GPU-s (sum of 16 exact rank interval unions).
- Compute union: 14,044.601886338 GPU-s (95.748%).
- NCCL union: 2,914.200114065 GPU-s (19.867%); 2,556.088471085 hidden by compute; 358.111642980 exposed (2.441%); rank-max exposed 29.282 s.
- NCCL classes: all-gather 2,024.157915590 GPU-s / 236.994416995 exposed; reduce-scatter 889.189210044 / 120.264237554 exposed; all-reduce 0.852988431 / 0.852988431 exposed.
- Boundary-offload candidate union: 244.744998890 GPU-s; 0.710071479 hidden; 244.034927411 exposed (1.664%); rank-max exposed 16.673 s.
- NCCL∪boundary-offload exposed union: 502.237218769 GPU-s (3.424%); rank-max 38.180 s. Shared exposed overlap is 99.909351622 GPU-s / rank-max 7.753 s, so NCCL and offload exposed values cannot be added.
- Candidate large-copy volume: 1,947,124,838,400 B H2D + the same D2H = 3,626.803 GiB round trip across world16; receipts attribute it to checkpoint-boundary CPU staging.
- No traced GPU activity: 121.078723070 GPU-s (0.825%), labelled UNKNOWN_NO_GPU_ACTIVITY, not CPU idle.

### Interpretation
Conclusion A+B with compute dominant. NCCL is mostly hidden (87.71% of NCCL union), but non-trivial NCCL exposure remains. Selective boundary movement is almost entirely outside the compute union in this diagnostic interval analysis and is a distinct exposed source. These numbers are not causal speedup estimates and do not update the 917.994 s formal result.

### Files changed / artifacts
All durable evidence is under `/home/lyy/Helios/research_artifacts/DMD-System/iteration_301_minimax_native_nsys_20260818/`:
- `reports/native_nsys_report.md`
- `reports/native_nsys_attribution.png`
- `native_nsys_attribution.json`
- `stage_breakdown.csv`, `dit_breakdown.csv`, `copy_size_histogram.csv`, `nccl_kernel_breakdown.csv`
- `manifest.json`, `artifact_sha256.json`
- `raw/` receipts and node logs; raw `.nsys-rep`/SQLite remain at the remote paths recorded in the manifest.

### Failure or limitation
The initial profile command used an unsupported Nsight option and is preserved as a bounded launcher failure. Two Student stage markers are absent; stage windows are inferred only for those ranks. Current capture lacks CPU sampling/API backtraces sufficient to close the 0.825% no-traced intervals. Kernel-name NCCL classes are not source-level proof of every operation.

### Gate status
`PASS_WITH_PARTIAL_STAGE_MARKER_PROVENANCE`; diagnostic attribution complete with exact copy reconciliation. Formal speedup eligibility: `false`.

### Next highest-value question
If a future iteration is authorized, measure consumer-ready dependency timing for the selective boundary copies and test one bounded legal hide/reduce mechanism. Do not treat exposed interval seconds as guaranteed speedup.

### Next command
No command executed automatically. Stop at this report as required.
