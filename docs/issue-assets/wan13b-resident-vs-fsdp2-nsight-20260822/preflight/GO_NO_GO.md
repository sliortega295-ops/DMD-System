# Iteration 327 preflight: matched Resident Nsight

- Current gate: diagnostic-only matched Resident profile corresponding to the valid Iteration 326 world16-FSDP2 profile.
- Authorized hypothesis: determine why same-GPU `NCCL not overlapped with compute` overstates the observed Resident-vs-FSDP wall delta by profiling the Resident arm with the identical Nsight/runtime/workload window.
- Frozen contract: real Wan2.1-T2V-1.3B, world16, B1/rank/global B16, Score packing off, G1/C5/EMA5, G2/C10/EMA10 -> G3/C15/EMA15, GPU residuals, GPU EMA, no parameter/activation/optimizer offload, no streaming/model swap, identical bootstrap/trace/model/runtime and Nsight 2024.6.2.
- Only legal variable: placement `world16-fsdp` -> `resident`; Resident retains its validated world16 mean LoRA-gradient all-reduce.
- Matching prior evidence: Iteration 321 diagnostic A/B observed Resident 55.868545s and FSDP2 66.762061s but cross-placement numerical parity failed; user explicitly authorized ignoring numerical parity for this diagnostic. Iteration 326 produced a valid FSDP2 Nsight profile but `exposed_vs_compute` is only a same-GPU overlap metric, not causal critical-path time.
- Rejected paths: no timing claim, no threshold relaxation, no workload or optimizer changes, no profiler-time speedup calculation, no reuse/overwrite of Iteration 326.
- New falsifiable question: under the same profiler and exact range, how much NCCL/copy/no-traced activity remains in Resident, and how much of the FSDP2 profile difference is actually placement-specific?
- Live substrate: both remote checkouts clean at c47135a; 16/16 A100 GPUs show 0 MiB and no compute clients; node locks unheld; isolated Nsight 2024.6.2 present; no active owned torchrun/nsys workers. Node0 has one historical defunct root torchrun with no GPU/context.
- Decision: GO for one diagnostic-only Resident world16 profile after an instrumentation-only placement-label/analyzer patch and focused tests.
