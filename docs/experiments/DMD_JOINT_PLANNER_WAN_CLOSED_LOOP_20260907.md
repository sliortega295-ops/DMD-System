# Wan measured-cost planner: closed-loop native diagnostic

Runtime source: `4e3ffb59c21429f3e9e714f14006f5ae6afbd002`. Later publication commits are not execution identities.

Seven implementations / 98 operation costs / 1,568 rank-stage records. CPU search agrees with a bounded exhaustive oracle.
The existing native dispatcher executed two cycles per trial with one initialization; only cycle 2 is compared. No Nsight.

| Phase | Implementation | Cycle 1 (s) | Cycle 2 (s) | Prediction (s) | Error |
|---|---|---:|---:|---:|---:|
| baseline | joint-wan1-fsdp-c512-e0 | 46.069719 | 35.975044 | 36.419292 | +1.23% |
| incumbent | joint-wan1-resident-c2048-e1 | 27.579021 | 24.428815 | 24.635730 | +0.85% |
| candidate | joint-wan1-resident-c2048-e0 | 28.341882 | 25.186929 | 25.265226 | +0.31% |
| candidate | joint-wan1-resident-c1024-e1 | 28.837623 | 25.618400 | 25.845579 | +0.89% |
| candidate | joint-wan1-resident-c1024-e0 | 29.368273 | 26.381003 | 26.310404 | -0.27% |
| confirmation | joint-wan1-resident-c2048-e1 | 27.703484 | 24.617367 | 24.635730 | +0.07% |

Confirmed diagnostic speedup vs matched FSDP: **1.461369x**.
Mean absolute prediction error: 0.60%; max: 1.23%.
All six native trials: 16/16 runtime/semantic/resource gates PASS; final G4/C20/EMA20; zero observed new CUDA/cgroup OOM.
Numerical status remains **REVIEW**, not strict P0 parity; strict_winner=null. The selected implementation rediscovered the incumbent; this is not a new optimization gain over it.

## Scope and remaining work

- Current Wan dimensions: placement, modulation chunk, existing no-grad LoRA epilogue. The native clustered schedule is fixed; this does not yet demonstrate searched sharing or G:C pool placement.
- Costs are synchronized diagnostic stage envelopes, including internal communication. They are not profile unions or formal native timing; the table above separates prediction from measurement.
- Native FSDP allocator peak can be reset in nested helpers. Use calibration peaks for the memory model, not the smaller native reported peak as a full-cycle assertion.
- One prerequisite failed on both SSH clients Broken pipe during loading. It was preserved and classified LAUNCHER_ERROR, not OOM. Fresh recovery epoch followed explicit cleanup/transport/preflight checks; no failed run was overwritten.
- Timing budget: 2 historical attempts + 6 current-source attempts = 8. Correctness/calibration costs are separate and not free; complete calibration of all seven combinations limits scalability claims.
- Canonical profiles and main are unchanged. Next: LiveWan generated adapter and measured pool/DAG costs, then H3 Grid1000, then both Helios workloads. These new model-based pilots are NOT_RUN.
