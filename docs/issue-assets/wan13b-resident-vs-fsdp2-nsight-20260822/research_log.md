# Append-only research log: Iteration 327

- Problem: Iteration 326 reported 29.502 s/GPU NCCL not overlapped with compute, apparently inconsistent with the roughly 20% raw Resident-vs-FSDP timing clue.
- Hypothesis: a matched Resident Nsight trace will show that same-GPU NCCL non-overlap is not causal critical-path time and that removed communication is partly replaced by other exclusive timeline categories.
- Why important: prevents converting a profiler overlap metric into an unsupported speedup estimate.
- Evidence required: identical profiler/workload window, 16/16 receipts, 8/8 devices per node, exact union/exclusive reconciliation, collective split, and post-run cleanup.
- Falsifier: Resident wall shrinks by approximately the full FSDP2 NCCL-not-overlapped delta without growth in other exclusive categories.
- Change: instrumentation-only placement-qualified outer NVTX label/report name and analyzer `--placement`; no training math change.
- Tests: remote focused test suite 10 passed; compile and shell syntax passed; local controller dry run passed.
- Run: one world16 Resident diagnostic profile under Nsight 2024.6.2.
- Correctness boundary: workload/versions/receipts PASS, cross-placement numerical correctness NOT CLAIMED by explicit user waiver.
- Result: FSDP2 minus Resident profiled wall is 6.686 s/GPU, while NCCL union falls 35.105 s/GPU. Resident gains 15.258 s compute-only, 3.914 s copy-only, and 9.247 s no-traced activity; exclusive categories reconcile the wall delta exactly.
- Negative result: 29.502 s/GPU FSDP2 NCCL not-overlapped is not removable critical-path time.
- Analyzer integration failure: official training Python lacked matplotlib after producing JSON/CSV. Failure log was preserved; analysis was rerun with system Python 3 plus matplotlib, without rerunning the GPU workload.
- Limits: profiled wall is diagnostic, stage ranges are controller-aligned rather than PID/TID-correlated per rank, D2H source and no-traced source remain unknown.
- Next question: attribute the placement-invariant D2H and Resident no-traced growth before proposing optimization.
