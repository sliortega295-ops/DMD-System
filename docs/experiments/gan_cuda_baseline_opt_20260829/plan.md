# Executable plan

1. Finish overlap-aware phase attribution for both node reports and preserve
   the remote JSON evidence.
2. Import the exact compute-fusion substrate through `6005c03` without taking
   its node-HSDP or No-GAN defaults.
3. Run focused tests and record the exact candidate diff/receipts.
4. Run one frozen GAN-on world16 correctness gate.
5. C1 failed closed because the official GAN-D Score call packs fake, real, and
   R1 into physical B3 while the imported kernels were registered only for B1.
   Preserve this run as rejected evidence; do not relax the fallback gate.
6. Extend only the exact tensor addressing to registered B1/B3, including
   batch-specific modulation-table/component strides, and prove bitwise B3
   forward/backward parity before another world16 run.
7. C1.1 passed focused CPU/CUDA tests and the frozen world16 correctness gate
   with zero residual/modulated-norm fallback.  Its matched unprofiled Full5
   is `151.851949039 s`, a `21.162533965 s` (`12.23%`) reduction from C0.
8. C1.1 remains `30.741810936 s` above the 30%-reduction threshold.  Analyze
   the remaining hotspot and iterate on the next single evidence-backed
   candidate; do not combine mechanisms before each one has its own gate.
9. Iteration 351 attributes the current `151.754 s` profile envelope to
   `130.967 s` compute union, `6.812 s` NCCL exposed versus compute, and
   `13.424 s` no-traced activity.  The first G optimizer and first C optimizer
   boundaries alone contain about `7.98 s` of setup-shaped no-traced time.
10. C2 changes only setup: extend the exact-restoring optimizer dispatch warmup
    from its validated node-HSDP layout to world16 `(Shard(),)`.  Run focused
    tests, a fresh 16-rank correctness gate, and only then one unprofiled
    timing.  Do not combine HSDP, attention, or another kernel in C2.
11. C2 passed after one preserved fail-closed run exposed logically equivalent
    DeviceMesh wrappers.  The bounded fix validates the logical mesh signature,
    not Python object identity.  Correctness is 16/16 PASS and formal Full5 is
    `144.621745256 s`, `7.230203783 s` faster than C1.1 and `28.392737748 s`
    faster than C0.
12. C2 is still `23.511607153 s` above the requested threshold.  The next arm
    is C3: only enable the previously validated pinned local FA2 backend on the
    C2 substrate.  Do not combine selective checkpoint-output retention, HSDP,
    or a new kernel until C3 has its own correctness and timing evidence.
13. C3 passed validate-only, 16-rank correctness, and matched unprofiled timing.
    Its Full5 is `143.379703687 s`, only `1.242041569 s` (`0.8588%`) faster than
    C2.  Promote the result, but do not mistake the pinned attention backend for
    the missing large optimization: `22.269565584 s` still separates C3 from
    the requested threshold.
14. The next candidate must attack measured compute/launch overhead rather than
    blind communication tuning.  Iteration 351 shows only `6.812 s` exposed
    NCCL versus compute, while the current critical path remains dominated by
    repeated Generator/Critic block compute.  Use a bounded exact chunk/launch
    canary or a GAN-specific packed execution only after its logical-row parity
    and inventory contract is preregistered.
15. C4 changes only the pinned Helios token grouping used by the existing
    modulation/residual/cross-norm helpers from 512 to 2048.  First run a real
    B1/B3 CUDA output-and-gradient canary, then focused tests, validate-only,
    world16 correctness, and at most one matched unprofiled timing.  Reject C4
    rather than combining another mechanism if it does not improve C3.
16. C4 passed the CUDA canary, 78 focused tests, validate-only, 16-rank
    correctness, and one matched timing.  Full5 is `143.206713110 s`, only
    `0.172990577 s` (`0.1207%`) faster than C3.  Retain it as the measured
    fastest substrate but classify the chunk-size gain as tiny and insufficient
    to explain the remaining `22.096575007 s` gap.
17. The broader failure-index search found the later schedule-compatible B2
    implementation, superseding the earlier pre-execution failure.  It already
    removed metadata-only mismatches, batched only rows with identical physical
    schedules, and ran per-row B1 parity.  Run
    `iter355_helios14b_hsdp_compatible_b2_correctness_r2_20260824` failed on all
    15 preserved rank receipts with normalized-L2 `0.058--0.071` and cosine
    `0.9977--0.9984`.  Record B2/B5 as a matching bounded negative and do not
    launch it again.  Real-Score packing remains independently NO-GO.
18. C5 changes only Critic checkpoint replay: for stride-2 blocks, save the
    exact pinned FA2 dispatcher result in the original forward and reuse it in
    non-reentrant replay.  Port the existing fail-closed receipt path to the C4
    substrate without checkpoint bypass, batching, topology, or math changes.
    Run focused tests, validate-only, then 16-rank GAN correctness/capacity.
    Timing is allowed only if every rank reports the exact selected-block,
    original-save, and replay-reuse inventory with zero OOM and no fallback.
