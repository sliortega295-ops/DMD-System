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
19. C5 passed all gates.  Its formal Full5 is `142.295341636 s`, a small
    `0.911371474 s` improvement over C4 and `17.7552%` over C0.  Promote the
    substrate, but do not test stride 1: peak reserved reached `38.533203 GiB`
    and the smallest driver-free end receipt was `0.012085 GiB`.
20. Capture one diagnostic C5 profile with the same correctness summary, then
    use nsys-ai overlap/idle/compute evidence and the kernel-design candidate
    ledger to choose one new mechanism.  Do not use profiled wall as formal
    timing and do not repeat B2/B5 or Score packing.
21. Iteration 356 is preregistered as that diagnostic-only profile.  Keep the
    exact C5 runtime at `e75c80e`, validate 16 range receipts and both node
    SQLite reports, analyze remotely, and promote at most one new mechanism
    from measured exposed time rather than aggregate kernel duration.
22. Iteration 356 passed: both node profiles and 16 range receipts are valid.
    The all-rank median range contains `128.837 s` compute union, `60.529 s`
    NCCL union with `53.170 s` hidden under compute, `7.354 s` exposed NCCL,
    `0.488 s` exposed copy and `6.128 s` no-traced activity.  D2H is
    effectively absent.  The remaining root cause is repeated model compute;
    communication-only work cannot close the target.
23. C6 is a bounded topology-only arm: run exact C5 with node-local HSDP
    `(replicate=2, shard=8)`.  Preserve B1/global-B16 and every rank-qualified
    sample and physical call.  Validate and run correctness/capacity before one
    formal timing.  Do not combine prefetch, batching, packing, or a new kernel.
24. C6 validate passed, but its first correctness run failed before workload
    execution: the resident HSDP Generator shard plus a full pre-shard Critic
    CUDA copy reached `38.27 GiB` allocated and a `144 MiB` allocation OOMed.
    This is the known full-device-before-FSDP construction mechanism, not a
    steady-state HSDP capacity result.  C6.1 reuses the bounded FSDP-first
    recovery: restore on CPU and invoke the same bottom-up `fully_shard`
    sequence directly, allowing each managed block to move and shard before
    the next.  Re-run the same C6 correctness gate; do not change runtime math.
25. C6.1 fixed construction without changing the timed workload.  Validate and
    correctness both pass on 16 ranks with HSDP `(replicate=2, shard=8)`, exact
    G1/C5/EMA1 and final G201/C1005/EMA801.  The matched unprofiled Full5 is
    `144.716706512 s`, however: `2.421364876 s` (`1.7016%`) slower than C5.
    Reject HSDP as a performance arm and return to the `142.295341636 s` C5
    world16-FSDP substrate.  Retain the FSDP-first change only as a bounded
    cold-construction compatibility path.
26. The next iteration must target compute.  Iteration 356 measures
    `128.836834 s` compute union, while C6 shows that shortening the shard group
    does not reduce formal wall.  Use a bounded NCU capture on a real dominant
    C5 GEMM/attention kernel before changing kernel math or schedule; do not
    infer MFU from Nsight Systems temporal coverage.
27. Iteration 358 passes its diagnostic gates.  The representative dominant
    Critic BF16 GEMM reaches `88.322976%` Tensor/SM throughput and only
    `25.233768%` DRAM throughput.  Direct tuning of that single already
    Tensor-bound cuBLAS kernel has an impossible all-occurrence upper bound of
    only about `3.31 GPU-s/GPU`; it cannot close the remaining `21.185 s` gap.
28. C8 ports the existing exact per-rollout Generator no-grad cross-attention
    encoder K/V reuse onto C5 and extends only its admission/gates to GAN-on.
    Keep five logical rollouts distinct, preserve all DiT/attention calls and
    FSDP collectives, clear the cache after each rollout, strengthen exact
    encoder storage/version identity, and compare every cached B1 output with
    the uncached C5 reference before one formal timing.
29. C8 passed validate-only and 16-rank correctness at `9cee963`, with every
    rank reporting 200 cache fills and 880 same-rollout reuses.  Its matched
    unprofiled Full5 is `141.235767135 s`, only `1.059574501 s` (`0.7446%`)
    faster than C5.  Promote it as the fastest substrate, but classify the
    isolated context-hoisting gain as small.
30. C8 is still `20.125629032 s` above the requested threshold.  The next
    iteration must not repeat B2/B5, Real-Score packing, HSDP, stride-1 retained
    checkpoint state, or direct tuning of the already Tensor-bound dominant
    GEMM.  Reconcile the remaining phase profile with source call inventory and
    preregister one structural exact-compute or launch-gap candidate before
    editing or launching.
31. C9 adds only the source-pinned restrict-self-attention history K/V cache to
    C8's five Generator no-grad B1 rollouts.  It reuses the invariant history
    prefix within one rollout while preserving all current-token, attention,
    block, DiT and FSDP calls.  Require exact 40-processor lifecycle receipts
    and an uncached B1 reference for every rollout before one formal timing.
32. C9 is rejected before timing.  Validate-only passed, but the first real
    call proved the frozen model has `restrict_self_attn=False` on all blocks;
    the wrapper failed closed on all ranks.  Changing that flag would change
    self-attention math, while skipping every block would save nothing.  Return
    to C8 and retain the failed run as the one bounded verification permitted
    by the repeated-failure rule.
33. Existing explicit-next FSDP2 prefetch is closed by the exact No-GAN matched
    result (`118.267264657 s` control versus `118.737632053 s` prefetch).  Do
    not rerun the same mechanism on GAN without materially new evidence.
34. C10 targets the measured BF16 add family at its largest source-qualified
    owner: scale-1 PEFT LoRA epilogues account for `214,960` invocations across
    node0's eight ranks versus `439,436` BF16 add launches in the C5 profile.
    First run one locked A100 B1/B3 `linear + add` versus beta-add GEMM canary,
    comparing output and all gradients under the frozen P0 thresholds and
    measuring warm CUDA-event latency.  If either parity or latency fails,
    reject C10 without a world16 launch.
35. Only after the canary passes, add the default-off, source-pinned runtime
    path and receipts; run focused tests, validate-only, 16-rank correctness,
    and one unprofiled timing.  Do not combine a second kernel, topology,
    prefetch, packing, batching, or timer change.
36. C10 is rejected before implementation.  Its locked A100 canary passes P0
    parity and has bitwise-identical input/LoRA-A/LoRA-B/base gradients, but
    no-grad B1 improves only `0.8--1.7%` and both B3 forward shapes regress
    slightly.  Because the preregistered gate required every registered shape
    to improve, world16 correctness and timing remain `NOT_RUN`; return to C8.
37. C11 targets the source-pinned modulated-norm backward's per-chunk autograd
    reconstruction.  The formal path executes about `4,720` such gradient
    chunks per rank.  First compare the source backward with an explicit
    native-LayerNorm backward on one locked A100 for real B1/B3 PhaseLocal
    layouts, including hidden and every timestep-input gradient plus warm CUDA
    event latency.  Reject before implementation if parity or both-shape
    latency does not pass.
38. C11 canary `iter362_gan_cuda_c11_modnorm_backward_canary_r2_20260830`
    passes: output and hidden gradients are bitwise, timestep gradients pass
    P0, and B1/B3 forward+backward improve `1.37195x`/`1.36542x`.  Implement
    the independently gated default-off backward, record exact execution
    receipts, then run focused tests, validate-only, correctness, and one
    formal timing without combining another candidate.
39. C11 completes all gates at `f3c5ed480001fb0007914de25264edd8439e9f50`.
    Formal Full5 is `138.462140212 s`, `2.773626923 s` faster than C8 and
    `19.970781%` below C0.  Promote C11 as the new substrate; the 30% target is
    still `17.352002109 s` away.
40. Iteration 364 profiles exact C11 without changing the formal denominator.
    It passes 16/16 range and two-node parser gates.  Median compute union is
    `126.112 s`; exposed NCCL is `7.121 s`, exposed copy `0.501 s`, and
    no-traced activity `5.903 s`.  Source correlation finds one bounded
    application-owned gap: one allocator flush and six explicit post-rollout
    device synchronizations inside the measured path.
41. C12 adds only a default-off policy to omit those six synchronizations and
    one `empty_cache`, retaining Full5 boundary synchronization and every
    dependency/semantic gate.  Run focused tests, validate-only, 16-rank
    correctness/capacity and, only on PASS, one unprofiled formal timing.  Do
    not combine finite-check removal, state-machine changes, prefetch, or a
    second optimization.
42. C12 passes all gates at `7811b20`: 16/16 ranks report zero in-window
    rollout fences/allocator flushes with outer boundary sync preserved.  Its
    formal Full5 is `138.176082544 s`, a real but tiny `0.206596%` gain over
    C11.  Promote the code as the fastest substrate, close broad sync removal
    as a major hypothesis, and return to profile-qualified compute/kernel
    ownership for the remaining `17.065944441 s` gap.
43. C13 targets repeated immutable Generator-rollout context preparation that
    C8 did not cache: history Conv3d patching, RoPE construction, history
    packing and PixArt text projection.  Reuse these values only inside one
    existing B1 no-grad rollout scope, with exact tensor identity/version and
    lifecycle receipts.  Keep current-latent patching, every DiT/attention/FFN
    call and every FSDP collective unchanged.  Require focused tests,
    validate-only, five uncached B1 parity rows per rank, 16-rank
    correctness/capacity and then one faster unprofiled timing.
44. The pre-launch source audit corrected one instrumentation-only assumption
    before any C13 GPU run: an official rollout traverses three pyramid
    layouts, so the changing current latent cannot have one layout contract.
    C13 therefore retains one exact history-patch/history-RoPE prefix and one
    text projection per rollout, while caching current-token RoPE separately
    for each of the three observed post-patch layouts.  Current-latent patching
    still executes on every physical call.  Require exactly three layout fills
    plus `call_count-3` layout reuses per rollout; do not weaken the exact
    history/index/text identity gates.
45. C13 passes all correctness and capacity gates at `02517f1`: 16/16 ranks,
    exact five-row uncached B1 parity, G1/C5/EMA1, final G201/C1005/EMA801,
    exact cache lifecycle, and zero CUDA/cgroup OOM. Its formal Full5 is
    `138.243935311 s`, however, which is `0.067852767 s` (`0.049106%`) slower
    than C12. Reject C13 for performance and return to C12 rather than stacking
    a negative cache mechanism into the next candidate.
