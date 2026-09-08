# H3 generated prepared-action native gate

> Subsequent update: the missing measured-cost/search/native performance unit
> is completed in [H3 action performance closure](DMD_H3_ACTION_PERFORMANCE_CLOSURE_20260907.md).
> Independent cycle2 is **769.785204601 s** against a **771.399059042 s** prediction.
> The correctness-only numbers and original stopping boundary below remain
> historical evidence, not native timing denominators.

**PASS: new native action execution and identity unit, not performance.**
The old depth1/depth2/self-repeat matrix was not repeated. This run exercised
the previously unvalidated generated **window=3** action order in the existing
H3 trainer on all 16 A100 ranks, two cycles in one process.

- Runtime source: `878aff490ce28c661ac1d927f08f8846992baf45`.
- Candidate: `joint-h3v2-d2-n128-w3`.
- Run: `jointp_h3_actions_20260907_r1-w3-correctness`.
- Evidence: `artifacts/experiments/joint_planner_h3_actions_20260907/`.
- This later report commit is not the measured runtime revision.

## The missing connection that is now exercised

The existing versioned cycle graph and `StaticRoleRunner` now call the real H3
helpers, rather than selecting a previously named monolithic recipe:

```text
g  = native _student_step (including its optimizer/LR commit)
rN = native _prepare_one_fake_group(N)
cN = native _apply_one_fake_group(prepared[N])

window 3: g → r0 → r1 → r2 → c0 → r3 → c1 → r4 → c2 → c3 → c4
```

Each prepared group is consumed once, with the original helper arguments and
objects, and released by the common runner after its last consumer. Window=3
means up to three retained **B1 input groups**, not rollout batching, three
concurrent models, or a different optimizer accumulation factor. Student/Fake
updates remain atomic and ordered. Plan construction occurs outside the cycle.

## Native gates

| Check | Actual result |
|---|---|
| Ranks / completed cycles | **16/16; 2/2** |
| Application forwards per rank/cycle | Student **24**, Fake **6**, Teacher **1** |
| Backward graphs per rank/cycle | Student **1**, Fake **5** |
| Final versions / optimizer commits | **G2/C10/EMA0; 2/10/0** |
| Unique rank-qualified samples | **96 per world cycle** |
| Actual dispatched actions | **352 = 16 × 2 × 11** |
| Peak retained input groups | **3** on every rank/cycle |
| Grid wraps / executions / hits / misses | **300/600/600/0**, every rank/cycle |
| Actual stochastic tensor full hashes | **1,152** |
| RNG-neutral Fake consumers | **160/160** |
| Logged losses | finite |
| CUDA/cgroup OOM | no observed CUDA OOM; **0** cgroup oom/oom_kill delta |
| Owned worker/GPU/port/lock teardown | **PASS**, both nodes |

The observer hashes the actual initial latents, renoise sigmas, and noise draws;
it does not draw replacement tensors. Action and native cycle default-Torch RNG
endpoints reconcile. Action scopes additionally observe Python random and loaded
NumPy legacy RNG states. Custom generators remain **unobserved**. This is not a
new cross-arm full-state parity result: the previous finite numerical drift
remains **REVIEW**, not a strict PASS.

## Diagnostic wall and capacity

| Cycle | Max rank-local CUDA-fenced host duration |
|---|---:|
| 1 | 773.958214879 s |
| 2 | 774.401585729 s |

**Both numbers include synchronizing correctness-only full draw hashes inside
the action scopes.** They are not admitted native timings, service costs or a
speedup denominator. No comparison against historical 772 s is made.
Cross-node clock alignment is not established.

- Max PyTorch allocated: **33.314137 GiB**; reserved: **38.365234 GiB**.
- Sampled node cgroup peaks: **452.542603 / 452.548363 GiB**. The kernel's
  `memory.peak` file is unavailable; these are sampled maxima, not exact peaks.
- Sampled minimum driver-free: **238 MiB / 526 MiB**. No-OOM completion does
  **not** establish a robust 2 GiB safety margin.
- The native per-rank memory boundary samples and process RSS/HWM remain in CSV.

Frozen workload: world16 FSDP2, shared H3 backbone, approximate Grid1000 sigma,
768×1344×124, joint video/audio, BF16, LoRA128, B1/globalB16, FA3, fixed exit3,
segment1, boundary activation policy **none**. FA3 output/LSE **cpu_staged**
storage remains enabled, depth2/N128 unchanged.

Rank0's independently logged FA3 cache reports 300 D2H entries, 300 H2D entries,
300 prefetches, zero prefetch misses, and 288 D2H backpressure waits **per cycle**.
Bytes in each direction are **164,794,448,000 B (cycle1)** and
**164,796,632,000 B (cycle2)**. These are rank0-only logged traffic counters;
do not extrapolate them as a measured all-rank byte total or exposed stall time.
No-boundary-offload does not mean no CPU transfer.

## Environment, test and failure evidence

- Both node checkouts were clean at runtime source `878aff4`; the existing
  MiniMax Python environment was reused (Torch 2.10.0+cu128, Diffusers 0.40.dev,
  Transformers 4.57.6, PEFT 0.18.1). Assets and selected source/binary hashes are
  in the preflight/deployment receipts, not guessed from old defaults.
- Pre-run focused source/dispatch/contract tests: **257 passed**. Read-only H3
  integration audit findings were fixed before launch.
- Initial no-weight preparation rejected a CUDA context created by the pinned
  LightX scheduler's `get_device()` constructor. Its stack trace proved there
  were **zero allocated/peak CUDA tensor bytes**, unchanged RNG and no weights.
  The second, locked no-weight probe correctly distinguished context creation
  from model allocation and passed on both nodes. No shared environment or
  workload code was changed to evade this failed check. Both attempts remain.
- The native controller exited zero; all raw rank receipts were revalidated
  remotely using `summarize_native_two_cycle` and `validate_cycle_actions`.
  Independent post-run readback found no GPU apps, owned workers, listeners on
  port31487, or held node locks. No process was killed by the readback.

## What remains, without repeating this gate

1. Admit action-specific calibration with output storage/alias lifetimes;
   do **not** reuse these host envelopes as isolated costs.
2. Build the H3 measured component model and search legal retained windows.
3. Execute a held-out lowered plan without these value-hash probes and record
   independent unprofiled cycle2 feedback. Keep numerical REVIEW separate.

H3 is not yet a completed measured-cost/search/held-out loop. No new kernel,
topology, canonical winner, PR push or main merge is part of this unit.

Raw receipts and logs:
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/h3_actions_r1/`.
Local small evidence:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/artifacts/h3_actions_r1/`.
