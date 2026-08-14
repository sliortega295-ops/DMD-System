# GAN Role-Phase performance v2

## Scope

This version keeps the frozen algorithmic boundary:

- Phase A: 8 Generator ranks + 8 Score ranks, Generator global batch 16;
- one Generator update with official DMD + GAN-G;
- Phase B: 4 no-grad Generator producers + 12 Critic ranks, Critic global batch 12;
- five Critic updates with official denoise + GAN-D + R1;
- `G200/C1000/EMA800 -> G201/C1005/EMA801`;
- Q2 producer/consumer window and payload credit returned on send completion.

The changes are systems-only. They do not remove Score forwards, alter random
trace consumption, change loss weights, or hide reverse handoff time.

## Why v2 is needed

Issue #9 shows that the existing semantic timeline is not a utilization trace.
Representative Role-Phase ranks have only about 19–21% compute-kernel interval
union. More importantly, A→B and B→A together consume roughly 170 seconds,
and the producer records long Critic-commit waits.

The old executor constructs most future Phase-B models after Phase A has
completed. That turns target construction, host preparation, engine
initialization, optimizer restoration, process-group lifecycle and peer waits
into one opaque transition bubble.

## Implemented changes

### Forward targets are prepared before formal timing

`probe_gan_rolephase_world16_v2.py` preconstructs and pins all future Phase-B
targets on ranks 0–7 before the formal Full5 timer:

- ranks 0–3: future resident no-grad Generator;
- ranks 4–7: future Critic;
- ranks 8–15: reuse the resident Score model.

The timed A→B path fails if it attempts lazy forward-target construction.

### Host-memory admission after the first real cgroup OOM

The first real v2 correctness attempt reached the node0 cgroup limit during
concurrent cold target preparation: `memory.max=816043786240` bytes,
`memory.peak=816039686144` bytes, with local rank 6 killed by SIGKILL and
`oom_kill` increasing by one. No Generator, Critic, or EMA update executed.

This is handled as a lifecycle/admission problem, not by changing the workload.
The final target policy remains **complete persistent-pinned CPU targets before
the formal timer**. The cold preparation is now:

1. admitted in deterministic singleton local-rank waves on node0;
2. wrapped in `suspend_transformers_zero3_init()` so future CPU targets do not
   inherit an active ZeRO-3 construction context;
3. pageable→persistent-pinned as before;
4. followed immediately by GC and host allocator trim before the next target;
5. guarded by cgroup-v2 `memory.current`, `memory.max`, `oom`, and `oom_kill`.

Default admission keeps at least both 10% of the finite cgroup and 64 GiB of
headroom before starting another full 14B target:

```text
DISTILLSCHEDULE_ROLEPHASE_HOST_ADMISSION_MAX_USED_FRACTION=0.90
DISTILLSCHEDULE_ROLEPHASE_HOST_ADMISSION_MIN_HEADROOM_GIB=64
```

If the stable inventory itself cannot satisfy that reserve, the run fails
before the next target is constructed instead of allowing a kernel OOM kill.
The values may be changed only as an explicit host-capacity policy; changing
batch, topology, ZeRO stage, loss, trace, EMA, forward inventory, offload, or
turning off forward preconstruction is not an allowed recovery.

### Reverse target is a separate capacity canary

Preconstructing another 14B Generator on every Flex rank can substantially
increase host residency. Reverse preconstruction is implemented but disabled
by default:

```text
DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_REVERSE=0
```

A dedicated capacity run may set it to `1`. It uses the same serialized
cgroup admission and cannot enter formal timing until it passes no-OOM and
lifecycle gates under the same policy.

### Transition decomposition receipts

Every rank emits append-only JSONL subranges for host-memory admission, source
export/release, process-group lifecycle, target preparation/materialization,
engine initialization, optimizer restore, broadcast/reshard, barrier/peer
wait, Phase-B producer/Critic service and reverse handoff total.

These receipts are diagnostic. They do not replace unprofiled Full5 timing.

### Service-rate-driven layout analysis

`analyze_gan_rolephase_service_rates.py` reports producer time, Critic
compute+commit time, send/receive waits and commit backpressure. It estimates a
bounded set: 4G+12C/B12, 6G+10C/B30 and 8G+8C/B16. Different batch contracts
are explicitly non-comparable and require new preregistration before timing.

### Stage-3 guard fix

`validate_zero_engine()` binds ZeRO through the public engine API and never
reads the Stage-1/2-only `partition_gradients` field on Stage-3. This removes
the guard failure that previously stopped the ZeRO-3 canary before model work.

## Entry point

```bash
python scripts/distillschedule/run_gan_rolephase_world16_v2.py \
  --hosts <node0>,<node1> \
  --run-id <immutable-id> \
  --mode correctness \
  --expected-head <git-sha> \
  --source-identity-hash <sha256>
```

## Validation order

1. dependency-light and Gloo tests;
2. validate-only topology/input gate;
3. one real correctness/capacity Full5 with serialized host admission;
4. inspect per-wave cgroup and transition receipts;
5. unprofiled service-rate canary;
6. at least three paired unprofiled timing trials;
7. one representative Nsight run for explanation only.

No GPU result is claimed by the source change alone.
