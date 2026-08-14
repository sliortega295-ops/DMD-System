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

### Reverse target is a separate capacity canary

Preconstructing another 14B Generator on every Flex rank can substantially
increase host residency. Reverse preconstruction is implemented but disabled
by default:

```text
DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_REVERSE=0
```

A dedicated capacity run may set it to `1`. It cannot enter formal timing until
it passes no-OOM and lifecycle gates under the same policy.

### Transition decomposition receipts

Every rank emits append-only JSONL subranges for source export/release,
process-group lifecycle, target preparation/materialization, engine
initialization, optimizer restore, broadcast/reshard, barrier/peer wait,
Phase-B producer/Critic service and reverse handoff total.

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
3. one real correctness/capacity Full5;
4. inspect transition receipts;
5. unprofiled service-rate canary;
6. at least three paired unprofiled timing trials;
7. one representative Nsight run for explanation only.

No GPU result is claimed by the source change alone.
