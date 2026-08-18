# BAAI matched MiniMax-H3: LightX2V native vs DistillSchedule

This is the claimable MiniMax runtime comparison. Both modes use the same pinned
`ModelTC/LightX2V` source, MiniMax-H3 weights, prompt cache, FSDP2 world16 mesh,
LoRA ranks, DMD/Fake losses, optimizer configuration, four-evaluation rollout,
and `1 Student + 5 Fake` update contract. Only the physical execution order is
changed.

## Fairness contract

Per rank, each cycle must record exactly:

```text
Student DiT forwards          24
Fake DiT forwards              6
Teacher DiT forwards           1
Total application forwards    31

Student grad-enabled forwards  1
Fake grad-enabled forwards     5
Teacher grad-enabled forwards  0

Student backward graphs        1
Fake backward graphs           5
Total backward graphs          6
```

Gradient-checkpoint recompute happens inside both modes and is not counted as a
new application-level DiT request. Both modes enable the same upstream gradient
checkpointing policy, so this internal recompute policy is matched as well.

The pair comparator additionally requires exact equality of:

- the six rank-qualified sample identities;
- model/config/prompt-cache/seed identities;
- final local Student/Fake LoRA parameter shard hashes;
- final local Student/Fake Adam state shard hashes.

If any gate differs, the pair is emitted as `NOT_COMPARABLE` and no speedup is
reported.

## The only scheduling change

LightX2V native order is:

```text
Student update
Fake-1: Student no-grad rollout -> Fake update
Fake-2: Student no-grad rollout -> Fake update
...
Fake-5: Student no-grad rollout -> Fake update
```

DistillSchedule uses the dependency fact that Student is fixed after the one
Student update and none of the five Student no-grad rollouts reads Fake state:

```text
Student update
Student no-grad rollout for Fake-1
Student no-grad rollout for Fake-2
...
Student no-grad rollout for Fake-5
Fake update 1
Fake update 2
...
Fake update 5
```

The first candidate deliberately does **not** batch multiple prompts into one
DiT call. Forward/backward counts therefore remain exactly unchanged. This
isolates phase clustering itself before testing a later B2/B5 packed candidate.

Because future-stage random inputs would otherwise be sensitive to reordering,
the candidate checks that each Fake forward/backward/update is globally RNG
neutral. If that canary fails, the candidate aborts rather than claiming exact
input replay.

## Prepare the environment once

Required on both nodes:

1. the exact DMD-System branch/commit with the pinned LightX2V submodule;
2. a Python environment that can run LightX2V MiniMax-H3 FSDP2 on A100;
3. the converted MiniMax-H3 model directory;
4. one prompt/condition cache with **at least 96 samples**.

The same Python environment, model path and cache are used by both modes. The
matched path runs directly from the pristine pinned LightX2V submodule; it does
not require a second model checkout.

## One-command acceptance pair

From the controller host:

```bash
python scripts/distillschedule/run_lightx2v_minimax_matched_pair.py \
  --run-id minimax_matched_p1 \
  --expected-head <FULL_BRANCH_SHA> \
  --model-path <MINIMAX_H3_PATH> \
  --prompt-cache <MINIMAX_PROMPT_CACHE> \
  --python <LIGHTX2V_PYTHON>
```

The pair runner uses the same 16 GPUs sequentially. It never launches native
and DistillSchedule concurrently, because concurrent execution would introduce
GPU contention and invalidate the timing comparison.

The output `pair.json` contains:

```text
native_seconds
distillschedule_seconds
speedup = native / distillschedule
wall_reduction_fraction
equivalence_gates
HBM peaks
```

`speedup` is populated only if all equivalence gates pass.

## Formal timing

Use at least three pairs and alternate order to reduce order/thermal bias:

```text
pair1: native -> distillschedule
pair2: distillschedule -> native
pair3: native -> distillschedule
```

Use `--order distillschedule-first` for pair2. Keep the exact same branch SHA,
model, prompt cache, attention backend and seed for all formal pairs.

Do not run Nsight during formal timing. After the winner is established with
unprofiled paired runs, profile one clean native cycle and one clean candidate
cycle separately to explain the difference.
