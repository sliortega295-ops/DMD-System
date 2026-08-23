# Wan1.3B fair Resident/FSDP2 comparison and Resident optimization ladder

## Scope

This path supersedes the numerical interpretation of PR #21 without deleting
its diagnostic evidence. It keeps the exact B1/global-B16 G1/C5/EMA5 workload
and removes the blanket-BF16 LoRA compute confound.

The four endpoints are:

1. `resident`: full model residency, manual post-backward LoRA all-reduce;
2. `resident-ddp`: full residency, DDP gradient buckets overlapped with backward;
3. `resident-fast`: `resident-ddp` plus fused AdamW and foreach EMA;
4. `world16-fsdp`: full-shard FSDP2 with original per-parameter dtypes.

All endpoints use BF16 frozen base parameters, FP32 LoRA parameters, FP32
gradient reduction, SUM followed by division by 16, the same AdamW
hyperparameters, and no offload/streaming/model swap.

## Required order

Do not launch timing first.

### 1. Validate

Run all four placements with the central controller:

```bash
python scripts/distillschedule/run_wan13b_fair_world16.py \
  --run-id <RUN_ID> \
  --mode validate \
  --placement <PLACEMENT> \
  --expected-head <HEAD> \
  --model-root <WAN13B_MODEL> \
  --bootstrap <BOOTSTRAP> \
  --bootstrap-hash <SHA256>
```

### 2. Correctness

Run all four placements with the same controller and `--mode correctness`. The controller invokes the following fail-closed summarizer:

```bash
python scripts/distillschedule/summarize_wan13b_fair_world16.py \
  --run-root <RUN_ROOT> \
  --placement <PLACEMENT> \
  --mode correctness \
  --output <PLACEMENT>_summary.json
```

The summary requires runtime LoRA dtype records from blocks 0, 15, and 29.

### 3. Numerical comparisons

Fair placement comparison:

```bash
python scripts/distillschedule/compare_wan13b_fair_correctness.py \
  --reference-summary resident_summary.json \
  --candidate-summary world16-fsdp_summary.json \
  --reference-placement resident \
  --candidate-placement world16-fsdp \
  --output resident_vs_fsdp2_correctness.json
```

Resident optimization gates:

```bash
python scripts/distillschedule/compare_wan13b_fair_correctness.py \
  --reference-summary resident_summary.json \
  --candidate-summary resident-ddp_summary.json \
  --reference-placement resident \
  --candidate-placement resident-ddp \
  --output resident_vs_ddp_correctness.json

python scripts/distillschedule/compare_wan13b_fair_correctness.py \
  --reference-summary resident_summary.json \
  --candidate-summary resident-fast_summary.json \
  --reference-placement resident \
  --candidate-placement resident-fast \
  --output resident_vs_fast_correctness.json
```

No tolerance may be changed. A candidate with `status != PASS` is diagnostic
only and cannot enter timing.

### 4. Timing

Only after the corresponding comparator is PASS, set:

```bash
export DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT=<comparison.json>
export DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_SHA256=<sha256>
```

Run three unprofiled pairs with alternating order through the pair driver:

```bash
python scripts/distillschedule/run_wan13b_fair_paired.py \
  --run-prefix <PREFIX> \
  --reference-placement resident \
  --candidate-placement <resident-ddp|resident-fast|world16-fsdp> \
  --expected-head <HEAD> \
  --model-root <WAN13B_MODEL> \
  --bootstrap <BOOTSTRAP> \
  --bootstrap-hash <SHA256> \
  --correctness-receipt <PAIR_CORRECTNESS_JSON> \
  --remote-root <DMD_SYSTEM_ROOT> \
  --master-addr <NODE0_IP> \
  --output <PAIRED_RESULT_JSON>
```

The driver performs one excluded warmup per arm and three alternating unprofiled
pairs. Profiled/Nsight wall is never used as a speed denominator.

## Per-node launch

Node 0:

```bash
bash scripts/distillschedule/launch_wan13b_fair_world16_node.sh \
  0 <RUN_ID> <MODE> <HEAD> <NODE0_IP> <PORT> <PLACEMENT>
```

Node 1:

```bash
bash scripts/distillschedule/launch_wan13b_fair_world16_node.sh \
  1 <RUN_ID> <MODE> <HEAD> <NODE0_IP> <PORT> <PLACEMENT>
```

Use a new immutable run ID for every arm and trial.

## Interpretation

`resident` vs `world16-fsdp` answers the fair placement question.

`resident` vs `resident-ddp` isolates the value of bucket views and
communication/backward overlap while keeping full residency.

`resident` vs `resident-fast` measures the complete optimized Resident
candidate. Fused AdamW and foreach EMA keep the same AdamW/EMA recurrences and
hyperparameters but may change floating-point order, so they remain subject to
the unchanged numerical thresholds.

The physical Adam/EMA work is intentionally not equalized: full replicated
state in Resident and local-shard state in FSDP2 are consequences of the
placement being measured. The logical optimizer recurrence, hyperparameters,
update counts, and model workload remain fixed.
