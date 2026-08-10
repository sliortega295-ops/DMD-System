# Runbook: BAAI Route-A GAN-on baseline

This runbook executes the frozen contract in [`../contracts/routea_gan_full5_world16.md`](../contracts/routea_gan_full5_world16.md). The default denominator remains sequential/full-offload. A separate capacity-search mode may vary only the frozen Generator/Score streaming set; it must not add pipeline overlap, prefetch, or change GAN math.

## 1. Prepare immutable GAN assets

Before reserving GPUs, prepare one source-faithful GAN continuation state:

- GAN Stage-3 data root;
- canonical `checkpoint-1000` produced under the GAN-on continuation contract;
- rank16 operation-trace bundle covering steps1000–1004 with the normal rank-synchronized DiT exit schedule.

Do **not** reuse the No-GAN checkpoint-10/trace10–14 and do not change `gan_start_step` to make an old artifact fit.

Copy the example site profile to an untracked path on both nodes and fill the GAN assets:

```bash
cp configs/distillschedule/server/baai_gan_2x8_a100.env.example /secure/path/baai_gan.env
```

Set and audit:

```bash
export HELIOS_GAN_DATA_ROOT=/path/to/gan/data
export DISTILLSCHEDULE_GAN_TRACE_BUNDLE=/path/to/rank16/trace1000_1004
export DISTILLSCHEDULE_GAN_CHECKPOINT=/path/to/checkpoint-1000
export DISTILLSCHEDULE_EXPECTED_HELIOS_HEAD="$(git -C "$HELIOS_RUNTIME_ROOT" rev-parse HEAD)"
export DISTILLSCHEDULE_EXPECTED_HELIOS_ENTRYPOINT_SHA256="$(sha256sum "$HELIOS_RUNTIME_ROOT/train_helios.py" | awk '{print $1}')"
```

The Helios checkout must be clean and must contain the official low-VRAM joint Critic-backward fix.

## 2. Structural dry run

```bash
python scripts/distillschedule/run_routea_gan_world16.py \
  --hosts <node0>,<node1> \
  --run-id gan-routea-dryrun-001 \
  --site-env /secure/path/baai_gan.env \
  --dry-run
```

## 3. Real functional baseline

```bash
python scripts/distillschedule/run_routea_gan_world16.py \
  --hosts <node0>,<node1> \
  --run-id gan-routea-functional-001 \
  --site-env /secure/path/baai_gan.env
```

The expected logical window is:

```text
1000: Generator+DMD+GAN-G → G update → Critic+GAN-D/R1/R2 → one C update → EMA
1001: Critic+GAN-D/R1/R2 → one C update → no EMA
1002: Critic+GAN-D/R1/R2 → one C update → no EMA
1003: Critic+GAN-D/R1/R2 → one C update → no EMA
1004: Critic+GAN-D/R1/R2 → one C update → no EMA
```

This follows the official post-`ema_start_step=750` DMD predicate. A source-faithful `checkpoint-1000` starts at EMA `optimization_step=800` and the five-step window ends at `801`.

## 4. Stochastic and DiT fairness gate

Do not use a shared seed as proof of fairness. The pipeline changes physical RNG call order by mapping one world16 logical batch to two eight-sample waves. Each real baseline/pipeline result must therefore point to two immutable artifacts:

```text
gan_fairness_receipt   # actual consumed source/history/random tensor hashes + stage exits
gan_compute_receipt    # exact rollout DiT counts + fixed Score/GAN call inventory
```

The formal pair checker requires identical logical slots0–15 after remapping:

```text
baseline: rank = slot
pipeline: wave=slot//8, G-rank=slot%8, Score-rank=8+(slot%8)
```

The consumed stochastic roles cover Generator rollout/re-noise/block noise, DMD timestep/noise, GAN-G noise/crop, Critic rollout/timestep/noise, GAN-D fake/real/R1 crops and R1 epsilon, plus the processed histories actually fed to DiT. R2 is not included because the frozen official contract has `r2_weight=0`.

The compute receipt additionally freezes fixed Score work per logical sample:

```text
step1000 Generator objective:
  1 Fake-Score conditional
  0 Fake-Score unconditional
  1 Real-Score conditional
  1 Real-Score unconditional
  1 GAN-G Score forward

all Critic steps:
  1 Critic denoise Score forward
  1 GAN-D Score forward packing fake + real + R1
```

Formal timing is rejected if either receipt differs or if the receipt rollout count disagrees with the recorded three-stage DiT exits.

## 5. Formal unprofiled paired timing

Use `scripts/distillschedule/run_gan_paired_formal.py`. It requires at least three pairs and alternates launch order. Only exact steps1000–1004 results with the same workload digest and matching fairness/compute receipts enter the speedup table. Any `profiled=true` or `diagnostic_only=true` result is rejected.

The speedup reported is the paired ratio:

```text
baseline 5-step wall time / pipeline 5-step wall time
```

Nsight time is never used for the formal speedup.

## 6. Nsight Systems diagnostic profile

Nsight is optional diagnostic attribution after correctness. Enable it on a fresh run by exporting:

```bash
export DISTILLSCHEDULE_GAN_NSYS_PROFILE=1
```

The launch shim starts one Nsight Systems session per node and traces the local Accelerate child processes. The exact capture range is synchronized across world16:

```text
START: all ranks reach step1000 before first model compute
       → node controllers rank0/rank8 call cudaProfilerStart
       → barrier releases model compute

STOP:  all ranks finish step1004 Critic commit
       → world16 barrier
       → rank0/rank8 call cudaProfilerStop
```

Every rank emits a range receipt; the profile is accepted only with 16 receipts, controller ranks `[0, 8]`, the exact `[1000,1004]` window, and non-empty CUDA/NVTX statistics. Nsight artifacts are explicitly `diagnostic_only=true` and `formal_speedup_eligible=false`.

## 7. Search the minimum frozen-block offload set

After the GAN assets and default baseline path are valid:

```bash
python scripts/distillschedule/run_gan_offload_search.py \
  --hosts <node0>,<node1> \
  --master-addr <node0-compute-network-ip> \
  --master-port <unused-port> \
  --search-id gan-minblock-001 \
  --site-env /secure/path/baai_gan.env \
  --repo-root "$PWD" \
  --resume
```

The searched variable is symmetric:

```text
k = streamed/offloaded frozen Transformer blocks per backbone
Generator blocks = Score blocks = [40-k, ..., 39]
```

Everything else remains frozen: official steps1000–1004, DMD+GAN, the one-update EMA predicate, GAN-G/GAN-D/R1 order, one joint Critic update per step, ZeRO-1 CPU optimizer offload, MB1/GAS1/global-batch16, checkpoint/trace, and activation policy. Search candidates are capacity/correctness evidence only and never enter the formal timing table.