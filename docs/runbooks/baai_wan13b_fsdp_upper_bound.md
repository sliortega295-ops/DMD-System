# BAAI Wan1.3B resident-vs-FSDP2 upper-bound experiment

This experiment uses the real `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
Transformer at revision `0fad780a534b6463e45facd96134c9f345acfa5b`.
It is a systems-only No-GAN Step10–14 workload, not a trained Wan continuation
and not a linear Helios14B extrapolation.

## Frozen comparison

Both arms are world16/B16, rollout B1, Score packing off, G1/C5/EMA5, LoRA
rank256, GPU EMA/residuals/optimizer, and no offload, streaming, or model swap.

- `resident`: both full frozen backbones stay on every GPU; only present
  trainable LoRA gradients are mean-all-reduced over world16.
- `world16-fsdp`: all 30 real Wan blocks and the root use composable FSDP2
  FULL_SHARD with `reshard_after_forward=True`.

## Assets and immutable bootstrap

Run once on node0 after syncing a clean experiment HEAD:

```bash
export DMD_SYSTEM_ROOT=/share/project/shuchenweng/yongyan_liu/DMD-System_wan13b_upper_bound_20260820
export HELIOS_RUNTIME_ROOT=/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812

bash "$DMD_SYSTEM_ROOT/scripts/distillschedule/prepare_wan13b_upper_bound_assets.sh" \
  <EXACT_HEAD> \
  /share/project/shuchenweng/DistillSchedule_assets/wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a \
  /share/project/shuchenweng/DistillSchedule_assets/wan13b/systems_only_bootstrap_v1
```

The bootstrap contains two deterministic FP32 LoRA inventories. Adam moments
are defined as zero and explicitly materialized before every run's timer; EMA
starts from the Generator trainables at logical version 10.

## Execution gates

For each placement run validate, then correctness:

```bash
python scripts/distillschedule/run_wan13b_upper_bound_world16.py \
  --run-id <UNIQUE_ID> --mode validate \
  --placement resident \
  --expected-head <EXACT_HEAD> \
  --master-addr <NODE0_IPV4> \
  --model-root <MODEL_ROOT> \
  --bootstrap <BOOTSTRAP_ROOT> \
  --bootstrap-hash <BOOTSTRAP_HASH>
```

Repeat with `--mode correctness`, then with placement `world16-fsdp`. Run the
cross-placement comparator on node0. Timing is forbidden until that receipt is
PASS.

## Formal timing

`run_wan13b_paired_formal.py` performs one excluded fresh-process warmup per
placement followed by three crossover pairs. It reports resident/FSDP raw
times, medians, CV, absolute FSDP cost, overhead fraction, and efficiency
relative to the resident proxy. Model load and optimizer/EMA materialization
remain outside Full5.

## Failure boundary

Do not lower resolution, batch, LoRA rank, call inventory, or update count.
Any trace/version/parity/OOM/cgroup failure makes the candidate ineligible.
The historical Helios14B timing may be shown as context only.
