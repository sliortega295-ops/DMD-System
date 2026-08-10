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

From the controller node:

```bash
python scripts/distillschedule/run_routea_gan_world16.py \
  --hosts <node0>,<node1> \
  --run-id gan-routea-dryrun-001 \
  --site-env /secure/path/baai_gan.env \
  --dry-run
```

A dry run only freezes the controller identity and prints the two remote commands. It does not construct a model.

## 3. Real functional baseline

Use a fresh immutable run ID:

```bash
python scripts/distillschedule/run_routea_gan_world16.py \
  --hosts <node0>,<node1> \
  --run-id gan-routea-functional-001 \
  --site-env /secure/path/baai_gan.env
```

Each host worker performs source/config/trace/checkpoint checks, builds a normalized two-node evidence contract, runs the world16 NCCL topology probe, and only then starts the official Helios training entrypoint.

The expected logical window is:

```text
1000: Generator+DMD+GAN-G → G update → Critic+GAN-D/R1/R2 → one C update → EMA
1001: Critic+GAN-D/R1/R2 → one C update → no EMA
1002: Critic+GAN-D/R1/R2 → one C update → no EMA
1003: Critic+GAN-D/R1/R2 → one C update → no EMA
1004: Critic+GAN-D/R1/R2 → one C update → no EMA
```

This follows the official post-`ema_start_step=750` DMD predicate: EMA is updated only when `TRAIN_GENERATOR` is true. A source-faithful `checkpoint-1000` therefore starts at EMA `optimization_step=800` and the five-step window ends at `801`.

## 4. Inspect the result

The curated result is written under:

```text
artifacts/manifests/<run-id>_result.json
```

`PASS` requires exact world16/trace/version/checkpoint/lifecycle/no-OOM evidence, including G200/C1000/EMA800 → G201/C1005/EMA801 and exactly one EMA event at step1000. A high cgroup peak without OOM is diagnostic only. The timing field from this first run is also diagnostic only.

After one functional PASS, run at least three low-overhead independent repetitions from the same canonical checkpoint/trace contract before freezing the formal GAN Route-A performance denominator.

## 5. Search the minimum frozen-block offload set

After the GAN assets and default baseline path are valid, the minimum-block search can be launched with one command:

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

The searched variable is deliberately simple and symmetric:

```text
k = streamed/offloaded frozen Transformer blocks per backbone
Generator blocks = Score blocks = [40-k, ..., 39]
```

Everything else remains frozen: official steps1000–1004, DMD+GAN branches, the official one-update EMA predicate, GAN-G/GAN-D/R1/R2 order, one joint Critic update per step, ZeRO-1 CPU optimizer offload, MB1/GAS1/global-batch16, 384×640, checkpoint/trace, activation policy, prefetch=0 and no pipeline overlap.

The driver first requires a fresh `k=40` PASS anchor under this GAN execution graph. It then tests `k=0`; when that is an actual capacity failure, it binary-searches the nested tail sets until it has adjacent fail/pass counts. Only strict `PASS` or actual CUDA/cgroup OOM may move a search bound. Infrastructure, evidence, trace, version or correctness failures abort the search instead of being interpreted as insufficient memory.

The selected `k*` is run once more under a fresh immutable run ID. The final result is written under:

```text
artifacts/g_offload_search/gan-<search-id>/result.json
```

Important interpretation: `k*` is **per backbone**. The symmetric baseline therefore streams `2 × k*` Transformer blocks in total across Generator and Score. These search runs are capacity/correctness evidence only and never enter the formal timing table. An asymmetric Generator-vs-Score search, if later needed, is a separate experiment.
