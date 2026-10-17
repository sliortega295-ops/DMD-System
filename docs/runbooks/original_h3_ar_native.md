# W6 task-native planner workflow

Read `docs/contracts/original_h3_ar_native_v1.md` first. This adapter is on the
same `workloads native --task` interface as the original Wan task, not a new
script training stack. `workloads list` intentionally still lists only the five
canonical frozen profiles; use:

```bash
python scripts/distillctl.py workloads native list-adapters
python scripts/distillctl.py workloads native validate \
  --task configs/distillschedule/original_h3_ar/w6_reference.json
```

All evidence/output paths below must be outside the checkout, in the authorized
remote persistent research-artifact directory. The coordinator owns GPUs,
source identity, two-node launch, owner leases, rendezvous, cancellation and
external resource/cleanup receipts. Do not start from an old shell command.
Use one clean integration commit for calibration, saved plan and execution.

## 1. CPU/I/O admission, before reserving GPUs

Use the **accepted two-cycle** native manifest, not its one-cycle parent:

```bash
python scripts/distillctl.py workloads native prepare-assets \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --native-manifest /share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/h3_diffsynth_dmd2_world16_20261010/zero3_200s_r2/steady_two_cycle_20261012_r1/configs/world16_r1_manifest.json \
  --checkpoint-audit /REMOTE/RUN/artifacts/checkpoint-audit.json \
  --output /REMOTE/RUN/configs/assets.json
```

This reads the old manifest solely to locate byte-pinned native source/helpers,
cache and existing libraries. It **never executes its old worker or command**.
The first invocation hashes the checkpoint once. An existing unchanged audit
is reused without payload scans. `assets.json` and its audit must remain fixed
for the campaign. Startup GPU-info receipts must be fresh, owner-qualified and
match each node's exact CUDA UUID ordering. If hosts/devices changed, regenerate
those small native metadata receipts through the existing owner first.

Freeze a common `capacity.json`. The original scalar-v1 form has exactly
`gpu_limit_bytes`, `gpu_reserve_bytes`, `host_limit_bytes`, `host_reserve_bytes`.
For different node host limits use `schema_version` equal to
`distillschedule.original_h3_ar_capacity.v2`, shared `gpu_limit_bytes`,
`gpu_reserve_bytes`, `host_reserve_bytes`, and `host_limits` with exactly string
keys `"0"` and `"1"`. Limits are explicit integer bytes, never historical raw
memory.max or inferred device totals. See the capacity-v2 contract. Both implementations and selected/
manual runs use the same policy. Driver reserve is sampled at cycle boundaries;
external owner sampling is still required for stronger capacity claims.
On kernels without `memory.peak`, the host policy must use the exact observed
finite cgroup hard limit and maintained host reserve 0, with unchanged OOM
counters. A tighter declared host budget or positive maintained reserve cannot
be proved from sampled current values and is rejected, not silently relaxed.

## 2. Calibrate both native implementations

Run the same CLI below under the existing owned **two-node x eight torchrun**
for each implementation, serially on the allocation. `--execute` is mandatory.
Use `--implementation zero3-native` then `zero3-ap`, separate output roots.

```bash
python scripts/distillctl.py workloads native calibrate \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --assets /REMOTE/RUN/configs/assets.json \
  --capacity-policy /REMOTE/RUN/configs/capacity.json \
  --implementation zero3-ap --output /REMOTE/RUN/calibration-ap --execute
```

Set `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`. Before importing Torch set
`PYTORCH_CUDA_ALLOC_CONF` (and `PYTORCH_ALLOC_CONF` if supplied) to exactly:

- native: `expandable_segments:True`
- AP: `expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:4`

Keep the validated existing Torch2.10.0+cu128 software environment and original
native offline/USE_TF/USE_FLAX/NCCL configuration; runtime identity records
versions, attention, TF32 and relevant communication environment. No package
installation is authorized. Both ranks need explicit WORLD_SIZE16,
LOCAL_WORLD_SIZE8 and correct global/local rank. CUDA_VISIBLE_DEVICES must be
the eight UUIDs in that node's fresh startup receipt, in the same order.

## 3. Assemble -> common search -> save plan

For each arm provide all16 rank files explicitly, including rank000 through
rank015; an incomplete/mixed/edited cohort fails closed.

```bash
python scripts/distillctl.py workloads native assemble \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --rank-receipts /REMOTE/RUN/calibration-ap/rank_*.json \
  --output /REMOTE/RUN/assembled-ap
python scripts/distillctl.py workloads native plan \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --calibration /REMOTE/RUN/assembled-native/collection.json /REMOTE/RUN/assembled-ap/collection.json \
  --output /REMOTE/RUN/artifacts/selected-plan.json
```

The planner reads only source-qualified measured native observations. It may
select either implementation based on actual duration and capacity. It never
imports 255.670644 s as a current cost. Current source/receipt changes require
recalibration; do not edit hashes to make old evidence current.

## 4. Execute saved plan; compare to manual incumbent

The coordinator first reads the saved `selected.implementation` and sets its
matching allocator environment, then starts the same owned two-node worker:

```bash
python scripts/distillctl.py workloads native run \
  --task configs/distillschedule/original_h3_ar/w6_reference.json \
  --assets /REMOTE/RUN/configs/assets.json \
  --capacity-policy /REMOTE/RUN/configs/capacity.json \
  --plan /REMOTE/RUN/artifacts/selected-plan.json \
  --output /REMOTE/RUN/selected-execution --execute
```

For the paired manual AP arm use `benchmark-manual --implementation zero3-ap`
with the identical task/assets/source/capacity/environment and two-cycle
boundary. That arm is not planner timing. Validate all16 real receipts, native
calls/versions/Adam/cache epochs/backend witnesses, owner resource/OOM/cleanup
and matched rank inputs/state/numerics. Only then compare actual max16 Cycle2
seconds. A successful CPU test, search, exit0, or historical receipt does not
establish <=3% manual matching, formal parity or a fresh speedup.

Capacity note: the original 64 GiB host headroom check was **startup-only**; it
is independently retained. No sustained GPU reserve was found in the old
contract. This adapter supplies no policy defaults: do not silently use GAN's
1 GiB reserve or convert the entry 64 GiB guard to a maintained host reserve.
See the contract's capacity-provenance section before choosing byte values.
