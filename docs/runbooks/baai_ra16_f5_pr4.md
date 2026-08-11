# Runbook: RA16-F5-PR4 functional gate

Use this runbook only for the experiment defined by
[`../contracts/ra16_f5_pr4.md`](../contracts/ra16_f5_pr4.md). The legacy
all-40 baseline remains `RA16-F5-FS40` and uses its existing plan.

## Required preflight

1. Both hosts must execute commands normally after SSH authentication.
2. Audit GPU processes, cgroup `memory.events`, shared mounts and assets.
3. Verify clean, identical DMD-System and external Helios runtime HEADs.
4. Verify the external runtime includes the phase-resident no-grad scope and its
   CUDA tests pass.
5. Hold the existing node GPU locks through construction, execution and cleanup.

Do not launch while any item is unknown.

## Command

```bash
cd /share/project/shuchenweng/yongyan_liu/DMD-System
python scripts/distillschedule/run_routea_world16.py \
  --hosts baai-helios-node0,baai-helios-node1 \
  --run-id ra16_f5_pr4_YYYYMMDD_NNN \
  --plan configs/distillschedule/plans/helios_ra16_f5_pr4.json \
  --remote-root /share/project/shuchenweng/yongyan_liu/DMD-System \
  --site-env configs/distillschedule/server/baai_2x8_a100.env.example
```

The coordinator now passes the selected plan path to both workers and rejects an
ambiguous run ID for this contract. A successful return code is insufficient;
accept only the exact 16-rank trace/version/EMA/lifecycle/checkpoint evidence in
the frozen contract.
