# Typed DistillSchedule plans

Start current training with the [recipes](../../recipes/README.md) and
`distillctl workloads list`. Native/FSDP2 plans retain their workload-specific
source, asset and numerical gates.

`fragments/` and `plans/` also support generic plan inspection. For example:

```bash
python scripts/distillctl.py validate configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json
python scripts/distillctl.py show configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json
```

This generic example is plan-only; it does not launch a model. Route-A/Route-B,
their legacy Helios launch adapter and manual tail-block/offload presets are
[retired](../../docs/architecture/legacy_route_retirement.md). They are not
fallbacks for current FSDP2 workloads.
