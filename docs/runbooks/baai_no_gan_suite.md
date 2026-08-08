# Runbook: staged BAAI No-GAN world16 suite

This suite is a fail-closed coordinator, not proof that every Route-B executor is
implemented. It reuses SHA-bound historical evidence, runs only selected gaps and
preserves downstream stages as `NOT_RUN` after a failed gate.

## Authoritative entrypoint

```bash
python scripts/distillschedule/run_no_gan_suite.py \
  --suite configs/distillschedule/suites/helios_no_gan_world16.yaml \
  --hosts <node0-ssh-target>,<node1-ssh-target> \
  --run-id <new-immutable-suite-id> \
  --repo-root "$PWD" \
  --resume
```

Use `--dry-run` first. Dry-run results are labelled `DRY_RUN` and are deliberately
ineligible for later `--resume` reuse. Real execution requires a clean repository.

## Frozen main contract

- world16, 2×8 A100-40GB, 384×640, MB1/GAS1/global batch 16;
- true 14B Generator and Score, GAN/Reward/VAE decode off;
- ZeRO-1 with CPU optimizer offload and no parameter offload;
- rank-qualified trace, canonical checkpoint and exact G/C/EMA version semantics;
- memory rejection only on actual CUDA OOM, cgroup OOM/OOM-kill or a
  memory-attributed nonzero exit.

## Stage boundary

The repository directly implements N0 preflight, N1 typed Route-A Attempt 004
coordination and the N2 bounded search controller. N2 first requires a fresh
same-contract conservative Generator-offload anchor; historical tail-1/3/5/all-40
artifacts are diagnostics only and are never search bounds.

The following real model executors are not yet built into this branch and must be
provided explicitly until their adapters are implemented:

```text
DISTILLSCHEDULE_GST_POOL_CANDIDATE_COMMAND
DISTILLSCHEDULE_GST_POOL_CONTINUOUS_COMMAND
DISTILLSCHEDULE_ROUTE_B_TRANSITION_COMMAND
DISTILLSCHEDULE_ROUTE_B_CONTINUOUS_COMMAND
DISTILLSCHEDULE_PAIRED_FORMAL_COMMAND
```

Leaving any required command unset produces `NOT_RUN` and blocks downstream stages;
it must never be interpreted as a successful one-command Full5 run. The optional
Nsight command is configured separately and may run only after formal timing.

## Evidence and recovery

Every stage receives a unique run ID and writes under
`artifacts/suites/<suite-id>/`. `--resume` reuses only a non-dry-run accepted result
whose stage-contract digest still matches. A failed immutable attempt is preserved;
`--force-rerun-stage` creates a new attempt ID rather than overwriting it.

No Route-A/Route-B speedup is claimable until continuous Route-B correctness and at
least three matched paired formal trials pass from the same logical contract.
