# Wan1.3B unprofiled Resident vs world16 FSDP2 diagnostic

This compact bundle publishes Iterations 321--322, the earlier **unprofiled**
diagnostic that preceded the matched Nsight experiment.

## Frozen workload

Both arms used real Wan2.1-T2V-1.3B, world16, B1/rank/global B16, Score packing
off, G1/C5/EMA5, final G3/C15/EMA15, 80 unique rank-step rollout identities,
GPU residuals/EMA/optimizer, and no parameter or activation offload, streaming,
or model swap. Source HEAD was
`a2cdefbebc90adca97fc3a9b4b90d26f0b1b5841`.

## Diagnostic results

| Arm | World-max unprofiled window | Max allocated | Max reserved | Minimum driver free |
|---|---:|---:|---:|---:|
| Resident | 55.868545 s | 18.434 GiB | 19.332 GiB | 19.245 GiB |
| world16 FSDP2 | 66.762061 s | 3.533 GiB | 6.014 GiB | 32.555 GiB |

The raw single-run ratio is 1.194985x: FSDP2 was 19.50% slower in these runs.
A Resident self-repeat took 57.595814 s.

## Correctness and claim boundary

Each arm individually completed 16/16 ranks with exact workload, identity,
update-count, version, queue and OOM gates. The cross-placement numerical gate
failed, and the same-layout Resident repeat also exceeded the absolute P0
rollout threshold. Cross-placement trainable and optimizer drift was materially
larger than same-layout drift.

Therefore:

- `formal_eligibility = false`;
- the three-pair formal driver was not run;
- these times are diagnostic performance clues, not a formal speedup;
- no tolerance was relaxed.

See [`correctness/report.md`](correctness/report.md) for the full comparison and
[`runs/`](runs/) for the two original summary receipts.
