# Iteration 336 post-fix memcpy attribution

## Identity and validity

- Run: `iter336_profile_resident_gpu_residual_20260823`
- DMD-System commit: `879816ad0a36ef718ae67dbf2c1159820bd29f4e`
- Helios transformer SHA256: `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`
- Nsight Systems: 2024.6.2.225 on both nodes
- Workload: world16/global-B16, resident-clean, B1, Score packing off, G1/C5/EMA5, final G3/C15/EMA15
- 16/16 range and workload receipts: PASS
- Capture/export logs: no `ProcessEventsError`, runtime API, bucket, string, or report-processing errors
- Exact range: 50.267583 s on node0/device0; profile wall is diagnostic-only

## Exact memcpy result

| Direction | Before events | After events | Before bytes | After bytes | Before union | After union | Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2H | 997 | 37 | 25.316620 GiB | 0.000000034 GiB | 8.029461 s | 0.000086 s | 99.999999864% bytes |
| H2D | 9,064 | 244 | 12.755536 GiB | 0.097226173 GiB | 1.032566 s | 0.007882 s | 99.237773% bytes |

Across all 16 GPUs in the post-fix exact range:

- D2H: **688 events / 688 bytes total**; per-GPU union median 0.111 ms.
- H2D: **3,904 events / 1.555620 GiB total**; per-GPU union median 8186.015 us.
- Old residual signatures (33,914,880 / 14,008,320 / 9,031,680 bytes): **0 events and 0 bytes on all 16 GPUs**.
- Range-crossing memcpy events: 0.

The remaining D2H is only one-byte control/check events (37 on representative GPU0; 688 bytes world total). Therefore the old blockwise residual D2H path is physically gone, not merely hidden. Remaining H2D has none of the old residual/reload signatures and is not classified as residual traffic.

## Representative GPU0 timeline change

| Category | Before | After | Delta |
|---|---:|---:|---:|
| Profile exact wall | 70.505866 s | 50.267583 s | -20.238283 s |
| non-NCCL kernel union | 25.222761 s | 25.122071 s | -0.100690 s |
| NCCL all-reduce union | 4.806667 s | 2.197168 s | -2.609499 s |
| D2H union | 8.029461 s | 0.000086 s | -8.029375 s |
| H2D union | 1.032566 s | 0.007882 s | -1.024684 s |
| D2D union | 0.485375 s | 0.481767 s | -0.003608 s |
| no traced GPU activity | 30.929036 s | 22.458610 s | -8.470426 s |

The after categories reconcile to 50.267582891 s versus a 50.267582891 s range (residual 0.000000000 s). Compute time is essentially unchanged. The diagnostic profile shrinks mainly through removal of D2H/H2D, an 8.470 s reduction in untraced gaps, and a 2.609 s reduction in the same 68 serialized NCCL all-reduce kernels. The latter two are observed changes, but the exact causal split is not proven by memcpy activity alone.

## Native timing boundary

- Pre-fix unprofiled diagnostic: 56.187420 s
- Post-fix unprofiled diagnostic: 35.757181 s
- Bounded single-run reduction: 20.430239 s (36.36%), equivalent to 1.571x

These are unprofiled diagnostics, not a three-pair formal speedup. Profiled wall is never used as the speed denominator.
