# Performance

## A100: six DMD workloads

16 A100-40GB GPUs, global batch 16. Times are complete update cycles;
Cycle 1 warms up the process and Cycle 2 is the reported comparison window.

| Workload | FSDP2 baseline / reference (s) | DMD-System (s) | Ratio |
| --- | ---: | ---: | ---: |
| W1 Wan1.3B DMD | 36.175 | 20.588 | 1.757× |
| W2 Helios No-GAN | 109.170 | 82.990 | 1.315× |
| W3 Helios GAN | 164.300 | 116.254 | 1.413× |
| W4 Self-Forcing DMD / LiveWan | 84.260 | 42.450 | 1.985× |
| W5 Audio–Video DMD, full-sequence Grid1000 | 911.770 | 767.020 | 1.189× |
| W6 Autoregressive Audio–Video DMD | 320.000 | 255.671 | 1.252× |

W1–W5 use G1/F5; W6 uses eight rollout steps and G1/F1. W5 uses
Grid1000; W6 uses continuous sigma. W3 uses the adopted unmerged-LoRA
configuration. W6's denominator is the reported FSDP2 reference.

The [six-workload data](https://github.com/sliortega295-ops/DMD-System/blob/main/artifacts/publications/20261019/results_correction/A100_user_reported_main_table.csv)
also includes Resident/offload and static-layout baselines.
The [W3 retiming record](https://github.com/sliortega295-ops/DMD-System/blob/main/artifacts/publications/20261019/results_correction/evidence/gan_nomerge/gan_nomerge_retime.json)
provides the unmerged-LoRA result used above. Launch with the
[native presets](native_presets.md) and the corresponding recipe.

## Official DMD2 and Self-Forcing

16 A100-40GB GPUs, global batch 16; unprofiled Cycle 2 (s).

| Workflow | FSDP2 baseline (s) | DMD-System (s) | Speedup |
| --- | ---: | ---: | ---: |
| SD1.5 DMD2 GAN, 512 × 512, G1/C10 | 7.514 | 3.475 | 2.162× |
| Official Self-Forcing Wan, 81 frames, 368 × 640, G1/C5 | 50.739 | 30.959 | 1.639× |

Both workflows completed 100 cycles, cycle-50 save and fresh-process
continuation, Student export and new-input generation. See
[configuration and commands](official_optimized_training.md) and
[run records](../status/official_optimized_framework.md).

## H100

Eight H100-80GB GPUs, logical global batch 16. Each row retains its own
reference and timing window; these are not hardware-normalized A100 comparisons.

| Workload | Reference | Reference (s) | DMD-System (s) | Ratio | Window |
| --- | --- | ---: | ---: | ---: | --- |
| W1 | Resident | 52.067 | 37.670 | 1.382× | C2 |
| W2 | Retained-G control | 95.535 | 94.311 | 1.013× | G1/C5/EMA5 Full5 |
| W3 | Whole-role swap, GPU ZeRO1 | 192.986 | 117.344 | 1.645× | G1/C5/EMA1 Full5 |
| W4 | Resident | 45.361 | 45.257 | 1.002× | C2 |
| W5 | Grid1000 FSDP | 638.522 | 539.718 | 1.183× | C2 |
| W6 | Native FSDP | 320.666 | 278.886 | 1.150× | C2, FA2 |

[Measurements, source revisions and contracts](https://github.com/sliortega295-ops/DMD-System/blob/main/artifacts/publications/20261019/h100/measured_results.csv).

## Planning and reproduction

[Manual versus saved-plan measurements](https://github.com/sliortega295-ops/DMD-System/blob/main/artifacts/publications/20261019/evidence/planner/six_workload_native_match.json)
record actual native execution, separately from end-to-end optimization gains.
Use an explicit strategy for direct training, or calibrate and search for your
hardware and inputs through the [planning workflow](../runbooks/native_planner_workflow.md).

Reuse existing training and recovery evidence for unchanged execution paths.
For new performance work, use the [harness timing protocol](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/experiment-protocol.md#cycle-2-timing).
Keep hot-cycle, startup and complete-job timing as separate fields.
