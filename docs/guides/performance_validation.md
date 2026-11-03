# Performance

## A100: six DMD workloads

16 A100-40GB GPUs, global batch 16. Times are complete update cycles;
Cycle 1 warms up the process and Cycle 2 is the reported comparison window.
Each table retains its baseline definition and uses the adopted DMD-System result.

### FSDP2 baseline

| Workload | FSDP2 baseline / reference (s) | DMD-System (s) | Ratio |
| --- | ---: | ---: | ---: |
| W1 Wan1.3B DMD | 36.175 | 20.588 | 1.757× |
| W2 Helios No-GAN | 109.170 | 82.990 | 1.315× |
| W3 Helios GAN | 164.300 | 116.254 | 1.413× |
| W4 Self-Forcing DMD / LiveWan | 84.260 | 42.450 | 1.985× |
| W5 Audio–Video DMD, full-sequence | 911.770 | 767.020 | 1.189× |
| W6 Autoregressive Audio–Video DMD | 320.000 | 255.671 | 1.252× |

W1–W5 use G1/F5; W6 uses eight rollout steps and G1/F1. Current H3
recipes use continuous sigma; W5 retains its previously reported cycle timings. W3 uses unmerged LoRA. W6's denominator
is the reported FSDP2 reference.

### Resident / offload baseline

| Workload | Resident / offload baseline (s) | DMD-System (s) | Ratio |
| --- | ---: | ---: | ---: |
| W1 Wan1.3B DMD | 35.050 | 20.588 | 1.702× |
| W2 Helios No-GAN | 688.730 | 82.990 | 8.299× |
| W3 Helios GAN | 762.440 | 116.254 | 6.558× |
| W4 Self-Forcing DMD / LiveWan | 69.250 | 42.450 | 1.631× |
| W5 Audio–Video DMD, full-sequence | — | 767.020 | — |
| W6 Autoregressive Audio–Video DMD | — | 255.671 | — |

W2 and W3 use the role-swap/offload reference; W1 and W4 use their resident
reference. A dash means no measurement is recorded for that baseline.

### RealHF-like baseline

| Workload | RealHF-like baseline (s) | DMD-System (s) | Ratio |
| --- | ---: | ---: | ---: |
| W1 Wan1.3B DMD | 39.430 | 20.588 | 1.915× |
| W2 Helios No-GAN | 156.250 | 82.990 | 1.883× |
| W3 Helios GAN | 203.740 | 116.254 | 1.753× |
| W4 Self-Forcing DMD / LiveWan | 62.380 | 42.450 | 1.469× |
| W5 Audio–Video DMD, full-sequence | — | 767.020 | — |
| W6 Autoregressive Audio–Video DMD | — | 255.671 | — |

RealHF-like identifies the static-layout comparison, not a run of the ReaLHF
software. [A100 result data](../../artifacts/results/a100.csv) contains all three
references. [W3 measurement](../../artifacts/results/gan_nomerge_retime.json)
binds the unmerged result to its source and execution counters.

Launch through the [native presets](native_workloads.md#native_presets) and the
corresponding recipe.

## Official DMD2 and Self-Forcing

16 A100-40GB GPUs, global batch 16; unprofiled Cycle 2 (s).

| Workflow | FSDP2 baseline (s) | DMD-System (s) | Speedup |
| --- | ---: | ---: | ---: |
| SD1.5 DMD2 GAN, 512 × 512, G1/C10 | 7.514 | 3.475 | 2.162× |
| Official Self-Forcing Wan, 81 frames, 368 × 640, G1/C5 | 50.739 | 30.959 | 1.639× |

Both workflows completed 100 cycles, cycle-50 saving and fresh-process
continuation, Student export and new-input generation.
See the [result data](../../artifacts/results/official.csv)
and [training commands](native_workloads.md#official_optimized_training).

## H100

Eight H100-80GB GPUs, logical global batch 16. Each row retains its own
reference and timing window; these are not hardware-normalized A100 comparisons.

| Workload | Reference | Reference (s) | DMD-System (s) | Ratio | Window |
| --- | --- | ---: | ---: | ---: | --- |
| W1 | Resident | 52.067 | 37.670 | 1.382× | C2 |
| W2 | Retained-G control | 95.535 | 94.311 | 1.013× | G1/C5/EMA5 Full5 |
| W3 | Whole-role swap, GPU ZeRO1 | 192.986 | 117.344 | 1.645× | G1/C5/EMA1 Full5 |
| W4 | Resident | 45.361 | 45.257 | 1.002× | C2 |
| W5 | Continuous FSDP2 | — | — | — | C2 |
| W6 | Native FSDP | 320.666 | 278.886 | 1.150× | C2, FA2 |

[Measurements and revisions](../../artifacts/results/h100.csv) ·
[source and timing contracts](../../artifacts/results/h100_sources.json).

## Planning and reproduction

[Manual versus saved-plan measurements](../../artifacts/results/planner.json)
record native plan execution separately from end-to-end optimization gains.
Use an explicit strategy for direct training, or calibrate and search for your
hardware and inputs through the [planning workflow](../runbooks/native_planner_workflow.md).

The [result manifest](../../artifacts/results/manifest.json) binds each data file
to its checksum and measurement sources. Reuse training and recovery evidence
for unchanged execution paths. For new measurements, use the
[harness timing protocol](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/experiment-protocol.md#cycle-2-timing).
Keep hot-cycle, startup and complete-job timing as separate fields.
