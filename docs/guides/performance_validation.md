# Performance

Complete-cycle latency on 16 A100-40GB GPUs and 8 H100-80GB GPUs.

## Workloads

| ID | Model and method | Teacher → Student | Frames | Resolution | Student steps | Fake score : Student | Recipe |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| W1 | MiniMax H3 PDMD | 33B → 33B | 124 | 768×1344 | 4 | 5:1 | [H3 full-sequence](../../recipes/h3_fullseq/README.md) |
| W2 | MiniMax H3 DMD2 | 33B → 33B | 112 | 768×1344 | 8 | 1:1 | [H3 autoregressive](../../recipes/h3_ar/README.md) |
| W3 | Wan2.1 DMD | 1.3B → 1.3B | 33 | 384×640 | 6 | 5:1 | [Wan 1.3B](../../recipes/wan_ar/README.md) |
| W4 | Wan2.1 SF-DMD | 14B → 1.3B | 81 | 368×640 | 4 | 5:1 | [LiveWan](../../recipes/self_forcing/README.md) |
| W5 | Helios Hier-DMD | 14B → 14B | 33 | 384×640 | 6 | 5:1 | [Helios No-GAN](../../recipes/helios_nogan/README.md) |
| W6 | Helios Hier-DMD2 | 14B → 14B | 33 | 384×640 | 6 | 5:1 | [Helios GAN](../../recipes/helios_gan/README.md) |

## A100

Latency labels are rounded to seconds.

| Workload | FSDP2 / reference (s) | Resident / offload (s) | ReaLHF-like (s) | DiFlow (s) | Speedup vs. reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 912 | OOM | 1054 | 767 | 1.19× |
| W2 | 320 | 932 | 353 | 256 | 1.25× |
| W3 | 36 | 35 | 39 | 21 | 1.76× |
| W4 | 84 | 69 | 62 | 42 | 1.98× |
| W5 | 109 | 689 | 156 | 83 | 1.32× |
| W6 | 164 | 762 | 162 | 117 | 1.41× |

Stateless base → full system: W4, 84.26 → 42.45 s; W6, 164.30 → 116.81 s.
[A100 data](../../artifacts/results/a100.csv)

## H100

| Workload | FSDP2 / reference (s) | Resident / offload (s) | ReaLHF-like (s) | DiFlow (s) | Speedup vs. reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 639 | 637 | 597 | 540 | 1.18× |
| W2 | 321 | 309 | 303 | 279 | 1.15× |
| W3 | 57 | 52 | 43 | 38 | 1.51× |
| W4 | 55 | 45 | 53 | 45 | 1.22× |
| W5 | 116 | 138 | 110 | 94 | 1.23× |
| W6 | 169 | 193 | 175 | 117 | 1.44× |

[H100 data](../../artifacts/results/h100.csv)

## Planning and reproduction

Cycle times for manually tuned and selected plans:

| Workload | Manual (s) | Selected (s) | Difference |
| --- | ---: | ---: | ---: |
| W2 | 255.72 | 255.67 | −0.02% |
| W3 | 20.524 | 20.59 | +0.32% |
| W4 | 42.514 | 42.45 | −0.15% |
| W5 | 82.90 | 82.99 | +0.11% |
| W6 | 114.88 | 116.81 | +1.68% |

Approximate profiling times for W1–W6 are 236, 217, 78, 142,
132 and 137 seconds, respectively.
[Planner data](../../artifacts/results/planner.json) ·
[Result manifest](../../artifacts/results/manifest.json)

Use a [native preset](native_workloads.md#native_presets) for direct training,
or [calibrate and search](../runbooks/native_planner_workflow.md) for your
hardware and inputs.
