# Wan1.3B world16 resident Nsight attribution

Status: **PASS** (diagnostic only).

Denominator: summed 16-rank GPU phase walls = 1130.406 GPU-s. Profile wall is not a formal latency result.

| Category | Union GPU-s | Hidden by compute GPU-s | Exposed vs compute GPU-s | % GPU-wall |
|---|---:|---:|---:|---:|
| Compute | 402.740 | -- | -- | 35.63% |
| NCCL | 101.377 | 0.000 | 101.377 | 8.97% |
| Memcpy | 152.419 | 0.000 | 152.419 | 13.48% |
| No traced GPU activity | 473.870 | -- | -- | 41.92% |

`NCCL exposed vs compute` is an overlap-aware upper bound, not automatically removable wall time. `No traced activity` is not automatically CPU idle.
