# Wan1.3B Resident modulation-chunk diagnostic (2026-08-24)

## Scope

This is a single-pair, diagnostic-only world16 experiment on 2×8 A100-40GB.
The frozen workload is No-GAN B1/global-B16, Score packing off, G1/C5/EMA5,
final G3/C15/EMA15, BF16 frozen base, FP32 LoRA, GPU-resident residuals, no
offload/streaming/model swap, and a 128 MiB gradient-ready reducer with
`NCCL_ALGO=Tree`.

The only measured variable is the token tile used by the already-existing
split-modulation/residual path:

- reference: 512 tokens;
- candidate: 2048 tokens.

Experiment source HEAD: `1fe2f0be33db83fdbea99bb2e786749adc776149`.

## Correctness and capacity

Both arms completed 16/16 ranks with 80 rank-step rollout identities,
G3/C15/EMA15, one Generator update, five Critic updates, and five EMA updates.
The candidate receipt records 30/30 Generator blocks and 30/30 Critic blocks at
2048 tokens. There was no CUDA or cgroup OOM.

Changing chunk size changes floating-point reduction ordering in the chunked
path, so this experiment does not claim bitwise equality. It preserves the
operation inventory and training contract; strict cross-arm numerical parity
was not used as a timing gate per the current diagnostic claim boundary.

## One matched unprofiled diagnostic pair

| Arm | Full5 wall | Peak allocated | Peak reserved | Minimum driver free |
|---|---:|---:|---:|---:|
| Tree + chunk512 | 34.473506 s | 21.8912 GiB | 22.3750 GiB | 16.2230 GiB |
| Tree + chunk2048 | 27.233468 s | 21.8867 GiB | 22.0156 GiB | 16.5824 GiB |

Observed reduction: `7.240038 s` (`21.0017%`), or `1.26585×` for this one
bounded pair. This reaches the requested ~28-second diagnostic target, but it is
not a three-pair formal performance result.

The large gain is consistent with reducing the number of small
elementwise/cast/norm launches in the 512-token loop. Iteration 343 had already
reduced representative-GPU exposed NCCL to about 1.37 seconds, so the 7.24-second
gain cannot be attributed to communication.

## Preserved failures and claim boundary

- Initial validate run: launcher-manifest variable expansion failed before any
  model/GPU workload started; fixed by commit `1fe2f0b` and rerun under a new ID.
- Validate R2: remote 16-rank validate passed, while the local forced-TTY fetch
  mixed a platform banner into JSON; the remote summary was refetched with a
  non-TTY SSH channel and verified.
- Reference diagnostic transport returned SSH 255 after all 16 receipts were
  written. No worker remained; the preserved receipts were summarized without
  rerunning the workload.
- Nsight and NCU were not run for chunk2048. No profiled wall is used here.

Persistent full evidence remains under
`/home/lyy/Helios/research_artifacts/DMD-System/wan13b_resident_chunk2048_20260824/`.
