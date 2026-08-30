# Iteration 386: C27 frozen cross-attention LayerNorm canary

## Pause-point summary

The accepted formal GAN-on denominator remains C25:

| Point | Full5 seconds | vs C0 | Status |
|---|---:|---:|---|
| C0 fresh CUDA-resident baseline | `173.014483004` | -- | PASS |
| C25 exact history/current fusion | `125.258735931` | `-47.755747073 s`, `-27.6022%`, `1.38126x` | PASS / current winner |
| Registered 30%-reduction target | `121.110138103` | `-30%` | not reached; gap `4.148597828 s` |

C27 is only a successful one-A100 canary.  It has **not** been integrated into
world16 and therefore does not change the formal `125.258735931 s` result.

## Source-qualified hypothesis

The pinned Helios `_ChunkedFrozenNorm` path owns the frozen affine
cross-attention `norm2`.  For every tile it currently performs:

```text
BF16 hidden
  -> explicit FP32 tensor
  -> patched FP32 Flash LayerNorm output
  -> BF16 cast
  -> copy into the preallocated output
```

Backward reconstructs the same graph with native autograd per tile.  C27 keeps
the width-5120 FP32 LayerNorm/affine formulas and frozen weight/bias semantics,
but directly reads BF16 and writes BF16 in a forward kernel and a hidden-only
backward kernel.

The selection evidence is C22 device 0: 8,760 `layernorm_forward` launches
totaling `0.520950048 GPU-s` and 1,800 `layernorm_backward` launches totaling
`0.171421406 GPU-s`.  These are kernel sums, not removable wall-clock time.

## Canary evidence

Exact code HEAD: `00ca399e109a5b1f6468dcbb3fc567417f26030d`.

| Batch | Source fwd+bwd | C27 fwd+bwd | Speedup | Output P0 | Hidden-grad P0 |
|---:|---:|---:|---:|---|---|
| B1 | `0.722380781 ms` | `0.242636800 ms` | `2.97721x` | PASS | PASS |
| B3 | `1.881651115 ms` | `0.273561597 ms` | `6.87835x` | PASS | PASS |

The output and hidden gradient are not bitwise identical, but pass the
unchanged `DEFAULT_P0_PARITY` thresholds by a wide margin.  The preregistered
gate was the existing multi-metric P0 contract, not bitwise equality.  Maximum
absolute differences are `0.015625` for output and `0.0078125/0.015625` for
B1/B3 hidden gradient; normalized-L2 is at most `1.2215e-5`.

The successful immutable run is:

```text
iter386_gan_cuda_c27_frozen_crossnorm_canary_r3_20260830
canary.json SHA-256:
2662e7497db269d4ff19868dd64f9cbb2937980b8acb50702e21f79d316cd1c6
kernel source SHA-256:
a743bb6631f0b5c42caa35be7d203383c16728f5f764558ecd91312b4e859eb8
```

Two earlier run IDs remain preserved:

- `r1`: launcher stopped before Python/CUDA because an unset `PYTHONPATH`
  expansion was fail-closed; no candidate verdict.
- `r2`: the full canary executed, then the result writer rejected a
  pre-created output directory; no result artifact and no candidate verdict.

`r3` has zero CUDA/cgroup OOM delta.  After stopping, both nodes have 8/8 idle
A100s, no compute processes, and free node locks.

## Decision and stop boundary

**CANARY_PASS / WORLD16_NOT_RUN / OPTIMIZATION_PAUSED_BY_USER.**

The next legal step, if resumed, is a default-off, source-pinned integration
with exact live call/chunk receipts followed by focused tests, validate-only,
16/16 correctness/capacity and only then one unprofiled Full5 timing.  A draft
integration patch was intentionally not pushed as accepted code; it is kept as
disposable local WIP under `/tmp/lyy-experiments/` so this pause leaves the
GitHub branch reconstructable and honest.

