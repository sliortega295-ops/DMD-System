# LiveWan continuous B16/B32 diagnostics (Iterations 143-174)

## Result

The persistent 8G + 8Score LiveWan cross-scale pipeline now reaches an actual
continuous B16 Full5 steady interval of `55.095211267 s`. Doubling the logical
global batch to B32 gives a real interval of `108.583634309 s` for twice the
work, or `54.291817154 s` per B16-equivalent 80 samples. This is a `1.48%`
throughput gain, not a 54-second B32 wall.

| Candidate | Runtime HEAD | Actual steady interval | Samples/s | B16-equivalent | Decision |
|---|---|---:|---:|---:|---|
| C21-0 continuous B16 | `e8283999274b0e6ba5a93cc322e79823a6b8cc80` | `55.095211267 s` | `1.452031822` | `55.095211267 s` | B16 winner |
| C21-C1 wave0 B1 overlap | `ad0b303356f05e5d1a812f01f1aacada06fb39e3` | `55.150482594 s` | - | `55.150482594 s` | reject, `+0.10%` |
| C22 checkpoint RNG elision | `e8283999274b0e6ba5a93cc322e79823a6b8cc80` | `55.153462575 s` | - | `55.153462575 s` | reject, `+0.11%` |
| C23 continuous B32 | `f8bbbac6c18872883dd7ed1bace9339784d8c51e` | `108.583634309 s` | `1.473518556` | `54.291817154 s` | throughput diagnostic |

Relative to C21 B16, C23 processes twice the samples with a normalized saving
of `0.803394113 s` per 80 samples (`1.458%`, `1.01480x`).

## Frozen systems contract

- 2 nodes x 8 A100-SXM4-40GB, world size 16.
- Persistent ranks 0-7 Generator and ranks 8-15 Score; no redistribution.
- Resident Wan1.3B Generator; resident Wan1.3B Critic and FSDP2 Wan14B Teacher.
- No model swap, parameter offload, block streaming, or role-pool transition.
- One Generator and five Critic commits per Full5; no EMA; final `G3/C15/EMA0`
  after the three-cycle continuous diagnostic.
- Same models, BF16/FP32 policy, loss, optimizer order, fixed trace semantics,
  continuous sigma, and logical DiT call inventory within each registered arm.

The B16 and B32 arms intentionally differ in logical global batch. B32 uses
four B1 waves per lane. Generator gradients are weighted by `0.25` and commit
once. Score executes two physical B2 Critic microbatches, each weighted by
`0.5`, accumulates both, and commits once per step.

## Execution sequence and outcomes

### C21-0: continuous B16 depth-one lookahead

The final timing run `iteration_170_c21_b16_finalhead_timing_20260901` passed
16/16 ranks with steady intervals:

```text
55.226561797 s
54.963860737 s
median 55.095211267 s
```

Peak allocated was `35.159714 GiB/rank`; minimum driver free was only
`0.242554 GiB/rank`. Generator backpressure and Score receive-wait medians were
`0.008812 s` and `0.000152 s`.

### C21-C1: overlap the first B1 Critic backward

C21-C1 begins the first B1 backward as soon as wave0 arrives, while the
Generator produces wave1. It still waits for both samples before the single
Critic commit. The candidate passed the inventory and version gates, but its
steady median was `55.150482594 s`, `0.055271327 s` slower than C21-0. It is
rejected rather than stacked into later candidates.

### C-pool compute audit and C22

The bounded NCU capture selected a real Critic/FlashAttention forward kernel;
it did not justify changing attention math or the almost-zero final exposed
NCCL. The lowest-risk follow-up was to avoid saving deterministic RNG state in
Critic checkpoint regions. C22 passed all runtime receipts but produced
`55.153462575 s`, also a regression. It is rejected.

### C23: continuous B32

C23 keeps at most two rollout streamers live while processing four B1 waves.
The first trace build failed closed because its manually derived trace-load
receipt used the wrong latent frame count; `iteration_171` is preserved as a
failure. The corrected trace-load receipts are `[24,22,20,24,22]` for current
and `[22,20,20,22,22]` for candidate inputs. Validate-only iteration 172,
correctness iteration 173, and timing iteration 174 then passed.

Timing iteration 174:

```text
raw steady intervals: 108.697360314 s, 108.469908303 s
raw median:           108.583634309 s
logical samples:      160 per Full5
throughput:             1.473518556 samples/s
B16-equivalent:        54.291817154 s per 80 samples
world three-cycle:    330.735920905 s
```

The run passed 16/16 ranks and ended at `G3/C15/EMA0`. Logical inventory was
Student/Critic/Teacher `12648/576/96`; physical Critic grad calls were `240`
because each lane performs two B2 microbatches per step. Peak allocated was
`35.175426 GiB/rank`, minimum driver free `1.006226 GiB`, with no CUDA/cgroup
OOM increment.

## Why B32 helps only slightly

The timing receipts directly show the pipeline is no longer dominated by an
empty transport bubble:

| Metric | C21 B16 | C23 B32 |
|---|---:|---:|
| Generator backpressure median | `0.008812 s` | `0.008950 s` |
| Score receive-wait median | `0.000152 s` | `0.000143 s` |

Additional waves can only amortize the fixed fill/drain edges, hence the
observed `1.48%` throughput gain. The remaining critical work is the dense
Critic B2 backward and step0 Teacher compute. C21-C1 also proves that splitting
the first B2 into an early B1 backward does not improve the critical path on
this implementation.

## Identity and claim boundary

- B16 trace SHA256: `e13c8716bd0756f380ba4d3521b2220e4ec4ce1c20dd5cd96598b021e8c8144c`
- B16 bootstrap SHA256: `2c70c70644e6b445af41d43a84d781f1ebf4fe4c4392015f20e151637f597c28`
- B32 trace SHA256: `51705edb7a31519010b9741bd35786be6d31edded99c8b644142d67eda807ec1`
- B32 trace file SHA256: `5e5906bb1a04ea314b5bf18ef76dcd41f7223d8d90f3d47b6bf41cf8f56e69d2`
- B32 bootstrap SHA256: `f30bbc6be8c902099ee29ba402fe1c5170f238b97a5b5f80fd0abec4e81b6da0`
- B32 bootstrap file SHA256: `956d8f0b68c5e5aca0af629cd679df58e1efb91a08ef3a5afddde21dc705dc39`
- Source-focused test gate: `71 passed, 1 deselected`; compile and shell syntax
  checks passed. The deselection is the known system-Python BF16 `torch.eye`
  limitation, not the validated Torch 2.10 training environment.

Only 80 source samples are present in the current asset bundle. C23 therefore
maps slots 16-31 to the same source rows as slots 0-15 while using distinct
operation keys and fresh RNG. This makes C23 valid for bounded scheduler and
throughput diagnosis, but not for convergence or data-diversity claims. The
historical strict sequential/pipeline numerical comparison also remains
outside its preregistered gate. All results in this report are diagnostic-only.

Persistent local summaries are under:

`/home/lyy/Helios/research_artifacts/DMD-System/livewan_crossscale_20260830/`

Raw remote evidence remains on the shared experiment storage; it is not copied
into GitHub.
