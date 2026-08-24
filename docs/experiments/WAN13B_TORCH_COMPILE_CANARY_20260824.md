# Wan1.3B Generator No-Grad `torch.compile` Canary (2026-08-24)

## Identity and scope

- Experiment HEAD: `9cca98381a36ef43b308863c6765785cce34483e`
- Workload: No-GAN world16, B1/global-B16, Score packing off,
  G1/C5/EMA5, final G3/C15/EMA15.
- Candidate: compile only the Generator's 30 Transformer-block forwards when
  autograd is disabled.
- Generator grad-bearing calls and all Critic calls remain eager.
- Inductor options: `mode=default`, `fullgraph=false`, `dynamic=false`.
- No model call, trace, communication, optimizer, EMA, offload, streaming, or
  swap policy was changed.

The runtime receipt observed 840 compiled and 180 eager Generator-block
dispatches per rank. Critic compile dispatches remained zero.

## Gates

| Gate | Result |
|---|---|
| validate R1 | FAIL before workers due to an unbound launcher-manifest local; preserved |
| validate R2 | PASS, 16/16 |
| compile semantic correctness/capacity | PASS, 16/16; 80 rank-step identities |
| eager same-HEAD semantic correctness | PASS, 16/16 |
| final versions | G3/C15/EMA15 |
| strict eager-vs-compile numerical parity | **FAIL** |
| formal paired timing eligibility | **false** |

All 80 rollout rows exceeded at least one fixed parity threshold. The worst
rollout normalized-L2 was `0.022686`; minimum cosine was `0.999737`. This arm
is therefore not an exact-equivalent optimization.

## One unprofiled diagnostic pair

| Arm | Full5 world max | Peak allocated | Minimum driver free |
|---|---:|---:|---:|
| same-HEAD eager | 27.181843 s | 21.887 GiB | 16.582 GiB |
| Generator no-grad compile | 33.730888 s | 21.887 GiB | 16.582 GiB |

The compile arm was 6.549045 s slower, a 24.09% regression. This is explicitly
diagnostic-only rather than a formal speedup result because strict parity did
not pass.

Cold correctness, including initial compilation, took 50.455033 s. After the
disk cache was populated, first block dispatch still cost median 3.806 s/rank
(min 3.492, max 4.100). Even subtracting this whole first-dispatch interval as
an explanatory upper bound leaves about 29.925 s, still slower than eager; that
subtraction is not itself a valid timing measurement.

## Decision

Reject the current block-level `torch.compile` arm. Do not promote it or expand
compile coverage automatically: it fails the fixed numerical gate and is
slower in the bounded diagnostic pair.

The preceding exact Nsight range reported 82.94% non-NCCL kernel temporal
coverage, only about 0.298 s median NCCL exposed versus non-NCCL kernels, and
350 B D2H per node. These are timeline metrics, not SM utilization or MFU.
Hardware SM/Tensor metrics were not captured, so MFU remains unresolved.
