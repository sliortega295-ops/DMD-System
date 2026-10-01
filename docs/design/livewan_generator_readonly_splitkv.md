# LiveWan C49: read-only no-grad attention tiling

C49 is an explicit, default-off child of C38. It changes only the pinned
FlashAttention-2 execution kernel for registered Generator no-grad self-attention:
B1, Q=2760, K=8280/11040/13800/16560/19320, H=12, D=128, contiguous BF16.
It consumes the complete current Q/K/V with split count 2. It does not reuse
previous-step K/V, append tokens, apply new rotary transforms, or remove any
application forward. All gradient/recompute paths, cross-attention casts, Score
FA3/SAC, batches, trace, losses, optimizers and versions remain C38.

The pinned read-only API has source SHA-256
`bb57b5afc7ac164376fd8c43921488a70f3c6a05568607759a1b497774f6dded`.
Source/options/shape mismatches fail closed. Runtime receipts record actual
eligible calls and reference calls and reconcile them to total attention calls.
Score ranks must report NOT_APPLICABLE. No CUDA synchronization, host tensor-value
read, tensor copy, or collective is added by the production dispatcher.

## Evidence and admission boundary

- C48 source-shaped NCU: current Q2760/K19320 FA2 kernel has Tensor Active
  55.325%, DRAM SOL 5.768%, 255 registers/thread and 11.015% occupancy. These
  isolated counters are not phase utilization or MFU.
- C49 bounded component screen tested all 14 observed shapes and unchanged
  representative gradient/checkpoint paths under DEFAULT_P0_PARITY.
- Candidate splits 1/2/4 were preregistered; one global split count is selected.
  Unchanged/ineligible timing variation must not be credited as optimization.
- Component improvement is not native cycle-2 speedup or a promoted default.
  Production-class canary and source-bound world16 trajectory qualification
  must precede matched unprofiled C1/C2 measurement.
- C1 is warmup; only matched native C2 is claimable. Original strict numerical
  thresholds and the existing ordinary-REVIEW admission policy are unchanged.

Select `c49-continuous-b16-generator-readonly-splitkv` explicitly with the
existing LiveWan world16 controller. It is registered only for trajectory
qualification and two-cycle native timing. C38 remains the default.
