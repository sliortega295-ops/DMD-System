# Static PP2 group2 / accumulate8 capacity path

This is an explicit, default-off generic pipeline memory policy, not a
DMD-System kernel, cache, offload, or role-reordering optimization. The default
remains `--pipeline-memory-policy gpipe8`. The new invocation is
`--pipeline-memory-policy group2-accumulate8`.

## Frozen semantics

- TP4 / DP2 / PP2 on world16; node-local TP and cross-node PP.
- Eight **independent B1** samples per DP replica, global logical batch16.
- Each actual DiT stage forward is B1. A GPipe invocation runs two B1
  microbatches, not one batch-of-two DiT forward.
- Four successive GPipe2 forward/score/backward groups accumulate gradients
  before **one** original optimizer update. No commit occurs between groups.
- A mean2 loss is multiplied by `2/8`; four such contributions implement the
  same mean8. The external-loss bridge applies no second division. DP SUM/2,
  model-global clipping, Adam, scheduler and EMA boundaries are unchanged.
- Original G update, then five G-rollout/C-update/EMA steps. No phase clustering.
- Each sample retains its own trace, noise, histories, indices and controls;
  the dispatcher draws no training RNG. No merged weights or parameter offload.

## Capacity rationale and limits

The native Helios GPipe8 run failed in cycle1 gradient rollout with a 536MiB
modulation allocation at roughly 36.75GiB allocated. GPipe2 limits simultaneous
microbatch graphs; it does not shrink a single B1 graph or native modulation.
It may therefore still fail. The original OOM remains a separate outcome.

PyTorch 2.10 GPipe rejects fewer microbatches than pipeline stages. A proposed
GPipe1 path was rejected by a CPU gate; it was never run on GPUs. This policy
uses the supported minimum of two, without changing upstream version checks.

## Admission and evidence

Real CPU PP2 checkpoint-forward/backward checks compare outputs and gradients
against eight original B1 forwards. A separate test proves per-sample DMD
normalization and grouped mean8 gradients; original GPipe8 and actual Gloo16
DP-mean/norm tests remain regression gates. GPU timing is admitted only after
same-source two-node imports, immutable asset identities and ownership checks.

Receipts additionally bind invocation size2 / accumulation groups4, all eight
source ranks per DP, four pair inventories per logical rollout, unchanged
physical B1 call totals, optimizer/EMA counts and no pending backward graph.
Only a complete 16-rank two-cycle run can provide a C2 baseline time. CPU
fixtures are not GPU capacity or full numerical-equivalence evidence.
