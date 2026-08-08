# No-GAN suite status

**Status date:** 2026-08-08  
**Current task:** N2 target-graph Generator minimum-offload search  
**Accepted capacity result:** Not Run

The frozen N2 graph is 8 Generator ranks on node 0 plus four resident Student
and four resident Teacher ranks on node 1. It replays the exact official
384x640 rank-qualified trace at isolated Step15-A, starts from G3/C11, performs
one global-batch-16 Generator update with ZeRO-1 and CPU optimizer offload, and
changes only the Generator frozen-block streamed set. Shared-file tensor
exchange is correctness transport and is not timing evidence.

## Attempt ledger

| Attempt | Result | Cause | Claim |
|---|---|---|---|
| `n2-g-offload-target-20260808-001-anchor-g40` | **FAIL before workload** | The new Generator executor disabled DeepSpeed BF16 while loading a native BF16 ZeRO-1 checkpoint. DeepSpeed therefore requested nonexistent `zero_pp_rank_*` optimizer shards instead of the present `bf16_zero_pp_rank_*` shards. The peer Score job was cancelled; neither DMD nor Generator training ran. | Runtime configuration failure only; no capacity or timing result |

The failure produced no new cgroup OOM event. It is preserved in
`artifacts/manifests/n2_target_g_offload_anchor_20260808_001_result.json` and
permits one bounded, non-invasive retry that restores the checkpoint's BF16
precision mode. The failed immutable run ID will not be overwritten or reused.

The official negative-prompt asset passed exact validation with manifest hash
`ca0de6928f83ebd450418d33f8990696763d3228069ec4ac0efb2444694cf78c`
and tensor hash
`470581384766363123713892abd21b6c2bd07a5569c90d2100a4c9aab84e504d`.
Historical Route-A tail-1/tail-3/tail-5/all-40 experiments remain diagnostics
and are not N2 search bounds.
