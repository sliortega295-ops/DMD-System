# RA16-F5-GLP-OFFLOAD: typed Generator LP-state phase offload

**Status:** bounded negative for Full5 admission; the storage mechanism itself
is verified. Formal timing is **Not Run**.

## Question and frozen contract

PR12 (`g12`, resident Score, A32 saved-tensor offload) completed the Step10
Generator update but all 16 ranks OOMed in Critic checkpoint backward. The
passing split-role Critic paths do not instantiate a Generator DeepSpeed
optimizer engine. This experiment therefore changed exactly one lifecycle
policy: after the Critic no-grad Generator rollout, offload the inactive
Generator ZeRO-1 BF16 `bit16_groups_flat` to pageable CPU, and require it to be
reloaded before EMA or any later Generator read.

Everything else stayed frozen: 2x8 A100-40GB, official 384x640, MB1/GAS1/global
batch 16, rank-qualified trace v3, checkpoint-10, steps 10--14, ZeRO-1 CPU
optimizer, `g12`, unstreamed Score, A32 activation offload, one G update, five C
updates and five EMA updates. No g16, ZeRO-2, threshold, shape, trace, loss or
update-order change was allowed.

## Gates

The typed controller first passed a small CUDA ZeRO-1/CPU-optimizer gate: the
LP flat buffer was released and restored exactly, parameter views were rebound,
the next optimizer step succeeded and all handles balanced. Runtime root tests
passed 309 tests; DMD-System passed 68 focused tests and both-node source/config
smoke.

The world16 attempt passed topology, construction, checkpoint resume, Step10
Generator backward and the Generator optimizer update on all ranks. The
Generator LP state was then offloaded on all 16 ranks.

## Measured causal result

| Metric | PR12 | GLP offload | Delta |
|---|---:|---:|---:|
| Generator LP storage/rank | resident | 2,502,885,376 B offloaded | -2.330994 GiB |
| Critic pre-backward allocated/rank | 36,226,001,920 B | 33,723,116,544 B | **-2,502,885,376 B exactly** |
| Critic pre-backward allocated | 33.738093 GiB | 31.407100 GiB | **-2.330994 GiB** |
| Critic backward ends | 0/16 | 0/16 | no admission |
| Critic optimizer steps | 0 | 0 | no admission |

This exact byte equality validates the suspected hidden state and the legal
phase release. It is not an allocator-cache illusion.

## Failure

Checkpoint recomputation consumed the newly available headroom and all ranks
still CUDA-OOMed on a 216-MiB allocation. At failure PyTorch had 37.84--38.01
GiB allocated and driver free space was 96.38--216.38 MiB. Fifteen tracebacks
failed at the later block residual add (`transformer_helios.py:1640`); one
failed while materializing modulation (`:767`). The immediate worst-rank
deficit is at least 119.62 MiB, but this is only a lower bound for the current
allocation, not a bound on all remaining backward work.

Sampled cgroup peaks were 701.622 and 712.753 GiB. Neither node added an OOM or
OOM-kill event. Steps 11--14, all Critic/EMA updates, checkpoint-15 and formal
timing are **Not Run**.

## Interpretation and next boundary

The hypothesis had two parts: the 2.33-GiB inactive Generator LP state exists
and can be removed safely (**verified**); removing it alone admits resident
Score/Critic backward (**refuted**). Near-capacity allocation at the new OOM
does not mean the offload had no effect: the exact pre-backward reduction shows
the released capacity, which recomputation later consumed.

Increasing Generator streaming from g12 to g16 cannot fix this measured phase:
Generator frozen-block residency is already zero at Critic entry. A new run
would need a separately preregistered Critic-phase mechanism, such as selective
Score frozen-block streaming, or a typed audit of another inactive runtime
state. No unchanged retry is authorized.

## Evidence

- `artifacts/manifests/ra16_f5_glp_offload_20260809_001_failure.json`
- `artifacts/manifests/ra16_f5_glp_offload_20260809_001_analysis.json`
- `artifacts/manifests/ra16_f5_glp_offload_20260809_001_result.json`
- `artifacts/runs/ra16_f5_glp_offload_20260809_001/`
