# Frozen contract: RA16-F5-GLP-SCORE4

`RA16-F5-GLP-SCORE4` is a new one-variable functional capacity experiment
following the bounded-negative `RA16-F5-GLP-OFFLOAD` result. It does not reopen
that result and does not authorize a sweep.

## Frozen workload and substrate

The parent contract remains exact: two nodes x eight A100-40GB, official
384x640, checkpoint-10, steps 10--14, rank-qualified trace v3, BF16,
MB1/GAS1/global batch 16, ZeRO-1 CPU optimizer, A32 saved-tensor offload, g12
Generator block selection, typed Generator LP-state Critic-phase offload, and
the official `G2/C10/EMA10 -> G3/C15/EMA15` update order.

## Only policy change

Add a Score frozen-block streamer with window 1 for exactly blocks
`[36, 37, 38, 39]`. The Generator retains its parent g12 set. The four Score
blocks remain CPU/mmap-backed outside their physical calls, are materialized
before each Score forward or checkpoint-recompute call, and are released only
after the CUDA use completes. This is the safe implementation of unloading
four blocks during backward; it does not skip or approximate block math.

The tail-four selection is preregistered rather than tuned after seeing the
result. Checkpoint backward traverses tail blocks first, so they can be released
before later recomputation while still reducing persistent Score residency.

## Fail-closed gates

The run is invalid unless compiled evidence records Generator indices
`[0,24,1,5,8,25,10,32,36,15,16,26]`, Score indices `[36,37,38,39]`, two streamer
inventories, exact trace/checkpoint/version contracts, and all other parent
settings unchanged. One functional world16 attempt is authorized. It passes
only if Step10 Critic backward/update and the complete steps10--14 Full5 state
transition finish without CUDA/cgroup OOM and checkpoint-15 validates. Formal
repeated timing remains **Not Run**.
