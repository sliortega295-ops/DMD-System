# Frozen contract: RA16-F5-PR4

This contract changes only the physical frozen-parameter lifetime policy of the
accepted `RA16-F5-FS40` Route-A workload. The algorithm, trace, checkpoint,
world size, batch/update semantics and EMA order are unchanged.

## Identity

- Experiment ID: `RA16-F5-PR4`
- Run-ID prefix: `ra16_f5_pr4_`
- Topology: 2 nodes x 8 A100-40GB, world size 16
- Scope: official 384x640 continuous steps 10--14
- Initial/final versions: `G2/C10/EMA10 -> G3/C15/EMA15`
- Updates: one Generator, five Critic, five EMA
- Backend: ZeRO-1 with CPU optimizer offload; no parameter sharding/offload
- Host admission: actual cgroup/CUDA OOM only; the retired 64-GiB reserve is
  still reported as a diagnostic but is not a rejection condition
- Inputs: the same rank-qualified trace v3 and checkpoint-10 as
  `RA16-F5-FS40`

## Only authorized lifecycle change

1. Step10-A Generator uses window-1 streaming for exactly blocks
   `[0,24,1,5]`.
2. Score has no block streamer. Its full immutable frozen state is materialized
   once per Score phase through the existing backing manager.
3. During each no-grad Generator rollout used by Critic, the four externally
   managed Generator blocks are materialized once at rollout entry, retained
   across all nested physical Transformer calls, and released at rollout exit.
4. `generator_selected` and `critic_train` retain the existing 32-MiB coarse
   activation-offload primitive.
5. Route-A still time-shares Generator and Score on each rank. This contract
   does not claim simultaneous full-model residency or a split-pool pipeline.

## Fail-closed evidence

An accepted functional run must prove:

- all 16 ranks and official rank-qualified trace fingerprints;
- steps 10--14 exactly once per rank;
- exact version/update counts above, including all five EMA updates;
- no Score block-stream events;
- Step10-A Generator uses only `[0,24,1,5]`;
- each Critic rollout has one balanced phase-resident enter/exit pair and no
  per-physical-call release/reload of those four blocks;
- zero final streamed resident bytes, activation handles and pending events;
- nonempty checkpoint-15 and no cgroup/CUDA OOM increment.

The isolated `GST16-A-G4` and resident Critic-only gates are supporting evidence,
not substitutes for this continuous Full5 acceptance. Formal speedup remains
`Not Run` until paired repeated timing against `RA16-F5-FS40` under an identical
trace/checkpoint contract.
