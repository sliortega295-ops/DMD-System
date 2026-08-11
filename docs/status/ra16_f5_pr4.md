# RA16-F5-PR4 status

**Evidence state:** continuous Full5 **FAIL -- CUDA OOM at Step10-A backward**
**Topology:** two nodes, 16 ranks, one complete Route-A pipeline per rank
**Logical scope:** official 384x640 steps 10--14 from canonical checkpoint-10

Admission is fail-closed on an actual cgroup/CUDA OOM increment. The previous
64-GiB reserve is retained only as an observed diagnostic, not as a rejection
condition for this experiment.

This is the enhanced Route-A candidate. It is distinct from the accepted
all-40-streaming baseline `RA16-F5-FS40` and from the isolated `GST16-A-G4`
capacity gate.

## Frozen lifecycle intent

- Step10-A: Generator streams exactly `g4=[0,24,1,5]` with window 1.
- Score/Teacher: no frozen-block streamer; immutable backing may materialize the
  full physical Score for its phase.
- Step10 Critic plus steps 11--14: Generator's four externally managed blocks
  are materialized once for each complete no-grad Critic rollout phase, not once
  per physical Transformer call.
- Official update order remains one Generator, five Critic and five EMA updates,
  ending at `G3/C15/EMA15`.
- Route-A still time-shares Generator and Score on each GPU. “Phase resident”
  does not claim simultaneous full Generator and Score residency.

## Measured continuous result

Run `ra16_f5_pr4_20260808_002` passed source/config contracts, exact world16
topology, model construction and checkpoint-10 resume. It then reached
`generator_loss_end` and `semantic_generator_pre_backward` at step 10, but all
16 ranks failed during checkpoint recomputation in Generator backward while
allocating a 216-MiB FP32 temporary. Observed driver-free memory was only
102--210 MiB; allocator reserved-but-unused memory was 205--371 MiB. No
Generator, Critic or EMA update completed, so versions remained
`G2/C10/EMA10`. Node cgroup OOM counters did not increment. Measured cgroup
peaks were 718.166 GiB on node 0 and 715.748 GiB on node 1.

This refutes admission of the exact PR4 policy. The isolated `GST16-A-G4`
result is not contradictory: it used a role-split Step15-A graph with
MB1/GAS2 and a lower Generator peak, rather than the combined continuous
Route-A graph.

Run `ra16_f5_pr4_20260808_001` is separately preserved as a pre-model
fail-closed launcher result: an untracked standalone topology artifact dirtied
the checkout. It contains no workload evidence.

## Existing supporting evidence

- `GST16-A-G4` passed the isolated Step10-A capacity/correctness gate.
- Resident official-shape no-grad Generator rollout and resident Score/Critic
  execution passed their separate gates.
- These isolated results do **not** establish continuous Full5 admission,
  correctness or timing for this combined policy.

## Bounded successor

`RA16-F5-PR4-EC` keeps g4 fixed and adds only the existing allocator-cache
release before backward. It receives one attempt under its own run ID and
contract. The original PR4 result remains failed and immutable; no performance
field is claimable.

That bounded attempt, `ra16_f5_pr4_ec_20260808_001`, reproduced the same OOM
on all 16 ranks: the same 216-MiB allocation, call site, and per-rank
free/reserved figures. Immediately before backward, only about 45 MiB was
allocator-reserved but unused; the larger unused figures in the exception were
created during backward and were not available to the pre-backward cache
release. PR4-EC is therefore closed as a bounded negative result. Increasing
the streamed set requires a separately named experiment.
