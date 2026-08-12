# GAN role-phase v2: 4 fixed G + 8 fixed C + 4 Flex

## Scope

This branch replaces the persistent-8G+8Score **performance state machine**, not the validated Helios math or evidence contracts from PR #7.

Reused unchanged:

- systems-only operation-keyed GAN trace;
- fairness and compute-inventory semantics;
- G/C/EMA version trajectory;
- GAN latent / latent-gradient / Critic request header identities;
- official global batch16 and logical steps1000--1004.

Still not claimed:

- real 14B end-to-end execution from this branch;
- numerical parity of the final DeepSpeed/FSDP C12 reduction backend;
- any speedup.

## Physical topology

Startup creates all groups once:

- fixed G: ranks 0--3;
- Flex: ranks 4--7;
- fixed C: ranks 8--15;
- Phase-A G8: ranks 0--7;
- Phase-A Score8: ranks 8--15;
- Phase-B G4: ranks 0--3;
- Phase-B Critic12: ranks 4--15.

No process-group creation belongs in the formal timing window.

## Phase A: real two-wave overlap contract

Phase A remains the already validated batch16 topology, but the DAG no longer collapses the whole Generator update into `score_flush -> G commit`.

The explicit dependency is:

```text
G forward wave0 -> latent send0 -----------------> Score boundary0
                         |
                         +------> G forward wave1 -> latent send1
                                      |                  |
                                      |                  +--> Score boundary1
                                      +----------------------> G backward0
Score boundary1 + G backward0 ---------------------------> G backward1
G backward1 ---------------------------------------------> one G commit
```

The critical invariant is that `G forward wave1` depends on **wave0 send completion**, not the wave0 Score gradient reply. This removes the old callback-level `send -> wait -> reply -> next wave` serialization while keeping one local Generator engine call at a time.

## Bounded handoff: use the proven PR #6 substrate first

GPU live-reshard is no longer the required first backend. The first real role-phase executor should reuse the no-GAN PR #6 transition family:

- future target model is constructed on CPU before formal timing;
- target host state may be registered/pinned before timing;
- after G201 and Score-gradient-service drain, Flex ranks perform one bounded role transition;
- stable Phase B does **not** repeatedly CPU-stream frozen Transformer blocks;
- CPU optimizer state may remain on CPU;
- forward and reverse transitions are both part of steady-state timing.

The old canonical `plan_contiguous_reshard()` remains a diagnostic/future optimization only. Its locality problem is therefore not a blocker for the first GPU role-phase run.

## Phase B: one collective-safe model call per C12 rank

The old `12 samples + 4 samples` description was only logical accounting. It is replaced by one physical model call on every Critic rank per logical step:

- Flex C ranks 4--7: local batch **2**;
- fixed C ranks 8--15: local batch **1**;
- all twelve ranks enter the same number of parameter collectives and backward boundaries;
- exactly one logical Critic optimizer commit occurs per step.

Logical slot ownership is unchanged:

- C4: slots 0,12
- C5: slots 1,13
- C6: slots 2,14
- C7: slots 3,15
- C8..C15: slots 4..11, one each.

Every logical sample retains weight `1/16`. Backend gradient reduction semantics must be explicitly bound:

- if the backend performs DP **SUM**, each local per-sample loss sum is scaled by `1/16`;
- if the backend performs DP **MEAN over 12 ranks**, each local per-sample loss sum is scaled by `12/16` before the backend mean.

No code may infer this from the configured rank count.

### Batch-aware operation trace

Flex B2 requires more than concatenating generated `x0`. The two logical samples also carry distinct operation-keyed:

- timestep/noise;
- GAN fake/real/R1 crops;
- R1 epsilon;
- history/source tensors.

`runtime/gan_rolephase_helios.py` batches these tensors along their explicit B1 dimension. Operation controls, including the DiT stage schedule, must be identical across the two traces; otherwise the batch adapter fails closed.

## Q2 wavefront and ordered transport

Producer side:

- two reusable payload buffers;
- payload credit returns on NCCL send completion;
- Critic optimizer ACK is not required to reuse the producer buffer.

Consumer side:

- at most two received logical batches may remain uncommitted;
- receiving batch `i+2` waits for Critic commit `i`;
- Critic compute for step `i+1` still waits for commit `i`.

P2P ordering is explicit. Every pending receive is bound to:

`(step, logical_slot, wave, src, dst, trace, G version, C before/after)`.

Each `(src,dst)` edge owns a FIFO. Multiple receives may be pre-posted, but consuming a later receive before the FIFO head fails. Pending objects are removed by object identity rather than Tensor dataclass equality.

## Implementation files

- `src/distillschedule/runtime/gan_rolephase.py`: topology, sample routing, Q2 and version state.
- `src/distillschedule/runtime/gan_rolephase_critic.py`: collective-safe B2/B1 batch16 and reduction contract.
- `src/distillschedule/runtime/gan_rolephase_transition.py`: PR #6 persistent-pinned transition contract.
- `src/distillschedule/runtime/gan_rolephase_schedule.py`: explicit Phase-A and Phase-B dependency DAG.
- `src/distillschedule/runtime/gan_rolephase_transport.py`: identity-bound per-edge FIFO P2P.
- `src/distillschedule/runtime/gan_rolephase_groups.py`: pre-created process groups.
- `runtime/gan_rolephase_helios.py`: batch-aware Helios operation-trace/source adapter and backward scaling wrapper.
- `configs/distillschedule/pipelines/helios_gan_world16_4g8c4flex_rolephase.json`: v2 prototype configuration.
- `scripts/distillschedule/inspect_gan_rolephase.py`: contract inspector; GPU reshard is diagnostic-only.

## Validation added in this round

Dependency-light CI validates:

- Phase-A wave1 no longer waits for Score wave0 reply;
- C12 local batches are exactly B2/B2/B2/B2 + eight B1 ranks;
- every C12 rank has one physical model-call contract;
- global-batch16 reduction scaling is explicit;
- pinned transition hides neither direction and forbids stable repeated CPU streaming.

Torch-enabled environments additionally run:

- a 2-rank Gloo FIFO/identity P2P test;
- B2 operation-trace/source/x0 batching tests.

## Remaining gate before a real 14B GPU launch

The next patch must bind these runtime primitives into a real Helios worker:

1. instantiate the Phase-A Generator/Score engines with the existing official DMD/GAN math;
2. create/restore the Phase-B C12 trainable engine using the PR #6 transition/checkpoint substrate;
3. bind the actual DeepSpeed/FSDP gradient reduction semantics to `SUM` or `DP_MEAN` and validate one toy distributed optimizer step against a world16 reference;
4. run one real world16 correctness/capacity gate before any timing trial.

PR remains Draft until those model bindings pass. No GPU performance run should start from the contract alone.
