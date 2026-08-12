# GAN role-phase v1: 4 fixed G + 8 fixed C + 4 Flex

## Scope

This branch replaces the persistent-8G+8Score **performance state machine**, not the validated Helios math or evidence contracts from PR #7.

Reused unchanged:

- systems-only operation-keyed GAN trace;
- fairness and compute-inventory semantics;
- G/C/EMA version trajectory;
- GAN latent / latent-gradient / Critic request header identities;
- official global batch16 and logical steps1000--1004.

Not claimed yet:

- real 14B GPU execution;
- numerical parity of a 12-rank Critic backend;
- FSDP/ZeRO-3 integration;
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

No NCCL group creation belongs in the formal timing window.

## Phase A: 8G + 8Score

The logical batch16 remains two waves of eight. This preserves the already validated PR #7 Generator-update semantics and one G commit.

The next executor must remove the current callback-level `send -> wait -> reply -> next wave` serialization. Wave1 Generator forward is allowed to proceed after the wave0 latent send is safe while Score computes the wave0 DMD/GAN-G boundary.

## Bounded handoff

After the G201 commit and Score-gradient-service drain:

- Generator forward state is resharded 8 -> 4;
- Score/Critic forward state is resharded 8 -> 12;
- Flex ranks switch G -> Critic;
- frozen-backbone movement is GPU-side NCCL chunked reshard, not repeated CPU block streaming;
- CPU optimizer state may remain on CPU.

`plan_contiguous_reshard()` defines an exact canonical flat-shard intersection plan, so a runtime never needs to materialize the full 14B payload on one GPU.

The reverse 4 -> 8 and 12 -> 8 handoff is part of a steady-state cycle and must be timed/reported.

## Phase B: 4G + 12Critic

Global batch16 is preserved exactly:

- Critic wave0: 12 samples, one per rank4--15;
- Critic wave1: 4 samples, one extra on Flex ranks4--7;
- every sample contributes weight `1/16`;
- the final reduction is a SUM of weighted per-sample gradients, not a mean over 12 ranks.

Each fixed G rank produces four logical samples:

- G0: 0,4,8,12;
- G1: 1,5,9,13;
- G2: 2,6,10,14;
- G3: 3,7,11,15.

## Q2 wavefront

The new state machine intentionally separates two resource constraints.

Producer side:

- two reusable payload buffers;
- payload credit returns on NCCL send completion;
- Critic optimizer ACK is not required to reuse the producer buffer.

Consumer side:

- at most two received logical batches may remain uncommitted;
- receiving batch `i+2` waits for Critic commit `i`;
- Critic compute for step `i+1` still waits for commit `i`, preserving C-version semantics.

Therefore G production can be one batch ahead while Critic updates remain strictly ordered.

## First implementation files

- `src/distillschedule/runtime/gan_rolephase.py`: topology, uneven batch mapping, G/C/EMA state, Q2 credits, GPU reshard plan.
- `src/distillschedule/runtime/gan_rolephase_schedule.py`: explicit dependency DAG for legal overlap.
- `src/distillschedule/runtime/gan_rolephase_transport.py`: pre-posted Critic receives and send-completion credit primitive reusing PR #7 headers.
- `src/distillschedule/runtime/gan_rolephase_groups.py`: pre-created Phase-A/Phase-B/Flex process groups.
- `configs/distillschedule/pipelines/helios_gan_world16_4g8c4flex_rolephase.json`: frozen prototype configuration.
- `scripts/distillschedule/inspect_gan_rolephase.py`: CPU-only contract/reshard inspector.

## Next implementation gate before GPU launch

The real executor still needs two concrete backends:

1. canonical model-forward-state pack/unpack + GPU chunked 8->4 / 8->12 reshard;
2. batch16-equivalent 12-rank Critic gradient SUM/reduction and optimizer-step boundary.

Those must pass CPU/unit tests plus a distributed toy-tensor test before reconnecting the 14B Helios workers. No GPU performance run should be launched from this v1 contract alone.
