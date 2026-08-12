# GAN role-phase v3: 4 fixed G + 8 fixed C + 4 Flex

## Scope

This branch replaces the persistent-8G+8Score performance state machine from PR #7 while preserving the validated Helios DMD/GAN math, G/C/EMA version order, operation-keyed stochastic replay and wire identities.

The central correction is that **Phase A and Phase B update different models and therefore do not need the same global batch**:

- Phase A updates Generator parameters and keeps global batch16.
- Phase B updates Critic/Score parameters and reuses the no-GAN global batch12 shape.

No speedup is claimed yet.

## Physical topology

- fixed Generator: ranks 0--3;
- Flex: ranks 4--7;
- fixed Critic/Score: ranks 8--15.

Phase A:

- G8 = ranks 0--7;
- Score8 = ranks 8--15;
- global batch16.

Phase B:

- G4 = ranks 0--3;
- Critic12 = ranks 4--15;
- global batch12.

## Phase A: 8G + 8Score, batch16

Phase A keeps the already validated PR #7 Generator-update semantics:

- two waves x eight lanes;
- each Generator rank owns two logical samples;
- Score computes the official DMD + GAN-G boundary;
- latent gradients return to Generator;
- exactly one Generator optimizer commit occurs after both waves.

The dependency DAG allows Generator wave1 forward to proceed after the wave0 latent send is safe while Score computes wave0. It must not regress to callback-level `send -> wait -> reply -> next wave` serialization.

## Bounded role transition

After G201 commits and the Score gradient service drains:

- ranks 0--3 remain no-grad Generator producers;
- ranks 4--7 switch from Generator to Critic;
- ranks 8--15 remain Critic/Score;
- the first real implementation reuses the PR #6 persistent-pinned bounded transition substrate;
- repeated CPU frozen-block streaming is forbidden inside stable phases;
- CPU optimizer state may remain on CPU;
- the reverse transition before the next Generator-update boundary is included in steady-state timing.

GPU live reshard remains a later diagnostic optimization rather than a prerequisite for the first real run.

## Phase B: 4G + 12Critic, batch12

Phase B directly reuses the simple no-GAN resource/batch shape:

- four Generator producers;
- each Generator produces exactly three B1 samples;
- twelve Critic ranks each receive exactly one B1 sample;
- total Critic global batch is exactly 12;
- each Critic rank enters exactly one physical `_critic_loss` call per logical step;
- each sample has weight `1/12` in the global Critic objective.

Canonical mapping:

- G0 -> logical slots 0, 4, 8;
- G1 -> slots 1, 5, 9;
- G2 -> slots 2, 6, 10;
- G3 -> slots 3, 7, 11;
- Critic ranks 4--15 consume slots 0--11 one-to-one.

This removes the previous prototype complexity entirely:

- no 12+4 second Critic wave;
- no Flex B2 vs fixed-rank B1 asymmetry;
- no batch16 special gradient normalization;
- no batched operation-trace adapter;
- no unequal model-call count problem for ZeRO/FSDP collectives.

For DP12/B1, the desired gradient is the ordinary batch12 mean. If the backend performs DP mean reduction, the official per-rank B1 loss needs no extra scale. If a backend exposes SUM semantics, each local loss is scaled by `1/12`.

## Q2 wavefront

The queue remains one-batch-ahead.

Producer side:

- two reusable payload buffers;
- buffer credit returns on send completion, not Critic optimizer ACK.

Consumer side:

- at most two logical batches remain received/uncommitted;
- receive of batch `i+2` waits for Critic commit `i`;
- Critic compute for step `i+1` still waits for commit `i`, preserving strict C-version order.

P2P routes are identity-bound by step, slot, source, destination, trace and G/C versions, with explicit FIFO on each `(src,dst)` edge. Phase B has one Critic wave only.

## Current implementation

- `src/distillschedule/runtime/gan_rolephase.py`: phase-local B16/B12 contracts, topology, versions and Q2 credits.
- `src/distillschedule/runtime/gan_rolephase_schedule.py`: explicit Phase-A overlap and Phase-B wavefront DAG.
- `src/distillschedule/runtime/gan_rolephase_critic.py`: uniform DP12/B1 batch12 reduction contract.
- `src/distillschedule/runtime/gan_rolephase_transport.py`: ordered identity-bound batch12 P2P.
- `src/distillschedule/runtime/gan_rolephase_transition.py`: PR #6-style bounded transition contract.
- `runtime/gan_rolephase_helios.py`: thin B1 Helios Critic/reduction adapter.
- `configs/distillschedule/pipelines/helios_gan_world16_4g8c4flex_rolephase.json`: frozen phase-local batch plan.

## Remaining gate before GPU

The complete real-14B worker/launcher still needs to be wired:

1. instantiate/restore the Phase-A G8 and Score8 engines using the existing official GAN helpers;
2. construct the Phase-B G4/C12 engines with the proven transition/checkpoint substrate;
3. bind the observed DeepSpeed gradient reduction mode for DP12/B1;
4. derive/replay the 12 Phase-B logical samples from the frozen workload while preserving stochastic semantics;
5. emit phase-local fairness/compute/batch/version/timing receipts;
6. run a real world16 capacity/correctness gate before any formal timing.

PR remains Draft until those bindings pass. No GPU performance run should start from the contract alone.
