# GAN role-phase node1 adaptation status (2026-08-12)

## Result

`PASS_CONTRACT_STATIC`; real 14B/world16 execution is `NOT_RUN`.

The PR8 topology is now bound to the eight fixed Score/Critic GPUs on node1:

| Global rank | Phase-A lane / slots | Phase-B slot | Phase-B producer |
|---:|---:|---:|---:|
| 8 | 0 / 0,8 | 4 | G0 |
| 9 | 1 / 1,9 | 5 | G1 |
| 10 | 2 / 2,10 | 6 | G2 |
| 11 | 3 / 3,11 | 7 | G3 |
| 12 | 4 / 4,12 | 8 | G0 |
| 13 | 5 / 5,13 | 9 | G1 |
| 14 | 6 / 6,14 | 10 | G2 |
| 15 | 7 / 7,15 | 11 | G3 |

Phase A remains B16/two waves/read-only Score: each node1 rank serves two
logical samples, with four official Score forwards per sample. Phase B is B12, DP12,
MB1, GAS1, ZeRO-1, CPU optimizer offload, and gradient clipping 1.0. Every
Critic rank receives one sample and enters one `_critic_loss` invocation. That
invocation still contains the official two Score forwards: denoising and the
packed GAN-D/real/R1 forward.

## Fixed-node transition

Node1 does **not** need to reconstruct or stream the Score backbone at the
phase boundary. The same resident physical module is rebound from the Phase-A
read-only service to the Phase-B DP12 engine; only the logical FP32 Adam state
is repartitioned into twelve CPU ZeRO-1 partitions. PR6 persistent-pinned target
materialization remains necessary for flex ranks 4--7 on node0, which actually
change physical role from Generator to Critic.

The worker distinguishes two different `micro_wave` meanings:

- wire/header identity is `micro_wave=0` because Phase B has one transport wave;
- official Helios `_critic_loss` receives `pipeline_micro_wave=None`, which is
  the existing one-call complete-batch path and commits exactly once. Passing
  `0` would incorrectly inherit the old PR7 GAS2 first-wave defer behavior.

## Validation

- DMD refactor tests: `159 passed`;
- external runtime contract tests: `4 passed`;
- rank contracts: `8/8 PASS`;
- GPUs after validation: `8/8 at 0 MiB`;
- cgroup: `oom=0`, `oom_kill=0`.

The first focused test command is preserved as a non-semantic failure: both
repositories expose a top-level `runtime` package and the combined PYTHONPATH
selected the wrong one for a DMD-only test. The second command separated the
two import roots and passed without changing source.

## Source identity

- PR8 base: `c221bc9f673f17dc26e4a07cd39b908c6b2b8bd1`;
- node1 implementation commit: `0b2caa6990afedca0d11065e8ab569446e27febc`;
- no-Torch CI compatibility guard: `7ff9b8309dcb1dd86ee155d107562eaf390ab478`;
- external runtime: `9ca730dfd40370b3a859c0fa5f3e554b6bd39223`;
- node1 DMD staging:
  `/share/project/shuchenweng/yongyan_liu/DMD-System_gan_rolephase_pr8_exec_20260812`;
- node1 runtime staging:
  `/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812`.

## Explicit Not Run boundary

No 14B model was constructed. Phase-A gradients, DP12 collectives,
C1000-to-C1005, node0 companion, flex transition, EMA ordering, reverse
handoff, full receipts, capacity, and timing remain `NOT_RUN`.

## Next gate

Implement node0 against the same pre-created group inventory, using PR6's
validated bounded transition only for flex ranks 4--7. Then run a no-model
world16 topology/transport gate before any 14B construction.
