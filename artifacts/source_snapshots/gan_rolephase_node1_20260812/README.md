# PR8 node1 real-worker adaptation snapshot (2026-08-12)

This directory mirrors the exact external DistillSchedule runtime source used
for the **low-cost node1 adaptation gate** of the GAN role-phase prototype.
It is review evidence, not an alternate launcher entry point.

## Source identity

- DMD-System PR8 source before this adaptation:
  `c221bc9f673f17dc26e4a07cd39b908c6b2b8bd1`
- External DistillSchedule runtime commit:
  `952ad750f92aa874f7974df32ed5b86560a51fe7`
- Node1 DMD staging root:
  `/share/project/shuchenweng/yongyan_liu/DMD-System_gan_rolephase_pr8_exec_20260812`
- Node1 runtime staging root:
  `/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812`

## Scope

The worker binds only global ranks 8--15:

- Phase A: fixed resident Score, two B1 gradient-service waves per rank;
- Phase B: fixed resident Critic, DP12/B1/GAS1/ZeRO-1 with CPU optimizer;
- node1 logical slots: 4--11;
- no repeated frozen-backbone streaming on fixed node1 ranks.

The physical Score module is retained across the boundary. Only its logical
optimizer state is repartitioned into DP12 CPU ZeRO-1 partitions.

## Evidence boundary

Contract/static tests passed on node1. No 14B model was constructed. The node0
companion, flex-rank materialization, real DP12 collective, reverse handoff,
world16 correctness, and formal timing remain `NOT_RUN`.
