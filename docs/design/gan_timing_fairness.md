# GAN Full5 timing fairness

Formal timing compares only steps1000–1004 and is based on **actual consumed values**, not shared RNG seeds. Physical scheduling may differ, but logical sample slot0–15 must consume identical source/history tensors, stochastic tensors, stage exits and fixed model-call inventory.

Required pair evidence:

- `gan_fairness_receipt`: source/history/random tensor hashes and three-stage DiT exits for every logical sample;
- `gan_compute_receipt`: variable rollout DiT-forward count plus the frozen fixed Score/GAN call inventory;
- same workload contract digest and exact G/C/EMA semantics;
- an unprofiled five-step wall-clock window.

The baseline world16 mapping is `rank=slot`. The persistent 8G+8Score mapping is `wave=slot//8`, `G=slot%8`, `Score=8+(slot%8)`. `scripts/distillschedule/run_gan_paired_formal.py` refuses to report speedup unless the canonical receipts are equal.

For the frozen official GAN configuration, step1000 has one Fake-Score conditional forward, no Fake-Score unconditional forward, one Real-Score conditional forward, one Real-Score unconditional forward, and one GAN-G Score forward per logical sample. Every Critic step has one denoising Score forward plus one GAN-mode Score forward packing fake, real and R1; R2 is disabled.

Nsight Systems is diagnostic only. `DISTILLSCHEDULE_GAN_NSYS_PROFILE=1` activates the node-local profiling shim. The exact range is synchronized across world16: collection starts before the first model compute at step1000 and stops only after every rank has completed the step1004 Critic commit. Profiled runs are never admitted to the formal timing table.

Current claim boundary: the schemas, validators, paired-timing gate, and Nsight capture path are implemented. Real 14B baseline/pipeline workers must still emit matching consumption receipts before stochastic fairness or speedup can be claimed.