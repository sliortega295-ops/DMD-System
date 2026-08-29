# C1 iteration preflight

- Gate: port the previously validated exact Helios compute-fusion substrate,
  then run frozen GAN-on correctness before timing.
- Live source before candidate: `c01cb7781c3c6feaec4e7a3f9128b45a263d5102`;
  baseline math source `55d8884cf394ecb2ddd9b762844466a82055b983`.
- Live resources: both assigned nodes expose eight idle A100-SXM4-40GB GPUs;
  no live CUDA worker was present and cgroup `oom/oom_kill` counters were zero.
- Environment: reuse the validated Torch 2.10 archive at
  `/share/project/shuchenweng/yongyan_liu/DistillSchedule_envs/torch210_20260824/`
  rather than constructing a new training environment.
- Frozen baseline and workload: see `draft.md`; formal denominator is
  `173.014483004 s`, target `121.110138103 s`.
- Matching prior evidence: No-GAN Iterations 353-367 exercised the same Helios
  LoRA/residual/modulation code paths with exact fail-closed receipts. The new
  GAN profile independently shows the corresponding high-count add, mul,
  cast, copy, norm, and materialization kernels.
- Rejected/superseded paths: no blind FSDP prefetch (NCCL is about 87% hidden),
  no CPU-offload repair (D2H is effectively absent), no B2/packing/topology
  change, and no reuse of the historical No-GAN timing as a GAN claim.
- Falsifiable question: does the exact fusion substrate preserve all GAN-on
  receipts and reduce the matched unprofiled Full5 below 173.014483004 s?
- Decision: **GO**, limited to the exact fusion implementation and its
  correctness/timing gates. A contract drift or fallback count rejects C1.

