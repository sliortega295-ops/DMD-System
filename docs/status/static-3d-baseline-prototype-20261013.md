# Static TP4/DP2/PP2 baseline prototype

This is a default-off prototype, not a performance result.

- Static layout: TP4 within a node, DP2 within each PP stage, PP2 across nodes.
- Logical batch: eight B1 microbatches per DP replica, global batch16. Collaborating TP/PP ranks are not extra samples.
- LoRA: column A replicated/B output-sharded, row A input-sharded/B replicated. Frozen base BF16, unmerged LoRA FP32; Teacher disable retains the original PEFT shell.
- Attention q/k RMSNorm uses the original global hidden-width statistic and its backward, not independent per-head-shard normalization.
- CPU gates: both actual A100-node Python environments passed10 tests, including a real4-process Gloo forward, input/LoRA gradient, non-reentrant checkpoint and Teacher-disable reference check. The small numerical fixture uses CPU FP64 base and FP32 LoRA, explicitly not GPU BF16 equivalence or timing.
- Transformer binder is source-hash guarded and opt-in. It validates the supported projection structure before mutating any module.

## Remaining NOT_RUN / NOT_IMPLEMENTED

- Native transformer model-binding GPU validation: NOT_RUN.
- PP microbatch executor, inter-role loss/backward dependency, DP synchronization/global clipping/optimizer binding: NOT_IMPLEMENTED.
- Original 31/6 (or workload-specific) application census and state equivalence: NOT_RUN.
- Native two-cycle timing: NOT_RUN. No RealHF-like seconds or speedup may be inferred from this prototype.

Evidence: /home/lyy/Helios/research_artifacts/DMD-System/main_experiments_20261013/manifests/static_3d_cpu_tp4_node{0,1}_r3.json. Small source and hashes are retained; no weights, environment or profiler database is included.
