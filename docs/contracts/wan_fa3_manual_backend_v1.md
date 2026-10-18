# Wan manual large-self-attention backend v1 (experimental, no planner)

Explicit immutable packet and `--execute` are mandatory. Reuse the original
manual two-cycle worker, bootstrap, trace, G1/C5/EMA5, B1/global16, full-state
exporter and all original gates. Both arms retain C26-29 with private frozen
bases and isolated graph pools. Failed shared-base/tail6 is NOT included.

Candidate replaces ONLY original SDPA BF16 CUDA self-attention of exact
B1/H12/D128, S2160/8640/11040, equal Q/K/V shape, contiguous last dimension,
no mask, dropout=0, noncausal, scale=None, GQA=False. Existing immutable
FlashAttention3 performs the same default scaled softmax attention. S540,
cross-attention and all other operators retain original SDPA. Do not change
FP32 LoRA computation or fuse weights. CPU metadata-only routing reads no
GPU tensor values. Backend shape/layout changes can affect rounding and
must retain numerical REVIEW; component parity is not training parity.

Loader is invoked only after original worker selects rank CUDA device,
before construction/capture. It temporarily borrows the already-validated
kernel loader import path, restores sys.path before model imports, and loads
only the pinned local source/binary. No downloads or dependency changes.
The scoped process-local SDPA hook is restored on success/failure. No source
profile, strategy, registry, default, scheduler or planner modification.

Receipts bind source HEAD, packet/assets/strategy and kernel-file hashes,
actual grad/nograd dispatch counts and lengths. Counts describe Python
eager/capture calls, NOT per-replay kernel launches. Original application,
version, graph, finite, dtype and capacity checks remain. Native timing
must use the same new source two arms and C2 only, excluding source/setup,
backend import, correctness/state export. No profiler; no primitive ratio
may be called a Full5 speedup.
