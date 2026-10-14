# Fresh Wan grid geometry binding

The original rollout primitives explicitly audit latent dimensions 48×80.
Higher-resolution fresh traces must change those audit/assertion dimensions,
not truncate, resize or reuse the original 384×640 input trace.

Only the opt-in parameter-grid entrypoints bind the requested registered
geometry. Source-derived AST lowering changes exactly two base_height and
two base_width tracer arguments per primitive, and the final latent shape
assertions. The original sampler/RNG, arithmetic, loss, budget, exit policy,
BF16/FP32 precision and update order are untouched. The three existing
primitives and all already-loaded aliases are restored by ExitStack.
No native model implementation or parallel API is substituted.

The canonical 384×640 workload is a no-op. Geometry receipts include source
and lowered AST hashes, exact substitutions and alias bindings. CUDA Graph
shape verification derives from the admitted fresh workload. A frozen plan
that does not fit a new resolution is reported as a capacity failure, never
silently retuned or replaced with an older-resolution measurement.

The A100 normal/global-synchronized control scope remains explicitly
disclosed. Native rank-local amplification is not part of this matrix.
Target GPU results are pending until the exact-head per-rank receipts pass.
