# Same-input raw FA3 backward diagnostic (default off)

This extends the existing source-bound first-Fake diagnostic only. The normal
training gradients, model, schedule, RNG, attention policy and cache are unchanged.
It authorizes no timing/calibration/search-cost or numerical admission.

Explicit plan marker and matching two-node opt-in are required. At rank0,
block49 of the first Fake backward, keep the original native gradients and call
that same raw backward three more times with deterministic=False, twice with
True, using the exact same six live inputs (dout/q/k/v/out/LSE), scale and other
options. Record full output difference norms/counts and input metadata/versions.
Use chunk-bounded temporary buffers, including strided inputs; do not allocate
full FP32 model-gradient copies. Repeated raw calls consume no training RNG.
Scalar tensor reads remain outside the original cycle stop fences. Return the
unmodified original gradients to native training, never the diagnostic outputs.

A complete original two-cycle native census is still collected, but all wall
fields remain diagnostic-only. Extra kernel counts are explicit. This can prove
repeatability or inequality for this one actual call, not whole-model numerical
equivalence, unbiased noise, optimizer stability, or prefetch/cache equivalence.
Existing strict H3 failures and the user's no-speed-pair direction remain.
