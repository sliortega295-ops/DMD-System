# W5 full FA3 cache round-trip diagnostic

Default off. Only `bind_plan(..., cache_roundtrip=True)` with the existing
correctness-only divergence plan can enable this diagnostic. Orphan flags,
changed helper/kernel hashes, mixed nodes, simultaneous raw-backward repeat,
timing/calibration and unknown schemas are rejected. Canonical profiles and
all existing numerical/admission thresholds are unchanged.

Read every logical element of original out/LSE before staging and of the
reloaded GPU tensors after H2D, using two index-mixed uint32 reductions. Store
the small results in int64 tensors; no full output remains resident for the
diagnostic. Hashing handles logical strides without materializing a contiguous
copy. Its bounded temporary reduction buffers are about 1 MiB at production
shape. It neither writes training tensors nor consumes RNG or changes grads.

Coverage is both native cycles, six grad graphs per cycle, 50 blocks per graph,
out plus LSE: 1200 pairs/rank. CUDA known-bit tests (including NaNs, signed zero,
strides and a changed value) execute before cycle1 and must pass. Host reads
occur only after the existing cycle stop fences; each cycle produces an
immutable rank record before the local mismatch check. No collective is added.
A mismatch is handled by the existing owned launcher and stops
the diagnostic and preserves the record. GPU self-test/compilation and all
diagnostic overhead are not native training performance evidence.

Matching fingerprints are not cryptographic or full bitwise equivalence
proofs, do not establish convergence or absence of all races, and do not grant
a speed-pair gate. Instrumentation may perturb timings/races. A mismatching
fingerprint is evidence to investigate the actual observed copy path and the
fingerprint implementation; it cannot be silently turned into PASS.
