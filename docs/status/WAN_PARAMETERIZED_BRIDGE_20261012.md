# Wan original-protocol parameterized strategy integration (2026-10-12)

## Publication and validation

The 10 source increments have been published on
`agent/original-wan-strategy-20261012` starting from
`088b8ad53f16051577aebb5d75eb6e2f5370fc13`.
The bridge is opt-in and uses the original Wan1.3B Helios-pyramid no-GAN
model and optimizer helpers; it does not replace them with the separate
Euler test protocol. Five canonical profiles and naive FSDP2 reference remain
unchanged.

The local CPU strategy and parameterized dispatch suite: **138 passed**.
The GitHub Actions job on the first published revision had 137 passed and one
missing-NumPy environment failure; the next revision declares NumPy in the
native optional dependencies and reruns the unchanged suite.
These are interface and CPU checks, **not** original-model multi-GPU
correctness, capacity, numerical equivalence, or speed evidence.

## Implemented surface

- Original protocol input contract: frame/height/width lattice, G1/Fn,
  LoRA rank, stage-specific denoising horizon, and EMA cadence.
- Trace admission checks sixteen rank-qualified payloads and geometry against
  a matched native bootstrap; existing Full5 traces cannot be truncated to
  impersonate a different task. Trace capture remains external.
- Existing world16 FSDP and Resident/async128/Tree layouts, modulation chunks,
  and no-grad LoRA epilogue enter a capability-gated candidate set.
- Validated generated orders lower through existing Wan math and actual worker
  functions. Two-node campaigns enforce finite GPU trial budgets and keep
  out-of-tolerance numerical results as REVIEW.

See `docs/workloads/parameterized_original_wan.md` for supported semantics,
commands, candidate restrictions, and remaining GPU/trace-capture work.
