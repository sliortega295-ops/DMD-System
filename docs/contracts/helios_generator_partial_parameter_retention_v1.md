# Manual No-GAN partial Generator parameter retention v1

This default-off experiment is not an automatic planner candidate or a new
cost-model dimension. `pg0-v1` and `pg24-v1` suffixes are accepted only on the
native No-GAN node-HSDP, chunk512, window5, safe bundle, exact Critic tail4
identity. Neither may inherit calibration data. Canonical profiles and the
automatic implementation enumerator remain unchanged.

Both arms group the existing five consecutive B1 rollout preparation actions
into one `r_all` metadata/action scope, using the existing native B1 helper five
times in exactly the original order. Expanding `r_all` gives the unchanged
operation order, versions, samples and updates. This is not B5 batching.

* `pg0-v1`: original per-block resharding. No parameter-retention hook.
* `pg24-v1`: first24 of40 Generator block FSDP units retain full parameters only
  inside this preparation phase. The root and16 other blocks are unmodified.
  Public `set_reshard_after_forward` and `reshard` are the only mutation APIs.
  All selected policies restore and all Generator units are sharded before any
  Critic update. The physical Generator version must remain fixed. Actual root
  forward hooks reject grad-enabled calls and version changes. No parameter
  fusion, offload, dtype change, checkpoint/cache/RNG/optimizer change occurs.

The legacy whole41-unit persistence flag stays false and its existing No-GAN
rejection remains intact. `enabled=false` in the legacy receipt refers only to
that whole-model policy; the new partial policy is separately named and audited
in `partial_parameter_retention`. Consumers require actual24 acquire/release
counts, a matching actual root-forward inventory, stable version and zero
remaining full units. The new helper does not force allocator cache flushing.

GPU qualification is NOT established by CPU mock FSDP tests. Both source-bound
correctness arms and the original comparison/admission precede native timing.
C1 warm-up, C2 MAX-rank original fenced timer; all residency/release costs are
inside that timer. Original allocated-highwater/host reserve and ownership
rules remain unchanged. REVIEW is not strict numerical equivalence.
