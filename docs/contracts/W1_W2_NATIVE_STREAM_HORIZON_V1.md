# W1/W2 bounded native fresh-stream horizon v1

This source/CPU contract extends the existing finite fresh-cohort interfaces
from 64 to **100 complete cycles**, superseding only the former 64-cohort
ceiling (including `helios_nogan_safe_stream_v1.md`), not the immutable fixed-trace experiment.
It does not establish 100-cycle pretrained GPU execution, numerical parity,
convergence, decoded output or quality. Existing research-only bundle status
and hardware/dependency admission remain unchanged.

- W1 reference/manual/selected fresh stream and explicit incumbent stream v2:
  2..100 cohorts, indexed 0..99. Every cohort retains its task/trace/source
  identity and per-rank feature contract. A single seed is valid with 100
  distinct cohort identities; repeating a fixed trace is not training.
- W2 original reference and SafeBundle fresh streams: 2..100 cohorts, retaining
  the original warm G2/C10/EMA10 bootstrap, No-GAN33/222/world16/B1 and G1/C5/EMA5.
  All 80 features per cohort remain globally path/hash distinct. CPU1 reference
  owners additionally enforce their existing `max_train_steps` capacity; a
  requested 100-cycle stream needs capacity at least 100 (source clock 510).
  Current-G CPU latent-only export accepts those completed boundaries without
  widening its precision, budget, topology, model or output scope.
- W1 source steps are `first_step + 5*i .. first_step + 5*i + 4` for the
  incumbent; reference tasks retain their actual fake-update count. W2 source
  steps at indices 63/64/99 are 325..329 / 330..334 / 505..509. Original
  constant LR is not a finite scheduler horizon. Native beta curriculum with
  `dynamic_step=500` clamps progress at one; neither source steps nor EMA
  clocks restart at that point. Resolution-dependent native shift and the
  original per-stage budgets are unchanged.
- W1 complete versions are bootstrap G/C/EMA plus actual updates and absolute
  EMA cadence. From cold G0/C0/EMA0, incumbent cycle100 is G100/C500/EMA500.
  W2 cycle100 is G102/C510/EMA510. A checkpoint at completed cycle50 resumes
  inputs/cycles 51..100, not 1..50 again. Exact source/assets/stream/topology
  identities and persistent-state/RNG checks remain mandatory.
- Graphs remain transient. First-process-cycle capture is bounded to the
  original 12 rollout / 3 critic signature capacities. A later uncaptured
  signature still fails; 100 admitted cohorts do not promise that arbitrary
  fresh inputs fit that capture inventory. No shape, stage budget, fake-update
  count, RNG draw or safety gate may be changed to force graph admission.
- **Public manual fixed trace and calibration remain exactly two cycles**.
  Existing internal/selected-plan repeated-fixed-trace diagnostics retain
  their old 2..64 bound and explicit diagnostic classification. Neither can
  gain a 65..100 fixed-trace repetition bypass from this extension. Integers
  above100, booleans and fractional fresh-stream horizons are rejected.

The stream schema names stay unchanged: source/content fingerprints bind this
explicit contract extension to the current implementation. Frozen source and
helper pins, source-import receipts, old benchmark/calibration contracts and
training math are not rewritten. Changed code inventories require new inputs
and normal source-bound admission; old source receipts are historical evidence,
not silently current training identities.
