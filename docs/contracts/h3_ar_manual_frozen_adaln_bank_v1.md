# Manual-only W6 exact frozen AdaLN bank, v1

Opt-in experiment only. Not an H3ARImplementation, search candidate, default,
calibration cost or saved-plan strategy. The normal API/CLI is unchanged.

On the original continuous-sigma, eight-step, Student1/Fake1 previous/current
chunk workload, relocate the same fifty frozen AdaLN projection modules from
ZeRO block leaves to a registered root bank. No parameter is copied, removed,
merged, quantized or offloaded. LoRA parameter order and optimizer groups remain
unchanged. A block leaf's explicit logical numel is 397290240 instead of
657499392; the total registered original AdaLN numel stays 13010457600.

Cache only the original projection's output tuple. Eight fixed rollout keys
are identified by original schedule scalar storage metadata, not rounded sigma
values. Two dynamic continuous-sigma pairs per update are held by strong
references and evicted after the original fenced boundary. Checkpoint functions
capture the key explicitly; replay does not rely on ambient ContextVar state.
No host-visible value read occurs in a training timer. Exact input-value checks
and frozen shard versions are audited outside that timer. Missing, mutated,
ambiguous or excessive keys fail without a projection fallback.

Each update retains the original twelve application forwards and two complete
backward graphs. Expected physical block calls: 700/update; checkpoint wraps:
100/update; scoped executions: 200/update. C1 bank calls/reads: 500/200;
C2: 100/600. Static output entries: 400; dynamic entries released: 100/update.
All original attention-cache, leaf-release, RNG, precision, update and resource
gates remain in force. Different receipt schema prevents manual measurements
being accepted as canonical planner calibration.

Validation levels must stay separate: synthetic CPU and native-size ZeRO3
primitive proofs do not establish full native capacity or speed. Native C2
requires a new-source manual control and candidate, each two consecutive
updates in one process; initialization, hashing and boundary auditing are not
part of the native update timer. No profiler. No long-trajectory claim.
