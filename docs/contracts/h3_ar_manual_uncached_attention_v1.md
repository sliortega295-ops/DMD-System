# Manual W6 uncached rollout attention v1

Explicit manual-only experiment. No canonical profile, search candidate,
planner calibration, saved plan or default changes.

The actual native architecture has56 attention heads of dimension128, not
hidden_size5376/128=42. Preserve both valid and padding sequence segments.
Use the already-pinned FA3 varlen build only on the seven UNSELECTED Student
no-grad rollout calls per update. Selected NG (source for later checkpoint
reuse), all grad forwards, all replay, Teacher/Fake score and refiner calls
keep the original Torch SDPA and original replay/output cache. Norm, RoPE,
QKV, LoRA weights and all model inputs/updates remain untouched.

Per update/rank:700 main attention executions,350 actual FA3 calls,350 native
main fallbacks,24 native refiner calls;12 application predicts. Application
Student9/Fake2/Teacher1 and backwardStudent1/Fake1 stay unchanged. Source-bound
original cache must retain its exact200 sources/300 consumers. Setup binds
original CUDA cu storage+version and reads bounds once before the training
window; dispatch does no host-visible GPU value reads. Missing/changed shape,
role, source, backend binary or census fails. Never silently use FA3 on the
selected source, because that would bypass the original ATen cache collector.

The rejected frozen-AdaLN bank is NOT combined with this experiment. Full
original block parameter numel657499392 and all registered weights stay put.
The existing original-native capacity/host/owner/timer policies remain.
Component synthetic QKV output/RNG proof and CPU routing/meta-module proofs
are not full-native correctness/speed proof. Native comparison must use the
same new source for both manual off and this explicit arm, each one process
and two consecutive updates; C1 warmup, C2 only, no profiler. Numerical REVIEW
must not be promoted to strict state/trajectory equivalence.
