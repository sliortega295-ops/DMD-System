# Generic workload program integration

Parent: PR #33 at 27b3a9d7bd135ed8ee02530507c86fbbf81cdf09. Isolated branch:
agent/generic-workload-capabilities-20261012. Existing workflows, canonical
profiles and protocol mathematics are not replaced.

Checkpoint 1: introduce a name-independent WorkloadProgram on the existing
ExecutionGraph, with semantic parameters, native call properties and identity.
Reject unresolved branches, missing input dependencies and stale version reads.
9 local contract tests passed. No execution, CUDA, performance or all-model
support is claimed by this checkpoint. Next: reusable capability matching,
shared bound-case scheduling, original-Wan planner integration, and a tested
callable runtime for previously unseen semantic graphs.
