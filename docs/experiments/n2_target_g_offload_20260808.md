# N2 target G/S/T Generator offload search

## Iteration preflight

**Status:** implementation `GO`; GPU launch `NO-GO` until the target executor and
source/config smoke pass.

The live checkout is `b728ac169d8c18f8e5d33eb06da464c952a7cd3b` on
`codex/pr3-ci-and-suite-fix`. Both remote hosts resolve the same detached HEAD,
have no GPU compute process, have no D-state process, and show no new cgroup OOM.
The existing untracked Attempt 005 result remains preserved on the shared tree.

The next falsifiable test is a **fresh target-graph all-40 anchor**, not a reuse of
the Route-A tail probes. The fixed target is:

- node 0: 8 Generator ranks, ZeRO-1, CPU optimizer offload, MB1/GAS2;
- node 1: four resident Student and four resident Teacher inference ranks;
- official 384x640, global batch 16, logical Step15-A, G3/C11;
- rank-qualified trace entries 0--15, including traced DMD timestep and noise;
- the official encoded negative prompt, identified by tensor SHA-256;
- only the Generator frozen-block streamed index set may change.

## Prior-path disposition

The Iteration 237 prototype is useful source evidence but is not executable under
this contract: it used ZeRO-3, synthetic contexts, newly sampled Score noise and
timestep, and a random negative embedding. ZeRO-2 remains a bounded known failure.
Historical Route-A `tail-1/3/5/all-40` results are diagnostics, not N2 bounds.

The machine-readable preflight is
`artifacts/manifests/n2_target_g_offload_preflight_20260808_001.json`.

## Implemented target executor

The narrow executor now binds every candidate to the immutable candidate manifest,
the rank-qualified trace bundle, G3/C11 target package and validated mixed-BF16
pre-cast manifests before model construction. Node 0 runs eight ZeRO-1 Generator
ranks with MB1/GAS2. Node 1 runs four resident C11 Student ranks and four resident
official-base Teacher ranks. A shared-file x0/gradient exchange is used only for
this capacity search and is explicitly excluded from timing claims.

The source audit caught two legacy-prototype differences before GPU execution:

- direct pre-cast construction must use Helios's validated
  `precast_direct_load=True` path rather than treating the mixed checkpoint as a
  generic Diffusers checkpoint;
- the old Iteration-237 Score worker used `eval`; the target executor preserves
  the official Stage-3 Score module mode while keeping all DMD calls under
  `no_grad`. This is a semantics repair and can never be credited as speedup.

The cheap generator-side input gate passes the exact trace-bundle hash, G3 native
world8 inventory (one model state plus eight ZeRO-1 optimizer shards), logical
target manifest and Generator pre-cast source identity. The full CPU test suite
passes in an isolated remote clone. The official negative-prompt tensor still has
to be generated and validated before the Score-side gate can pass.

No N2 performance or capacity result exists yet.

## Next bounded execution

1. Freeze and validate the official negative-prompt asset under a node GPU lock.
2. Re-run the exact committed source/config gates on both nodes.
3. Run the fresh all-40 target anchor.
4. Only if the anchor passes, freeze its measured block order and enter the
   preregistered coarse-to-linear search. A non-capacity failure stops the search;
   historical Route-A probes are never substituted.
