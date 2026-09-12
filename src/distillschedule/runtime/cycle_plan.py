"""Composable G1/R5/C5 cycle actions over the existing execution graph.

This defines dependencies and bounded retained-rollout schedules, not model math.
Native adapters must bind input/RNG identities and verify actual update/call
counts independently. It does not enable a backend merely by naming a workload.
"""
from __future__ import annotations

from typing import Mapping

from distillschedule.core.contracts import require
from distillschedule.core.enums import GradientMode as Grad, OperationKind as Kind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation


def build_cycle_graph(*, source_steps: tuple[int, ...],
                      initial_versions: Mapping[str, int],
                      ema_after: tuple[int, ...]) -> ExecutionGraph:
    require(len(source_steps) == 5 and len(set(source_steps)) == 5
            and all(type(s) is int and s >= 0 for s in source_steps),
            "exactly five distinct source steps required", path="cycle.steps")
    require(set(initial_versions) == {"generator", "critic", "ema"}
            and all(type(v) is int and v >= 0 for v in initial_versions.values()),
            "explicit nonnegative G/C/EMA versions required", path="cycle.versions")
    require(all(type(i) is int and 0 <= i < 5 for i in ema_after)
            and tuple(sorted(set(ema_after))) == ema_after,
            "invalid EMA commit positions", path="cycle.ema")
    g, c, e = (initial_versions[k] for k in ("generator", "critic", "ema"))
    ops = [Operation("g", source_steps[0], "native_runtime", Kind.GENERATOR_UPDATE,
                     Grad.GRAD, required_versions={"generator": g, "critic": c},
                     produced_versions={"generator": g+1}, metadata={"inputs": ()})]
    edges = []
    prior_commit = "g"
    for i, step in enumerate(source_steps):
        r, update = f"r{i}", f"c{i}"
        ops.append(Operation(r, step, "native_runtime", Kind.CRITIC_ROLLOUT, Grad.NO_GRAD,
                             required_versions={"generator": g+1},
                             metadata={"inputs": (), "fake_index": i}))
        # Preserves the actual operation-qualified preparation order. Arbitrary
        # permutations require prebound random/sample contexts, not a seed claim.
        edges.append(Dependency("g" if i == 0 else f"r{i-1}", r, "preparation_order"))
        ops.append(Operation(update, step, "native_runtime", Kind.CRITIC_UPDATE, Grad.GRAD,
                             required_versions={"generator": g+1, "critic": c+i},
                             produced_versions={"critic": c+i+1},
                             metadata={"inputs": (r,), "fake_index": i}))
        edges.extend((Dependency(r, update, "retained_input"),
                      Dependency(prior_commit, update, "optimizer_order")))
        prior_commit = update
        if i in ema_after:
            name = f"e{i}"
            ops.append(Operation(name, step, "native_runtime", Kind.EMA_UPDATE, Grad.NO_GRAD,
                                 required_versions={"generator": g+1, "critic": c+i+1, "ema": e},
                                 produced_versions={"ema": e+1}, metadata={"inputs": ()}))
            edges.append(Dependency(update, name, "ema_order"))
            prior_commit, e = name, e+1
    return ExecutionGraph(tuple(ops), tuple(edges), {
        "schema_version": "distillschedule.dmd_cycle_actions.v1",
        "initial_versions": dict(initial_versions),
        "final_versions": dict(generator=g+1, critic=c+5, ema=e),
        "source_steps": source_steps, "ema_after": ema_after,
        "input_binding": "NATIVE_ADAPTER_MUST_VERIFY_OPERATION_QUALIFIED_INPUTS",
    })


def rollout_window_order(graph: ExecutionGraph, *, max_live_rollouts: int) -> tuple[str, ...]:
    require(type(max_live_rollouts) is int and 1 <= max_live_rollouts <= 5,
            "rollout window must be an integer in [1,5]", path="cycle.window")
    require(graph.metadata.get("schema_version") == "distillschedule.dmd_cycle_actions.v1",
            "not a DMD cycle graph", path="cycle.graph")
    result, produced = ["g"], 0
    for i in range(5):
        while produced < min(i + max_live_rollouts, 5):
            result.append(f"r{produced}")
            produced += 1
        result.append(f"c{i}")
        if i in graph.metadata["ema_after"]:
            result.append(f"e{i}")
    return tuple(result)
