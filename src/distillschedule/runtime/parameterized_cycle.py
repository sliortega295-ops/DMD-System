"""Versioned, parameterized G1/Rn/Cn program over the existing ExecutionGraph.

The protocol is deliberately explicit: one G commit, n independently bound
post-commit rollouts, n ordered C commits, and EMA at declared C boundaries.
It is NOT a generic claim about LiveWan pre-commit latents, arbitrary GAN loss
ordering, stochastic sampler exits, or a new model backend. The old Full5 API
and canonical worker contracts remain unchanged.
"""
from __future__ import annotations

from typing import Mapping

from distillschedule.core.contracts import require
from distillschedule.core.enums import GradientMode as Grad, OperationKind as Kind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation

SCHEMA = "distillschedule.parameterized_cycle.v2"
PROTOCOL = "post_generator_commit_independent_rollouts"


def build_parameterized_cycle_graph(*, source_steps: tuple[int, ...],
                                    initial_versions: Mapping[str, int],
                                    ema_after: tuple[int, ...] = (),
                                    protocol: str = PROTOCOL) -> ExecutionGraph:
    """Compile n=1..64 logical C updates; adapters still own actual inputs/math."""
    path = "cycle.v2"
    require(protocol == PROTOCOL, "unsupported cycle protocol; provide a native protocol adapter", path=path)
    count = len(source_steps)
    require(1 <= count <= 64 and len(set(source_steps)) == count
            and all(type(s) is int and s >= 0 for s in source_steps),
            "need 1..64 distinct nonnegative source steps", path=path)
    require(set(initial_versions) == {"generator", "critic", "ema"}
            and all(type(v) is int and v >= 0 for v in initial_versions.values()),
            "explicit nonnegative G/C/EMA versions required", path=path)
    require(all(type(i) is int and 0 <= i < count for i in ema_after)
            and tuple(sorted(set(ema_after))) == ema_after, "invalid EMA positions", path=path)
    g, c, e = (initial_versions[k] for k in ("generator", "critic", "ema"))
    ops = [Operation("g", source_steps[0], "native_runtime", Kind.GENERATOR_UPDATE, Grad.GRAD,
                     required_versions={"generator": g, "critic": c},
                     produced_versions={"generator": g + 1}, metadata={"inputs": ()})]
    edges = []
    previous = "g"
    for i, step in enumerate(source_steps):
        r, update = f"r{i}", f"c{i}"
        ops.append(Operation(r, step, "native_runtime", Kind.CRITIC_ROLLOUT, Grad.NO_GRAD,
                             required_versions={"generator": g + 1}, metadata={"inputs": (), "fake_index": i}))
        # Preserve original preparation order until an adapter binds randomness
        # by logical sample identity. Same global seed is not sufficient.
        edges.append(Dependency("g" if i == 0 else f"r{i-1}", r, "preparation_order"))
        ops.append(Operation(update, step, "native_runtime", Kind.CRITIC_UPDATE, Grad.GRAD,
                             required_versions={"generator": g + 1, "critic": c + i},
                             produced_versions={"critic": c + i + 1},
                             metadata={"inputs": (r,), "fake_index": i}))
        edges.extend((Dependency(r, update, "retained_input"), Dependency(previous, update, "optimizer_order")))
        previous = update
        if i in ema_after:
            name = f"e{i}"
            ops.append(Operation(name, step, "native_runtime", Kind.EMA_UPDATE, Grad.NO_GRAD,
                                 required_versions={"generator": g + 1, "critic": c + i + 1, "ema": e},
                                 produced_versions={"ema": e + 1}, metadata={"inputs": ()}))
            edges.append(Dependency(update, name, "ema_order"))
            previous, e = name, e + 1
    return ExecutionGraph(tuple(ops), tuple(edges), {
        "schema_version": SCHEMA, "protocol": protocol, "critic_updates": count,
        "initial_versions": dict(initial_versions),
        "final_versions": dict(generator=g + 1, critic=c + count, ema=e),
        "source_steps": source_steps, "ema_after": ema_after,
        "input_binding": "NATIVE_ADAPTER_MUST_VERIFY_OPERATION_QUALIFIED_INPUTS",
        "native_support": "REQUIRES_PARAMETERIZED_WORKLOAD_ADAPTER",
    })


def parameterized_window_order(graph: ExecutionGraph, *, max_live_rollouts: int) -> tuple[str, ...]:
    require(graph.metadata.get("schema_version") == SCHEMA, "not a v2 cycle graph", path="cycle.v2")
    n = graph.metadata["critic_updates"]
    require(type(n) is int and 1 <= n <= 64, "invalid update count", path="cycle.v2")
    require(type(max_live_rollouts) is int and 1 <= max_live_rollouts <= n,
            "window must be in [1,critic_updates]", path="cycle.v2")
    result, prepared = ["g"], 0
    for i in range(n):
        while prepared < min(i + max_live_rollouts, n):
            result.append(f"r{prepared}")
            prepared += 1
        result.append(f"c{i}")
        if i in graph.metadata["ema_after"]:
            result.append(f"e{i}")
    return tuple(result)
