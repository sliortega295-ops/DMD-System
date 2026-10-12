"""Parameterized dispatch into the original Wan/Helios-pyramid math helpers.

No Euler surrogate protocol, model replacement or canonical Full5 modification.
The caller supplies an independently validated, operation-qualified trace for
this task. Device placement is established before preparing a cycle.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from distillschedule.core.contracts import require
from distillschedule.core.enums import OperationKind as Kind
from distillschedule.runtime.parameterized_cycle import (
    build_parameterized_cycle_graph, parameterized_window_order,
)
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger


class ParameterizedWanCycle:
    def __init__(self, prepared, records, read_versions, count):
        self.prepared, self.records = prepared, records
        self.read_versions, self.count = read_versions, count
        self.native_validation_error = None

    def run(self):
        results = self.prepared.run()
        if self.read_versions() != self.prepared.receipt()["final_versions"]:
            self.native_validation_error = "native engine commits disagree with selected plan"
            require(False, self.native_validation_error, path="wan.parameterized.versions")
        return {"generator": results["g"], "rollouts": self.records,
                "critic": [results[f"c{i}"] for i in range(self.count)]}

    def receipt(self):
        result = self.prepared.receipt()
        result["native_versions"] = self.read_versions()
        if self.native_validation_error:
            result.update(status="FAIL", native_validation_error=self.native_validation_error)
        return result


def prepare_parameterized_wan_cycle(*, p0: Any, source_steps: tuple[int, ...],
        ema_after: tuple[int, ...], max_live_rollouts: int,
        rank, device, generator_engine, critic_engine, ema, trace, negative,
        diagnostics, policy, clean_runtime, runtime_cache,
        deferred_finite_checks, collect_scalars=False, operation_scope=None,
        order: tuple[str, ...] | None = None) -> ParameterizedWanCycle:
    """Use the original G update / rollout / C update / EMA implementations.

    Trace inputs, not a new RNG stream, bind every logical sample. Serial rank-
    local dispatch works with the existing world16 Resident/FSDP engines; it
    does not create or simulate inter-pool overlap. CPU handler tests only
    validate dispatch until real trace/backend validation has run.
    """
    require(tuple(trace.logical_steps) == source_steps,
            "trace steps must match task; cannot relabel a Full5 trace", path="wan.parameterized")
    require(trace.variant == "no-gan", "GAN requires its own native protocol adapter", path="wan.parameterized")
    require(policy.rollout_batch_size == 1 and not policy.pack_real_score_pair,
            "this binding preserves physical B1 and unpacked Score requests", path="wan.parameterized")
    require(not policy.generator_nograd_persistent_unshard,
            "persistent unshard requires a declared multi-action scope", path="wan.parameterized")
    require(ema is not None, "explicit native EMA state required, even with zero EMA updates", path="wan.parameterized")

    def versions():
        return dict(generator=int(generator_engine.global_steps),
                    critic=int(critic_engine.global_steps), ema=int(ema.optimization_step))

    graph = build_parameterized_cycle_graph(source_steps=source_steps,
                initial_versions=versions(), ema_after=ema_after)
    default_order = parameterized_window_order(graph, max_live_rollouts=max_live_rollouts)
    selected = default_order if order is None else tuple(order)
    live = peak = 0
    for key in selected:
        if key.startswith("r"): live += 1
        elif key.startswith("c"): live -= 1
        peak = max(peak, live)
    require(peak <= max_live_rollouts, "selected order exceeds retained window", path="wan.parameterized")
    records = []
    common = dict(rank=rank, device=device, trace=trace, collect_diagnostics=False,
                  diagnostics=diagnostics, runtime_cache=runtime_cache)

    def scope(ctx):
        return nullcontext() if operation_scope is None else operation_scope(ctx.operation)

    def generator(ctx):
        with scope(ctx):
            result = p0._run_nogan_generator_update_p0(**common,
                generator_engine=generator_engine, critic_engine=critic_engine, negative=negative,
                collect_scalars=collect_scalars, policy=policy, score_pack_parity_check=False,
                clean_runtime=clean_runtime, deferred_finite_checks=deferred_finite_checks)
        # The old helper loads trace.load_generator_entry() correctly but its
        # diagnostic return contains a literal 10. Relabel only that diagnostic.
        return {**result, "logical_step": ctx.operation.logical_step}

    def rollout(ctx):
        step, entries = ctx.operation.logical_step, {}
        with scope(ctx) as sample:
            generated, rows, _ = p0._precompute_critic_rollouts(**common, steps=(step,),
                generator_engine=generator_engine, policy=policy, rollout_parity_check=False,
                clean_runtime=clean_runtime, entry_outputs=entries)
            require(set(generated) == set(entries) == {step} and len(rows) == 1,
                    "native B1 rollout inventory changed", path="wan.parameterized")
            records.append({**rows[0], "batch_index": ctx.operation.metadata["fake_index"]})
        if sample is not None:
            sample.record_output((generated[step], entries[step].tensors))
        return generated[step], entries[step]

    def critic(ctx):
        x0, entry = ctx.values[ctx.operation.metadata["inputs"][0]]
        with scope(ctx):
            return p0._run_nogan_critic_precomputed(**common,
                step=ctx.operation.logical_step, x0=x0, critic_engine=critic_engine,
                entry=entry, collect_scalars=collect_scalars,
                deferred_finite_checks=deferred_finite_checks)

    def update_ema(ctx):
        with scope(ctx): ema.step()

    handlers = {Kind.GENERATOR_UPDATE.value: generator, Kind.CRITIC_ROLLOUT.value: rollout,
                Kind.CRITIC_UPDATE.value: critic, Kind.EMA_UPDATE.value: update_ema}
    runner = StaticRoleRunner("native_runtime", handlers, versions=VersionLedger(versions()))
    prepared = runner.prepare(graph, order=selected,
                               retain_outputs=("g", *(f"c{i}" for i in range(len(source_steps)))))
    return ParameterizedWanCycle(prepared, records, versions, len(source_steps))
