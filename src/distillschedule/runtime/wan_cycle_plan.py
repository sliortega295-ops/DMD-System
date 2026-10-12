"""Bind prepared actions to the existing exact-trace Wan native functions.

The dispatcher is shared with other native adapters; there is no alternate
model, loss, optimizer or rollout implementation in this module.
"""
from contextlib import nullcontext
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.core.enums import OperationKind as Kind
from .cycle_plan import build_cycle_graph, rollout_window_order
from .parameterized_cycle import (
    build_parameterized_cycle_graph,
    parameterized_window_order,
)
from .role_runner import StaticRoleRunner
from .version_sync import VersionLedger


class WanPreparedCycle:
    def __init__(self, prepared, records, read_versions, *, critic_updates=5):
        require(type(critic_updates) is int and 1 <= critic_updates <= 64,
                "invalid native Critic count", path="wan.cycle")
        self.prepared, self.records, self.read_versions = prepared, records, read_versions
        self.critic_updates = critic_updates
        self.native_validation_error = None

    def run(self):
        results = self.prepared.run()
        if self.read_versions() != self.prepared.receipt()["final_versions"]:
            self.native_validation_error = "native engine commits disagree with dispatched plan"
            require(False, self.native_validation_error, path="wan.cycle.versions")
        return {"generator": results["g"], "rollouts": self.records,
                "critic": [results[f"c{i}"] for i in range(self.critic_updates)]}

    def receipt(self):
        value = self.prepared.receipt()
        value["native_versions"] = self.read_versions()
        if self.native_validation_error:
            value.update(status="FAIL", native_validation_error=self.native_validation_error)
        return value


def validate_wan_prepared_receipts(rows, *, window: int | None, cycles: int) -> None:
    """Coordinator-side dispatch check in addition to native trace/math gates."""
    if window is None:
        require(not rows, "legacy run contains undeclared prepared actions", path="wan.cycle.receipts")
        return
    require(isinstance(rows, list) and len(rows) == cycles, "prepared cycle receipts missing", path="wan.cycle.receipts")
    for cycle, row in enumerate(rows):
        initial = dict(generator=2+cycle, critic=10+5*cycle, ema=10+5*cycle)
        graph = build_cycle_graph(source_steps=(10,11,12,13,14), initial_versions=initial, ema_after=(0,1,2,3,4))
        expected = {"schema_version":"distillschedule.prepared_role_execution.v1", "status":"PASS",
                    "executed_operations":list(rollout_window_order(graph, max_live_rollouts=window)),
                    "initial_versions":initial, "final_versions":graph.metadata["final_versions"],
                    "native_versions":graph.metadata["final_versions"], "failed_operation":None,
                    "peak_live_inputs":window}
        require(all(row.get(key) == value for key,value in expected.items()),
                f"cycle {cycle+1} did not execute the selected schedule", path="wan.cycle.receipts")


def prepare_wan_cycle(*, p0: Any, source_steps: tuple[int, ...], max_live_rollouts: int,
                      rank, device, generator_engine, critic_engine, ema, trace, negative,
                      diagnostics, policy, clean_runtime, runtime_cache,
                      deferred_finite_checks, collect_scalars=False,
                      observe_grad_norm=None, operation_scope=None) -> WanPreparedCycle:
    """Compile outside the cycle timer; all stochastic tensors come from trace.

    Persistent full-G unsharding cannot be broken into per-action scopes without
    changing its residency/communication contract; reject it until explicitly
    represented. No rollout batching is introduced by retained-context windows.
    """
    require(policy.rollout_batch_size == 1, "prepared Wan requires physical B1", path="wan.cycle")
    require(not policy.generator_nograd_persistent_unshard,
            "persistent unshard needs an explicit multi-action residency scope", path="wan.cycle")
    def versions():
        return {"generator": int(generator_engine.global_steps), "critic": int(critic_engine.global_steps),
                "ema": int(ema.optimization_step)}
    graph = build_cycle_graph(source_steps=source_steps, initial_versions=versions(), ema_after=(0,1,2,3,4))
    return _bind_wan_cycle(
        graph=graph, order=rollout_window_order(graph, max_live_rollouts=max_live_rollouts),
        p0=p0, rank=rank, device=device, generator_engine=generator_engine,
        critic_engine=critic_engine, ema=ema, trace=trace, negative=negative,
        diagnostics=diagnostics, policy=policy, clean_runtime=clean_runtime,
        runtime_cache=runtime_cache, deferred_finite_checks=deferred_finite_checks,
        collect_scalars=collect_scalars, observe_grad_norm=observe_grad_norm,
        operation_scope=operation_scope, read_versions=versions)


def prepare_parameterized_wan_cycle(
    *, p0: Any, source_steps: tuple[int, ...], ema_after: tuple[int, ...],
    max_live_rollouts: int, trace_contract: Mapping[str, Any],
    rank, device, generator_engine, critic_engine, ema, trace, negative,
    diagnostics, policy, clean_runtime, runtime_cache, deferred_finite_checks,
    collect_scalars=False, observe_grad_norm=None, operation_scope=None,
) -> WanPreparedCycle:
    """Use the *original* native helpers under an explicit new trace contract.

    This binds existing mathematics; it does not invent a new sampler/loss.
    A caller must supply a parameterized trace adapter that has independently
    validated its payloads and native control flow. The old StaticWorld16Trace
    and canonical timing/trajectory launchers retain their fixed contracts.
    Consequently this API alone does not enable new-shape/new-frequency GPU runs.
    """
    path = "wan.parameterized_cycle"
    require(isinstance(trace_contract, Mapping), "explicit trace contract required", path=path)
    expected_keys = {"schema_version", "protocol", "source_steps", "generator_source_step",
                     "ema_after", "physical_batch", "world_size", "input_binding"}
    require(set(trace_contract) == expected_keys, "incomplete/unknown trace contract fields", path=path)
    require(trace_contract["schema_version"] == "distillschedule.wan_prepared_trace_contract.v1",
            "unsupported Wan trace contract", path=path)
    require(trace_contract["protocol"] == "wan_native_nogan_post_g_independent_rollouts",
            "not the native Wan no-GAN protocol", path=path)
    require(trace_contract["input_binding"] == "validated_operation_qualified_trace",
            "a global RNG seed is not a bound trace", path=path)
    require(tuple(trace_contract["source_steps"]) == source_steps
            and tuple(getattr(trace, "logical_steps", ())) == source_steps,
            "trace/source steps differ; do not truncate an old Full5 trace", path=path)
    require(tuple(trace_contract["ema_after"]) == ema_after,
            "EMA policy differs from the declared trace contract", path=path)
    # The unchanged generator helper loads and labels the generator entry at 10.
    # Do not advertise arbitrary source offsets until that helper is adapted.
    require(source_steps and source_steps[0] == 10
            and trace_contract["generator_source_step"] == 10,
            "native Generator helper remains anchored at source step 10", path=path)
    require(getattr(trace, "variant", None) == "no-gan", "GAN needs its own native adapter", path=path)
    require(type(trace_contract["world_size"]) is int and trace_contract["world_size"] == 16
            and type(rank) is int and 0 <= rank < 16,
            "native Wan engine still requires world16", path=path)
    require(type(trace_contract["physical_batch"]) is int and trace_contract["physical_batch"] == 1
            and policy.rollout_batch_size == 1, "native Wan helper requires physical B1", path=path)
    require(not policy.generator_nograd_persistent_unshard,
            "persistent unshard requires a multi-action residency scope", path=path)
    def versions():
        return {"generator": int(generator_engine.global_steps), "critic": int(critic_engine.global_steps),
                "ema": int(ema.optimization_step)}
    graph = build_parameterized_cycle_graph(source_steps=source_steps, initial_versions=versions(),
                                             ema_after=ema_after)
    return _bind_wan_cycle(
        graph=graph, order=parameterized_window_order(graph, max_live_rollouts=max_live_rollouts),
        p0=p0, rank=rank, device=device, generator_engine=generator_engine,
        critic_engine=critic_engine, ema=ema, trace=trace, negative=negative,
        diagnostics=diagnostics, policy=policy, clean_runtime=clean_runtime,
        runtime_cache=runtime_cache, deferred_finite_checks=deferred_finite_checks,
        collect_scalars=collect_scalars, observe_grad_norm=observe_grad_norm,
        operation_scope=operation_scope, read_versions=versions)


def _bind_wan_cycle(*, graph, order, p0, rank, device, generator_engine, critic_engine,
                    ema, trace, negative, diagnostics, policy, clean_runtime,
                    runtime_cache, deferred_finite_checks, collect_scalars,
                    observe_grad_norm, operation_scope, read_versions):
    """One set of native handlers for both legacy and parameterized programs."""
    critic_ids = tuple(op.operation_id for op in graph.operations if op.kind == Kind.CRITIC_UPDATE)
    count = len(critic_ids)
    require(critic_ids == tuple(f"c{i}" for i in range(count)), "invalid Critic action identities", path="wan.cycle")
    records = []
    common = dict(rank=rank, device=device, trace=trace, collect_diagnostics=False,
                  diagnostics=diagnostics, runtime_cache=runtime_cache)
    def scope(ctx):
        return nullcontext() if operation_scope is None else operation_scope(ctx.operation)
    def gupdate(ctx):
        with scope(ctx):
            result = p0._run_nogan_generator_update_p0(**common,
                generator_engine=generator_engine, critic_engine=critic_engine, negative=negative,
                collect_scalars=collect_scalars, policy=policy, score_pack_parity_check=False,
                clean_runtime=clean_runtime, deferred_finite_checks=deferred_finite_checks)
            if observe_grad_norm is not None:
                result["prepared_gradient_norm"] = observe_grad_norm(generator_engine)
            return result
    def rollout(ctx):
        step = ctx.operation.logical_step
        entries = {}
        with scope(ctx) as sample:
            generated, rows, _ = p0._precompute_critic_rollouts(**common, steps=(step,),
                generator_engine=generator_engine, policy=policy, rollout_parity_check=False,
                clean_runtime=clean_runtime, entry_outputs=entries)
            require(set(generated) == set(entries) == {step} and len(rows) == 1,
                    "native single-rollout inventory changed", path="wan.cycle.rollout")
            rows[0]["batch_index"] = ctx.operation.metadata["fake_index"]
            records.extend(rows)
        if sample is not None:
            # Calibration-only storage metadata; never read latent values and
            # never include metadata inspection in the calibrated action range.
            sample.record_output((generated[step], entries[step].tensors))
        return generated[step], entries[step]
    def update(ctx):
        x0, entry = ctx.values[ctx.operation.metadata["inputs"][0]]
        with scope(ctx):
            result = p0._run_nogan_critic_precomputed(**common, step=ctx.operation.logical_step,
                x0=x0, critic_engine=critic_engine, entry=entry, collect_scalars=collect_scalars,
                deferred_finite_checks=deferred_finite_checks)
            if observe_grad_norm is not None:
                result["prepared_gradient_norm"] = observe_grad_norm(critic_engine)
            return result
    def ema_update(ctx):
        with scope(ctx):
            ema.step()
    handlers = {Kind.GENERATOR_UPDATE.value:gupdate, Kind.CRITIC_ROLLOUT.value:rollout,
                Kind.CRITIC_UPDATE.value:update, Kind.EMA_UPDATE.value:ema_update}
    runner = StaticRoleRunner("native_runtime", handlers, versions=VersionLedger(read_versions()))
    prepared = runner.prepare(graph, order=order, retain_outputs=("g", *critic_ids))
    return WanPreparedCycle(prepared, records, read_versions, critic_updates=count)
