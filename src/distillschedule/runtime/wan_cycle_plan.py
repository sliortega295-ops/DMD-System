"""Bind prepared actions to the existing exact-trace Wan native functions.

The dispatcher is shared with other native adapters; there is no alternate
model, loss, optimizer or rollout implementation in this module.
"""
from contextlib import nullcontext
from typing import Any

from distillschedule.core.contracts import require
from distillschedule.core.enums import OperationKind as Kind
from .cycle_plan import build_cycle_graph, rollout_window_order
from .role_runner import StaticRoleRunner
from .version_sync import VersionLedger


class WanPreparedCycle:
    def __init__(self, prepared, records, read_versions):
        self.prepared, self.records, self.read_versions = prepared, records, read_versions
        self.native_validation_error = None

    def run(self):
        results = self.prepared.run()
        if self.read_versions() != self.prepared.receipt()["final_versions"]:
            self.native_validation_error = "native engine commits disagree with dispatched plan"
            require(False, self.native_validation_error, path="wan.cycle.versions")
        return {"generator": results["g"], "rollouts": self.records,
                "critic": [results[f"c{i}"] for i in range(5)]}

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
    runner = StaticRoleRunner("native_runtime", handlers, versions=VersionLedger(versions()))
    prepared = runner.prepare(graph, order=rollout_window_order(graph, max_live_rollouts=max_live_rollouts),
                              retain_outputs=("g", *(f"c{i}" for i in range(5))))
    return WanPreparedCycle(prepared, records, versions)
