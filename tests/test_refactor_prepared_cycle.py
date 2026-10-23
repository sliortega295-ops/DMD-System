"""CPU infrastructure checks, not evidence of GPU model equivalence or speed."""
from dataclasses import replace

import pytest

from distillschedule.core.enums import OperationKind
from distillschedule.core.graph import ExecutionGraph
from distillschedule.runtime.cycle_plan import build_cycle_graph, rollout_window_order
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger


INITIAL = dict(generator=2, critic=10, ema=10)


def graph(ema_after=(0, 1, 2, 3, 4)):
    return build_cycle_graph(source_steps=(10, 11, 12, 13, 14),
                             initial_versions=INITIAL, ema_after=ema_after)


def runner(events, *, failing=None):
    def call(ctx):
        name = ctx.operation.operation_id
        if name == failing:
            raise RuntimeError("native failure")
        if name.startswith("c"):
            assert ctx.values[f"r{name[1:]}"] == ("latent", int(name[1:]))
        events.append((name, tuple(ctx.values)))
        return ("latent", int(name[1:])) if name.startswith("r") else name
    return StaticRoleRunner("native_runtime", {k.value: call for k in (
        OperationKind.GENERATOR_UPDATE, OperationKind.CRITIC_ROLLOUT,
        OperationKind.CRITIC_UPDATE, OperationKind.EMA_UPDATE)},
        versions=VersionLedger(INITIAL))


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ema_after", [(), (0,), (0, 1, 2, 3, 4)])
def test_generated_order_executes_exactly_once_and_releases_latents(window, ema_after):
    g = graph(ema_after)
    order = rollout_window_order(g, max_live_rollouts=window)
    events = []
    r = runner(events)
    prepared = r.prepare(g, order=order, retain_outputs=tuple(f"c{i}" for i in range(5)))
    result = prepared.run()
    assert [e[0] for e in events] == list(order)
    assert len(events) == 11 + len(ema_after)
    assert set(result) == {f"c{i}" for i in range(5)}
    assert prepared.receipt()["status"] == "PASS"
    assert prepared.receipt()["executed_operations"] == list(order)
    assert prepared.receipt()["peak_live_inputs"] == window
    assert r.versions.snapshot().values == dict(generator=3, critic=15, ema=10+len(ema_after))
    # No returned latent survives the final native consumer.
    for name, live in events:
        if name.startswith("r") and int(name[1:]) >= window:
            assert f"r{int(name[1:])-window}" not in live


def test_window_three_is_generated_not_a_registered_recipe():
    assert rollout_window_order(graph(()), max_live_rollouts=3) == (
        "g", "r0", "r1", "r2", "c0", "r3", "c1", "r4", "c2", "c3", "c4")


@pytest.mark.parametrize("window", [0, 6, True, 1.5])
def test_invalid_window_fails_before_execution(window):
    with pytest.raises(ValueError):
        rollout_window_order(graph(), max_live_rollouts=window)


def test_invalid_order_handler_or_versions_rejected_before_side_effects():
    g = graph()
    events = []
    r = runner(events)
    order = rollout_window_order(g, max_live_rollouts=5)
    for bad in (order[:-1], order + ("g",), tuple(reversed(order))):
        with pytest.raises(ValueError):
            r.prepare(g, order=bad)
    r.handlers.pop(OperationKind.CRITIC_UPDATE.value)
    with pytest.raises(ValueError, match="handler"):
        r.prepare(g, order=order)
    assert events == []


def test_prepared_plan_is_one_shot_and_failure_does_not_advance_commit():
    r = runner([], failing="c1")
    prepared = r.prepare(graph(()), order=rollout_window_order(graph(()), max_live_rollouts=1))
    with pytest.raises(RuntimeError, match="native failure"):
        prepared.run()
    assert r.versions.get("critic") == 11
    assert prepared.receipt()["status"] == "FAIL"
    assert prepared.receipt()["failed_operation"] == "c1"
    with pytest.raises(ValueError, match="one-shot"):
        prepared.run()


def test_missing_input_dependency_cannot_be_hidden_by_selected_order():
    g = graph(())
    bad = replace(g.operations[-1], metadata={**g.operations[-1].metadata, "inputs": ("missing",)})
    g = ExecutionGraph((*g.operations[:-1], bad), g.dependencies, g.metadata)
    with pytest.raises(ValueError, match="input"):
        runner([]).prepare(g, order=rollout_window_order(graph(()), max_live_rollouts=5))


def test_no_implicit_cross_role_execution_or_condition_fallback():
    g = graph(())
    for op in (replace(g.operations[0], role="remote_pool"), replace(g.operations[0], condition="maybe")):
        bad = ExecutionGraph((op, *g.operations[1:]), g.dependencies, g.metadata)
        with pytest.raises(ValueError):
            runner([]).prepare(bad, order=rollout_window_order(g, max_live_rollouts=5))


def test_external_version_change_after_preparation_fails_before_native_call():
    events = []
    r = runner(events)
    prepared = r.prepare(graph(), order=rollout_window_order(graph(), max_live_rollouts=5))
    r.versions.advance("generator")
    with pytest.raises(ValueError, match="version"):
        prepared.run()
    assert events == []


def test_completed_receipt_is_snapshot_not_later_runner_state():
    r = runner([])
    p = r.prepare(graph(), order=rollout_window_order(graph(), max_live_rollouts=3))
    p.run()
    r.versions.advance("critic")
    assert p.receipt()["final_versions"]["critic"] == 15
