from __future__ import annotations

from distillschedule.runtime.gan_rolephase_schedule import (
    GanRolePhasePlanner,
    RolePhaseTaskKind,
)


def test_rolephase_full5_graph_is_acyclic_and_contains_reverse_handoff() -> None:
    planner = GanRolePhasePlanner()
    tasks = planner.compile_full5()
    order = planner.topological_order(tasks)
    assert len(order) == len(tasks)
    assert "handoff:to_b" in order
    assert "handoff:to_a" in order
    assert order.index("handoff:to_b") < order.index("b:1000:produce")
    assert order.index("b:1004:commit") < order.index("handoff:to_a")


def test_next_generator_batch_does_not_wait_for_previous_critic_commit() -> None:
    tasks = {task.key: task for task in GanRolePhasePlanner().compile_full5()}
    for step in range(1001, 1005):
        produce = tasks[f"b:{step}:produce"]
        assert f"b:{step - 1}:commit" not in produce.dependencies


def test_critic_compute_remains_version_serial() -> None:
    tasks = {task.key: task for task in GanRolePhasePlanner().compile_full5()}
    for step in range(1001, 1005):
        consume = tasks[f"b:{step}:consume"]
        assert f"b:{step - 1}:commit" in consume.dependencies


def test_q2_backpressure_is_two_batches_not_per_route_ack() -> None:
    tasks = {task.key: task for task in GanRolePhasePlanner().compile_full5()}
    for step in range(1002, 1005):
        assert f"b:{step - 2}:commit" in tasks[f"b:{step}:recv_post"].dependencies
        assert (
            f"b:{step - 2}:send_complete"
            in tasks[f"b:{step}:produce"].dependencies
        )


def test_only_step1000_has_ema_commit() -> None:
    tasks = GanRolePhasePlanner().compile_full5()
    ema = [task for task in tasks if task.kind is RolePhaseTaskKind.EMA_COMMIT]
    assert len(ema) == 1
    assert ema[0].logical_step == 1000
    assert ema[0].dependencies == ("b:1000:commit",)
