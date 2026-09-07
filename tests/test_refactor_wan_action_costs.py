"""Component reuse is tested independently of measured GPU durations."""
from dataclasses import replace

import pytest

from distillschedule.search.wan_action_costs import action_graph, component_contract
from distillschedule.search.wan_implementation import WanImplementation
from distillschedule.search.plan_ir import validate_versions


WORK = {"input_layouts": [{"shape": [1, 16, 9, 48, 80], "dtype": "bfloat16"}],
        "dit_inventory": {"call_count": 6, "stage_call_count": [2, 2, 2]},
        "logical_batch": 16, "physical_batch": 1}


def op(key):
    return next(x for x in action_graph().operations if x.operation_id == key)


def test_two_cycles_have_actual_actions_and_version_dependencies():
    graph = action_graph()
    assert len(graph.operations) == 32
    assert graph.metadata["cycle_ends"] == {"1": "cycle1.e4", "2": "cycle2.e4"}
    assert graph.metadata["final_versions"] == {"generator": 4, "critic": 20, "ema": 20}
    validate_versions(graph)
    for window in range(1, 6):
        ordered = action_graph(window)
        validate_versions(ordered)
        order = ordered.topological_order()
        assert order.index("cycle1.c0") == window + 1
        assert order.index("cycle2.g") == 16


def test_cost_reused_across_windows_and_occurrences_not_recipe_ids():
    a = WanImplementation("resident", 2048, True, rollout_window=1)
    b = replace(a, rollout_window=5)
    first = component_contract(op("cycle2.r0"), a, WORK)
    second = component_contract(op("cycle2.r1"), b, WORK)
    assert first.digest == second.digest
    assert a.id not in str(first.to_dict())
    assert "cycle2.r0" not in str(first.to_dict())
    assert "rollout_window" not in str(first.to_dict())


def test_kernel_cost_identity_includes_only_consumed_choices():
    a = WanImplementation("resident", 2048, False, rollout_window=3)
    b = replace(a, lora_epilogue=True)
    for key in ("cycle2.c0", "cycle2.e0"):
        assert component_contract(op(key), a, WORK).digest == component_contract(op(key), b, WORK).digest
    assert component_contract(op("cycle2.r0"), a, WORK).digest != component_contract(op("cycle2.r0"), b, WORK).digest
    assert component_contract(op("cycle2.e0"), a, WORK).digest == component_contract(op("cycle2.e0"), replace(a,chunk_tokens=512), WORK).digest


def test_cold_state_work_and_placement_cannot_reuse_costs():
    a = WanImplementation("resident", 512, False, rollout_window=3)
    base = component_contract(op("cycle2.r0"), a, WORK).digest
    assert component_contract(op("cycle1.r0"), a, WORK).digest != base
    assert component_contract(op("cycle2.r0"), replace(a,placement="fsdp"), WORK).digest != base
    changed = {**WORK,"dit_inventory":{"call_count":5,"stage_call_count":[2,2,1]}}
    assert component_contract(op("cycle2.r0"), a, changed).digest != base


def test_unbound_or_legacy_operation_is_not_a_component():
    a = WanImplementation("resident", 2048, True, rollout_window=3)
    with pytest.raises(ValueError):
        component_contract(op("cycle2.r0"), replace(a,rollout_window=None), WORK)
    with pytest.raises(ValueError):
        component_contract(op("cycle2.r0"), a, {})
