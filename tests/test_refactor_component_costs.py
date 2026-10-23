"""Component reuse is identity-bound; no synthetic costs are GPU evidence."""
import copy
from dataclasses import replace

import pytest

from distillschedule.search.cost_model import ComponentContract, CostModel, MissingCalibration, OperationCost


BINDING = {"source_head": "a"*40, **{k: "b"*64 for k in (
    "hardware_fingerprint", "environment_fingerprint", "asset_fingerprint",
    "semantic_fingerprint", "precision_fingerprint")}}


def component(**kwargs):
    return ComponentContract(family="wan.critic_backward_commit",
        work={"shape": [1,16,9,48,80], "forward_calls": 1, "backward_graphs": 1, "updates": 1},
        precision={"base": "bf16", "lora": "fp32", "reduce": "fp32"},
        placement={"mesh": [16], "kind": "fsdp", "ranks": list(range(16))},
        implementation={"critic_norm": "native", "optimizer": "adamw"},
        context={"optimizer_materialized": True, "fsdp_entry": "resharded",
                 "overlap": "serial", "retained_input_accounting": "separate_lifetimes"}, **kwargs)


def test_equivalent_occurrences_reuse_one_measured_component_key():
    a = component()
    # No absolute cycle number or unrelated G-kernel assignment is in the key.
    b = ComponentContract.from_dict(copy.deepcopy(a.to_dict()))
    assert a.digest == b.digest
    r = OperationCost(a.digest, (20,30), 10, {"kind": "CPU_FIXTURE_ONLY", "component_contract": a.to_dict()})
    costs = CostModel(BINDING, (r,), scope="CPU_FIXTURE_ONLY")
    assert costs.get_component(a).duration_ns == costs.get_component(b).duration_ns == 25


@pytest.mark.parametrize("field, key, value", [
    ("work", "shape", [2,16,9,48,80]),
    ("work", "forward_calls", 2),
    ("precision", "lora", "bf16"),
    ("placement", "mesh", [2,8]),
    ("implementation", "optimizer", "fused_adamw"),
    ("context", "optimizer_materialized", False),
    ("context", "overlap", "concurrent_nccl"),
])
def test_local_or_interaction_change_requires_new_calibration(field, key, value):
    a = component()
    changed = replace(a, **{field: {**getattr(a, field), key: value}})
    assert changed.digest != a.digest
    costs = CostModel(BINDING, (OperationCost(a.digest,(10,),0,{"component_contract":a.to_dict()}),),
                      scope="CPU_FIXTURE_ONLY")
    with pytest.raises(MissingCalibration):
        costs.get_component(changed)


def test_matching_hash_without_matching_component_receipt_is_not_enough():
    a = component()
    costs = CostModel(BINDING, (OperationCost(a.digest,(10,),0,{}),), scope="CPU_FIXTURE_ONLY")
    with pytest.raises(ValueError, match="component"):
        costs.get_component(a)


def test_incomplete_or_recipe_identity_rejected():
    raw = component().to_dict()
    for key in ("work", "precision", "placement", "implementation", "context"):
        with pytest.raises(ValueError):
            ComponentContract.from_dict({**raw, key: {}})
    for key in ("candidate_id", "cycle", "operation_id"):
        with pytest.raises(ValueError):
            ComponentContract.from_dict({**raw, key: "recipe"})
