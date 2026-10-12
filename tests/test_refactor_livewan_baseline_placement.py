"""Placement contracts and actual frozen-parameter staging behavior."""
from __future__ import annotations

import copy
import os
from types import SimpleNamespace
import unittest

from distillschedule.runtime.livewan_baseline_placement import (
    BaselinePlacement, FrozenTeacherCPUStaging, apply_contract,
    check_collective_trace, from_plan,
)


def plan():
    return {"metadata": {"execution_mode": "baseline"},
            "workload": {"shape": {"batch_size": 16}},
            "training_backends": {role: {"kind": "native"}
                for role in ("generator", "score", "teacher")}}


def test_default_no_mutation():
    value = plan(); before = copy.deepcopy(value)
    apply_contract(value, mode="baseline", name="legacy-hybrid", teacher_offload="off")
    assert value == before


def test_fsdp_configuration_and_no_cpu_offload():
    for name in ("world16-fsdp", "world16-fsdp-student-retained"):
        value = plan()
        spec = apply_contract(value, mode="baseline", name=name, teacher_offload="off")
        assert spec == from_plan(value)
        assert all(row["kind"] == "fsdp" for row in value["training_backends"].values())
        assert all(row["parameter_offload"] == "none" for row in value["training_backends"].values())
        assert spec.student_reshard == (name == "world16-fsdp")


def test_invalid_options_fail_closed():
    for name, offload in (("bad", "off"), ("world16-fsdp", "cpu-block"),
                          ("legacy-hybrid", "cpu-block")):
        with unittest.TestCase().assertRaises(ValueError): BaselinePlacement(name, offload)
    with unittest.TestCase().assertRaises(ValueError):
        apply_contract(plan(), mode="pipeline", name="full-resident", teacher_offload="off")


def test_unequal_rollout_collectives_rejected_before_construction():
    rows = [SimpleNamespace(logical_step=0, new_blocks=i) for i in (4, 5, 6)]
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "FAIL_FSDP_COLLECTIVE_TRACE"):
        check_collective_trace(BaselinePlacement("world16-fsdp"), rows)
    check_collective_trace(BaselinePlacement("world16-fsdp-student-retained"), rows)
    check_collective_trace(BaselinePlacement("full-resident"), rows)


def test_frozen_staging_runs_forward_restores_weights_and_no_extra_calls():
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("Torch unavailable; run on the validated training environment")
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.causal_layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            for name in ("causal_head", "patch_embedding", "text_embedding",
                         "time_embedding", "time_projection"):
                setattr(self, name, torch.nn.Linear(4, 4))
        def forward(self, x):
            for layer in self.causal_layers: x = layer(x)
            return self.causal_head(x)
    torch.manual_seed(13)
    model = Teacher().requires_grad_(False)
    reference = copy.deepcopy(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(2, 4)
    staging = FrozenTeacherCPUStaging(torch, model, device="cpu",
        attach_causal_modules=lambda m: m, pin_memory=False)
    for _ in range(3):
        assert torch.equal(model(x), reference(x))
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    receipt = staging.receipt()
    assert receipt["unit_calls"] == 6
    assert receipt["active_units_at_receipt"] == 0
    assert receipt["max_active_units"] == 1
    assert receipt["d2h_parameter_writeback_bytes"] == 0
    assert receipt["trainable_parameters_offloaded"] == 0
    model.requires_grad_(True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "frozen"):
        FrozenTeacherCPUStaging(torch, model, device="cpu", attach_causal_modules=lambda m: m,
                               pin_memory=False)


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items()) if name.startswith("test_"))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
