"""Construction-only Wan compatibility; no weights or CUDA are loaded."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


def _check_wan_worker_loader_keyword_contract():
    tree = ast.parse((ROOT / "runtime/critic_pool_helios.py").read_text())
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "build_helios_pool_model")
    supported = {n.arg for n in loader.args.kwonlyargs}
    for name in ("probe_wan13b_fair_world16.py", "probe_wan13b_upper_bound_world16.py"):
        worker = ast.parse((ROOT / "scripts/distillschedule" / name).read_text())
        calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)
                 and ((isinstance(n.func, ast.Name) and n.func.id == loader.name)
                      or (isinstance(n.func, ast.Attribute) and n.func.attr == loader.name))]
        assert calls, name
        for call in calls:
            assert {k.arg for k in call.keywords} <= supported, name


def _model(torch, device="meta"):
    model = torch.nn.Module()
    for name in ("patch_short", "patch_mid", "patch_long"):
        model.add_module(name, torch.nn.Conv3d(2, 3, (1, 2, 2), device=device))
    return model


class WanLoaderContract(unittest.TestCase):
    def test_worker_keywords(self):
        _check_wan_worker_loader_keyword_contract()


class WanLoaderCPU(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable; run CPU tensor gates in validated remote environment")
        from runtime.critic_pool_helios import _materialize_missing_pyramid_patches
        self.torch = torch
        self.materialize = _materialize_missing_pyramid_patches

    def test_opt_in_matches_historical_seeded_constructor_and_preserves_rng(self):
        torch, materialize = self.torch, self.materialize
        model = _model(torch)
        rng = torch.get_rng_state().clone()
        assert materialize(model, 1234) == ["patch_short", "patch_mid", "patch_long"]
        assert torch.equal(torch.get_rng_state(), rng)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            expected = _model(torch, device="cpu").to(dtype=torch.bfloat16)
        for name, value in model.named_parameters():
            assert value.device.type == "cpu" and value.dtype == torch.bfloat16
            assert torch.equal(value, expected.get_parameter(name)), name


    def test_existing_checkpoint_parameters_are_never_reinitialized(self):
        torch, materialize = self.torch, self.materialize
        model = _model(torch, "cpu")
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        assert materialize(model, None) == []
        assert materialize(model, 1234) == []
        assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())


    def test_default_does_not_fill_missing_weights(self):
        torch, materialize = self.torch, self.materialize
        with self.assertRaisesRegex(RuntimeError, "left meta parameters"):
            materialize(_model(torch), None)


    def test_unrelated_missing_weights_fail_closed(self):
        torch, materialize = self.torch, self.materialize
        model = _model(torch)
        model.add_module("unexpected", torch.nn.Linear(2, 2, device="meta"))
        with self.assertRaisesRegex(RuntimeError, "unexpected.weight"):
            materialize(model, 1234)


    def test_partial_pyramid_materialization_fails(self):
        torch, materialize = self.torch, self.materialize
        model = _model(torch)
        model.patch_short.bias = torch.nn.Parameter(torch.zeros(3))
        with self.assertRaisesRegex(AssertionError, "partially materialized"):
            materialize(model, 1234)


    def test_helios_loader_opt_in_defaults_and_residual_capability_gate(self):
        from runtime.critic_pool_helios import build_helios_pool_model
        assert inspect.signature(build_helios_pool_model).parameters["missing_pyramid_patch_seed"].default is None
        assert "validate_chunked_residual_runtime(" in inspect.getsource(build_helios_pool_model)


if __name__ == "__main__":
    unittest.main()
