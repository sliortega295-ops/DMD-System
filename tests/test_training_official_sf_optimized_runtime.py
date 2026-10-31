"""CPU contract tests, not evidence of world16 GPU qualification or speed.

The small state-machine fixture is SYNTHETIC_TEST_ONLY. The optional RMS test
executes the actual pinned Wan class and the imported measured implementation.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def test_locked_definition_digest_is_stable_on_pre312_empty_type_parameter_ast():
    import ast
    from distillschedule.training.official_profiles import _definition_sha256
    source = "class Helper:\n    def forward(self, value):\n        return value + 1\n"
    old = ast.parse(source).body[0]
    modern = ast.parse(source).body[0]
    for node in ast.walk(old):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node._fields = tuple(key for key in node._fields if key != "type_params")
            if hasattr(node, "type_params"):
                del node.type_params
    for node in ast.walk(modern):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if "type_params" not in node._fields:
                node._fields = (*node._fields, "type_params")
            node.type_params = []
    assert _definition_sha256(old, set()) == _definition_sha256(modern, set())


def test_source_bytes_still_fail_closed_after_cross_version_ast_normalization(tmp_path, monkeypatch):
    import shutil
    from distillschedule.training import official_profiles
    folder = tmp_path / "official_optimizations"
    shutil.copytree(Path(official_profiles.__file__).parent / "official_optimizations", folder)
    monkeypatch.setattr(official_profiles, "_TRAINING_ROOT", tmp_path)
    monkeypatch.setattr(official_profiles, "_SOURCE_LOCK", folder / "source_lock.json")
    official_profiles.verify_sources("self-forcing-wan-fsdp2-r6-v1")
    changed = folder / "_native/sf_phase_retention.py"
    changed.write_bytes(changed.read_bytes() + b"\n# deliberately changed source\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        official_profiles.verify_sources("self-forcing-wan-fsdp2-r6-v1")


def test_bound_loader_refuses_unverified_sources_and_unknown_helpers():
    from distillschedule.training.official_optimizations.bound_source import SourceBoundHelpers
    with pytest.raises(ValueError, match="PinnedSource"):
        SourceBoundHelpers(SimpleNamespace(namespace="wan", root=Path(".")))


def test_helper_discovery_and_admission_do_not_initialize_cuda():
    from distillschedule.training.official_optimizations.self_forcing import admission_requirements
    value = admission_requirements()
    assert value["profile"] == "self-forcing-wan-fsdp2-r6-v1"
    assert value["integration_gpu_validation"] == "NOT_RUN"
    assert value["runtime"] == "SOURCE_BOUND_NATIVE_HELPERS_WITH_CONTINUOUS_BOUNDARY_API"
    assert value["native_checkpoint_and_backward"] == "REQUIRED_UNCHANGED"
    assert not torch.cuda.is_initialized()


def test_optimized_runtime_never_accepts_plain_reference_owner_or_changes_attention():
    from distillschedule.training.official_optimizations.self_forcing import SelfForcingOptimizedRuntime
    before = torch.cuda.is_initialized()
    owner = SimpleNamespace(optimization_profile=None)
    with pytest.raises(ValueError, match="explicit optimized owner profile"):
        SelfForcingOptimizedRuntime(owner, SimpleNamespace())
    assert torch.cuda.is_initialized() is before


def test_canary_clock_is_local_but_training_and_parameter_clocks_are_untouched():
    from distillschedule.training.official_optimizations.self_forcing import _CanaryClock
    owner = SimpleNamespace(step=35, generator_updates=7, critic_updates=35)
    view = _CanaryClock(owner)
    assert view.step == 0 and owner.step == 35
    owner.step = 40
    assert view.step == 5 and owner.step == 40
    assert view.generator_updates == 7 and view.critic_updates == 35


def test_partial_patch_rollback_restores_methods_and_abort_hooks_not_tensors():
    from distillschedule.training.official_optimizations.self_forcing import _PatchSnapshot
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    owner = SimpleNamespace(model=model, generator_optimizer=optimizer, critic_optimizer=optimizer)
    api = SimpleNamespace(wan=SimpleNamespace(rope_apply=lambda x: x, flash_attention=lambda x: x),
                          causal=SimpleNamespace(causal_rope_apply=lambda x: x, attention=lambda x: x))
    before = model.weight.detach().clone()
    forward = model.forward
    rope = api.wan.rope_apply
    scope = _PatchSnapshot(owner, api)
    model.forward = lambda x: x
    model.register_forward_hook(lambda *args: None)
    optimizer.register_step_pre_hook(lambda *args: None)
    api.wan.rope_apply = lambda x: x + 1
    scope.restore(abort=True)
    assert model.forward == forward and api.wan.rope_apply is rope
    assert not model._forward_hooks and not optimizer._optimizer_step_pre_hooks
    torch.testing.assert_close(before, model.weight, rtol=0, atol=0)


def test_actual_upstream_rms_runs_without_public_wan_alias_and_keeps_canaries():
    source = os.environ.get("SELF_FORCING_SOURCE_ROOT")
    if not source:
        pytest.skip("set SELF_FORCING_SOURCE_ROOT for actual upstream class execution")
    from distillschedule.training.self_forcing_source import original
    from distillschedule.training.official_optimizations.bound_source import SourceBoundHelpers
    api = original(source)
    before = {name: value for name, value in sys.modules.items() if name == "wan" or name.startswith("wan.")}
    modules = SourceBoundHelpers(api.source)
    with pytest.raises(ValueError, match="unknown helper"):
        modules.module("not_a_helper")
    helper_module = modules.module("sf_nograd_rms")
    trainer = SimpleNamespace(step=0, model=SimpleNamespace(**{
        role: torch.nn.Sequential(api.wan.WanRMSNorm(4)).bfloat16()
        for role in ("generator", "fake_score", "real_score")}))

    def cpu_op(x, weight, eps):
        # Exact pinned WanRMSNorm._norm(x.float()).type_as(x) boundary.
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

    helper = helper_module.NoGradRMS(trainer, op=cpu_op, device_type="cpu")
    try:
        x = torch.linspace(-1, 1, 8).reshape(1, 2, 4).bfloat16()
        with torch.no_grad():
            first = trainer.model.generator(x)
            trainer.step = 10
            second = trainer.model.generator(x)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert helper.receipt()["calls"]["generator"] == 2
        assert helper.receipt()["canary_keys"] == 1
        assert helper_module.__name__.startswith("_distillschedule_sf_helpers_")
        assert {name: value for name, value in sys.modules.items() if name == "wan" or name.startswith("wan.")} == before
        modules.validate()
    finally:
        helper.close()
    assert not torch.cuda.is_initialized()


def test_host_geometry_indices_preserve_kv_geometry_restore_and_multiple_pipelines():
    from distillschedule.training.self_forcing_source import GeometryCache
    from distillschedule.training.official_optimizations.self_forcing import HostGeometryIndices
    attention = torch.nn.Identity()
    native = SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)], patch_size=(1, 2, 2),
                             num_heads=2, dim=8, text_len=4, local_attn_size=-1)
    model = SimpleNamespace(generator=SimpleNamespace(model=native), inference_pipeline=None)

    def initialize():
        pipeline = GeometryCache()
        pipeline.generator = model.generator
        pipeline.kv_cache1 = pipeline.kv_cache2 = pipeline.crossattn_cache = None
        pipeline.configure_geometry(shape=[1, 21, 16, 4, 8])
        model.inference_pipeline = pipeline

    model._initialize_inference_pipeline = initialize
    trainer = SimpleNamespace(model=model)
    scope = HostGeometryIndices(trainer)
    pipelines = []
    try:
        for _ in range(2):
            model._initialize_inference_pipeline()
            pipeline = model.inference_pipeline
            pipeline._initialize_kv_cache(1, torch.bfloat16, torch.device("cpu"))
            assert pipeline.kv_cache1[0]["k"].shape == (1, 168, 2, 4)
            assert pipeline.kv_cache1[0]["k"].dtype is torch.bfloat16
            assert pipeline.kv_cache1[0]["global_end_index"].device.type == "cpu"
            pipelines.append(pipeline)
        assert scope.receipt()["pipeline_initializations"] == 2
        assert scope.receipt()["index_tensors_constructed"] == 4
    finally:
        scope.close()
    assert model._initialize_inference_pipeline is initialize
    assert all(p._initialize_kv_cache.__func__ is GeometryCache._initialize_kv_cache for p in pipelines)


class _FixtureTrainer:
    """SYNTHETIC_TEST_ONLY lifecycle fixture with real autograd and AdamW."""
    def __init__(self, *, return_alias=False):
        self.step = 0
        self.net = torch.nn.Linear(2, 2)
        self.generator_optimizer = torch.optim.AdamW(self.net.parameters(), lr=.001)
        self.critic_optimizer = torch.optim.AdamW(self.net.parameters(), lr=.001)
        self.model = SimpleNamespace(inference_pipeline=None, fake_score=torch.nn.Identity())
        self.model._consistency_backward_simulation = self.simulate
        self.return_alias = return_alias

    def simulate(self):
        cache = {"k": torch.ones(1, 2), "v": torch.zeros(1, 2),
                 "global_end_index": torch.zeros(1, dtype=torch.int64),
                 "local_end_index": torch.zeros(1, dtype=torch.int64)}
        self.model.inference_pipeline = SimpleNamespace(kv_cache1=[cache], kv_cache2=None,
                                                        crossattn_cache=[dict(k=torch.zeros(1, 2))])
        return cache["k"] if self.return_alias else torch.ones(1, 2)

    def fwdbwd_one_step(self, batch, train_generator):
        optimizer = self.generator_optimizer if train_generator else self.critic_optimizer
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_generator):
            latent = self.model._consistency_backward_simulation()
        if not train_generator:
            assert self.model.inference_pipeline.kv_cache1 is None
        loss = self.net(latent).square().sum()
        loss.backward()
        return {"loss": loss.detach()}


def test_early_kv_release_is_after_rollout_for_c_and_after_commit_for_g():
    from distillschedule.training.official_optimizations.self_forcing import EarlyConsumedKV
    trainer = _FixtureTrainer()
    original = trainer.fwdbwd_one_step
    simulation = trainer.model._consistency_backward_simulation
    before = deepcopy(trainer.net.state_dict())
    scope = EarlyConsumedKV(trainer)
    try:
        trainer.fwdbwd_one_step({}, True)
        assert trainer.model.inference_pipeline.kv_cache1 is not None
        trainer.generator_optimizer.step()
        assert trainer.model.inference_pipeline.kv_cache1 is None
        trainer.fwdbwd_one_step({}, False)
        trainer.critic_optimizer.step()
        row = scope.receipt()
        assert row["generator_commit_releases"] == row["critic_early_releases"] == 1
        assert row["pending"] is None and row["host_tensor_value_reads"] == 0
        assert any(not torch.equal(value, trainer.net.state_dict()[name]) for name, value in before.items())
    finally:
        scope.close()
    assert trainer.fwdbwd_one_step == original
    assert trainer.model._consistency_backward_simulation == simulation
    assert not trainer.generator_optimizer._optimizer_step_post_hooks
    assert not trainer.critic_optimizer._optimizer_step_post_hooks


def test_early_kv_alias_and_incomplete_backward_fail_closed_and_restore():
    from distillschedule.training.official_optimizations.self_forcing import EarlyConsumedKV
    trainer = _FixtureTrainer(return_alias=True)
    original = trainer.fwdbwd_one_step
    scope = EarlyConsumedKV(trainer)
    try:
        with pytest.raises(ValueError, match="KV_STORAGE_ALIAS"):
            trainer.fwdbwd_one_step({}, False)
        with pytest.raises(ValueError, match="INCOMPLETE"):
            scope.receipt()
    finally:
        scope.close(abort=True)
    assert trainer.fwdbwd_one_step == original
    assert trainer.model.inference_pipeline.kv_cache1 is None


def test_dynamic_phase_receipt_not_fixed_to_two_cycles_or_step_ten():
    from distillschedule.training.official_optimizations.self_forcing import phase_boundary
    trainer = SimpleNamespace(step=35, generator_updates=7, critic_updates=35)
    calls = []
    phase = SimpleNamespace(trainer=trainer, active=False, closed=False, roles={"generator": None, "fake_score": None},
        live={"generator": True, "fake_score": False}, counts={"generator_commits": 7, "critic_commits": 35},
        record={"contract_failures": 0}, bytes={"generator": 16, "fake_score": 16}, max_bytes=100,
        g_version=7, c_version=35)
    phase.assert_released = lambda: None
    phase._release = lambda role: (calls.append(role), phase.live.__setitem__(role, False))
    phase._master_check = lambda role: None
    receipt = phase_boundary(phase)
    assert receipt["native_step"] == 35 and receipt["generator_version"] == 7
    assert receipt["live_at_boundary"] == {"generator": False, "fake_score": False}
    assert receipt["closed"] is False
    assert calls == ["generator", "fake_score"]
