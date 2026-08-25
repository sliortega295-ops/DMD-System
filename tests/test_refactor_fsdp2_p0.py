from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.fsdp2_p0 import (
    BatchedOperationTrace,
    StaticFSDP2P0Policy,
    build_p0_execution_order,
    chunked,
    estimate_rollout_collective_groups,
    make_batched_loaded_entry,
    mesh_shape_for_policy,
    pack_score_conditions,
    split_packed_prediction,
    validate_rollout_entries,
)
from distillschedule.runtime.p0_parity import tensor_parity_metrics


class FakeTrace:
    def __init__(self, step: int, marker: float) -> None:
        self.step = step
        self.trace_hash = f"trace-{step}"
        self.manifest = {"contract": {"dtype": "bfloat16"}}
        self._marker = marker

    def operation(self, key: str):
        if key == f"step/{self.step}/branch":
            return {"key": key, "operation": "stage3_branch", "control": {"global_step": self.step}}
        assert key == f"step/{self.step}/critic/rollout"
        return {
            "key": key,
            "operation": "generator_rollout",
            "control": {
                "global_step": self.step,
                "call_count": 6,
                "pyramid_stage_call_sequence": [
                    {"stage": 0, "call_count": 2, "denoising_exit": 1},
                    {"stage": 1, "call_count": 1, "denoising_exit": 0},
                    {"stage": 2, "call_count": 3, "denoising_exit": 2},
                ],
            },
        }

    def tensor(self, key: str, *, clone: bool = True):
        expected = f"step/{self.step}/critic/rollout_noise"
        assert key == expected
        value = torch.full((1, 2), self._marker)
        return value.clone() if clone else value


def fake_entry(step: int, marker: float):
    tensors = {
        "entry_noise": torch.full((1, 2), marker),
        "text_embedding": torch.full((1, 3, 4), marker),
        "continuous_history": torch.full((1, 1, 1), marker),
        "history_short": torch.full((1, 1, 1), marker),
        "history_mid": torch.full((1, 1, 1), marker),
        "history_long": torch.full((1, 1, 1), marker),
        "indices_hidden_states": torch.arange(2),
        "indices_history_short": torch.arange(2),
        "indices_history_mid": torch.arange(2),
        "indices_history_long": torch.arange(2),
    }
    return SimpleNamespace(
        pool_entry=SimpleNamespace(
            logical_step=step,
            rollout_operation=f"step/{step}/critic/rollout",
            identity_sha256=f"identity-{step}",
        ),
        operation_trace=FakeTrace(step, marker),
        tensors=tensors,
    )


def test_policy_and_mesh_contract() -> None:
    conservative = StaticFSDP2P0Policy()
    assert conservative.mesh_policy == "world16-fsdp"
    assert conservative.rollout_batch_size == 1
    assert conservative.pack_real_score_pair is False

    policy = StaticFSDP2P0Policy(
        mesh_policy="node-hsdp", rollout_batch_size=5, pack_real_score_pair=True
    )
    assert policy.id == "node-hsdp-rollout-b5-realpair-1"
    assert mesh_shape_for_policy("world16-fsdp") == (16,)
    assert mesh_shape_for_policy("node-hsdp") == (2, 8)
    with pytest.raises(ValueError):
        StaticFSDP2P0Policy(rollout_batch_size=3)
    with pytest.raises(ValueError):
        StaticFSDP2P0Policy(rollout_backend="vllm-omni")
    warmed = StaticFSDP2P0Policy(
        mesh_policy="node-hsdp", optimizer_dispatch_warmup=True
    )
    assert warmed.id == "node-hsdp-rollout-b1-realpair-0-optwarm"
    with pytest.raises(ValueError, match="node-HSDP"):
        StaticFSDP2P0Policy(optimizer_dispatch_warmup=True)
    fused = StaticFSDP2P0Policy(
        mesh_policy="node-hsdp",
        generator_nograd_residual_fusion=True,
        exact_grad_residual_fusion=True,
    )
    assert fused.id.endswith("optwarm") is False
    assert fused.id.endswith("residualfusion-gradresidualfusion")
    with pytest.raises(ValueError, match="Exact grad"):
        StaticFSDP2P0Policy(
            rollout_batch_size=2,
            exact_grad_residual_fusion=True,
        )
    modulated_norm = StaticFSDP2P0Policy(
        exact_modulated_norm_epilogue=True,
    )
    assert modulated_norm.id.endswith("modnormepilogue")
    with pytest.raises(ValueError, match="modulated-norm"):
        StaticFSDP2P0Policy(
            rollout_batch_size=2,
            exact_modulated_norm_epilogue=True,
        )
    materialization = StaticFSDP2P0Policy(
        exact_modulated_norm_epilogue=True,
        exact_modulated_norm_materialization=True,
    )
    assert materialization.id.endswith("modnormmaterialization")
    with pytest.raises(ValueError, match="requires the exact epilogue"):
        StaticFSDP2P0Policy(
            exact_modulated_norm_materialization=True,
        )
    checkpoint_bypass = StaticFSDP2P0Policy(checkpoint_bypass_stride=4)
    assert checkpoint_bypass.id.endswith("ckptbypass10")
    with pytest.raises(ValueError, match="checkpoint_bypass_stride"):
        StaticFSDP2P0Policy(checkpoint_bypass_stride=5)


def test_execution_order_precomputes_all_rollouts_before_critic() -> None:
    order = build_p0_execution_order(
        (1000, 1001, 1002, 1003, 1004),
        rollout_batch_size=2,
        ema_steps=(1000,),
    )
    assert order[:4] == (
        "generator_update",
        "generator_rollout_batch:1000,1001",
        "generator_rollout_batch:1002,1003",
        "generator_rollout_batch:1004",
    )
    first_critic = order.index("critic_update:1000")
    assert all("generator_rollout_batch" in item for item in order[1:first_critic])
    assert order[first_critic + 1] == "ema_update:1000"


def test_collective_group_estimate_tracks_true_batching() -> None:
    estimate = estimate_rollout_collective_groups(
        physical_calls_per_rollout=6, logical_rollouts=5, batch_size=5
    )
    assert estimate["baseline_forward_groups"] == 30
    assert estimate["optimized_forward_groups"] == 6
    assert estimate["ideal_group_reduction_fraction"] == pytest.approx(0.8)
    assert chunked(tuple(range(5)), 2) == ((0, 1), (2, 3), (4,))


def test_batched_trace_stacks_values_and_rewrites_step_prefixes() -> None:
    entries = (fake_entry(10, 1.0), fake_entry(11, 2.0))
    validation = validate_rollout_entries(entries)
    assert validation["call_count"] == 6
    assert validation["global_steps"] == [10, 11]
    batched = make_batched_loaded_entry(entries)
    assert tuple(batched.tensors["entry_noise"].shape) == (2, 2)
    assert batched.tensors["entry_noise"].tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert batched.tensors["indices_hidden_states"].tolist() == [0, 1]
    trace = BatchedOperationTrace(entries, synthetic_prefix="batched/critic")
    operation = trace.operation("batched/critic/rollout")
    assert operation["control"]["call_count"] == 6
    # global_step is retained only as first-member metadata. Real correctness
    # runs must compare every B2/B5 row against its original-step B1 result.
    assert operation["control"]["global_step"] == 10
    tensor = trace.tensor("batched/critic/rollout_noise")
    assert tensor.tolist() == [[1.0, 1.0], [2.0, 2.0]]


def test_rollout_global_step_may_live_on_branch_metadata() -> None:
    entries = (fake_entry(10, 1.0), fake_entry(11, 2.0))
    originals = [entry.operation_trace.operation for entry in entries]

    def rollout_without_step(index: int):
        original = originals[index]

        def operation(key: str):
            value = original(key)
            value["control"].pop("global_step", None)
            if key == f"step/{entries[index].pool_entry.logical_step}/branch":
                return {"control": {"global_step": entries[index].pool_entry.logical_step}}
            return value

        return operation

    for index, entry in enumerate(entries):
        entry.operation_trace.operation = rollout_without_step(index)

    validation = validate_rollout_entries(entries)
    assert validation["global_steps"] == [10, 11]


def test_rollout_batchability_fails_closed_on_schedule_drift() -> None:
    good = fake_entry(10, 1.0)
    bad = fake_entry(11, 2.0)
    original = bad.operation_trace.operation

    def changed(key: str):
        value = original(key)
        value["control"]["call_count"] = 7
        return value

    bad.operation_trace.operation = changed
    with pytest.raises(ValueError, match="controls differ"):
        validate_rollout_entries((good, bad))


def test_real_score_pair_packing_preserves_one_request_per_row() -> None:
    batch = {
        "hidden_states": torch.randn(1, 4, 2),
        "timestep": torch.tensor([500.0]),
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "indices_hidden_states": torch.arange(4),
        "latents_history_short": torch.randn(1, 2, 2),
        "generated_x0": torch.randn(1, 4, 2),
        "sigmas": torch.linspace(1, 0, 10),
    }
    cond = torch.randn(1, 3, 8)
    uncond = torch.randn(1, 3, 8)
    packed, conditions = pack_score_conditions(batch, (cond, uncond))
    assert tuple(packed["hidden_states"].shape) == (2, 4, 2)
    assert tuple(packed["timestep"].shape) == (2,)
    assert tuple(packed["indices_hidden_states"].shape) == (4,)
    assert "generated_x0" not in packed and "sigmas" not in packed
    assert tuple(conditions.shape) == (2, 3, 8)
    prediction = torch.randn(2, 4, 2)
    left, right = split_packed_prediction(prediction, 2)
    assert tuple(left.shape) == (1, 4, 2)
    assert tuple(right.shape) == (1, 4, 2)

    exact = tensor_parity_metrics(left, left.clone())
    assert exact["status"] == "PASS"
    sparse_bad = left.clone()
    sparse_bad.view(-1)[0] += 10.0
    bad = tensor_parity_metrics(left, sparse_bad)
    assert bad["status"] == "FAIL"
    assert bad["max_abs"] > bad["max_abs_limit"]


def test_gan_real_pair_patch_reduces_three_dmd_requests_to_two_forwards(monkeypatch) -> None:
    import sys
    import types

    from runtime.gan_score_packing import (
        compute_kl_grad_real_pair_packed,
        packed_gan_real_score_pair,
    )

    helios_pkg = types.ModuleType("helios")
    helios_utils_pkg = types.ModuleType("helios.utils")
    helios_post = types.ModuleType("helios.utils.utils_helios_post")
    helios_post.compute_kl_grad = lambda *args, **kwargs: None
    helios_post.convert_flow_pred_to_x0 = lambda *, flow_pred, **kwargs: flow_pred
    monkeypatch.setitem(sys.modules, "helios", helios_pkg)
    monkeypatch.setitem(sys.modules, "helios.utils", helios_utils_pkg)
    monkeypatch.setitem(sys.modules, "helios.utils.utils_helios_post", helios_post)

    diffusers_pkg = types.ModuleType("diffusers")
    diffusers_utils_pkg = types.ModuleType("diffusers.utils")
    torch_utils = types.ModuleType("diffusers.utils.torch_utils")
    torch_utils.is_compiled_module = lambda model: False
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_pkg)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils_pkg)
    monkeypatch.setitem(sys.modules, "diffusers.utils.torch_utils", torch_utils)

    class Accelerator:
        @staticmethod
        def unwrap_model(model):
            return model

    class Model:
        def __init__(self) -> None:
            self.adapters = True
            self.calls = []

        def disable_adapters(self) -> None:
            self.adapters = False

        def enable_adapters(self) -> None:
            self.adapters = True

        def __call__(self, *, hidden_states, encoder_hidden_states, **kwargs):
            del kwargs
            self.calls.append((int(hidden_states.shape[0]), self.adapters))
            condition = encoder_hidden_states.float().mean(dim=tuple(range(1, encoder_hidden_states.ndim)))
            condition = condition.view(-1, *([1] * (hidden_states.ndim - 1)))
            adapter_term = 1.0 if self.adapters else 0.0
            return (hidden_states.float() + condition + adapter_term,)

    model = Model()
    noisy = torch.ones(1, 1, 1, 1, 1)
    clean = torch.full_like(noisy, 0.5)
    prompt = torch.full((1, 1, 1), 2.0)
    negative = torch.full((1, 1, 1), -1.0)
    history = torch.zeros(1, 1, 1, 1, 1)
    gt_all_data = (
        clean,
        None,
        None,
        None,
        None,
        history,
        history,
        history,
        history,
    )
    with packed_gan_real_score_pair(parity_check=True) as state:
        grad, ca_grad, dm_grad, logs = compute_kl_grad_real_pair_packed(
            Accelerator(),
            scheduler=None,
            real_fake_score_model=model,
            noisy_image_or_video=noisy,
            estimated_clean_image_or_video=clean,
            prompt_embeds=prompt,
            negative_prompt_embeds=negative,
            timestep=torch.tensor([500.0]),
            sigmas=torch.linspace(1.0, 0.0, 10),
            timesteps=torch.linspace(1000.0, 0.0, 10),
            fake_guidance_scale=0.0,
            real_guidance_scale=3.0,
            is_use_gt_history=True,
            gt_all_data=gt_all_data,
        )
    assert ca_grad is None and dm_grad is None
    assert torch.isfinite(grad).all()
    assert logs["dmdtrain_gradient_norm"].numel() == 1
    assert state.logical_score_requests == 3
    assert state.physical_dmd_score_forwards == 2
    assert model.calls[:2] == [(1, True), (2, False)]
    assert state.parity is not None and state.parity["status"] == "PASS"
    for key in ("real_cond", "real_uncond", "dmd_gradient"):
        metrics = state.parity[key]
        assert metrics["finite"] is True
        assert metrics["normalized_l2"] <= metrics["thresholds"]["normalized_l2"]
        assert metrics["cosine"] >= metrics["thresholds"]["min_cosine"]
        assert metrics["max_abs"] <= metrics["max_abs_limit"]


def test_p0_plan_contracts_are_valid_json() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "helios_nogan_world16_fsdp2_p0.json",
        "helios_gan_world16_fsdp2_p0.json",
    ):
        path = root / "configs/distillschedule/baselines" / name
        value = json.loads(path.read_text())
        assert value["schema_version"] == "distillschedule.static_fsdp2_p0_plan.v1"
        assert value["topology"]["world_size"] == 16
        assert value["default_candidate"]["mesh_policy"] == "world16-fsdp"
        assert value["default_candidate"]["rollout_batch_size"] == 1
        assert value["default_candidate"]["pack_real_score_pair"] is False
