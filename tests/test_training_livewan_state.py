"""Actual original LoRA/Adam CPU persistence, NOT native C51 lifecycle parity.

The tiny container names the real licensed LoRALinear; it is not a WanModel or
replacement DMD loop. CUDA reducer/transport/shadow drain remains NOT_RUN.
"""
from collections import OrderedDict
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from distillschedule.runtime.livewan_crossscale_worker import (
    ModelBundle, _bind_trajectory_optimizer_parameter_names, materialize_adam_state,
)
from distillschedule.training.checkpoint import (
    pack_state, restore_checkpoint, rng_state, save_checkpoint, unpack_state,
)
from distillschedule.workloads.best_known.registry import file_sha256

PIN = "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
LORA_SHA = "71bbcdc932e1290734df55e15cf5ed998bda1f41a3cb607684f28da71fcf0824"
LICENSE_SHA = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"


@pytest.fixture(scope="module")
def original_source():
    value = os.environ.get("DMD_LIVEWAN_TEST_UPSTREAM")
    if not value:
        pytest.skip("real original LoRA requires DMD_LIVEWAN_TEST_UPSTREAM")
    root = Path(value).resolve()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"])
    assert file_sha256(root / "wanstreamer/lora.py") == LORA_SHA
    assert file_sha256(root / "LICENSE") == LICENSE_SHA
    return root


@pytest.fixture(scope="module")
def original_lora(original_source):
    # Baseline import of the actual admitted file, without importing the eager
    # wanstreamer package, copying its class, or constructing package mocks.
    spec = importlib.util.spec_from_file_location("livewan_original_lora_state_baseline",
                                                original_source / "wanstreamer/lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def owner(original_lora, role="critic"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        linear = torch.nn.Linear(3, 4)
        layer = (original_lora.LoRALinear(linear.to(torch.bfloat16), rank=32)
                 if role == "critic" else linear)
        model = torch.nn.Sequential(OrderedDict(projection=layer))
    model.eval()
    model.register_buffer("fixture_buffer", torch.tensor([2., 3.]))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=4e-5 if role == "critic" else 2e-6,
                                  betas=(0., .99), weight_decay=0., eps=1e-8, foreach=True)
    names = _bind_trajectory_optimizer_parameter_names(
        named_parameters=list(model.named_parameters()), optimizer=optimizer,
        expected_parameter_ids={id(p) for p in parameters})
    adam_receipt = materialize_adam_state(torch, optimizer, step=0)
    metadata = {"generator_ranks": list(range(8)), "score_ranks": list(range(8, 16)),
                "critic_ranks": list(range(8, 16)), "student": None, "critic": None}
    if role == "critic":
        metadata["critic"] = {"placement": "gpu_resident_replica", "base_sha256": "a" * 64,
                              "lora_rank": 32, "lora_modules": 1,
                              "lora_numel": sum(p.numel() for p in parameters), "optimizer": adam_receipt}
    else:
        metadata["student"] = {"placement": "gpu_resident_replica", "optimizer_impl": "foreach_adamw",
                               "state": {"kind": "stage1_state", "sha256": "b" * 64,
                                         "load_policy": "complete_stage1_direct_mmap"},
                               "optimizer": adam_receipt}
    # Actual dataclass and actual owners. Missing teacher/shadow/groups here do
    # not establish a physically runnable C51 bundle or a drained cycle.
    bundle = ModelBundle(
        student=model if role == "generator" else None, critic=model if role == "critic" else None,
        critic_shadow=None, teacher=None,
        generator_optimizer=optimizer if role == "generator" else None,
        critic_optimizer=optimizer if role == "critic" else None,
        generator_optimizer_parameter_names=names if role == "generator" else {},
        critic_optimizer_parameter_names=names if role == "critic" else {},
        critic_lora_params=parameters if role == "critic" else [], critic_shadow_lora_params=[],
        critic_gradient_reducer=None, generator_group=None, score_group=None, critic_group=None,
        score_local_group=None, score_leader_group=None, runahead_host_group=None,
        critic_shadow_sync_group=None, score_local_ranks=(), score_leader_ranks=(),
        generator_group_size=8, score_group_size=8, critic_group_size=8, metadata=metadata)
    flat = torch.zeros(sum(p.numel() for p in parameters))
    offset, counts, handles = 0, {}, []
    for p in parameters:
        p.grad = flat[offset:offset + p.numel()].view_as(p)
        offset += p.numel()
        counts[id(p)] = 0
        def hook(parameter, calls=counts):
            calls[id(parameter)] += 1
        handles.append(p.register_post_accumulate_grad_hook(hook))
    return SimpleNamespace(bundle=bundle, model=model, optimizer=optimizer, parameters=parameters,
                           role=role, flat=flat, counts=counts, handles=handles)


def views(current, original_source):
    from distillschedule.training.livewan_state import model_bundle_state
    return model_bundle_state(current.bundle, source_root=original_source)


def update(current):
    x = torch.randn(2, 3).to(torch.bfloat16 if current.role == "critic" else torch.float32)
    target = torch.randn(2, 4) + random.random() + float(np.random.random())
    output = current.model(x)
    (output.float() - target).square().mean().backward()
    current.optimizer.step()
    current.flat.zero_()  # Deliberately preserve attached consumed grad views.
    return {"x": x, "target": target, "output": output.detach().clone()}


def snapshot(current):
    return {
        "model": deepcopy(current.model.state_dict()), "optimizer": deepcopy(current.optimizer.state_dict()),
        "modes": {n: m.training for n, m in current.model.named_modules()},
        "trainability": {n: p.requires_grad for n, p in current.model.named_parameters()},
        "lora": {n: (m.enabled, m.rank, m.scale) for n, m in current.model.named_modules()
                 if type(m).__name__ == "LoRALinear"},
        "parameter_ids": [(n, id(p)) for n, p in current.model.named_parameters()],
        "optimizer_ids": [[id(p) for p in group["params"]] for group in current.optimizer.param_groups],
        "gradients": [(id(p.grad), p.grad.data_ptr(), p.grad.untyped_storage().data_ptr(), p.grad.clone())
                      if p.grad is not None else None for p in current.model.parameters()],
        "hooks": [[(k, id(h)) for k, h in (p._post_accumulate_grad_hooks or {}).items()]
                  for p in current.model.parameters()], "hook_calls": dict(current.counts), "rng": rng_state(),
    }


@pytest.mark.parametrize("role", ["critic", "generator"])
def test_actual_owner_checkpoint_continuation_preserves_parameters_grad_views_hooks_rng(
        tmp_path, original_source, original_lora, role):
    random.seed(29); np.random.seed(30); torch.manual_seed(31)
    control = owner(original_lora, role)
    state = views(control, original_source)
    frozen = {n: p.detach().clone() for n, p in control.model.named_parameters() if not p.requires_grad}
    update(control); update(control)
    identity = {"evidence": "CPU_ORIGINAL_LORA_ADAM_STATE_OWNER_NOT_NATIVE_C51", "role": role}
    path = save_checkpoint(tmp_path / "owner", state, identity=identity,
                           cycle=2, cursor=2, versions={role: 2})
    saved = pack_state(state)
    weights = saved[role + ".weights"]["value"]
    assert weights["identity"]["source"]["files"]["wanstreamer/lora.py"] == LORA_SHA
    if role == "critic":
        assert set(weights["tensors"]) == {"projection.a", "projection.b"}
        assert original_lora.lora_state_dict(control.model).keys() == weights["tensors"].keys()
    draws = [update(control) for _ in range(3)]
    expected, expected_rng = pack_state(state), rng_state()

    restored = owner(original_lora, role)
    resumed = views(restored, original_source)
    restored.model.train()
    restored.model.fixture_buffer.fill_(7.)
    ids = snapshot(restored)
    restore_checkpoint(path, resumed, identity=identity)
    assert equal(saved, pack_state(resumed))
    after = snapshot(restored)
    for key in ("parameter_ids", "optimizer_ids", "gradients", "hooks", "hook_calls", "lora", "trainability"):
        assert equal(ids[key], after[key]), key
    assert equal(draws, [update(restored) for _ in range(3)])
    assert equal(expected, pack_state(resumed))
    assert equal(expected_rng, rng_state())
    assert sorted(restored.counts.values()) == [3, 3]
    assert all(torch.equal(p, frozen[n]) for n, p in restored.model.named_parameters() if not p.requires_grad)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", [
    "moment_dtype", "moment_shape", "moment_stride", "step_dtype", "step_vector", "step_negative", "step_fractional",
    "step_nonfinite", "missing_slot", "extra_slot", "extra_parameter_state", "group_policy", "name_order",
    "base_identity", "source_identity", "lora_rank", "lora_scale", "lora_enabled", "missing_trainable",
])
def test_malformed_actual_lora_state_is_rejected_before_any_owner_or_rng_mutation(
        original_source, original_lora, mutation):
    current = owner(original_lora)
    state = views(current, original_source)
    update(current)
    packed = pack_state(state)
    # Make a visible pending weight restore, so late optimizer admission would
    # also be detected as a partial model mutation.
    with torch.no_grad():
        current.model.projection.a.add_(1.)
    value = packed["critic.optimizer"]["value"]
    saved = value["optimizer"]
    first = next(iter(saved["state"]))
    slots = saved["state"][first]
    identity = packed["critic.weights"]["value"]["identity"]
    if mutation == "moment_dtype": slots["exp_avg"] = slots["exp_avg"].double()
    elif mutation == "moment_shape": slots["exp_avg_sq"] = torch.zeros(1)
    elif mutation == "moment_stride": slots["exp_avg"] = slots["exp_avg"].t().contiguous().t()
    elif mutation == "step_dtype": slots["step"] = slots["step"].double()
    elif mutation == "step_vector": slots["step"] = slots["step"].reshape(1)
    elif mutation == "step_negative": slots["step"].fill_(-1)
    elif mutation == "step_fractional": slots["step"].fill_(.5)
    elif mutation == "step_nonfinite": slots["step"].fill_(float("nan"))
    elif mutation == "missing_slot": del slots["exp_avg"]
    elif mutation == "extra_slot": slots["max_exp_avg_sq"] = slots["exp_avg_sq"].clone()
    elif mutation == "extra_parameter_state": saved["state"][123456] = deepcopy(slots)
    elif mutation == "group_policy": saved["param_groups"][0]["lr"] = 123.
    elif mutation == "name_order": value["parameters"][0].reverse()
    elif mutation == "base_identity": identity["base"]["sha256"] = "c" * 64
    elif mutation == "source_identity": identity["source"]["files"]["wanstreamer/lora.py"] = "d" * 64
    elif mutation == "lora_rank": identity["lora"]["projection"]["rank"] = 16
    elif mutation == "lora_scale": identity["lora"]["projection"]["scale"] = 2.
    elif mutation == "lora_enabled": identity["lora"]["projection"]["enabled"] = False
    elif mutation == "missing_trainable": del packed["critic.weights"]["value"]["tensors"]["projection.b"]
    before = snapshot(current)
    with pytest.raises(ValueError):
        unpack_state(packed, state)
    assert equal(before, snapshot(current))


@pytest.mark.parametrize("mutation", ["binding", "parameter", "extra_trainable", "order", "base", "rank", "scale", "enabled"])
def test_live_owner_drift_is_rejected_nonmutating(original_source, original_lora, mutation):
    current = owner(original_lora)
    state = views(current, original_source)
    packed = pack_state(state)
    if mutation == "binding":
        key = next(iter(current.bundle.critic_optimizer_parameter_names))
        current.bundle.critic_optimizer_parameter_names[key] = "renamed.a"
    elif mutation == "parameter":
        current.model.projection.a = torch.nn.Parameter(current.model.projection.a.detach().clone())
    elif mutation == "extra_trainable":
        current.model.register_parameter("unexpected", torch.nn.Parameter(torch.ones(1)))
    elif mutation == "order": current.optimizer.param_groups[0]["params"].reverse()
    elif mutation == "base": current.bundle.metadata["critic"]["base_sha256"] = "e" * 64
    elif mutation == "rank": current.model.projection.rank = 16
    elif mutation == "scale": current.model.projection.scale = 2.
    elif mutation == "enabled": current.model.projection.enabled = False
    before = snapshot(current)
    with pytest.raises(ValueError):
        unpack_state(packed, state)
    assert equal(before, snapshot(current))


def test_owner_boundary_refuses_unconsumed_gradients_without_unpublishing_or_zeroing(
        tmp_path, original_source, original_lora):
    current = owner(original_lora)
    state = views(current, original_source)
    current.flat.fill_(1.)
    before = snapshot(current)
    path = tmp_path / "not-a-drained-checkpoint"
    with pytest.raises(ValueError, match="gradients"):
        save_checkpoint(path, state, identity={"evidence": "CPU_LOCAL_OWNER_BOUNDARY_ONLY"},
                        cycle=1, cursor=1, versions={"critic": 1})
    assert not path.exists() and not list(tmp_path.glob(".*.tmp-*"))
    assert equal(before, snapshot(current))


@pytest.mark.parametrize("mutation", ["placement", "topology", "optimizer", "base_receipt"])
def test_only_source_qualified_resident_actual_bundle_owners_are_admitted(original_source, original_lora, mutation):
    current = owner(original_lora)
    if mutation == "placement": current.bundle.metadata["critic"]["placement"] = "world16_fsdp2_keep_unsharded"
    elif mutation == "topology": current.bundle.metadata["critic_ranks"] = list(range(16))
    elif mutation == "optimizer": current.bundle.critic_optimizer = torch.optim.SGD(current.parameters, lr=4e-5)
    elif mutation == "base_receipt": del current.bundle.metadata["critic"]["base_sha256"]
    before = snapshot(current)
    with pytest.raises(ValueError):
        views(current, original_source)
    assert equal(before, snapshot(current))


def test_source_root_is_not_silently_substituted(tmp_path, original_lora):
    current = owner(original_lora)
    before = snapshot(current)
    with pytest.raises(ValueError, match="source"):
        views(current, tmp_path)
    assert equal(before, snapshot(current))
