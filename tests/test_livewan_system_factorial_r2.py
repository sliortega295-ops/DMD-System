"""CPU-only R2 contract tests. Mock timings are never native evidence."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import pytest

from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime import livewan_system_factorial_r2 as r2


ROOT = Path(__file__).resolve().parents[1]
HEAD = "a" * 40


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                 "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                 "LIVEWAN_SYSTEMS_BOOTSTRAP"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "CPU_SYNTHETIC_TEST_ONLY")
    monkeypatch.setenv("LIVEWAN_EXPECTED_HEAD", HEAD)
    monkeypatch.setenv("DMD_SYSTEM_ROOT", str(ROOT))
    return tmp_path


@pytest.mark.parametrize("arm", ("baseline", "only_A", "full"))
def test_real_lowering_and_runtime_flags(env, arm):
    p = r2.make_plan(ROOT, arm=arm, source_head=HEAD)
    mode = "baseline" if arm == "baseline" else "pipeline"
    r2.validate(p, mode=mode, repo_root=ROOT)
    worker._validate_plan_identity(p, mode=mode)
    assert p["evidence"]["steps"] == 20
    assert p["workload"]["shape"]["batch_size"] == 16
    assert p["workload"]["shape"]["resolution"] == {"height": 368, "width": 640}
    assert p["workload"]["shape"]["latent"]["frames"] == 21
    assert worker._generator_runahead_transport_enabled(p, mode=mode) == (arm != "baseline")
    assert worker._split_dmd_shadow_enabled(p, mode=mode) == (arm != "baseline")
    assert worker._critic_attention_checkpoint_enabled(p, mode=mode) == (arm == "full")
    assert worker._generator_rollout_cast_cache_enabled(p, mode=mode) == (arm == "full")
    assert worker._score_attention_backend_enabled(p, mode=mode) == (arm == "full")
    if arm == "baseline":
        spec = p["metadata"]["baseline_placement"]
        assert spec["name"] == "world16-fsdp"
        assert spec["student_reshard_after_forward"] is True
        assert spec["critic_fsdp"] is True
    else:
        c = worker._continuous_pipeline_contract(p, mode=mode)
        assert c["cycles"] == 4 and c["lookahead_depth"] == 1
        assert c["critic_shadow_sync_versions"] == [5, 10, 15]


def test_only_B_is_blocked_not_a_silently_smaller_package(env):
    with pytest.raises(ValueError, match="BLOCKED"):
        r2.make_plan(ROOT, arm="only_B", source_head=HEAD)


@pytest.mark.parametrize("mutation", ("shape", "seed", "source", "batch", "kernel", "shadow"))
def test_tampered_bindings_rejected(env, mutation):
    p = r2.make_plan(ROOT, arm="only_A", source_head=HEAD)
    if mutation == "shape":
        p["workload"]["shape"]["resolution"]["height"] = 320
    elif mutation == "batch":
        p["workload"]["shape"]["batch_size"] = 8
    elif mutation == "seed":
        p["metadata"][r2.KEY]["seed"] += 1
    elif mutation == "source":
        p["metadata"][r2.KEY]["source_head"] = "b" * 40
    elif mutation == "kernel":
        p["metadata"]["score_modulation_kernel"] = {"enabled": True}
    else:
        p["metadata"].pop("split_dmd_shadow")
    with pytest.raises((ValueError, RuntimeError)):
        r2.validate(p, mode="pipeline", repo_root=ROOT)


def test_canonical_two_cycle_plan_and_rejections_remain(env):
    from distillschedule.core.config import load_plan
    p = load_plan(ROOT / r2.FULL_PLAN).plan.to_dict()
    worker._validate_plan_identity(p, mode="pipeline")
    assert worker._continuous_pipeline_contract(p, mode="pipeline")["cycles"] == 3
    with pytest.raises(RuntimeError, match="one of"):
        worker._apply_trajectory_contract(deepcopy(p), cycles=4, mode="pipeline")
    p["metadata"].pop("generator_cross_attention_kv_cache")
    with pytest.raises(RuntimeError, match="C19"):
        worker._generator_runahead_transport_enabled(p, mode="pipeline")


def test_full_keeps_every_original_B_value(env):
    from distillschedule.core.config import load_plan
    parent = load_plan(ROOT / r2.FULL_PLAN).plan.to_dict()
    p = r2.make_plan(ROOT, arm="full", source_head=HEAD)
    for name in r2.B_METADATA:
        assert p["metadata"].get(name) == parent["metadata"].get(name)
    assert p["training_backends"] == parent["training_backends"]


def test_only_A_recompute_proxy_preserves_real_cpu_grad_rng_adam():
    torch = pytest.importorskip("torch")
    from torch.utils import checkpoint
    torch.manual_seed(7)
    first = torch.nn.Linear(4, 4)
    second = deepcopy(first)
    x = torch.randn(3, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    opt1 = torch.optim.AdamW(first.parameters())
    opt2 = torch.optim.AdamW(second.parameters())
    state = torch.get_rng_state()
    expected = checkpoint.checkpoint(lambda a: first(a).square(), x, use_reentrant=False)
    expected.sum().backward()
    rng_after = torch.get_rng_state()
    torch.set_rng_state(state)
    proxy = r2.RecomputeCheckpointAudit(checkpoint)
    observed = proxy.checkpoint(lambda a: second(a).square(), y, use_reentrant=False)
    observed.sum().backward()
    assert torch.equal(expected, observed) and torch.equal(x.grad, y.grad)
    assert torch.equal(torch.get_rng_state(), rng_after)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a.grad, b.grad)
    opt1.step(); opt2.step()
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    receipt = proxy.receipt()
    assert receipt["checkpoint_calls"] == 1
    assert receipt["function_entries"] == 2 and receipt["replay_entries"] == 1
    assert receipt["cache_reads"] == 0
    assert not torch.cuda.is_initialized()
