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


def test_only_B_compatible_is_separate_from_blocked_full_B(env):
    arm = "only_B_fsdp_compatible"
    p = r2.make_plan(ROOT, arm=arm, source_head=HEAD)
    r2.validate(p, mode="baseline", repo_root=ROOT)
    worker._validate_plan_identity(p, mode="baseline")
    baseline = r2.make_plan(ROOT, arm="baseline", source_head=HEAD)
    assert p["rank_groups"] == baseline["rank_groups"]
    assert p["metadata"]["baseline_placement"] == baseline["metadata"]["baseline_placement"]
    assert p["workload"] == baseline["workload"]
    assert p["training_backends"]["generator"]["options"]["optimizer_impl"] == "fused_adamw"
    assert p["metadata"][r2.COMPATIBLE_KEY]["scope"] == "explicit_critic_teacher_calls_all_ranks"
    assert not worker._generator_runahead_transport_enabled(p, mode="baseline")
    assert not worker._split_dmd_shadow_enabled(p, mode="baseline")
    assert not worker._critic_attention_checkpoint_enabled(p, mode="baseline")
    assert not worker._generator_rollout_cast_cache_enabled(p, mode="baseline")
    assert set(p["metadata"][r2.COMPATIBLE_KEY]["operators"]) == {
        "critic_lora_kernel", "score_modulation_kernel", "score_rmsnorm_kernel", "score_rope_kernel"}
    with pytest.raises(ValueError, match="BLOCKED"):
        r2.make_plan(ROOT, arm="only_B", source_head=HEAD)
    for key in ("score_rope_kernel", "generator_rollout_cast_cache"):
        modified = deepcopy(p)
        modified["metadata"][key] = {"enabled": True}
        with pytest.raises(ValueError):
            r2.validate(modified, mode="baseline", repo_root=ROOT)


def test_score_scope_is_nested_exception_safe_and_preserves_G_and_checkpoint():
    torch = pytest.importorskip("torch")
    from torch.utils import checkpoint
    from types import SimpleNamespace as NS
    counts = dict(original=0, fused=0)
    def original(x):
        counts["original"] += 1
        return x.square()
    def fused(x):
        counts["fused"] += 1
        return x.square()
    target = NS(op=original)
    scope = r2.ScopedScoreKernels({"score_rmsnorm_kernel": (target, "op", original, fused)})
    x = torch.randn(3, requires_grad=True)
    assert torch.equal(target.op(x), original(x))
    with pytest.raises(RuntimeError, match="synthetic"):
        with scope.role("teacher"):
            with scope.role("critic"):
                target.op(x)
            assert scope.current_role == "teacher"
            raise RuntimeError("synthetic")
    assert scope.current_role is None
    assert scope.receipt()["scope_failures"] == 1
    proxy = r2.ScopedRecomputeCheckpointAudit(checkpoint, scope)
    with scope.role("critic"):
        y = proxy.checkpoint(target.op, x, use_reentrant=False)
        y.sum().backward()
    assert torch.equal(x.grad, 2 * x.detach())
    assert proxy.receipt()["checkpoint_calls"] == 1
    assert proxy.receipt()["replay_entries"] == 1
    # G is outside the explicit Score scopes, including G checkpoint replay.
    before = counts["fused"]
    g = x.detach().clone().requires_grad_(True)
    proxy.checkpoint(target.op, g, use_reentrant=False).sum().backward()
    assert counts["fused"] == before and torch.equal(g.grad, x.grad)
    assert proxy.receipt()["checkpoint_calls"] == 1
    assert scope.receipt()["role_kernel_calls"]["critic"]["score_rmsnorm_kernel"] == 3
    assert not torch.cuda.is_initialized()


def test_real_CPU_checkpoint_early_stop_has_dispatch_without_completed_leaf():
    torch = pytest.importorskip("torch")
    from torch.utils import checkpoint
    from types import SimpleNamespace as NS
    completed = []
    def optimized(x):
        value = x.square().sin()
        # Non-reentrant recompute stops inside the preceding eager operators
        # once its saved tensors have been recovered, before the leaf marker.
        completed.append(1)
        return value
    target = NS(op=lambda x: x.square().sin())
    scope = r2.ScopedScoreKernels({"score_modulation_kernel": (target, "op", target.op, optimized)})
    audit = r2.ScopedRecomputeCheckpointAudit(checkpoint, scope)
    x = torch.tensor([.2, .4], requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    ref.square().sin().sum().backward()
    with scope.role("critic"):
        audit.checkpoint(target.op, x, use_reentrant=False).sum().backward()
    assert torch.equal(x.grad, ref.grad)
    assert scope.calls["critic"]["score_modulation_kernel"] == 2
    assert len(completed) == 1
    assert audit.receipt()["checkpoint_calls"] == audit.receipt()["replay_entries"] == 1
    assert scope.current_role is None and scope.failures == 0
    # The collector must admit positive completed <= dispatch attempts, but
    # never zero completed or more completions than attempted dispatches.
    row = _synthetic_compatible_mechanisms()
    row["physical_call_inventory"] = {"critic/grad/checkpoint": 20}
    row["score_modulation_kernel"]["invocation_count"] = len(completed)
    assert not r2.compatible_mechanism_errors(row)
    for count in (0, 3):
        row["score_modulation_kernel"]["invocation_count"] = count
        assert r2.compatible_mechanism_errors(row)
    assert not torch.cuda.is_initialized()


def test_tiny_real_CPU_DTENSOR_fused_Adam_matches_foreach(tmp_path):
    """Target-Torch CPU preflight, not distributed/CUDA model qualification."""
    torch = pytest.importorskip("torch")
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    assert not dist.is_initialized(), "test must not reuse an unowned process group"
    assert not torch.cuda.is_initialized()
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "gloo"), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        initial = torch.arange(24, dtype=torch.float32).reshape(4, 6).div(24)
        p = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(0)]))
        ref = torch.nn.Parameter(initial.clone())
        kwargs = dict(lr=2e-6, betas=(0., .99), eps=1e-8, weight_decay=0.)
        opt = torch.optim.AdamW([p], fused=True, **kwargs)
        reference = torch.optim.AdamW([ref], foreach=True, **kwargs)
        # Same producer as build_models, including the non-DTensor step scalar.
        assert worker.materialize_adam_state(torch, opt, step=0) == dict(
            tensor_count=1, local_numel_each_moment=24, step=0)
        worker.materialize_adam_state(torch, reference, step=0)
        assert not hasattr(opt.state[p]["step"], "to_local")
        assert opt.state[p]["step"].device == p.device
        for step in range(4):
            gradient = torch.full_like(ref, (step + 1) / 16)
            p.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
            ref.grad = gradient.clone()
            opt.step(); reference.step()
            assert p.placements == p.grad.placements == (Shard(0),)
            torch.testing.assert_close(p.to_local(), ref, rtol=1e-6, atol=0.)
            for key in ("exp_avg", "exp_avg_sq"):
                assert opt.state[p][key].placements == p.placements
                assert bool(torch.isfinite(opt.state[p][key].to_local()).all())
                torch.testing.assert_close(opt.state[p][key].to_local(), reference.state[ref][key], rtol=1e-6, atol=0.)
            assert torch.equal(opt.state[p]["step"], reference.state[ref]["step"])
        assert opt.defaults["fused"] is True and not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_compatible_native_controller_shell_and_collector_CLI(env):
    import os
    import subprocess
    import sys
    from test_livewan_system_factorial_chain import controller, controller_args
    c = controller()
    args = controller_args("baseline")
    args[args.index("--system-factorial-arm") + 1] = r2.COMPATIBLE_ARM
    parsed = c.parser().parse_args(args)
    for node in (0, 1):
        command = " ".join(c.command(parsed, node, "192.0.2.1"))
        assert "LIVEWAN_SYSTEM_FACTORIAL_ARM=" + r2.COMPATIBLE_ARM in command
        assert "LIVEWAN_BASELINE_PLACEMENT=world16-fsdp" in command
        assert "LIVEWAN_STUDENT_ROLLOUT_PADDING=max-noop" in command
    text = (ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh").read_text()
    start = text.index('if [[ "${RUN_KIND}" == system-factorial-timing ]]')
    branch = text[start:text.index("\nfi", start) + 3]
    environment = dict(os.environ, RUN_KIND=r2.RUN_KIND, LIVEWAN_SYSTEM_FACTORIAL_SEED=str(r2.SEED),
        PROFILE_TWO_CYCLES="0", DISTILLSCHEDULE_NSYS_BIN="", DISTILLSCHEDULE_NCU_BIN="",
        LIVEWAN_SYSTEM_FACTORIAL_ARM=r2.COMPATIBLE_ARM, EXECUTION_MODE="baseline", PIPELINE_VARIANT="b1",
        LIVEWAN_BASELINE_PLACEMENT="world16-fsdp", LIVEWAN_STUDENT_ROLLOUT_PADDING="max-noop")
    assert subprocess.run(["bash", "-c", branch], env=environment, capture_output=True).returncode == 0
    environment["LIVEWAN_STUDENT_ROLLOUT_PADDING"] = "off"
    assert subprocess.run(["bash", "-c", branch], env=environment, capture_output=True).returncode != 0
    result = subprocess.run([sys.executable, str(ROOT / "scripts/distillschedule/summarize_livewan_system_factorial_r2.py"),
        "--help"], text=True, capture_output=True)
    assert result.returncode == 0 and r2.COMPATIBLE_ARM in result.stdout


def test_production_installer_and_real_Score_helpers_use_scoped_original_kernels(monkeypatch, env):
    """Mock only CUDA kernel installers and model math; execute production dispatch."""
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace as NS
    import sys
    import importlib
    torch = pytest.importorskip("torch")
    modules = {name: ModuleType(name) for name in
        ("wan", "wan.modules", "wan.modules.model", "wanstreamer", "wanstreamer.blockcausal", "wanstreamer.lora", "wanstreamer.dmd")}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    class LoRA:
        scale = 1.
        def forward(self, x): return x
    class RMS:
        def forward(self, x): return x
    bc, lora, dmd = (modules["wanstreamer." + name] for name in ("blockcausal", "lora", "dmd"))
    lora.LoRALinear = LoRA
    lora.lora_enabled = lambda *_: nullcontext()
    modules["wan.modules.model"].WanRMSNorm = RMS
    bc._modulate = lambda x: x
    bc.apply_rope = lambda x: x
    entered_installers = []
    for key, module_name, installer_name, owner, attribute in (
        ("critic_lora_kernel", "livewan_lora_kernel", "install_scale_one_lora_epilogue", LoRA, "forward"),
        ("score_modulation_kernel", "livewan_modulation_kernel", "install_score_modulation_kernel", bc, "_modulate"),
        ("score_rmsnorm_kernel", "livewan_score_rmsnorm", "install_frozen_score_rmsnorm", RMS, "forward"),
        ("score_rope_kernel", "livewan_rope_kernel", "install_score_rope_kernel", bc, "apply_rope"),
    ):
        def install(_torch, _key=key, _owner=owner, _attribute=attribute, **kwargs):
            entered_installers.append((_key, kwargs))
            original = getattr(_owner, _attribute)
            setattr(_owner, _attribute, lambda *args: original(*args))
            return dict(enabled=True, install_status="SYNTHETIC_TEST_ONLY")
        module = importlib.import_module("distillschedule.runtime." + module_name)
        monkeypatch.setattr(module, installer_name, install)
    loras = [LoRA() for _ in range(180)]
    bundle = NS(student=NS(), critic=NS(modules=lambda: loras), teacher=NS())
    p = r2.make_plan(ROOT, arm=r2.COMPATIBLE_ARM, source_head=HEAD)
    receipts = r2.install_fsdp_compatible_score_kernels(torch, bundle=bundle, plan=p, mode="baseline")
    assert set(receipts) == set(r2.COMPATIBLE_OPERATORS)
    assert entered_installers[0] == ("critic_lora_kernel", {"autocast_input_elision": True})
    scope = bundle.system_factorial_score_scope
    def call_score(model, z, *args):
        assert scope.current_role == ("teacher" if model is bundle.teacher else "critic")
        bc._modulate(z); bc.apply_rope(z); RMS().forward(z)
        if model is bundle.critic: loras[0].forward(z)
        return z
    dmd.cfg_velocity = call_score
    dmd.bidirectional_velocity = call_score
    value = torch.ones(1)
    flow = NS(z_t=value, t=value)
    context = NS(negative_teacher_context=value, rope=value)
    sample = NS(teacher=value, critic=value)
    worker._dmd_teacher_velocity(torch, bundle=bundle, score_context=context, flow=flow,
                                sample_context=sample, context_lens=value)
    worker._dmd_fake_velocity(torch, critic_model=bundle.critic, score_context=context,
                             flow=flow, critic_context=value, context_lens=value)
    assert scope.receipt()["scope_entries"] == dict(critic=1, teacher=1)
    assert scope.calls["teacher"]["score_rmsnorm_kernel"] == 1
    assert scope.calls["critic"]["critic_lora_kernel"] == 1
    with r2.score_scope(bundle.student, role="critic"):
        assert scope.current_role is None
        bc._modulate(value); bc.apply_rope(value); RMS().forward(value)
    assert all(scope.original_calls[key] == 1 for key in r2.COMPATIBLE_OPERATORS if key != "critic_lora_kernel")
    assert not torch.cuda.is_initialized()


def test_real_baseline_Critic_call_holds_scope_through_backward_and_reentry(monkeypatch, env):
    """Real baseline loop + CPU checkpoint; only model/transport leaves mocked."""
    from types import SimpleNamespace as NS
    from torch.utils import checkpoint
    from distillschedule.runtime.livewan_baseline_placement import BaselinePlacement
    torch = pytest.importorskip("torch")
    target = NS(op=lambda x: x.square())
    scope = r2.ScopedScoreKernels({"score_rmsnorm_kernel": (target, "op", target.op, lambda x: x.square())})
    audit = r2.ScopedRecomputeCheckpointAudit(checkpoint, scope)
    g, c = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    student, critic = NS(parameters=lambda: [g]), NS()
    object.__setattr__(critic, "_livewan_fsdp_score_scope", scope)
    bundle = NS(student=student, critic=critic, teacher=NS(), critic_lora_params=[c],
        generator_optimizer=torch.optim.AdamW([g]), critic_optimizer=torch.optim.AdamW([c]),
        metadata=dict(baseline_placement=BaselinePlacement("world16-fsdp", student_padding="max-noop").receipt()))
    def rollout(*args, **kwargs):
        assert scope.current_role is None
        target.op(g)  # Student's exact original implementation is exercised.
        return NS(x0=g, raw_context=None, context_lens=None)
    def generator_backward(*args, **kwargs):
        assert scope.current_role is None
        g.square().sum().backward()
        return []
    def critic_backward(*args, **kwargs):
        assert scope.current_role == "critic"
        value = audit.checkpoint(target.op, c, use_reentrant=False)
        value.sum().backward()
        assert scope.current_role == "critic"
        return value.detach().sum()
    monkeypatch.setattr(worker, "prepare_rollout", rollout)
    monkeypatch.setattr(worker, "prepare_score_sample_context", lambda *a, **kw: NS())
    monkeypatch.setattr(worker, "dmd_gradient_for_sample", lambda *a, **kw: torch.zeros(3))
    monkeypatch.setattr(worker, "generator_backward_for_sample", generator_backward)
    monkeypatch.setattr(worker, "critic_backward_for_sample", critic_backward)
    versions = dict(generator=0, critic=0, ema=0)
    worker._run_baseline(torch=torch, rank=0, trace=NS(row=lambda *a: NS()), loader=None,
        bundle=bundle, streamers=[None], score_context=None, run_kind=r2.RUN_KIND,
        evidence=worker.RunEvidence(), versions=versions, device=None)
    assert versions == dict(generator=1, critic=5, ema=0)
    assert scope.current_role is None and scope.entries == dict(critic=5, teacher=0)
    assert scope.original_calls["score_rmsnorm_kernel"] == 5
    assert scope.calls["critic"]["score_rmsnorm_kernel"] == 10
    assert audit.receipt()["checkpoint_calls"] == audit.receipt()["replay_entries"] == 5
    assert not torch.cuda.is_initialized()


def _synthetic_compatible_mechanisms():
    """Counter fixtures only; never native kernel or timing evidence."""
    from distillschedule.runtime.livewan_score_rmsnorm import REFERENCE_CLASS_SHA256
    from distillschedule.runtime.livewan_lora_kernel import EXPECTED_FORWARD_SHA256
    from distillschedule.runtime.livewan_modulation_kernel import EXPECTED_MODULATE_SHA256
    from distillschedule.runtime.livewan_rope_kernel import EXPECTED_APPLY_ROPE_SHA256
    from distillschedule.runtime.livewan_baseline_placement import BaselinePlacement
    counters = {role: {key: (0 if role == "teacher" and key == "critic_lora_kernel" else 1)
                      for key in r2.COMPATIBLE_OPERATORS} for role in ("critic", "teacher")}
    row = {r2.COMPATIBLE_KEY: dict(name="fsdp_scoped_score_operators_v1",
        scope="explicit_critic_teacher_calls_all_ranks", current_role=None, scope_entries=dict(critic=24, teacher=4), scope_failures=0,
        counter_scope="DISPATCH_ATTEMPTS_INCLUDING_CHECKPOINT_EARLY_STOP",
        role_kernel_calls=counters, original_outside_scope_calls={key: 1 for key in r2.COMPATIBLE_OPERATORS},
        parameter_views_cached=0, complete_B=False, observed_generator_optimizer_fused=True,
        observed_generator_optimizer_dtensor_parameters=True)}
    row.update(dict(build=dict(student=dict(optimizer_impl="fused_adamw"),
        baseline_placement=BaselinePlacement("world16-fsdp", student_padding="max-noop").receipt(),
        baseline_placement_runtime=dict(status="PASS", roles={role: dict(sharded=True, storage_device="cuda")
            for role in ("student", "critic", "teacher")})),
        critic_lora_kernel=dict(enabled=True, install_status="PASS", autocast_input_elision_invocation_count=1,
            autocast_contract_failure_count=0, module_count=180, all_scales_one=True, autocast_input_elision=True,
            name="scale_one_autocast_bf16_input_elision_v1", source_forward_sha256=EXPECTED_FORWARD_SHA256),
        score_modulation_kernel=dict(enabled=True, install_status="PASS", invocation_count=2, backward_invocation_count=1,
            name="score_bf16_modulation_fusion_v1", source_modulate_sha256=EXPECTED_MODULATE_SHA256),
        score_rope_kernel=dict(enabled=True, install_status="PASS", invocation_count=2, contract_failure_count=0,
            backward_invocation_count=1, name="score_bf16_rope_fusion_v1", source_apply_rope_sha256=EXPECTED_APPLY_ROPE_SHA256),
        score_rmsnorm_kernel=dict(enabled=True, install_status="PASS", name="score_frozen_bf16_rmsnorm_v1",
            source_class_sha256=REFERENCE_CLASS_SHA256, contract_failures=0, forward_calls=2,
            backward_calls=1, shape_histogram={"1x1x1536": 1, "2x1x5120": 1}),
        system_factorial_recompute=dict(checkpoint_calls=600, function_entries=1200, replay_entries=600,
            cache_reads=0, saved_tensor_policy="ORIGINAL", rng_policy="ORIGINAL")))
    return row


def test_compatible_real_producer_writer_collector_and_native_CLI_refuses_CPU_mock(monkeypatch, env):
    import subprocess
    import sys
    from distillschedule.runtime import livewan_system_factorial_collect as collector
    from test_livewan_system_factorial_chain import trace_fixture, execute_mock_rank
    trace = trace_fixture(env)
    monkeypatch.setenv("LIVEWAN_OPERATION_TRACE", str(trace.path))
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(env))
    p = r2.make_plan(ROOT, arm=r2.COMPATIBLE_ARM, source_head=HEAD)
    for node in (0, 1):
        worker._write_effective_plan_for_run(plan=p, output_dir=env, run_kind=r2.RUN_KIND, local_rank=0, node_rank=node)
    for rank in range(16):
        # Reuse the fixture's baseline physics; the real bound plan carries the compatible arm.
        timing, inventory, evidence = execute_mock_rank(monkeypatch, plan=p, trace=trace, rank=rank, arm="baseline")
        row = dict(rank=rank, run_id=p["evidence"]["run_id"], status="PASS", run_kind=r2.RUN_KIND,
            mode="baseline", head=HEAD, trace_sha256=trace.sha256, bootstrap_sha256="3" * 64,
            source_receipt_sha256="4" * 64, effective_plan_sha256=r2.digest(p), profiled=False,
            system_factorial_timing=timing, final_versions=dict(generator=4, critic=20, ema=0),
            logical_generator_updates=4, logical_critic_updates=20, physical_generator_commits=4,
            physical_critic_commits=20, system_factorial_c1_finite=dict(finite=True, count=6),
            generator_state_metrics=dict(test_only=0.), critic_state_metrics=dict(test_only=0.),
            cgroup_oom_before=dict(oom=0, oom_kill=0), cgroup_oom_after=dict(oom=0, oom_kill=0),
            physical_call_inventory=inventory.counts, logical_call_inventory=inventory.counts,
            peak_memory=dict(max_allocated_bytes=0, max_reserved_bytes=0, driver_free_bytes=0))
        row.update(_synthetic_compatible_mechanisms())
        worker._write_rank_receipt(output_dir=env, rank=rank, value=row)
    summary = collector.collect(env, arm=r2.COMPATIBLE_ARM, expected_head=HEAD, trace=trace, allow_synthetic=True)
    assert summary["status"] == "PASS_CPU_MOCK_ONLY" and summary["C2_C4_mean_seconds"] is None
    assert not summary["native_descriptive_timing_eligible"]
    output = env / "cli_summary.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/distillschedule/summarize_livewan_system_factorial_r2.py"),
        "--run-root", str(env), "--arm", r2.COMPATIBLE_ARM, "--trace", str(trace.path),
        "--expected-head", HEAD, "--output", str(output)], text=True, capture_output=True)
    assert result.returncode == 2
    assert any("CPU mock" in error for error in json.loads(output.read_text())["errors"])
    path = env / "receipts/rank_000.json"
    row = json.loads(path.read_text())
    row[r2.COMPATIBLE_KEY]["role_kernel_calls"]["teacher"]["score_rope_kernel"] = 0
    worker.atomic_json(path, row)
    summary = collector.collect(env, arm=r2.COMPATIBLE_ARM, expected_head=HEAD, trace=trace, allow_synthetic=True)
    assert summary["status"] == "FAIL" and any("role scope" in error for error in summary["errors"])
