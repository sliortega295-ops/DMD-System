"""Default-off C44, executable SAC composition and observed receipt gates."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_ffn_down_checkpoint import (
    CONTRACT, VARIANT, FrozenFFNDownPolicies, enabled, ffn_down_metadata,
    runtime_receipt_errors,
)
from distillschedule.runtime.livewan_attention_checkpoint import AttentionCheckpointContexts

ROOT = Path(__file__).resolve().parents[1]


def test_import_is_lazy_default_off():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c",
        "import sys; from distillschedule.runtime.livewan_ffn_down_checkpoint import enabled;"
        "assert not enabled({}, mode='pipeline');"
        "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],
        env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "additional_policy_factory: Any = None" in inspect.getsource(AttentionCheckpointContexts)
    assert not any(t in inspect.getsource(FrozenFFNDownPolicies) + inspect.getsource(ffn_down_metadata)
                   for t in (".cpu(", ".tolist(", ".item(", "synchronize(", "copy_("))


@pytest.fixture
def plans(monkeypatch):
    from distillschedule.core.config import load_plan
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/unread/" + key)
    paths = ROOT / "configs/distillschedule/plans"
    return tuple(load_plan(paths / f).plan.to_dict() for f in (
        "livewan_wan13b_14b_sf_dmd_c38_continuous_b16_score_fa3.json",
        "livewan_wan13b_14b_sf_dmd_c44_continuous_b16_critic_ffn_down_cache.json"))


def test_one_factor_plan_and_complete_worker_validation(plans):
    from distillschedule.runtime.livewan_crossscale_worker import (
        _validate_plan_identity, _apply_trajectory_contract)
    a, b = copy.deepcopy(plans)
    assert not enabled(a, mode="pipeline")
    assert enabled(b, mode="pipeline")
    assert b["metadata"].pop("critic_ffn_down_checkpoint") == CONTRACT
    b["id"] = a["id"]
    assert a == b
    b = copy.deepcopy(plans[1])
    _apply_trajectory_contract(b, cycles=2, mode="pipeline")
    _validate_plan_identity(b, mode="pipeline")
    with pytest.raises(RuntimeError):
        enabled(b, mode="baseline")


@pytest.mark.parametrize("field,value", [
    ("cpu_offload", True), ("application_calls_unchanged", False),
    ("allow_cache_entry_mutation", True), ("alpha", 2), ("dtype", "float32"),
    ("input_shape", [19320, 8960]), ("source_blockcausal_sha256", "f" * 64),
])
def test_plan_fail_closed(plans, field, value):
    p = copy.deepcopy(plans[1])
    p["metadata"]["critic_ffn_down_checkpoint"][field] = value
    with pytest.raises(RuntimeError):
        enabled(p, mode="pipeline")


def metadata_tensors():
    def tensor(shape, strides, ptr, grad=False):
        return SimpleNamespace(shape=shape, stride=lambda: strides, dtype="bf16",
            device="cuda:0", is_cuda=True, requires_grad=grad, _version=0,
            data_ptr=lambda: ptr, element_size=lambda: 2)
    return (tensor((1536,), (1,), 1),
            tensor((38640, 8960), (8960, 1), 2, True),
            tensor((8960, 1536), (1, 8960), 3))


def test_metadata_binds_frozen_storage_not_recomputed_activation():
    torch = SimpleNamespace(bfloat16="bf16")
    args = metadata_tensors()
    ref = ffn_down_metadata(torch, args, {})
    args[1].data_ptr = lambda: 4
    assert ffn_down_metadata(torch, args, {}) == ref
    args[2]._version = 1
    assert ffn_down_metadata(torch, args, {}) != ref
    args[1].shape = (38640, 1536)  # FFN-up/QKV never saved by C44.
    assert ffn_down_metadata(torch, args, {}) is None


@pytest.mark.parametrize("kind", ["gradient", "dtype", "device", "bias", "alpha"])
def test_eligible_operation_mismatch_does_not_fallback(kind):
    args = metadata_tensors()
    kw = {}
    if kind == "gradient": args[2].requires_grad = True
    if kind == "dtype": args[1].dtype = "fp32"
    if kind == "device": args[1].is_cuda = False
    if kind == "bias": args[0].shape = (1, 1536)
    if kind == "alpha": kw["alpha"] = 2
    with pytest.raises(RuntimeError):
        ffn_down_metadata(SimpleNamespace(bfloat16="bf16"), args, kw)


def valid_receipt():
    return dict(name=CONTRACT["name"], contract=copy.deepcopy(CONTRACT),
        enabled=True, execution_status="PASS", production_installed=True,
        contract_failures=0, checkpoint_calls=300, expected_checkpoint_calls=300,
        original_ffn_down_calls=300, replay_ffn_down_cache_reads=300,
        all_checkpoint_ledgers_one_to_one=True,
        cumulative_cached_output_bytes=300 * 38640 * 1536 * 2,
        cache_location="cuda_per_checkpoint", cpu_offload=False,
        allow_cache_entry_mutation=False)


@pytest.mark.parametrize("rank", range(16))
def test_actual_runtime_census_required(rank):
    row = valid_receipt() if rank >= 8 else dict(
        enabled=True, execution_status="NOT_APPLICABLE", checkpoint_calls=0)
    assert not runtime_receipt_errors(row, rank=rank, expected_calls=300)
    row["checkpoint_calls"] = 1
    assert runtime_receipt_errors(row, rank=rank, expected_calls=300)
    if rank >= 8:
        for key, value in (("replay_ffn_down_cache_reads", 0),
                ("all_checkpoint_ledgers_one_to_one", False), ("contract_failures", 1),
                ("cumulative_cached_output_bytes", 0), ("cpu_offload", True)):
            assert runtime_receipt_errors(dict(valid_receipt(), **{key: value}),
                                          rank=rank, expected_calls=300)


@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing"])
def test_controller_routes_exact_child_and_shell_header_accepts_it(tmp_path, kind):
    args = [sys.executable, str(ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"),
        "--hosts", "node0,node1", "--run-id", "c44-dryrun", "--run-kind", kind,
        "--execution-mode", "pipeline", "--pipeline-variant", VARIANT,
        "--expected-head", "a" * 40, "--master-addr", "127.0.0.1", "--dry-run",
        "--local-evidence-root", str(tmp_path)]
    for key in ("remote-root", "remote-output-root", "student-checkpoint", "critic-checkpoint",
                "teacher-checkpoint", "trace", "bootstrap", "source-root", "source-receipt",
                "python-overlay"):
        args += ["--" + key, str(tmp_path / key)]
    if kind == "trajectory":
        args += ["--trajectory-cycles", "2", "--trajectory-arm-id", "c44",
                 "--trajectory-seed", "20260830"]
    else:
        args += ["--two-cycle-arm-id", "c44", "--two-cycle-seed", "20260830",
                 "--correctness-receipt", str(tmp_path / "gate.json"),
                 "--correctness-sha256", "b" * 64]
    r = subprocess.run(args, capture_output=True, check=True, text=True, timeout=15)
    assert all(VARIANT in " ".join(v) for v in json.loads(r.stdout)["commands"])
    shell = (ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh").read_text()
    assert shell[:shell.index('PYTHON=')].count(VARIANT) >= 3
    assert 'livewan_wan13b_14b_sf_dmd_c44_continuous_b16_critic_ffn_down_cache.json' in shell


def test_real_nonreentrant_cpu_backward_composes_both_policies(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint
    import distillschedule.runtime.livewan_ffn_down_checkpoint as module
    # Scaled CPU proof of the actual policy machinery. GPU metadata and real
    # dimensions have independent tests and the source-shaped GPU canary.
    def reader(t, args, kwargs):
        bias, h, w = args
        assert not w.requires_grad and not bias.requires_grad
        return tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype),
                      x.data_ptr() if i != 1 else None, x._version if i != 1 else None)
                     for i, x in enumerate(args))
    monkeypatch.setattr(module, "ffn_down_metadata", reader)
    torch.manual_seed(44)
    x = torch.randn(3, 5, requires_grad=True)
    w = torch.randn(5, 4); bias = torch.randn(4)
    a = torch.randn(5, 2, requires_grad=True)
    b = torch.randn(2, 4, requires_grad=True)
    def fn(z):
        h = z.sin()
        return torch.addmm(bias, h, w) + h @ a @ b
    policies = FrozenFFNDownPolicies(torch)
    contexts = AttentionCheckpointContexts(torch, torch.ops.aten.sin.default, {},
        metadata_reader=lambda t, args, kwargs: tuple(args[0].shape),
        additional_policy_factory=policies)
    y0 = checkpoint(fn, x, use_reentrant=False)
    grads0 = torch.autograd.grad(y0.square().sum(), (x, a, b))
    y1 = checkpoint(fn, x, use_reentrant=False, context_fn=contexts)
    grads1 = torch.autograd.grad(y1.square().sum(), (x, a, b))
    assert torch.equal(y0, y1)
    assert all(torch.equal(g, h) for g, h in zip(grads0, grads1))
    assert contexts.receipt()["original_fa_calls"] == 1
    assert contexts.receipt()["replay_fa_cache_reads"] == 1
    ledger = policies.checkpoints[0]
    assert len(ledger["original"]) == ledger["replay_calls"] == 1
    assert ledger["failures"] == 0
    assert ledger["cached_output_bytes"] == 3 * 4 * 4
