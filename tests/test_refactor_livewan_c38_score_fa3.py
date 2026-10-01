"""Source-bound C38 kernel replacement; no schedule/precision/census changes."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_score_fa3 import (
    BACKEND_CONTRACT, SAC_CONTRACT, VARIANT, OP_NAME, ARGUMENT_NAMES,
    attention_metadata, backend_receipt_errors, runtime_receipt_errors,
)
from distillschedule.runtime.livewan_crossscale_worker import (
    _score_attention_backend_enabled, _critic_attention_checkpoint_enabled,
    _critic_step1_overlap_enabled, _apply_trajectory_contract, _validate_plan_identity,
)

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "configs/distillschedule/plans"
PARENT = P / "livewan_wan13b_14b_sf_dmd_c36_continuous_b16_score_layernorm.json"
CHILD = P / "livewan_wan13b_14b_sf_dmd_c38_continuous_b16_score_fa3.json"


@pytest.fixture
def plans(monkeypatch):
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/unread/" + key)
    return load_plan(PARENT).plan.to_dict(), load_plan(CHILD).plan.to_dict()


def test_one_factor_defaults_and_c36_parent(plans):
    a, b = json.loads(PARENT.read_text()), json.loads(CHILD.read_text())
    assert b["metadata"].pop("score_attention_backend") == BACKEND_CONTRACT
    assert b["metadata"].pop("critic_attention_checkpoint") == SAC_CONTRACT
    a["metadata"].pop("critic_attention_checkpoint")
    assert a == b  # Score backend comes from the same inherited C21 plan.
    a, b = plans
    assert a["metadata"]["score_attention_backend"]["name"] == "score_flash_attn2_unmasked_bf16_v1"
    assert _score_attention_backend_enabled(b, mode="pipeline")
    assert _critic_attention_checkpoint_enabled(b, mode="pipeline")
    assert not _critic_step1_overlap_enabled(b, mode="pipeline")
    _apply_trajectory_contract(b, cycles=2, mode="pipeline")
    _validate_plan_identity(b, mode="pipeline")
    with pytest.raises(RuntimeError): _score_attention_backend_enabled(b, mode="baseline")


@pytest.mark.parametrize("field,value", [("num_splits", 2), ("head_dimension", 64),
    ("input_output_dtype", "float32"), ("artifact_interface_sha256", "f" * 64),
    ("scope", "all_ranks"), ("enabled", False)])
def test_backend_fail_closed(plans, field, value):
    p = copy.deepcopy(plans[1]); p["metadata"]["score_attention_backend"][field] = value
    with pytest.raises(RuntimeError): _validate_plan_identity(p, mode="pipeline")


def inputs():
    def t(shape):
        return SimpleNamespace(ndim=4, is_cuda=True, dtype="bf16", device="cuda:0",
            shape=shape, stride=lambda dim=None: 1 if dim == -1 else (shape[1]*1536,1536,128,1))
    return (t((2,19320,12,128)), t((2,512,12,128)), t((2,512,12,128)))


def test_metadata_defaults_keyword_positional_and_no_value_read():
    torch = SimpleNamespace(bfloat16="bf16")
    assert len(attention_metadata(torch, inputs(), {})) == 3
    full = tuple(dict(zip(ARGUMENT_NAMES[:3], inputs()), **__import__(
        "distillschedule.runtime.livewan_score_fa3", fromlist=["OPTION_DEFAULTS"]
        ).OPTION_DEFAULTS)[n] for n in ARGUMENT_NAMES)
    assert attention_metadata(torch, full, {}) == attention_metadata(torch, (), dict(zip(ARGUMENT_NAMES, full)))
    s = inspect.getsource(attention_metadata)
    assert not any(x in s for x in (".cpu(", ".tolist(", ".item(", "synchronize(", "copy_("))


@pytest.mark.parametrize("key,value", [("num_splits", 2), ("causal", True),
    ("softmax_scale", .5), ("out_", object()), ("q_descale", object()),
    ("k_new", object()), ("attention_chunk", 16), ("window_size_left", 32),
    ("softcap", .1), ("unknown", None)])
def test_metadata_rejects_semantic_or_staging_change(key, value):
    with pytest.raises(RuntimeError):
        attention_metadata(SimpleNamespace(bfloat16="bf16"), inputs(), {key: value})


def test_receipts_prove_backend_and_actual_replay_not_config_only():
    backend = dict(BACKEND_CONTRACT, install_status="PASS", execution_status="PASS",
        function_patched=True, contract_failure_count=0, invocation_count=1680,
        backward_invocation_count=600, artifact_root="/pinned")
    assert not backend_receipt_errors(backend, rank=8)
    for key, value in [("invocation_count", 0), ("num_splits", 2), ("function_patched", False)]:
        assert backend_receipt_errors(dict(backend, **{key:value}), rank=8)
    sac = dict(name=SAC_CONTRACT["name"],
        source=dict(interface_sha256=SAC_CONTRACT["interface_sha256"],
                    operator=OP_NAME.replace("::", ".") + ".default"),
        execution_status="PASS", production_installed=True, self_attention_only=True,
        contract_failures=0, checkpoint_calls=300, original_fa_calls=300,
        replay_fa_cache_reads=300, expected_checkpoint_calls=300,
        allow_cache_entry_mutation=False, cpu_offload=False)
    assert not runtime_receipt_errors(sac, rank=8, expected_calls=300)
    for key, value in [("replay_fa_cache_reads", 0), ("cpu_offload", True), ("checkpoint_calls", 360)]:
        assert runtime_receipt_errors(dict(sac, **{key:value}), rank=8, expected_calls=300)


@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing"])
def test_controller_explicit_child_route(tmp_path, kind):
    import subprocess, sys
    args = [sys.executable, str(ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"),
        "--hosts", "node0,node1", "--run-id", "c38-dryrun", "--run-kind", kind,
        "--execution-mode", "pipeline", "--pipeline-variant", VARIANT,
        "--expected-head", "a" * 40, "--master-addr", "127.0.0.1", "--dry-run",
        "--local-evidence-root", str(tmp_path)]
    for key in ("remote-root", "remote-output-root", "student-checkpoint", "critic-checkpoint",
                "teacher-checkpoint", "trace", "bootstrap", "source-root", "source-receipt", "python-overlay"):
        args += ["--" + key, str(tmp_path / key)]
    if kind == "trajectory": args += ["--trajectory-cycles", "2", "--trajectory-arm-id", "c38", "--trajectory-seed", "20260830"]
    else: args += ["--two-cycle-arm-id", "c38", "--two-cycle-seed", "20260830",
                  "--correctness-receipt", str(tmp_path / "gate.json"), "--correctness-sha256", "b" * 64]
    p = subprocess.run(args, check=True, capture_output=True, text=True, timeout=15)
    assert all(VARIANT in " ".join(v) for v in json.loads(p.stdout)["commands"])
