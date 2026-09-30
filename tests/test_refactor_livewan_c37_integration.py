"""One scheduling factor, actual census, default-off, unchanged B16 mean."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    _critic_step1_overlap_enabled, _apply_trajectory_contract, _validate_plan_identity)
from distillschedule.runtime.livewan_step1_overlap import PLAN_CONTRACT, VARIANT, SCHEDULE, runtime_receipt_errors

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "configs/distillschedule/plans"
PARENT = P / "livewan_wan13b_14b_sf_dmd_c36_continuous_b16_score_layernorm.json"
CHILD = P / "livewan_wan13b_14b_sf_dmd_c37_continuous_b16_step1_overlap.json"


@pytest.fixture
def plans(monkeypatch):
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/unread/" + key)
    return load_plan(PARENT).plan.to_dict(), load_plan(CHILD).plan.to_dict()


def test_single_factor_and_default_off(plans):
    a, b = json.loads(PARENT.read_text()), json.loads(CHILD.read_text())
    assert b["metadata"].pop("critic_step1_overlap") == PLAN_CONTRACT
    assert a == b
    a, b = plans
    assert not _critic_step1_overlap_enabled(a, mode="pipeline")
    assert _critic_step1_overlap_enabled(b, mode="pipeline")
    with pytest.raises(RuntimeError):
        _critic_step1_overlap_enabled(b, mode="baseline")
    _apply_trajectory_contract(b, cycles=2, mode="pipeline")
    _validate_plan_identity(b, mode="pipeline")


@pytest.mark.parametrize("key,value", [("loss_weights", [1, 1]),
    ("global_batch", 8), ("scope", "all_steps"),
    ("physical_calls_by_step", [2, 2, 2, 2, 2]),
    ("single_reduction_and_commit_after_both_waves", False),
    ("numerical_policy_unchanged", False)])
def test_metadata_fail_closed(plans, key, value):
    p = copy.deepcopy(plans[1])
    p["metadata"]["critic_step1_overlap"][key] = value
    with pytest.raises(RuntimeError):
        _validate_plan_identity(p, mode="pipeline")


def test_c36_parent_required(plans):
    p = copy.deepcopy(plans[1])
    p["metadata"].pop("score_layernorm")
    with pytest.raises(RuntimeError):
        _critic_step1_overlap_enabled(p, mode="pipeline")


def test_c37_cannot_masquerade_as_c36_profile(plans):
    from distillschedule.runtime.livewan_two_cycle_profile import plan_variant
    with pytest.raises(RuntimeError, match="no qualified"):
        plan_variant(plans[1], mode="pipeline")


def test_observed_census_not_config_only():
    c = dict(critic_step1_physical_schedule=SCHEDULE, critic_step1_single_commit=True,
             critic_physical_batch_size="mixed_step1_b1_other_b2",
             critic_physical_backward_calls_per_step=[1, 2, 1, 1, 1],
             critic_physical_calls_by_cycle=[[1, 2, 1, 1, 1]] * 2)
    assert not runtime_receipt_errors(c, rank=8, cycles=2,
                                     inventory={"critic/grad/checkpoint": 12})
    assert runtime_receipt_errors(c, rank=8, cycles=2,
                                  inventory={"critic/grad/checkpoint": 10})
    for key, value in [("critic_physical_calls_by_cycle", [[1, 1, 1, 1, 1]] * 2),
                       ("critic_step1_single_commit", False)]:
        assert runtime_receipt_errors(dict(c, **{key: value}), rank=8, cycles=2,
                                      inventory={"critic/grad/checkpoint": 12})
    assert not runtime_receipt_errors(dict(c, critic_physical_calls_by_cycle=[]),
                                     rank=0, cycles=2, inventory={})


def test_physical_inventory_48_not_40_per_cycle():
    path = ROOT / "scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py"
    spec = importlib.util.spec_from_file_location("c37_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = SimpleNamespace(cycles=2, rows=[SimpleNamespace(new_blocks=4)] * 160)
    logical, default = module._expected_inventory(trace, mode="pipeline")
    same_logical, candidate = module._expected_inventory(trace, mode="pipeline", pipeline_variant=VARIANT)
    assert logical == same_logical
    assert default["critic/grad/checkpoint"] == 80
    assert candidate["critic/grad/checkpoint"] == 96
    assert module._expected_inventory(trace, mode="baseline", pipeline_variant=VARIANT)[1]["critic/grad/checkpoint"] == 160


@pytest.mark.parametrize("script", [
    "summarize_livewan_trajectory_world16.py",
    "summarize_livewan_two_cycle_timing_world16.py",
])
@pytest.mark.parametrize("physical_forwards", [10, 12])
def test_summary_sac_gate_uses_actual_physical_forwards(script, physical_forwards):
    """Execute each real summary's SAC validation expression, not a string check.

    C36 remains 10 physical forwards/300 block checkpoints for two cycles;
    C37 is 12/360. Source, replay, missing-inventory and offload gates remain strict.
    """
    import ast
    from distillschedule.runtime.livewan_attention_checkpoint import (
        INTERFACE_SHA256, PLAN_CONTRACT as SAC_CONTRACT,
        runtime_receipt_errors as sac_errors,
    )

    path = ROOT / "scripts/distillschedule" / script
    calls = [n for n in ast.walk(ast.parse(path.read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "sac_errors"]
    assert len(calls) == 1
    code = compile(ast.Expression(calls[0]), str(path), "eval")
    checkpoints = physical_forwards * 30
    sac = dict(name=SAC_CONTRACT["name"],
               source={"interface_sha256": INTERFACE_SHA256},
               execution_status="PASS", production_installed=True,
               self_attention_only=True, contract_failures=0,
               checkpoint_calls=checkpoints, expected_checkpoint_calls=checkpoints,
               original_fa_calls=checkpoints, replay_fa_cache_reads=checkpoints,
               cpu_offload=False, allow_cache_entry_mutation=False)
    receipt = dict(critic_attention_checkpoint=sac,
                   physical_call_inventory={"critic/grad/checkpoint": physical_forwards})

    def validate(value):
        return eval(code, {"sac_errors": sac_errors, "rank": 8,
                           "row": value, "final": value})

    assert not validate(receipt)
    for key, value in [("checkpoint_calls", checkpoints - 1),
                       ("replay_fa_cache_reads", 0), ("cpu_offload", True),
                       ("expected_checkpoint_calls", 300 if checkpoints == 360 else 360)]:
        changed = copy.deepcopy(receipt)
        changed["critic_attention_checkpoint"][key] = value
        assert validate(changed), (script, key)
    missing = copy.deepcopy(receipt)
    missing.pop("physical_call_inventory")
    assert validate(missing)


def test_two_weighted_b1_backwards_equal_b2_mean_one_update():
    import torch
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(2, 7, dtype=torch.float64, generator=generator)
    y = torch.randn(2, 4, dtype=torch.float64, generator=generator)
    w = torch.randn(7, 4, dtype=torch.float64, generator=generator)
    ref = torch.nn.Parameter(w.clone())
    split = torch.nn.Parameter(w.clone())
    ((x @ ref - y) ** 2).mean().backward()
    for wave in range(2):
        (0.5 * ((x[wave:wave+1] @ split - y[wave:wave+1]) ** 2).mean()).backward()
    torch.testing.assert_close(ref.grad, split.grad, rtol=1e-12, atol=1e-12)
    # Both contributions precede exactly one optimizer commit; no wave0 step.
    opt_a = torch.optim.AdamW([ref], lr=1e-3)
    opt_b = torch.optim.AdamW([split], lr=1e-3)
    opt_a.step(); opt_b.step()
    torch.testing.assert_close(ref, split, rtol=1e-12, atol=1e-12)


def test_actual_execution_reuses_early_wave_branch_and_counter():
    s = (ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py").read_text()
    assert "or _critic_step1_overlap_enabled(plan, mode=mode)" in s
    assert "critic_inventory=inventory if critic_step1_split_overlap else None" in s
    assert 'critic_physical_calls_by_global_step[str(global_step)] = observed' in s
    assert "reducer_accumulation=(0, 2)" in s and "reducer_accumulation=(1, 2)" in s


@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing"])
def test_controller_explicit_child_route(tmp_path, kind):
    import subprocess, sys
    args = [sys.executable, str(ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"),
            "--hosts", "node0,node1", "--run-id", "c37-dryrun", "--run-kind", kind,
            "--execution-mode", "pipeline", "--pipeline-variant", VARIANT,
            "--expected-head", "a" * 40, "--master-addr", "127.0.0.1", "--dry-run",
            "--local-evidence-root", str(tmp_path)]
    for key in ["remote-root", "remote-output-root", "student-checkpoint", "critic-checkpoint",
                "teacher-checkpoint", "trace", "bootstrap", "source-root", "source-receipt", "python-overlay"]:
        args += ["--" + key, str(tmp_path / key)]
    if kind == "trajectory":
        args += ["--trajectory-cycles", "2", "--trajectory-arm-id", "c37", "--trajectory-seed", "20260830"]
    else:
        args += ["--two-cycle-arm-id", "c37", "--two-cycle-seed", "20260830",
                 "--correctness-receipt", str(tmp_path / "gate.json"), "--correctness-sha256", "b" * 64]
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=15)
    payload = json.loads(result.stdout)
    assert len(payload["commands"]) == 2
    assert all(VARIANT in " ".join(v) for v in payload["commands"])
