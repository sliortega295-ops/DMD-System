"""A different external stateful workflow through the existing Program runtime."""
import json
import subprocess
import sys
from pathlib import Path

import torch
from test_training_end_to_end import equal, run_cli

from distillschedule.data import prepare
from distillschedule.training.config import load_training_config
from distillschedule.training.plugins import load_callable

TEMPLATE = Path(__file__).resolve().parents[1] / "examples/framework/newworkflow.py"


def workflow_fixture(tmp_path):
    root = tmp_path / "workflow"
    result = subprocess.run([sys.executable, str(TEMPLATE), "--output", str(root), "--samples", "6",
                             "--validation-samples", "2"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    prepare(root / "raw/manifest.json", root / "prepared")
    prepare(root / "raw/validation-manifest.json", root / "validation-prepared")
    return root


def test_external_workflow_fresh_process_validation_restart_matches_every_persistent_state(tmp_path):
    root = workflow_fixture(tmp_path)
    config = root / "train.json"
    first = run_cli("train", config, "--cycles", 2, "--output", root / "first")
    resumed = run_cli("train", config, "--resume", first["checkpoint"], "--cycles", 6,
                      "--output", root / "resumed")
    whole = run_cli("train", config, "--cycles", 6, "--output", root / "whole")
    a = torch.load(Path(resumed["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(whole["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
        assert equal(a[key], b[key]), key
    for name in ("student", "fake", "student_optimizer", "fake_optimizer", "student_ema",
                 "student_scheduler", "fake_scheduler", "history"):
        assert name in a["state"]
    versions = whole["final_versions"]
    assert versions["student"] == versions["history"] == versions["student_ema"] == 6
    assert versions["fake"] == 12 and versions["teacher"] == 0
    assert versions["student_scheduler"] == versions["fake_scheduler"] == 6
    assert versions["student_optimizer"] == 12 and versions["fake_optimizer"] == 18
    whole_rows = [json.loads(line) for line in (root / "whole/validation.jsonl").read_text().splitlines()]
    resumed_rows = [json.loads(line) for line in (root / "resumed/validation.jsonl").read_text().splitlines()]
    assert [row["cycle"] for row in whole_rows] == [2, 4, 6]
    assert resumed_rows == whole_rows[1:]
    assert all(row["sample_ids"] == ["sample-000006", "sample-000007"] for row in whole_rows)
    assert set(whole_rows[-1]["metrics"]) == {"student_teacher_flow_mse"}
    assert whole["quality_validation"] == "NOT_RUN"
    assert whole["identity"]["model"]["reference"].endswith("newworkflow.py:build_model")


def test_workflow_declares_real_conditioning_and_post_cycle_commits(tmp_path):
    from distillschedule.data import PreparedDataset
    from distillschedule.training.session import build_cycle
    root = workflow_fixture(tmp_path)
    config = load_training_config(root / "train.json")
    factory, _ = load_callable(config.model["factory"])
    algorithm, _ = load_callable(config.algorithm["factory"])
    model = factory(config.model["parameters"])
    workload = build_cycle(config, model, PreparedDataset(config.data["prepared"]), algorithm, cycle=0, cursor=0)
    calls = workload.program.calls
    forward = next(call for call in calls if call.binding == "student_forward")
    assert "history" in forward.reads
    commit = calls[-1]
    assert commit.binding == "commit_cycle" and commit.grad == "no_grad"
    assert set(commit.writes) == {"student_ema", "history", "student_scheduler", "fake_scheduler",
                                  "student_optimizer", "fake_optimizer"}
    from distillschedule.open_workload.torch_backend import Context
    feeds = dict(workload.feeds)
    context = Context(feeds, model.state, forward.attributes)
    with torch.no_grad():
        before = workload.bindings["student_forward"](context)
        deployment_before = model.generate(model.state, feeds, {})
        model.state["history"].fill_(.2)
        after = workload.bindings["student_forward"](context)
        deployment_after = model.generate(model.state, feeds, {})
    assert not torch.equal(before, after)
    assert torch.equal(deployment_before, deployment_after)
    assert model.metadata["generation"] == "CURRENT_STUDENT_WITH_ZERO_HISTORY_NOT_EMA"
    initial_ema = {name: value.clone() for name, value in model.state["student_ema"].items()}
    initial_lrs = {role: model.state[role + "_optimizer"].param_groups[0]["lr"] for role in ("student", "fake")}
    from distillschedule.open_workload.torch_backend import Session
    session = Session(workload, {"backend": "torch-callable-resident-v1", "share_immutable": False,
                                 "checkpoint_bindings": []}, copy_state=False)
    session.run()
    for name, parameter in model.state["student"].named_parameters():
        assert torch.equal(model.state["student_ema"][name], initial_ema[name].lerp(parameter, 1 - .9))
    assert torch.equal(model.state["history"], after.mean(dim=(0, 2, 3, 4), keepdim=True))
    for role in ("student", "fake"):
        assert model.state[role + "_scheduler"].last_epoch == 1
        assert model.state[role + "_optimizer"].param_groups[0]["lr"] == initial_lrs[role] * .95
