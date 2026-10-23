"""Production-path regressions found by the framework's pre-release review."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.open_workload.torch_backend import Workload, signature, storage_bytes
from distillschedule.training.checkpoint import restore_checkpoint, save_checkpoint
from distillschedule.training.policy import CyclePolicy
from distillschedule.workloads.program import Call, Program


def update_with_scheduler(ctx):
    state = ctx.state
    loss = state["model"](ctx.inputs["x"]).square().mean()
    loss.backward()
    state["optimizer"].step()
    state["optimizer"].zero_grad(set_to_none=True)
    state["scheduler"].step()
    with torch.no_grad():
        state["ema"].lerp_(state["model"].weight, .2)
        state["history"].add_(1)
    return loss.detach()


def scheduled_workload(state, cycle):
    names = tuple(state)
    return Workload(Program((Call("update", "update", ("x",), names, names, grad="grad"),),
                            dict.fromkeys(names, cycle), ("x",), ("update",)),
                    {"update": update_with_scheduler}, state, {"x": torch.ones(1, 2)}, seed=cycle)


def test_scheduler_auto_reference_and_resume_include_all_persistent_state(tmp_path):
    from distillschedule.open_workload.torch_backend import compare
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    state = {"model": model, "optimizer": optimizer,
             "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9),
             "ema": model.weight.detach().clone(), "history": torch.zeros(2)}
    reference = deepcopy(state)
    automatic = CyclePolicy({"mode": "auto", "memory_limit_bytes": 1_000_000,
                             "max_candidates": 1, "samples": 1, "max_program_runs": 2})
    control = CyclePolicy({"mode": "reference"})
    expected = None
    for cycle in range(2):
        session, outputs = automatic.run(scheduled_workload(state, cycle), tmp_path / "auto")
        state = session.state
        expected, expected_outputs = control.run(scheduled_workload(reference, cycle), tmp_path / "control")
        reference = expected.state
        assert compare(session.result_state(outputs), expected.result_state(expected_outputs))["status"] == "PASS"
    checkpoint = save_checkpoint(tmp_path / "checkpoint", state, identity={"source": "test"},
                                 cycle=2, cursor=2, versions=dict.fromkeys(state, 2), policy=automatic.bound)
    restored = deepcopy(state)
    restored["scheduler"].step()
    restored["history"].zero_()
    payload = restore_checkpoint(checkpoint, restored, identity={"source": "test"})
    resumed = CyclePolicy(automatic.execution)
    resumed.restore(payload["policy"])
    session, outputs = resumed.run(scheduled_workload(restored, 2), tmp_path / "resume")
    expected, expected_outputs = control.run(scheduled_workload(reference, 2), tmp_path / "control")
    assert compare(session.result_state(outputs), expected.result_state(expected_outputs))["status"] == "PASS"
    assert session.state["scheduler"].last_epoch == 3
    assert session.state["scheduler"].optimizer is session.state["optimizer"]


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adafactor, torch.optim.LBFGS])
def test_non_pointwise_native_optimizer_checkpoint_continuation(tmp_path, optimizer_type):
    from test_training_end_to_end import equal

    def build(seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = torch.nn.Linear(3, 2)
        options = {"max_iter": 3, "history_size": 3} if optimizer_type is torch.optim.LBFGS else {}
        return {"model": model, "optimizer": optimizer_type(model.parameters(), lr=.1, **options)}

    def update(state, offset):
        def closure():
            state["optimizer"].zero_grad(set_to_none=True)
            prediction = state["model"](torch.tensor([[1., -2., 3.], [-1., 1., 2.]]))
            loss = (prediction - offset).square().mean()
            loss.backward()
            return loss
        state["optimizer"].step(closure)
        state["optimizer"].zero_grad(set_to_none=True)

    control = build(5)
    update(control, .5)
    weight_state = control["optimizer"].state[control["model"].weight]
    if optimizer_type is torch.optim.Adafactor:
        assert weight_state["row_var"].shape == (2, 1)
        assert weight_state["col_var"].shape == (1, 3)
        assert control["optimizer"].state[control["model"].bias]["variance"].shape == (2,)
    else:
        assert weight_state["d"].shape == (8,)
        assert weight_state["old_dirs"]
    identity = {"test": "native_optimizer_layout"}
    checkpoint = save_checkpoint(tmp_path / "checkpoint", control, identity=identity,
                                 cycle=1, cursor=1, versions=dict.fromkeys(control, 1))
    restarted = build(19)
    parameters = tuple(restarted["model"].parameters())
    restore_checkpoint(checkpoint, restarted, identity=identity)
    assert all(a is b for a, b in zip(parameters, restarted["model"].parameters()))
    assert all(a is b for a, b in zip(parameters, restarted["optimizer"].param_groups[0]["params"]))
    assert equal(control["model"].state_dict(), restarted["model"].state_dict())
    assert equal(control["optimizer"].state_dict(), restarted["optimizer"].state_dict())
    update(control, -.75)
    update(restarted, -.75)
    assert equal(control["model"].state_dict(), restarted["model"].state_dict())
    assert equal(control["optimizer"].state_dict(), restarted["optimizer"].state_dict())


@pytest.mark.parametrize("field", ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"])
@pytest.mark.parametrize("fault", ["shape", "dtype"])
def test_pointwise_adam_checkpoint_rejects_malformed_moments_before_mutation(field, fault):
    from distillschedule.training.checkpoint import pack_state, unpack_state
    from test_training_end_to_end import equal
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    model(torch.ones(1, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = {"model": model, "optimizer": optimizer}
    before = pack_state(state)
    malformed = deepcopy(before)
    values = malformed["optimizer"]["value"]["state"][0]
    values[field] = torch.zeros(1) if fault == "shape" else values[field].double()
    with pytest.raises(ValueError, match="shape/dtype"):
        unpack_state(malformed, state)
    assert equal(pack_state(state), before)


@pytest.mark.parametrize("fault,match", [("ownership", "ownership"), ("class", "class"),
                                        ("groups", "parameter group")])
def test_optimizer_checkpoint_preserves_named_class_and_group_admission(fault, match):
    from distillschedule.training.checkpoint import pack_state, unpack_state
    from test_training_end_to_end import equal
    model = torch.nn.Linear(3, 2)
    state = {"model": model, "optimizer": torch.optim.AdamW(model.parameters())}
    before = pack_state(state)
    malformed = deepcopy(before)
    if fault == "ownership":
        malformed["optimizer"]["parameters"][0].reverse()
    elif fault == "class":
        malformed["optimizer"]["class"] = "torch.optim.sgd.SGD"
    else:
        malformed["optimizer"]["value"]["param_groups"][0]["params"].pop()
    with pytest.raises(ValueError, match=match):
        unpack_state(malformed, state)
    assert equal(pack_state(state), before)


def test_stateful_signature_accounts_tensors_and_allows_scheduler_infinity():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    assert signature(scheduler)
    with pytest.raises(ValueError, match="serializable"):
        signature(object())
    scheduler.extra_tensor = torch.ones(3)
    assert storage_bytes(scheduler) == 12
    scheduler.extra_tensor[0] = 2
    first = signature(scheduler)
    scheduler.extra_tensor[0] = 3
    assert signature(scheduler) != first


def read_shared(ctx):
    return ctx.state["first"](ctx.inputs["x"]) + ctx.state["second"](ctx.inputs["x"])


def shared_workload(state):
    return Workload(Program((Call("read", "read", ("x",), tuple(state)),),
                            dict.fromkeys(state, 0), ("x",), ("read",)),
                    {"read": read_shared}, state, {"x": torch.ones(1, 2)},
                    immutable_modules=("first", "second"))


def test_immutable_alias_survives_later_cycles_and_checkpoint_restart(tmp_path):
    first = torch.nn.Linear(2, 2).requires_grad_(False)
    state = {"first": first, "second": deepcopy(first)}
    implementation = {"backend": "torch-callable-resident-v1", "share_immutable": True,
                      "checkpoint_bindings": []}
    policy = CyclePolicy({"mode": "manual", "implementation": implementation})
    session, _ = policy.run(shared_workload(state), tmp_path)
    assert session.state["first"].weight is session.state["second"].weight
    session, _ = policy.run(shared_workload(session.state), tmp_path)
    assert session.state["first"].weight is session.state["second"].weight
    checkpoint = save_checkpoint(tmp_path / "shared", session.state, identity={"test": "shared"},
                                 cycle=2, cursor=2, versions=dict.fromkeys(state, 0), policy=policy.bound)
    restarted = {"first": deepcopy(first), "second": deepcopy(first)}
    payload = restore_checkpoint(checkpoint, restarted, identity={"test": "shared"})
    resumed = CyclePolicy(policy.execution)
    resumed.restore(payload["policy"])
    session, _ = resumed.run(shared_workload(restarted), tmp_path)
    assert session.state["first"].weight is session.state["second"].weight


def test_source_identity_binds_loaded_compatibility_runtime(tmp_path, monkeypatch):
    from distillschedule.open_workload.planning import source_identity
    original = importlib.util.find_spec
    root = tmp_path / "runtime"
    root.mkdir()
    helper = root / "dmd_boundary.py"
    helper.write_text("coefficient = 0.5\n")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: (
        SimpleNamespace(submodule_search_locations=[str(root)]) if name == "runtime" else original(name)))
    first = source_identity()
    helper.write_text("coefficient = 5.0\n")
    assert source_identity() != first


def test_auto_plan_does_not_require_an_installed_git_executable(tmp_path, monkeypatch):
    import distillschedule.open_workload.planning as planning
    original = planning.subprocess.run
    def without_git(command, *args, **kwargs):
        if command[0] == "git":
            raise FileNotFoundError("no optional Git executable")
        return original(command, *args, **kwargs)
    monkeypatch.setattr(planning.subprocess, "run", without_git)
    first = torch.nn.Linear(2, 2).requires_grad_(False)
    state = {"first": first, "second": deepcopy(first)}
    policy = CyclePolicy({"mode": "auto", "memory_limit_bytes": 1_000_000,
                          "max_candidates": 1, "samples": 1, "max_program_runs": 2})
    session, _ = policy.run(shared_workload(state), tmp_path)
    assert session.receipt["status"] == "EXECUTED_OPEN_PROGRAM"
    calibration = json.loads((tmp_path / "planning/calibration/calibration.json").read_text())
    assert calibration["source_commit_available"] is False
    costs = json.loads(Path(calibration["variants"][0]["cost_model"]).read_text())
    assert costs["binding"]["source_head"] == "0" * 40


class ExtraStateModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.history = {"count": 0, "previous": torch.zeros(2)}

    def get_extra_state(self):
        return self.history

    def set_extra_state(self, value):
        self.history = value


def test_module_extra_state_is_validated_and_restored_as_a_safe_tree(tmp_path):
    module = ExtraStateModule()
    module.history["count"] = 3
    module.history["previous"].fill_(4)
    checkpoint = save_checkpoint(tmp_path / "extra", {"model": module}, identity={"test": "extra"},
                                 cycle=1, cursor=1, versions={"model": 1})
    module.history = {"count": 0, "previous": torch.zeros(2)}
    restore_checkpoint(checkpoint, {"model": module}, identity={"test": "extra"})
    assert module.history["count"] == 3
    assert torch.equal(module.history["previous"], torch.full((2,), 4.))
    module.history["previous"] = torch.zeros(3)
    with pytest.raises(ValueError, match="shape/dtype"):
        restore_checkpoint(checkpoint, {"model": module}, identity={"test": "extra"})


def test_public_model_contract_accepts_frozen_subtrees_not_trainable_ones():
    from distillschedule.training.adapters import ModelComponents
    student = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    student[0].requires_grad_(False)
    components = ModelComponents({"student": student}, immutable_modules=("student.0",))
    assert components.validate() is components
    with pytest.raises(ValueError, match="immutable|trainable"):
        ModelComponents({"student": student}, immutable_modules=("student.1",)).validate()
    with pytest.raises((ValueError, AttributeError), match="missing|submodule|attribute"):
        ModelComponents({"student": student}, immutable_modules=("student.missing",)).validate()


def test_failed_first_resumed_cycle_preserves_prior_complete_checkpoint(tmp_path, monkeypatch):
    from distillschedule.data import prepare
    from distillschedule.examples.tiny_av import build_fixture
    from distillschedule.training.config import TrainingConfig, load_training_config
    from distillschedule.training.session import train
    root = tmp_path / "fixture"
    build_fixture(root, samples=5)
    prepare(root / "raw/manifest.json", root / "prepared")
    config = load_training_config(root / "train.json")
    completed = train(config)
    raw = config.to_dict()
    raw["trainer"].update(cycles=5, resume_from=completed["checkpoint"], output_dir=str(root / "resumed"))
    def fail(*args, **kwargs):
        raise ValueError("injected policy admission failure")
    monkeypatch.setattr(CyclePolicy, "run", fail)
    with pytest.raises(ValueError, match="injected"):
        train(TrainingConfig.from_dict(raw, source=config.source))
    failure = json.loads((root / "resumed/failure.json").read_text())
    assert failure["completed_cycles_this_process"] == 0
    assert failure["last_complete_checkpoint"] == completed["checkpoint"]
