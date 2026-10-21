"""Real user-facing lifecycle acceptance; no pretrained-quality claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_persistent_session_keeps_optimizer_parameter_ownership():
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.workloads.program import Call, Program
    from distillschedule.open_workload.torch_backend import Workload
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    state = {"model": model, "optimizer": optimizer}
    workload = Workload(Program((Call("read", "read", ("x",), ("model",)),),
                                {"model": 0, "optimizer": 0}, ("x",), ("read",)),
                        {"read": predict}, state, {"x": torch.ones(1, 2)})
    session = Session(workload, {"backend": "torch-callable-resident-v1",
                                "share_immutable": False, "checkpoint_bindings": []},
                      copy_state=False)
    assert session.state["model"] is model
    assert session.state["optimizer"] is optimizer
    session.run()


def predict(ctx):
    return ctx.state["model"](ctx.inputs["x"])


def test_checkpoint_complete_boundary_roundtrip_and_rejects_tampering(tmp_path):
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = {"model": model, "optimizer": optimizer, "history": torch.ones(2)}
    expected = {key: val.clone() for key, val in model.state_dict().items()}
    checkpoint = save_checkpoint(tmp_path / "checkpoint", state, identity={"config": "abc"},
                                 cycle=2, cursor=2, versions={"model": 2, "optimizer": 2})
    with torch.no_grad():
        model.weight.zero_()
    restored = restore_checkpoint(checkpoint, state, identity={"config": "abc"})
    assert restored["cycle"] == restored["cursor"] == 2
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(checkpoint, state, identity={"config": "different"})
    (checkpoint / "_SUCCESS").unlink()
    with pytest.raises(ValueError, match="complete"):
        restore_checkpoint(checkpoint, state, identity={"config": "abc"})
