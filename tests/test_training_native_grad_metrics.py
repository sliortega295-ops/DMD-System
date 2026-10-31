"""Native owner row ABI through the common metric publication path.

Rows mirror DMD2Owner._native_iteration and SelfForcingOwner.run. These are
metadata fixtures, not model training or numerical-equivalence evidence.
"""
from copy import deepcopy
import json

import pytest
import torch

from distillschedule.training.adapters import ModelComponents
from distillschedule.training.sd import collect_metrics as sd_metrics
from distillschedule.training.self_forcing import collect_metrics as sf_metrics
from distillschedule.training.session import _append_json, _training_metrics


@pytest.mark.parametrize("native,collector,other,generator_loss,loss", [
    ("dmd2.native_cycle", sd_metrics, "guidance", "loss_dm", "loss_fake_mean"),
    ("self_forcing.native_cycle", sf_metrics, "critic", "generator_loss", "critic_loss"),
])
def test_native_iteration_grad_norms_keep_loss_names_and_optional_generator(
        native, collector, other, generator_loss, loss):
    rows = [
        {"native_step": 35, "generator_turn": True,
         "losses": {generator_loss: 1.25, loss: 2.5},
         "generator_grad_norm": 3.75, f"{other}_grad_norm": 4.5},
        {"native_step": 36, "generator_turn": False, "losses": {loss: 2.25},
         f"{other}_grad_norm": 4.25},
        {"native_step": 37, "generator_turn": False, "losses": {loss: 2.0},
         f"{other}_grad_norm": 0.0},
    ]
    outputs = {native: {"iterations": rows, "native_step": 38}}
    before = deepcopy(outputs)
    metrics = collector(outputs)
    assert metrics == {
        f"native_0/{generator_loss}": 1.25, f"native_0/{loss}": 2.5,
        "native_0/generator_grad_norm": 3.75, f"native_0/{other}_grad_norm": 4.5,
        f"native_1/{loss}": 2.25, f"native_1/{other}_grad_norm": 4.25,
        f"native_2/{loss}": 2.0, f"native_2/{other}_grad_norm": 0.0,
    }
    assert outputs == before
    assert all(type(value) is float for value in metrics.values())


@pytest.mark.parametrize("native,collector,loss", [
    ("dmd2.native_cycle", sd_metrics, "loss_fake_mean"),
    ("self_forcing.native_cycle", sf_metrics, "critic_loss"),
])
def test_older_rows_without_norms_keep_existing_metrics(native, collector, loss):
    assert collector({native: {"iterations": [{"losses": {loss: 1.25}}]}}) == {
        f"native_0/{loss}": 1.25,
    }


@pytest.mark.parametrize("native,collector,other,loss", [
    ("dmd2.native_cycle", sd_metrics, "guidance", "loss_fake_mean"),
    ("self_forcing.native_cycle", sf_metrics, "critic", "critic_loss"),
])
def test_common_publication_uses_host_norms_without_tensor_reads(
        native, collector, other, loss, monkeypatch, tmp_path):
    debug_tensor = torch.tensor([101.0, 202.0])
    outputs = {native: {"iterations": [
        {"native_step": 10, "generator_turn": True,
         "losses": {loss: 1.5}, "generator_grad_norm": 2.5,
         f"{other}_grad_norm": 3.5, "debug": debug_tensor},
        {"native_step": 11, "generator_turn": False,
         "losses": {loss: 1.75}, f"{other}_grad_norm": 3.75,
         "debug": debug_tensor},
    ], "debug": debug_tensor}}
    expected = {f"native_0/{loss}": 1.5, "native_0/generator_grad_norm": 2.5,
                f"native_0/{other}_grad_norm": 3.5, f"native_1/{loss}": 1.75,
                f"native_1/{other}_grad_norm": 3.75}
    if native == "self_forcing.native_cycle":
        outputs[native]["cache_lifetime_effects"] = {
            "releases": {"generator": 1, "critic": 5},
            "released_owner_storage_bytes": 256,
        }
        expected.update({"cache_lifetime/releases_generator": 1.0,
                         "cache_lifetime/releases_critic": 5.0,
                         "cache_lifetime/released_owner_storage_bytes": 256.0})
    components = ModelComponents(state={"native_cycle": object()}, collect_metrics=collector)

    def forbidden(*args, **kwargs):
        raise AssertionError("metric publication must reuse native host values")

    path = tmp_path / "metrics.jsonl"
    with monkeypatch.context() as guarded:
        for name in ("cpu", "to", "detach", "numpy", "tolist", "item", "__bool__", "__float__"):
            guarded.setattr(torch.Tensor, name, forbidden)
        guarded.setattr(torch.cuda, "synchronize", forbidden)
        for cycle in (1, 2):
            metrics = _training_metrics(components, outputs)
            _append_json(path, {"cycle": cycle, "metrics": metrics})
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"cycle": cycle, "metrics": expected} for cycle in (1, 2)
    ]
    assert outputs[native]["iterations"][0]["debug"] is debug_tensor
    assert outputs[native]["debug"] is debug_tensor
    assert not torch.cuda.is_initialized()
