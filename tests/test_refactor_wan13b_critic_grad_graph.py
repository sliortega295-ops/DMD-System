from types import SimpleNamespace
import pytest
from distillschedule.runtime.wan13b_critic_grad_graph import (
    ENV, PLACEMENT, CriticGradGraph, requested, check_admission, install,
    validate_build,
)


def test_default_off_invalid_and_placement(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not requested()
    assert not install(object(), role="critic", placement=PLACEMENT,
                       reference_checks=True)["enabled"]
    monkeypatch.setenv(ENV, "bad")
    with pytest.raises(ValueError):
        requested()
    with pytest.raises(ValueError):
        check_admission("world16-fsdp", enabled=True, role="critic")
    assert not check_admission(PLACEMENT, enabled=True, role="generator")


def test_cpu_dispatch_update_backward_and_rng_contract(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(CriticGradGraph, "_audit_source", lambda self: None)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(4, 4)
            self._distillschedule_resident_clean_state = SimpleNamespace(
                training=True, adapter_enabled=True)

        def forward(self, hidden_states, return_dict=False):
            assert return_dict is False
            return (self.lora_A(hidden_states),)

    model = Model()
    native = model.forward

    def capture(self, keys, values):
        self.receipt["capture_entries"] += 1
        self.receipt["cold_original_forward_calls"] += 4
        self.receipt["cold_original_backward_calls"] += 4
        return lambda *xs: self.original_forward(**dict(zip(keys, xs)),
                                                 return_dict=False)[0]

    monkeypatch.setattr(CriticGradGraph, "_capture", capture)
    receipt = install(model, role="critic", placement=PLACEMENT,
                      reference_checks=True)
    rng = torch.get_rng_state().clone()
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    for i in range(10):
        with torch.no_grad():
            model.lora_A.weight.add_(0.001)
            torch.testing.assert_close(model(hidden_states=x, return_dict=False)[0],
                                       native(hidden_states=x)[0], rtol=0, atol=0)
        model.lora_A.zero_grad()
        model(hidden_states=x+i, return_dict=False)[0].sum().backward()
        grad = model.lora_A.weight.grad.clone()
        model.lora_A.zero_grad()
        native(hidden_states=x+i)[0].sum().backward()
        torch.testing.assert_close(grad, model.lora_A.weight.grad, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    build = {"critic": {"metadata": {"critic_grad_cuda_graph": receipt}}}
    validate_build(build, cycles=2, reference_checks=True)
    with pytest.raises(ValueError, match="signature after cold"):
        model(hidden_states=torch.zeros(3, 4), return_dict=False)


def test_default_off_and_capture_boundaries_source():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (root/"src/distillschedule/runtime/wan13b_critic_grad_graph.py").read_text()
    assert "torch.cuda.make_graphed_callables" in source
    assert "loss.backward()" not in source.split("class CriticGradGraph:")[1]
    assert "num_warmup_iters=3, allow_unused_input=False" in source
    assert "ALL_GRAPH_GRADS_READY_AFTER_BACKWARD_REPLAY" in source
    assert "float().cpu()" not in source
    assert "optimizer.step" not in source
    for name in ("probe_wan13b_trajectory_world16.py", "probe_wan13b_two_cycle_timing_world16.py"):
        assert "critic_grad_cuda_graph_receipt=" in (root/"scripts/distillschedule"/name).read_text()
