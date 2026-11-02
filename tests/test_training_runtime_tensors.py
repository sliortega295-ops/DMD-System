"""Explicit live placement avoids treating serialized host metadata as compute."""
import pytest
import torch
from distillschedule.open_workload.torch_backend import tensors, signature


class NativeOwner:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([2.]))
        self.rng = torch.get_rng_state()
    def state_dict(self):
        return {'weight': self.parameter, 'rng': self.rng}
    def load_state_dict(self, value):
        self.parameter.data.copy_(value['weight']); self.rng = value['rng']
    def runtime_tensors(self):
        yield self.parameter


def test_native_compute_placement_excludes_serialized_host_metadata_but_identity_keeps_it():
    owner = NativeOwner()
    assert list(tensors(owner)) == [owner.parameter]
    assert 'rng' in signature(owner)['state']


def test_runtime_tensor_protocol_rejects_non_tensor_values():
    class BadOwner(NativeOwner):
        def runtime_tensors(self):
            yield 'not a live tensor'
    with pytest.raises(ValueError, match='runtime_tensors'):
        list(tensors(BadOwner()))


def test_finite_checks_scan_unique_tensors_and_read_one_scalar_per_device(monkeypatch):
    from distillschedule.training.session import _finite
    a, b = torch.ones(4), torch.tensor([2., float('nan')])
    reads = []
    real_bool = torch.Tensor.__bool__
    def observe(value):
        reads.append(value.device)
        return real_bool(value)
    monkeypatch.setattr(torch.Tensor, '__bool__', observe)
    assert _finite({'a': a, 'again': a, 'b': b}, {}) is False
    assert len(reads) == 1
