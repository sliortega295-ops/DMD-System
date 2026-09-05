import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.runtime.livewan_rollout_cast_cache import rollout_cast_cache_scope


def test_default_off_does_not_even_access_torch():
    with rollout_cast_cache_scope(None):
        pass


def test_scope_has_no_sync_copy_rng_model_or_optimizer_calls():
    source = inspect.getsource(rollout_cast_cache_scope)
    for token in ("synchronize(", ".cpu(", ".item(", ".to(", "copy_(", "manual_seed(", ".step("):
        assert token not in source


def test_plan_only_needs_no_torch_or_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(root / 'scripts/distillschedule/canary_livewan_rollout_cast_cache.py'),
                        '--plan-only', '--trace', '/unread', '--bootstrap', '/unread',
                        '--output', str(tmp_path/'result.json')], capture_output=True, text=True, check=True)
    assert json.loads(p.stdout)['status'] == 'NOT_RUN'
    assert not (tmp_path/'result.json').exists()


def test_cpu_checkpoint_gradient_and_cache_lifecycle():
    torch = pytest.importorskip('torch')
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(53)
    layer = torch.nn.Linear(16, 8)
    x = torch.randn(4, 16)
    rng = torch.get_rng_state().clone()

    def forward():
        with torch.autocast('cpu', dtype=torch.bfloat16):
            return layer(x)

    with torch.no_grad():
        reference = forward()
        with rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
            assert not torch.is_autocast_enabled('cpu')
            a, b = forward(), forward()
        assert torch.equal(reference, a) and torch.equal(a, b)
        # An optimizer-equivalent in-place parameter version update occurs only
        # AFTER the cache scope exits; the next scope must not use stale casts.
        layer.weight.add_(0.125)
        changed = forward()
        with rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
            assert torch.equal(changed, forward())
        assert not torch.equal(reference, changed)
    assert torch.equal(rng, torch.get_rng_state())

    def gradient(after_cache):
        layer.zero_grad(set_to_none=True)
        if after_cache:
            with torch.no_grad(), rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
                forward()
        y = checkpoint(lambda value: forward(), x, use_reentrant=False)
        y.float().square().sum().backward()
        return {name: p.grad.clone() for name, p in layer.named_parameters()}
    a, b = gradient(False), gradient(True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(p.dtype == torch.float32 for p in layer.parameters())


def test_cpu_rejects_grad_and_restores_exception_state():
    torch = pytest.importorskip('torch')
    with pytest.raises(RuntimeError, match='requires no_grad'):
        with rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
            pass
    before = torch.get_autocast_dtype('cpu')
    with torch.no_grad():
        with pytest.raises(ValueError, match='deliberate'):
            with rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    raise ValueError('deliberate')
        assert not torch.is_autocast_enabled('cpu')
        assert torch.get_autocast_dtype('cpu') == before
        with torch.autocast('cpu'):
            with pytest.raises(RuntimeError, match='outer autocast disabled'):
                with rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
                    pass
        with torch.autocast('cpu', enabled=False, cache_enabled=False):
            with pytest.raises(RuntimeError, match='original cache enabled'):
                with rollout_cast_cache_scope(torch, enabled=True, device_type='cpu'):
                    pass
