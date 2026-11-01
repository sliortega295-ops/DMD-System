"""CPU compatibility regressions; no model weights or weakened numerical gates."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.adapters.helios.reference_helpers import _bind_block_noise
from runtime.official_baseline import stable_block_noise

ROOT = Path(__file__).resolve().parents[1]


def original_sampler():
    # Execute this actual pinned primitive without importing the full model stack.
    # The full-source CI also runs the unchanged original G/C/EMA owners.
    import math
    import os
    path = ROOT / 'third_party/helios_compat/source/helios/utils/utils_helios_post.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_block_noise')
    namespace = dict(torch=torch, math=math, os=os)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['sample_block_noise']


def scheduler(gamma):
    return SimpleNamespace(config=SimpleNamespace(gamma=gamma))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reference_psd_sampler_preserves_covariance_shape_rng_and_dtype(monkeypatch, dtype):
    monkeypatch.delenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE', raising=False)
    original = original_sampler()
    sample = _bind_block_noise(original)
    before_dtype, before_rng = torch.get_default_dtype(), torch.get_rng_state()
    try:
        torch.set_default_dtype(dtype)
        torch.manual_seed(17)
        actual = sample(scheduler(1/3), 1, 2, 3, 4, 4)
        after = torch.get_rng_state()
        torch.manual_seed(17)
        raw = stable_block_noise(24, dtype=dtype)
        expected = raw.view(1,2,3,2,2,2,2).permute(0,1,2,3,5,4,6).reshape(1,2,3,4,4)
        assert torch.equal(actual, expected)
        assert torch.equal(after, torch.get_rng_state())
        assert actual.dtype == dtype and actual.device.type == 'cpu'
        assert torch.isfinite(actual).all()
        # Exact covariance of sqrt(4/3) * (I - 11^T/4); no jitter term.
        projection = (torch.eye(4,dtype=torch.float64) - torch.ones(4,4,dtype=torch.float64)/4) * (4/3)**.5
        covariance = projection @ projection.T
        target = torch.eye(4,dtype=torch.float64)*(4/3) - torch.ones(4,4,dtype=torch.float64)/3
        torch.testing.assert_close(covariance, target, rtol=0, atol=3e-16)
        # No skipped or extra random draws relative to the native standard-normal shape.
        torch.manual_seed(17)
        torch.empty((24,4), dtype=dtype).normal_()
        assert torch.equal(after, torch.get_rng_state())
    finally:
        torch.set_default_dtype(before_dtype)
        torch.set_rng_state(before_rng)


@pytest.mark.parametrize('gamma', [0., .2])
def test_nonsingular_gamma_keeps_original_samples_and_rng(monkeypatch, gamma):
    monkeypatch.delenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE', raising=False)
    original = original_sampler()
    sample = _bind_block_noise(original)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        expected = original(scheduler(gamma),1,2,3,4,4)
        state = torch.get_rng_state()
        torch.manual_seed(51)
        actual = sample(scheduler(gamma),1,2,3,4,4)
        assert torch.equal(actual,expected) and torch.equal(state,torch.get_rng_state())


def test_sampler_preserves_explicit_policy_and_invalid_covariance_rejection(monkeypatch):
    original = original_sampler()
    sample = _bind_block_noise(original)
    monkeypatch.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE','1')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9); expected = original(scheduler(1/3),1,1,1,2,2)
        torch.manual_seed(9); actual = sample(scheduler(1/3),1,1,1,2,2)
        assert torch.equal(actual,expected)
    with pytest.raises(ValueError,match='gamma=1/3'):
        sample(scheduler(.2),1,1,1,2,2)
    monkeypatch.delenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE')
    with pytest.raises(ValueError):
        sample(scheduler(.4),1,1,1,2,2)
    assert sample.__wrapped__ is original
