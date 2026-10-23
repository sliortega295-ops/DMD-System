"""Lightweight factory discovery, source locks and CPU attention semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_factory_discovery_never_imports_models_torch_or_cuda():
    script = '''
import json, sys
from distillschedule.training.self_forcing import build_model, build_self_forcing
assert callable(build_model) and callable(build_self_forcing)
assert 'torch' not in sys.modules
assert not any(name.startswith('_distillschedule_upstream_') for name in sys.modules)
print(json.dumps({'discovery': 'NO_MODEL_CUDA_OR_NETWORK_IMPORT'}))
'''
    result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True, env=os.environ, check=True)
    assert json.loads(result.stdout)['discovery'] == 'NO_MODEL_CUDA_OR_NETWORK_IMPORT'


def test_source_hash_and_uncached_causal_failure_are_explicit(tmp_path):
    source = os.environ.get('SELF_FORCING_SOURCE_ROOT')
    if not source:
        pytest.skip('optional official source: set SELF_FORCING_SOURCE_ROOT')
    import shutil
    from distillschedule.training.self_forcing import original
    from distillschedule.training.self_forcing_source import _uncached_not_admitted
    copied = tmp_path / 'changed-source'; shutil.copytree(source, copied)
    path = copied / 'model/dmd.py'; path.write_text(path.read_text() + '\n# changed source\n')
    with pytest.raises(ValueError, match='SHA256'):
        original(copied)
    with pytest.raises(ValueError, match='uncached causal/flex'):
        _uncached_not_admitted()


def test_cpu_attention_is_math_sdpa_with_varlen_and_bottom_right_causal_masks():
    import torch
    from distillschedule.training.self_forcing_source import cpu_attention
    q = torch.linspace(-1., 1., 24).reshape(1, 3, 2, 4).requires_grad_(True)
    k = torch.linspace(.3, -.4, 40).reshape(1, 5, 2, 4).requires_grad_(True)
    v = torch.linspace(.7, .1, 40).reshape(1, 5, 2, 4).requires_grad_(True)
    actual = cpu_attention(q, k, v, causal=True, window_size=(2, 0), softmax_scale=.7, q_scale=1.2)
    query = q.transpose(1, 2) * 1.2
    scores = query @ k.transpose(1, 2).transpose(-1, -2) * .7
    qp = torch.arange(3)[:, None] + 2; kp = torch.arange(5)[None, :]
    mask = (kp <= qp) & (kp >= qp - 2)
    expected = (scores.masked_fill(~mask, -float('inf')).softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual_grad = torch.autograd.grad(actual.sum(), (q, k, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (q, k, v))
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    padded = cpu_attention(q.detach(), k.detach(), v.detach(), q_lens=torch.tensor([2]), k_lens=torch.tensor([4]))
    assert padded.shape == q.shape and torch.equal(padded[:, 2:], torch.zeros_like(padded[:, 2:]))
    with pytest.raises(ValueError, match='FP32'):
        cpu_attention(q.detach().double(), k.detach().double(), v.detach().double())


def test_math_attention_clones_packed_cache_reads_and_rejects_cross_device_inputs():
    import torch
    from distillschedule.training.self_forcing_source import fp32_attention
    q = torch.linspace(-.4, .8, 24).reshape(1, 3, 2, 4).requires_grad_(True)
    k = torch.linspace(.1, .9, 40).reshape(1, 5, 2, 4)
    v = k.flip(1).clone()
    expected = fp32_attention(q, k.clone(), v.clone(), causal=True, window_size=(2, 0))
    expected_grad, = torch.autograd.grad(expected.sum(), (q,))
    actual = fp32_attention(q, k, v, causal=True, window_size=(2, 0))
    # Native subsequent block writes must not invalidate backward saved views.
    k.add_(2); v.mul_(3)
    actual_grad, = torch.autograd.grad(actual.sum(), (q,))
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    with pytest.raises(ValueError, match='single-device'):
        fp32_attention(q, k.to('meta'), v)
