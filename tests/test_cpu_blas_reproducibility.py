"""Exact CPU serialization checks need a fixed BLAS arithmetic policy.

Safetensors/default loaders may preserve tensor values and strides but change
storage alignment. This regression varies only that alignment, not precision,
weights, input, or the equality requirement. It is not a performance benchmark.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_cpu_linear_exact_across_storage_offsets_in_fresh_process():
    code = r'''
import json, torch

torch.set_num_threads(1)
checked = 0
for seed in range(20):
    torch.manual_seed(seed)
    reference = torch.nn.Linear(32, 1).eval()
    inputs = torch.randn(3, 32)
    with torch.no_grad():
        expected = reference(inputs)
    for offset in range(16):
        backing = torch.empty(32 + offset)
        weight = backing[offset:].view_as(reference.weight)
        weight.copy_(reference.weight.detach())
        restored = torch.nn.Linear(32, 1).eval()
        restored.load_state_dict({
            'weight': weight,
            'bias': reference.bias.detach().clone(),
        }, assign=True)
        assert torch.equal(restored.weight, reference.weight)
        assert restored.weight.stride() == reference.weight.stride()
        assert restored.weight.data_ptr() == weight.data_ptr()
        with torch.no_grad():
            actual = restored(inputs)
        assert torch.equal(actual, expected), (seed, offset, (actual - expected).abs().max())
        checked += 1
assert not torch.cuda.is_initialized()
print(json.dumps({'checked': checked, 'policy': 'MKL_CBWR=COMPATIBLE'}))
'''
    # Must be set before importing Torch/MKL in this fresh process. Do not mutate
    # the caller's BLAS settings or apply the CI policy to production workloads.
    env = dict(os.environ, MKL_CBWR='COMPATIBLE', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['checked'] == 320


def test_helios_exact_ci_sets_policy_before_python_starts():
    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/helios-reference-model-cpu.yml').read_text())
    job = workflow['jobs']['original-full-model']
    assert job['env']['MKL_CBWR'] == 'COMPATIBLE'
    assert job['env']['OMP_NUM_THREADS'] == job['env']['MKL_NUM_THREADS'] == '1'
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    assert 'test_cpu_blas_reproducibility.py' in commands
    assert 'test_helios_default_loader_diagnostics.py' in commands
    assert 'test_helios_reference_model_cpu.py' in commands
