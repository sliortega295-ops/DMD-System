"""C31 is initially canary-only; the C29/C30 runtime stays default unchanged."""
import inspect
import json
from pathlib import Path
import subprocess
import sys

from distillschedule.runtime.livewan_generator_rms_epilogue import (
    build_cast_square, build_nograd_generator_rms_epilogue, PLAN_CONTRACT,
)


def test_explicit_default_off_and_registered_runtime_call_site():
    parameter = inspect.signature(build_nograd_generator_rms_epilogue).parameters['fuse_cast_square']
    assert parameter.default is False
    root = Path(__file__).resolve().parents[1]
    worker = (root/'src/distillschedule/runtime/livewan_crossscale_worker.py').read_text()
    assert 'fuse_cast_square=_generator_rms_cast_square_enabled(plan, mode=mode)' in worker
    assert PLAN_CONTRACT['native_pow_mean_preserved'] is True


def test_cast_square_has_no_reduction_or_cpu_side_effect():
    source = inspect.getsource(build_cast_square)
    assert 'value * value' in source
    assert 'torch.bfloat16' in source and 'torch.float32' in source
    assert 'enable_fp_fusion=False' in source
    for token in ('tl.sum', '.mean(', '.cpu(', '.item(', '.tolist(', 'synchronize(', 'all_reduce', '.step('):
        assert token not in source


def test_original_native_mean_and_truthful_receipt():
    source = inspect.getsource(build_nograd_generator_rms_epilogue)
    assert 'x.float().pow(2).mean(dim=-1, keepdim=True)' in source
    assert 'square(x).mean(dim=-1, keepdim=True)' in source
    assert 'native_pow_mean_preserved=False' in source
    assert 'native_mean_preserved=True' in source


def test_c31_plan_only_does_not_load_assets_or_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'unused.json'
    process = subprocess.run([sys.executable, str(root/'benchmarks/kernels/canary_livewan_generator_rms_epilogue.py'),
        '--trace', '/unread', '--bootstrap', '/unread', '--output', str(output),
        '--fuse-cast-square', '--plan-only'], check=True, capture_output=True, text=True)
    result = json.loads(process.stdout)
    assert result['status'] == result['world16'] == 'NOT_RUN'
    assert result['parent'] == 'C30'
    assert result['square_and_variance_parity'] == 'BITWISE_REQUIRED'
    assert result['production_installed'] is False
    assert not output.exists()
