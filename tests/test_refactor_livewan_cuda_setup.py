import ast
import inspect
from pathlib import Path
import pytest

from distillschedule.runtime.livewan_cuda_setup import materialize_blas_handles


def test_library_setup_never_touches_model_rng_optimizer_or_collectives():
    source = inspect.getsource(materialize_blas_handles)
    for forbidden in ('randn(', 'manual_seed(', 'empty_cache(', 'optimizer.step(',
                      'dist.barrier(', 'all_reduce(', 'model.forward('):
        assert forbidden not in source
    assert 'torch.cuda.current_blas_handle()' in source
    assert 'SETUP_ONLY' in source


def test_both_arms_share_setup_before_build_and_training():
    source = (Path(__file__).resolve().parents[1] /
              'src/distillschedule/runtime/livewan_crossscale_worker.py').read_text()
    tree = ast.parse(source)
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_worker')
    calls = [(n.lineno, n.func.id) for n in ast.walk(worker)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    setup = [line for line, name in calls if name == 'materialize_blas_handles']
    assert len(setup) == 1
    assert setup[0] < min(line for line, name in calls if name == 'build_models')
    assert setup[0] < min(line for line, name in calls if name == '_run_pipeline_8x8_continuous')
    # The call is unconditional in run_worker, not specific to the candidate.
    assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name)
               and n.value.func.id == 'materialize_blas_handles' for n in worker.body)


def test_cuda_setup_rng_parity():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('explicit owned-GPU canary required')
    torch.manual_seed(20260905)
    cpu, gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    receipt = materialize_blas_handles(torch, torch.device('cuda:0'))
    assert torch.equal(cpu, torch.get_rng_state())
    assert torch.equal(gpu, torch.cuda.get_rng_state())
    assert receipt['status'] == 'PASS'
    assert receipt['model_calls'] == receipt['optimizer_updates'] == receipt['random_draws'] == 0
