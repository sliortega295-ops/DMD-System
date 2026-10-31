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
    p = subprocess.run([sys.executable, str(root / 'benchmarks/kernels/canary_livewan_rollout_cast_cache.py'),
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


def test_c28_plan_changes_only_rollout_cast_cache(monkeypatch):
    import copy
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_rollout_cast_cache import PLAN_CONTRACT
    from distillschedule.runtime.livewan_crossscale_worker import _generator_rollout_cast_cache_enabled
    root = Path(__file__).resolve().parents[1] / 'configs/distillschedule/plans'
    parent = root/'livewan_wan13b_14b_sf_dmd_c25_continuous_b16_score_rmsnorm.json'
    candidate = root/'livewan_wan13b_14b_sf_dmd_c28_continuous_b16_rollout_cast_cache.json'
    a, b = json.loads(parent.read_text()), json.loads(candidate.read_text())
    assert b['metadata'].pop('generator_rollout_cast_cache') == PLAN_CONTRACT
    assert a == b
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key, '/unread/'+key)
    plan = load_plan(candidate).plan.to_dict()
    assert _generator_rollout_cast_cache_enabled(plan, mode='pipeline')
    assert not _generator_rollout_cast_cache_enabled(load_plan(parent).plan.to_dict(), mode='pipeline')
    with pytest.raises(RuntimeError):
        _generator_rollout_cast_cache_enabled(plan, mode='baseline')
    for key in ('outer_autocast_enabled','changes_effective_dtype','cache_cross_optimizer'):
        changed = copy.deepcopy(plan)
        changed['metadata']['generator_rollout_cast_cache'][key] = True
        with pytest.raises(RuntimeError):
            _generator_rollout_cast_cache_enabled(changed, mode='pipeline')
    for key in ('score_prompt_staging','critic_attention_checkpoint'):
        changed = copy.deepcopy(plan)
        changed['metadata'][key] = {}
        with pytest.raises(RuntimeError):
            _generator_rollout_cast_cache_enabled(changed, mode='pipeline')


def test_default_model_scope_and_actual_receipt_gate():
    from types import SimpleNamespace
    from distillschedule.runtime.livewan_rollout_cast_cache import model_rollout_cast_cache_scope, runtime_receipt_errors, NAME
    with model_rollout_cast_cache_scope(None, SimpleNamespace()):
        pass
    receipt = dict(name=NAME, execution_status='PASS', scope_entries=20, scope_exits=20,
                   active=False, contract_failures=0, effective_dtype_changed=False, cache_cross_optimizer=False)
    assert not runtime_receipt_errors(receipt, rank=0, cycles=2)
    for key, value in [('scope_entries',19),('scope_exits',19),('active',True),('contract_failures',1),('cache_cross_optimizer',True)]:
        assert runtime_receipt_errors(dict(receipt, **{key:value}), rank=0, cycles=2)
    assert runtime_receipt_errors(receipt, rank=8, cycles=2)
    assert not runtime_receipt_errors({'execution_status':'NOT_APPLICABLE'}, rank=8, cycles=2)


def test_scope_surrounds_only_native_nograd_rollout():
    import ast
    from distillschedule.runtime.livewan_crossscale_worker import prepare_rollout, generator_backward_for_sample
    source = inspect.getsource(prepare_rollout)
    tree = ast.parse(source)
    scopes = [n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        ast.unparse(item.context_expr) == 'model_rollout_cast_cache_scope(torch, student)'
        for item in n.items)]
    assert len(scopes) == 1
    assert ast.unparse(scopes[0].items[0].context_expr) == 'torch.no_grad()'
    assert ast.unparse(scopes[0].items[1].context_expr) == 'model_rollout_cast_cache_scope(torch, student)'
    lifetimes = [n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        ast.unparse(item.context_expr) == 'generator_text_kv_payload_lifetime(torch, streamer)'
        for item in n.items)]
    assert len(lifetimes) == 1 and scopes[0] in lifetimes[0].body
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) in ('student.text_embedding', 'streamer.rollout_record')]
    assert len(calls) == 2 and all(call in list(ast.walk(scopes[0])) for call in calls)
    assert 'cast_cache_scope' not in inspect.getsource(generator_backward_for_sample)
    assert '.step(' not in source
