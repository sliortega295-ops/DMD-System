"""Continuous W5 runtime contracts; tiny CPU tensors, no benchmark evidence."""
import ast
from pathlib import Path

import pytest

from distillschedule.workloads.best_known.registry import load_profile, repository_root


def test_current_profile_and_both_native_entries_use_continuous_sigma():
    profile = load_profile('minimax_h3_continuous_best')
    assert profile['semantics'] == profile['contract']['sigma_semantics'] == 'continuous'
    assert 'adaln_grid_manifest' not in profile['required_assets']
    assert 'adaln_grid' not in profile['runtime']['model_config']['training']['a100']
    backend = repository_root() / 'src/distillschedule/backends/training/minimax_h3'
    for name in ('entrypoint.py', 'integrated_entrypoint.py'):
        tree = ast.parse((backend / name).read_text())
        assert not any('grid' in alias.name for node in ast.walk(tree)
                       if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names)
    assert 'current_training' in (backend / 'entrypoint.py').read_text()
    assert not (backend / 'grid_entrypoint.py').exists()
    with pytest.raises(ValueError):
        load_profile('minimax_h3_grid1000_best')


def test_exact_cache_stores_contiguous_modulations_without_changing_values():
    import torch
    from distillschedule.backends.training.minimax_h3.adaln_cache import AdaLNCacheController
    class Projection(torch.nn.Module):
        def forward(self, temb):
            return (temb @ torch.arange(48, dtype=temb.dtype).reshape(2, 24)).chunk(6, dim=-1)
    controller = AdaLNCacheController()
    value = torch.ones(2, 2)
    expected = Projection()(value)
    with controller.scope(('score', 0)):
        actual = controller.get_or_compute(0, value, Projection())
        again = controller.get_or_compute(0, value, Projection())
    assert all(torch.equal(left, right) and right.is_contiguous() for left, right in zip(expected, actual))
    assert all(left is right for left, right in zip(actual, again))
    assert controller.stats().misses == 1 and controller.stats().hits == 1
    assert not torch.cuda.is_initialized()


def test_native_checkpoint_replay_restores_exact_continuous_cache_scope(monkeypatch):
    import torch
    from torch.utils.checkpoint import checkpoint
    from distillschedule.backends.training.minimax_h3.adaln_cache import AdaLNCacheController
    from distillschedule.backends.training.minimax_h3.exact_adaln_replay import install_exact_adaln_checkpoint_replay_scope
    controller = AdaLNCacheController()
    model = torch.nn.Module()
    model.gradient_checkpointing = True
    model._gradient_checkpointing_func = lambda fn, *args: checkpoint(fn, *args, use_reentrant=False)
    class Block(torch.nn.Module):
        def forward(self, value):
            modulation = controller.get_or_compute(0, value, torch.nn.Identity())[0]
            return value.sin() * modulation
    model.transformer_blocks = torch.nn.ModuleList([Block()])
    key = ('continuous', .123456789)
    with controller.scope(key):
        controller.get_or_compute(0, torch.ones(2, 3), torch.nn.Identity())
    registration = install_exact_adaln_checkpoint_replay_scope(model, controller, expected_block_count=1)
    def forbidden_scan():
        raise AssertionError('checkpoint replay must not scan the whole cache')
    monkeypatch.setattr(controller, 'stats', forbidden_scan)
    x = torch.randn(2, 3, requires_grad=True)
    with controller.scope(key):
        y = model._gradient_checkpointing_func(model.transformer_blocks[0], x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    assert registration.stats['checkpoint_wrap_count'] == 1
    assert registration.stats['scoped_execution_count'] == registration.stats['cache_hit_count'] == 2
    assert registration.stats['cache_miss_count'] == 0
    assert controller.current_scope().key is None
    registration.remove()
    registration.remove()  # Idempotent teardown leaves the original owner installed.
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['missing_key', 'incomplete_key', 'miss', 'extra_hit', 'block_error'])
def test_exact_checkpoint_replay_rejects_bad_scopes_and_restores_context(fault):
    import torch
    from distillschedule.backends.training.minimax_h3.adaln_cache import AdaLNCacheController
    from distillschedule.backends.training.minimax_h3.exact_adaln_replay import install_exact_adaln_checkpoint_replay_scope
    controller = AdaLNCacheController()
    model = torch.nn.Module()
    model.gradient_checkpointing = True
    original = lambda fn, *args: fn(*args)
    model._gradient_checkpointing_func = original
    registration = install_exact_adaln_checkpoint_replay_scope(model, controller, expected_block_count=1)
    key = ('continuous', .125)
    x = torch.ones(2, 3)
    if fault != 'missing_key':
        with controller.scope(key):
            if fault != 'incomplete_key':
                controller.get_or_compute(0, x, torch.nn.Identity())

    def block(value):
        controller.get_or_compute(0, value, torch.nn.Identity())
        if fault == 'miss':
            controller.get_or_compute(1, value, torch.nn.Identity())
        elif fault == 'extra_hit':
            controller.get_or_compute(0, value, torch.nn.Identity())
        elif fault == 'block_error':
            raise ValueError('original block error')
        return value

    errors = dict(missing_key='no active cache key', incomplete_key='incomplete',
                  miss='cache miss', extra_hit='one cache hit', block_error='original block error')
    with pytest.raises((RuntimeError, ValueError), match=errors[fault]):
        with controller.scope(None if fault == 'missing_key' else key):
            model._gradient_checkpointing_func(block, x)
    assert controller.current_scope().key is None
    assert registration.stats['captured_key_missing_count'] == int(fault == 'missing_key')
    assert registration.stats['captured_key_incomplete_count'] == int(fault == 'incomplete_key')
    assert registration.stats['cache_miss_count'] == int(fault == 'miss')
    assert registration.stats['unexpected_hit_delta_count'] == int(fault == 'extra_hit')
    registration.remove()
    assert model._gradient_checkpointing_func is original
    assert not torch.cuda.is_initialized()


def test_real_setup_installs_enabled_fa3_after_native_checkpoint_and_before_exact_scope(monkeypatch):
    """Execute production installation statements; model/kernel boundaries are CPU fixtures."""
    from types import SimpleNamespace as NS
    import torch
    from distillschedule.backends.training.minimax_h3 import fa3_replay_cache as fa3
    from distillschedule.backends.training.minimax_h3 import fa3_replay_setup as setup
    from distillschedule.backends.training.minimax_h3.adaln_cache import AdaLNCacheController
    from distillschedule.backends.training.minimax_h3.exact_adaln_replay import install_exact_adaln_checkpoint_replay_scope
    path = repository_root() / 'src/distillschedule/backends/training/minimax_h3/trainer.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'MiniMaxH3A100DmdTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'setup')
    statements = []
    for node in method.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) in (
                'self.gradient_checkpointing', 'self.activation_policy == CHECKPOINT_BOUNDARY_CPU'):
            statements.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'install_configured_fa3_replay':
            statements.append(node)
    assert len(statements) == 4  # Neither omit nor reorder the production installation branches.
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(50))
    controller = AdaLNCacheController()
    installed = {}

    def enable():
        transformer.gradient_checkpointing = True
        transformer._gradient_checkpointing_func = lambda fn, *args, **kwargs: fn(*args)
        installed['native'] = transformer._gradient_checkpointing_func

    def boundary(module, **kwargs):
        assert module._gradient_checkpointing_func is installed['native']
        module._gradient_checkpointing_func = lambda fn, *args, **kw: installed['native'](fn, *args, **kw)
        installed['boundary'] = module._gradient_checkpointing_func

    kernel = NS(kernel_fn=lambda *args, **kwargs: None)
    monkeypatch.setattr(fa3, '_RUNTIME', None)
    def install(module, **kwargs):
        assert module._gradient_checkpointing_func is installed['boundary']
        return fa3.install_fa3_replay_cache(module, **kwargs, kernel_config=kernel,
            raw_forward=lambda *args: None, raw_backward=lambda *args: None)
    monkeypatch.setattr(setup, 'install_fa3_replay_cache', install)
    shared = NS(enable_gradient_checkpointing=enable, denoiser_module=lambda: transformer,
                adaln_cache=lambda: controller, _role=NS(get=lambda: 'student'))
    owner = NS(gradient_checkpointing=True, activation_policy='boundary', boundary_offload_events=False,
               boundary_offload_pin_memory=True, shared_model=shared, fa3_replay_cache_blocks=tuple(range(50)),
               fa3_replay_cache_storage='cuda', fa3_replay_cache_max_d2h_inflight=2,
               fa3_replay_cache_trim_before_backward=False,
               fa3_nograd_split_registration=NS(enabled=True, grad_num_splits=2))
    namespace = dict(self=owner, model=shared, bank=NS(projections=range(50)),
                     CHECKPOINT_BOUNDARY_CPU='boundary', install_checkpoint_boundary_cpu_offload=boundary,
                     install_configured_fa3_replay=setup.install_configured_fa3_replay,
                     install_exact_adaln_checkpoint_replay_scope=install_exact_adaln_checkpoint_replay_scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(path), 'exec'), namespace)
    try:
        assert owner.fa3_replay_cache_registration.enabled
        assert owner.fa3_replay_cache_registration.original_checkpoint is installed['boundary']
        assert transformer._gradient_checkpointing_func is not owner.exact_adaln_replay_registration.original_checkpoint
    finally:
        owner.exact_adaln_replay_registration.remove()
        owner.fa3_replay_cache_registration.pre_hook.remove()
        owner.fa3_replay_cache_registration.post_hook.remove()
    assert not torch.cuda.is_initialized()
