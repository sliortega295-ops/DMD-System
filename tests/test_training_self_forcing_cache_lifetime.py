"""Consumed native KV ownership, not an attention/rollout approximation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.training.checkpoint import cpu_tree, rng_state
from test_training_end_to_end import equal, run_cli
from test_training_self_forcing_cpu import fixture_assets, model_and_data, source_root


POLICY = 'release-after-commit-v1'


def policy_config(files):
    path = Path(files['config'])
    value = json.loads(path.read_text())
    value['model']['parameters']['native_cache_lifetime'] = POLICY
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize('checkpoint', [False, True])
def test_release_after_commit_keeps_exact_native_training(fixture_assets, checkpoint):
    path = Path(fixture_assets['config'])
    value = json.loads(path.read_text())
    value['model']['parameters']['settings']['gradient_checkpointing'] = checkpoint
    path.write_text(json.dumps(value))
    _, control_components, inputs = model_and_data(fixture_assets)
    policy_config(fixture_assets)
    _, selected_components, _ = model_and_data(fixture_assets)
    control = control_components.state['native_cycle']
    selected = selected_components.state['native_cycle']
    traces = [[], []]
    for owner, target in zip((control, selected), traces):
        owner.observer = lambda event, values, target=target: target.append((event, cpu_tree(values)))
    commits = []
    native_commit = selected._commit

    def observe(role):
        pipeline = selected.model.inference_pipeline
        assert pipeline.kv_cache1 is not None and pipeline.crossattn_cache is not None
        native_commit(role)
        assert pipeline.kv_cache1 is pipeline.kv_cache2 is pipeline.crossattn_cache is None
        commits.append(role)

    selected._commit = observe
    outer = rng_state()
    for cycle in range(3):
        feeds = inputs.cycle_sample(cycle)
        expected = control.run(feeds, cycle=cycle)
        actual = selected.run(feeds, cycle=cycle)
        effect = actual.pop('cache_lifetime_effects')
        assert effect['policy'] == POLICY and effect['scope'] == 'THIS_COMPLETED_CYCLE'
        assert effect['releases'] == {'generator': 1, 'critic': 5}
        assert effect['released_owner_storage_bytes'] > 0
        assert effect['helper_host_tensor_value_reads'] == 0
        assert equal(actual, expected) and equal(traces[0], traces[1])
        a, b = [cpu_tree(owner.state_dict()) for owner in (control, selected)]
        a.pop('identity'); b.pop('identity')
        assert equal(a, b), 'model, gradients, Adam, EMA, clocks, input progress and RNG must match'
        assert selected.model.inference_pipeline is None
    assert commits == ['generator', *(['critic'] * 5)] * 3
    assert equal(outer, rng_state())


def test_release_is_not_run_before_commit_on_failure(fixture_assets):
    policy_config(fixture_assets)
    _, components, inputs = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    def fail(event, value):
        if event == 'generator_gradients':
            raise RuntimeError('test stops before commit')
    owner.observer = fail
    with pytest.raises(RuntimeError, match='before commit'):
        owner.run(inputs.cycle_sample(0), cycle=0)
    assert owner.generator_updates == owner.critic_updates == 0
    assert owner.model.inference_pipeline.kv_cache1 is not None


@pytest.mark.parametrize('checkpoint', [False, True])
def test_consumed_cache_policy_public_fresh_process_restore(fixture_assets, tmp_path, checkpoint):
    path = policy_config(fixture_assets)
    value = json.loads(path.read_text())
    value['model']['parameters']['settings']['gradient_checkpointing'] = checkpoint
    path.write_text(json.dumps(value))
    first = run_cli('train', path, '--cycles', '1', '--output', tmp_path / 'first')
    resumed = run_cli('train', path, '--cycles', '3', '--output', tmp_path / 'resumed',
                       '--resume', first['checkpoint'])
    full = run_cli('train', path, '--cycles', '3', '--output', tmp_path / 'full')
    import torch
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b)
    owner = b['state']['native_cycle']['value']
    assert owner['native_step'] == owner['critic_updates'] == 15
    assert owner['generator_updates'] == 3
    assert not torch.cuda.is_initialized()


def test_release_accounts_aliased_storage_without_host_value_reads(monkeypatch):
    # SYNTHETIC_TEST_ONLY: the real helper's accounting/host-read boundary;
    # the native three-cycle tests above establish training behavior separately.
    import torch
    from distillschedule.training.self_forcing_owner import SelfForcingOwner
    base = torch.arange(32, dtype=torch.float32)
    extra = torch.ones(3, dtype=torch.int64)
    expected = base.untyped_storage().nbytes() + extra.untyped_storage().nbytes()
    pipeline = SimpleNamespace(kv_cache1={'k': base, 'v': base[1:3]},
        kv_cache2=[base.view(4, 8)], crossattn_cache={'k': base[3:], 'end': extra})
    owner = SelfForcingOwner.__new__(SelfForcingOwner)
    owner.model = SimpleNamespace(inference_pipeline=pipeline)
    owner._cache_cycle_effects = dict(releases=dict(generator=0, critic=0), released_owner_storage_bytes=0)

    def forbidden(*args, **kwargs):
        raise AssertionError('release helper must not read tensor values, transfer or synchronize')

    with monkeypatch.context() as guarded:
        for name in ('cpu', 'to', 'numpy', 'tolist', 'item', '__bool__', '__float__'):
            guarded.setattr(torch.Tensor, name, forbidden)
        guarded.setattr(torch.cuda, 'synchronize', forbidden)
        owner._release_consumed_rollout_cache('critic')
    assert pipeline.kv_cache1 is pipeline.kv_cache2 is pipeline.crossattn_cache is None
    assert owner._cache_cycle_effects == dict(releases=dict(generator=0, critic=1),
                                             released_owner_storage_bytes=expected)


def test_cache_metrics_preserve_common_float_contract():
    # SYNTHETIC_TEST_ONLY: collector ABI, not native execution evidence.
    from distillschedule.training.self_forcing import collect_metrics
    value = {'self_forcing.native_cycle': dict(iterations=[{'losses': {'critic_loss': 1.25}}],
        cache_lifetime_effects=dict(releases=dict(generator=1, critic=5), released_owner_storage_bytes=256))}
    metrics = collect_metrics(value)
    assert metrics == {'native_0/critic_loss': 1.25, 'cache_lifetime/releases_generator': 1.,
                       'cache_lifetime/releases_critic': 5., 'cache_lifetime/released_owner_storage_bytes': 256.}
    assert all(type(x) is float for x in metrics.values())


@pytest.mark.parametrize('value', ['unknown', None, True, 1, {}])
def test_unknown_lifetime_policy_refuses_before_native_source(value):
    from distillschedule.training.self_forcing import _arguments
    with pytest.raises(ValueError, match='cache lifetime'):
        _arguments(dict(source_root='/does/not/exist', assets='/does/not/exist',
                        native_cache_lifetime=value))
