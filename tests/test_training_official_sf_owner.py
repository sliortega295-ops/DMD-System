"""Production admission is distinct from CPU source/helper qualification."""
import pytest


def test_fsdp_profile_requires_live_cuda_before_loading_assets(tmp_path, monkeypatch):
    from distillschedule.training.self_forcing import _arguments
    monkeypatch.setenv('NCCL_CROSS_NIC', '1')
    with pytest.raises(ValueError, match='CUDA.*world16|world16.*CUDA'):
        _arguments(dict(source_root=str(tmp_path), assets=str(tmp_path / 'absent.json'),
            device='cpu', optimization_profile='self-forcing-wan-fsdp2-r6-v1'))


@pytest.mark.parametrize('value', [None, '', '0', '2', ' 1', '1 '])
def test_native_fsdp_requires_exact_explicit_launch_policy_without_mutation(monkeypatch, value):
    import os
    from distillschedule.training.self_forcing_fsdp import validate_environment
    if value is None:
        monkeypatch.delenv('NCCL_CROSS_NIC', raising=False)
    else:
        monkeypatch.setenv('NCCL_CROSS_NIC', value)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='explicit NCCL_CROSS_NIC=1'):
        validate_environment()
    assert dict(os.environ) == before


def test_native_fsdp_launch_policy_does_not_require_dmd2_settings_or_mutate_environment(monkeypatch):
    import os
    from distillschedule.training.self_forcing_fsdp import validate_environment
    monkeypatch.setenv('NCCL_CROSS_NIC', '1')
    monkeypatch.delenv('NCCL_ALGO', raising=False)
    monkeypatch.delenv('NCCL_MIN_CTAS', raising=False)
    before = dict(os.environ)
    validate_environment()
    assert dict(os.environ) == before


@pytest.mark.parametrize('entrypoint', ['reference-admit', 'optimized-admit', 'placement', 'live'])
def test_native_fsdp_checks_launch_policy_before_admission_or_collectives(monkeypatch, entrypoint):
    import os
    from distillschedule.training.self_forcing_fsdp import SelfForcingFSDP2Owner, admit, place_roles
    monkeypatch.delenv('NCCL_CROSS_NIC', raising=False)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='explicit NCCL_CROSS_NIC=1'):
        if entrypoint.endswith('-admit'):
            parameters = {'native_role_backend': 'fsdp2-native-bf16-v1'}
            if entrypoint == 'optimized-admit':
                parameters['optimization_profile'] = 'self-forcing-wan-fsdp2-r6-v1'
            admit(parameters)
        elif entrypoint == 'placement':
            place_roles({}, None)
        else:
            # Only the pre-placement live policy guard is exercised on CPU.
            owner = object.__new__(SelfForcingFSDP2Owner)
            owner._cache_lifetime = 'native-v1'
            owner.identity = {}
            owner._validate_live()
    assert dict(os.environ) == before


def test_native_fsdp_profile_has_no_second_training_loop():
    from distillschedule.training.self_forcing_owner import SelfForcingOwner
    from distillschedule.training.self_forcing_fsdp import SelfForcingFSDP2Owner
    # Same loop, native fwdbwd, Adam/EMA and state schema; physical seams only.
    assert SelfForcingFSDP2Owner._commit is SelfForcingOwner._commit
    assert SelfForcingFSDP2Owner.state_dict is SelfForcingOwner.state_dict
    assert SelfForcingFSDP2Owner.load_state_dict is SelfForcingOwner.load_state_dict
    assert SelfForcingFSDP2Owner.fwdbwd_one_step is SelfForcingOwner.fwdbwd_one_step


def test_component_disposal_on_normal_and_failed_training_scope():
    from distillschedule.training.adapters import ModelComponents
    closed = []
    components = ModelComponents(state={'placeholder': 0}, close=lambda: closed.append('closed')).validate()
    with components.managed():
        assert not closed
    assert closed == ['closed']
    with pytest.raises(RuntimeError, match='native failure'):
        with components.managed():
            raise RuntimeError('native failure')
    assert closed == ['closed', 'closed']


@pytest.fixture
def construction_provider(tmp_path, monkeypatch):
    """CPU lifecycle fixture; not a qualification of the FSDP2 provider."""
    from types import SimpleNamespace
    import torch
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training import session

    closed, hook_calls = [], []
    module = torch.nn.Identity()
    handle = module.register_forward_pre_hook(lambda *_: hook_calls.append('active'))

    def close():
        closed.append('closed')
        handle.remove()

    components = ModelComponents(state={'owner': module}, close=close)
    config = SimpleNamespace(model={'factory': 'lifetime:model'},
        algorithm={'factory': 'lifetime:algorithm'}, execution={}, identity={'fixture': 'lifetime'},
        trainer={'device': 'cpu', 'cycles': 1, 'output_dir': str(tmp_path / 'train')})
    dataset = SimpleNamespace(identity={'fixture': 'data'}, manifest={})
    factories = {'lifetime:model': lambda _: components, 'lifetime:algorithm': lambda *_: None}
    monkeypatch.setattr(session, 'training_data', lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(session, 'load_callable', lambda reference:
        (factories[reference], {'reference': reference, 'sha256': 'lifetime-fixture'}))
    return SimpleNamespace(config=config, components=components, closed=closed, hook_calls=hook_calls,
                           module=module)


@pytest.mark.parametrize('include_validation', [False, True])
def test_production_construct_keeps_provider_until_managed_end(construction_provider, include_validation):
    import torch
    from distillschedule.training import session
    fixture = construction_provider
    result = session.construct(fixture.config, include_validation=include_validation)
    assert result[0] is fixture.components
    assert len(result) == (5 if include_validation else 4)
    assert fixture.closed == []
    with result[0].managed():
        fixture.module(torch.ones(1))
        assert fixture.hook_calls == ['active']
        assert fixture.closed == []
    assert fixture.closed == ['closed']
    fixture.module(torch.ones(1))
    assert fixture.hook_calls == ['active']


@pytest.mark.parametrize('fail_owned', [False, True])
def test_public_train_keeps_constructed_provider_through_owned_scope(construction_provider, monkeypatch, fail_owned):
    import torch
    from distillschedule.training import session
    fixture = construction_provider

    def use_provider(config, components, *_):
        assert components is fixture.components
        assert fixture.closed == []
        components.state['owner'](torch.ones(1))
        assert fixture.hook_calls == ['active']
        if fail_owned:
            raise RuntimeError('owned training failure')
        return 'completed'

    monkeypatch.setattr(session, '_train_owned', use_provider)
    if fail_owned:
        with pytest.raises(RuntimeError, match='owned training failure'):
            session.train(fixture.config)
    else:
        assert session.train(fixture.config) == 'completed'
    assert fixture.closed == ['closed']


@pytest.mark.parametrize('failure_stage', ['algorithm', 'admission'])
def test_production_construct_failure_closes_provider_once(construction_provider, monkeypatch, failure_stage):
    from distillschedule.training import session
    fixture = construction_provider

    def fail(*_args, **_kwargs):
        raise RuntimeError('construction failure')

    if failure_stage == 'algorithm':
        load = session.load_callable
        monkeypatch.setattr(session, 'load_callable', lambda reference:
            fail() if reference == fixture.config.algorithm['factory'] else load(reference))
    else:
        fixture.components.admit_training = fail
    with pytest.raises(RuntimeError, match='construction failure'):
        session.construct(fixture.config)
    assert fixture.closed == ['closed']


@pytest.mark.parametrize('command', ['evaluate', 'export'])
def test_public_output_failure_disposes_owned_provider(tmp_path, monkeypatch, command):
    from types import SimpleNamespace
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training import session
    closed = []
    components = ModelComponents(state={'owner': 0}, close=lambda: closed.append('closed'))
    config = SimpleNamespace(trainer={'device': 'cpu'}, model={'parameters': {}, 'factory': 'unused'})
    monkeypatch.setattr(session, '_load_for_output', lambda *_: (components, None, {'cycle': 0}, {'model': {}}, None))
    # Failure after construction must release providers on both output routes.
    if command == 'export':
        def fail(*args, **kwargs):
            raise RuntimeError('injected export failure')
        import distillschedule.training.inference as inference
        monkeypatch.setattr(inference, 'publish_export', fail)
        expected = RuntimeError
    else:
        expected = TypeError  # Deliberately absent dataset fails input admission.
    with pytest.raises(expected):
        getattr(session, command)(config, tmp_path / 'checkpoint', tmp_path / command)
    assert closed == ['closed']


def test_completed_sf_census_reaches_public_execution_receipt(monkeypatch):
    from types import SimpleNamespace
    import torch.distributed as dist
    from distillschedule.training.self_forcing_fsdp import SelfForcingFSDP2Owner
    monkeypatch.setattr(dist, 'get_rank', lambda: 0)
    owner = SimpleNamespace(_validate_live=lambda: None, generator_updates=1, critic_updates=5,
        _role_forwards={'generator': 7, 'real_score': 2, 'fake_score': 6},
        _last_optimization_receipt={'profile': 'fixture', 'kv': {'critic_early_releases': 5}})
    result = SelfForcingFSDP2Owner.validate_collective_execution(owner)
    assert result['optimization_effects'] == owner._last_optimization_receipt
    result['optimization_effects']['kv']['critic_early_releases'] = 0
    assert owner._last_optimization_receipt['kv']['critic_early_releases'] == 5


def test_sharded_inventory_reports_the_actual_native_backend(monkeypatch):
    from distillschedule.open_workload import collective_owner
    monkeypatch.setattr(collective_owner, 'checked', lambda call, label: call())
    monkeypatch.setattr(collective_owner, 'contract', lambda _: {'spec': {'backend': 'fsdp2'}})
    row = collective_owner.inventory(object())
    assert row['layout'] == 'native_owner_fsdp2'
    assert 'native mixed FSDP2' not in row['unsupported']
