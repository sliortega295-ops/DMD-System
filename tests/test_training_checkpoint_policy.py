"""Public checkpoint cadence, retention and complete-cycle timing on CPU."""
import json
from pathlib import Path

import pytest
import torch

from distillschedule.training.config import TrainingConfig
from distillschedule.training.session import train
from test_training_end_to_end import fixture, equal


def test_finite_checker_reuses_unchanged_state_but_checks_updates_and_outputs(monkeypatch):
    from distillschedule.training.finite import FiniteCheck
    frozen = torch.nn.Linear(3, 2).requires_grad_(False)
    trainable = torch.tensor([1., 2.], requires_grad=True)
    optimizer = torch.optim.AdamW([trainable], lr=.01)
    state = {'teacher': frozen, 'parameter': trainable, 'optimizer': optimizer}
    checked = []
    original = torch.isfinite
    def observed(value):
        checked.append(value)
        return original(value)
    monkeypatch.setattr(torch, 'isfinite', observed)
    guard = FiniteCheck()
    assert guard(state, {'loss': torch.tensor(1.)})
    checked.clear()
    assert guard(state, {'loss': torch.tensor(2.)})
    assert len(checked) == 1  # Only the new output, not all unchanged weights.
    trainable.sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    checked.clear()
    assert guard(state, {})
    assert any(value is trainable for value in checked)
    assert not any(value is frozen.weight or value is frozen.bias for value in checked)
    with torch.no_grad():
        frozen.weight[0, 0] = float('nan')
    assert not guard(state, {})
    assert not guard(state, {})  # A failed scan must never cache successful admission.
    with torch.no_grad():
        frozen.weight[0, 0] = 1.
    assert guard(state, {})
    assert not guard(state, {'loss': torch.tensor(float('inf'))})
    checked.clear()
    assert guard(state, {}, audit=True)
    assert any(value is frozen.weight for value in checked)


def test_finite_checker_has_no_tensor_retention_and_checks_counterless_state():
    import gc
    import weakref
    from distillschedule.training.finite import FiniteCheck
    guard = FiniteCheck()
    value = torch.ones(2)
    reference = weakref.ref(value)
    assert guard({'value': value}, {})
    del value
    gc.collect()
    assert reference() is None
    with torch.inference_mode():
        counterless = torch.ones(2)
        assert guard({'value': counterless}, {})
        counterless[0] = float('nan')
        assert not guard({'value': counterless}, {})


def test_finite_outputs_cannot_inherit_an_aliased_state_cache():
    from distillschedule.training.finite import FiniteCheck
    guard = FiniteCheck()
    value = torch.ones(2)
    assert guard({'weight': value}, {})
    # A raw write bypasses the version counter. State checks require an audit
    # for that case, but outputs must always be checked, even if also in state.
    value.data.fill_(float('nan'))
    assert not guard({'weight': value}, {'output': value})


def _livewan_diagnostic_continuation(tmp_path, rank, *, output_only):
    """Actual owner/control objects; no model, CUDA allocation or output claim."""
    from types import SimpleNamespace
    from distillschedule.training import checkpoint, livewan_c51_training as training
    from distillschedule.training.livewan_native_export import NativeOutputRequest
    identity = dict(schema_version=training.SCHEMA, source_head='1' * 40,
        source_files=training.source_files(), assets_sha256='2' * 64,
        input_sha256='3' * 64, horizon=4)
    header = dict(schema_version=checkpoint.SCHEMA, kind='RESUMABLE_COMPLETE_CYCLE',
        identity=identity, cycle=2, cursor=10, versions=dict(generator=2, critic=10, ema=0),
        policy=training.checkpoint_policy(identity)) if output_only else None
    value = training.Continuation(config=SimpleNamespace(trainer=dict(stop_after_cycle=2)),
        admitted={}, identity=identity, output=tmp_path / 'owner', bootstrap=None,
        loader=SimpleNamespace(inputs=SimpleNamespace(cycles=4)), header=header,
        capacity_check=lambda: None,
        native_output=NativeOutputRequest({}, tmp_path / 'export') if output_only else None)
    inventory = {str(i): f'00000000-0000-0000-0000-{rank // 8 * 8 + i + 1:012d}'
                 for i in range(8)}
    value.device_binding = dict(rank=rank, node_rank=rank // 8, cuda_device=rank % 8,
        physical_device=rank % 8, uuid=inventory[str(rank % 8)], physical_inventory=inventory)
    return value


@pytest.mark.parametrize('rank', [0, 8], ids=['generator', 'score'])
def test_livewan_output_only_records_real_construction_without_training_phases(tmp_path, monkeypatch, rank):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training import livewan_c51_diagnostics as diagnostics
    from distillschedule.training.livewan_c51_state import physical_versions
    monkeypatch.setenv(diagnostics.FLAG, 'v1')
    monkeypatch.setenv('DISTILLSCHEDULE_RUN_ID', 'SYNTHETIC_TEST_ONLY_output_diagnostics')
    reads = []
    memory = {key: 10 for key in diagnostics.MEMORY_FIELDS}
    monkeypatch.setattr(worker, 'memory_snapshot', lambda *args: reads.append(True) or dict(memory))
    value = _livewan_diagnostic_continuation(tmp_path, rank, output_only=True)
    before = torch.get_rng_state().clone()
    value.configure_phase_diagnostics(rank=rank)
    with pytest.raises(ValueError, match='phase diagnostic'):
        value.phase_receipt(torch=torch, rank=rank, device=f'cuda:{rank % 8}', phase='construction_entry',
                            cycle=2, versions=physical_versions(rank, 2))
    assert not reads and not value.phase_diagnostics.path.exists()
    for phase in ('construction_entry', 'construction_exit'):
        value.phase_receipt(torch=torch, rank=rank, device=f'cuda:{rank % 8}', phase=phase,
                            cycle=2, versions=physical_versions(rank, 0))
    result = json.loads(value.phase_diagnostics.path.read_text())
    assert result['schema_version'] == diagnostics.OUTPUT_SCHEMA
    assert result['identity']['target_checkpoint_cycle'] == 2
    assert result['observation_limit'] == len(result['observations']) == len(reads) == 2
    assert [row['phase'] for row in result['observations']] == ['construction_entry', 'construction_exit']
    assert all(row['global_step'] is None and row['wave_scope'] == 'NO_TRAINING_WAVES'
               and row['physical_versions'] == physical_versions(rank, 0)
               and row['physical_versions_scope'] == 'BOOTSTRAP_PRE_RESTORE'
               for row in result['observations'])
    assert result['status'] == 'DIAGNOSTIC_ONLY_PARTIAL_HISTORY'
    assert value.native_output.result is None and not list(value.output.rglob('_SUCCESS'))
    assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
    original = value.phase_diagnostics.path.read_bytes()
    for phase in diagnostics.SETUP_PHASES + diagnostics.CYCLE_PHASES:
        with pytest.raises(ValueError, match='phase diagnostic'):
            value.phase_receipt(torch=torch, rank=rank, device=f'cuda:{rank % 8}', phase=phase,
                                cycle=2, versions=physical_versions(rank, 0))
    assert len(reads) == 2 and value.phase_diagnostics.path.read_bytes() == original


@pytest.mark.parametrize('damage', ['callback', 'header', 'identity', 'policy', 'cycle', 'uuid', 'source'])
def test_livewan_output_diagnostics_reject_incomplete_restore_identity_before_sampling(tmp_path, monkeypatch, damage):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training import livewan_c51_diagnostics as diagnostics
    monkeypatch.setenv(diagnostics.FLAG, 'v1')
    monkeypatch.setenv('DISTILLSCHEDULE_RUN_ID', 'SYNTHETIC_TEST_ONLY_output_diagnostics')
    monkeypatch.setattr(worker, 'memory_snapshot', lambda *args: pytest.fail('sample before admission'))
    value = _livewan_diagnostic_continuation(tmp_path, 8, output_only=True)
    if damage == 'callback': value.native_output = object()
    elif damage == 'header': value.header = None
    elif damage == 'identity': value.header['identity'] = {}
    elif damage == 'policy': value.header['policy'] = {}
    elif damage == 'cycle': value.start_cycle = value.end_cycle = 0
    elif damage == 'uuid': value.device_binding['uuid'] = 'invalid'
    else: value.identity['source_files'] = {}
    with pytest.raises(ValueError, match='phase diagnostic|C51'):
        value.configure_phase_diagnostics(rank=8)
    assert not value.output.exists()


def test_livewan_training_diagnostics_and_disabled_path_keep_existing_contract(tmp_path, monkeypatch):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training import livewan_c51_diagnostics as diagnostics
    from distillschedule.training.livewan_c51_state import physical_versions
    monkeypatch.setenv(diagnostics.FLAG, '0')
    assert diagnostics.from_environment(object(), rank=-1) is None
    monkeypatch.setenv(diagnostics.FLAG, 'v1')
    monkeypatch.setenv('DISTILLSCHEDULE_RUN_ID', 'SYNTHETIC_TEST_ONLY_training_diagnostics')
    value = _livewan_diagnostic_continuation(tmp_path, 8, output_only=False)
    monkeypatch.setattr(worker, 'memory_snapshot', lambda *args: {key: 10 for key in diagnostics.MEMORY_FIELDS})
    value.configure_phase_diagnostics(rank=8)
    for phase in diagnostics.SETUP_PHASES:
        value.phase_receipt(torch=torch, rank=8, device='cuda:0', phase=phase, cycle=0,
            versions=physical_versions(8, 0),
            **({'memory': {key: 10 for key in diagnostics.MEMORY_FIELDS}} if phase == 'setup_ready' else {}))
    value.phase_receipt(torch=torch, rank=8, device='cuda:0', phase='full5_entry', cycle=1,
                        versions=physical_versions(8, 0))
    result = json.loads(value.phase_diagnostics.path.read_text())
    assert result['schema_version'] == diagnostics.SCHEMA
    assert result['observation_limit'] == 12 and len(result['observations']) == 5
    assert result['observations'][-1]['global_step'] == 0


def test_livewan_output_diagnostic_failed_write_does_not_advance_history(tmp_path, monkeypatch):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training import livewan_c51_diagnostics as diagnostics
    monkeypatch.setenv(diagnostics.FLAG, 'v1')
    monkeypatch.setenv('DISTILLSCHEDULE_RUN_ID', 'SYNTHETIC_TEST_ONLY_output_diagnostics')
    monkeypatch.setattr(worker, 'memory_snapshot', lambda *args: {key: 10 for key in diagnostics.MEMORY_FIELDS})
    value = _livewan_diagnostic_continuation(tmp_path, 0, output_only=True)
    value.configure_phase_diagnostics(rank=0)
    arguments = dict(torch=torch, rank=0, device='cuda:0', cycle=2,
                     versions=dict(generator=0, critic=0, ema=0))
    value.phase_receipt(phase='construction_entry', **arguments)
    original = value.phase_diagnostics.path.read_bytes()
    def fail(*args, **kwargs): raise OSError('atomic publication failed')
    monkeypatch.setattr(worker, 'atomic_json', fail)
    with pytest.raises(OSError, match='atomic publication'):
        value.phase_receipt(phase='construction_exit', **arguments)
    assert len(value.phase_diagnostics.observations) == 1
    assert value.phase_diagnostics.path.read_bytes() == original


def test_default_is_thirty_minutes_and_final_cycle():
    from distillschedule.training.checkpoint import CheckpointPolicy
    policy = CheckpointPolicy.from_trainer({})
    assert policy.keep_last == 3
    assert not policy.due(cycle=1, total=100, elapsed=1799.)
    assert policy.due(cycle=2, total=100, elapsed=1800.)
    assert policy.due(cycle=100, total=100, elapsed=1.)
    explicit = CheckpointPolicy.from_trainer({'checkpoint_every': 2})
    assert not explicit.due(cycle=1, total=4, elapsed=9999.)
    assert explicit.due(cycle=2, total=4, elapsed=1.)
    assert not CheckpointPolicy.from_trainer({'checkpoint_every': 0}).due(
        cycle=4, total=4, elapsed=9999.)


def test_checkpoint_settings_are_configurable_but_not_mathematical_identity(tmp_path):
    from test_training_config import request
    raw = request()
    original = TrainingConfig.from_dict(raw, source=tmp_path / 'config.json')
    raw['trainer'].update(checkpoint_interval_seconds=60., checkpoint_keep_last=5)
    configured = TrainingConfig.from_dict(raw, source=original.source)
    assert configured.identity == original.identity


@pytest.mark.parametrize('field,value', [
    ('checkpoint_interval_seconds', 0), ('checkpoint_interval_seconds', float('inf')),
    ('checkpoint_interval_seconds', True), ('checkpoint_keep_last', -1),
    ('checkpoint_keep_last', True), ('checkpoint_keep_last', 1.5),
])
def test_invalid_checkpoint_settings_fail_before_construction(tmp_path, field, value):
    from test_training_config import request
    raw = request(); raw['trainer'][field] = value
    with pytest.raises(ValueError, match='checkpoint'):
        TrainingConfig.from_dict(raw, source=tmp_path / 'config.json')


def test_retention_keeps_latest_three_and_never_touches_unpublished_or_foreign(tmp_path, monkeypatch):
    from distillschedule.training.checkpoint import save_checkpoint, prune_checkpoints
    root = tmp_path / 'checkpoints'
    state = {'weight': torch.tensor([2.])}
    for cycle in range(1, 6):
        save_checkpoint(root / f'cycle-{cycle:06d}', state, identity={'run': 'ours'},
                        cycle=cycle, cursor=cycle, versions={})
    save_checkpoint(root / 'cycle-000000', state, identity={'run': 'other'},
                    cycle=0, cursor=0, versions={})
    (root / 'cycle-000006').mkdir()  # Incomplete: not eligible for deletion.
    (root / '.cycle-000007.pending').mkdir()
    external = tmp_path / 'outside'; external.mkdir()
    (root / 'cycle-000008').symlink_to(external, target_is_directory=True)
    from distillschedule.training import distributed
    def forbidden():
        raise AssertionError('rank-zero retention entered collective topology discovery')
    monkeypatch.setattr(distributed, 'topology', forbidden)
    removed = prune_checkpoints(root, identity={'run': 'ours'}, keep_last=3)
    assert [Path(p).name for p in removed] == ['cycle-000001', 'cycle-000002']
    assert all((root / f'cycle-{n:06d}').is_dir() for n in [0, 3, 4, 5, 6])
    assert external.exists() and (root / 'cycle-000008').is_symlink()
    assert (root / '.cycle-000007.pending').exists()


def test_default_run_saves_final_only_and_explicit_retention_preserves_training(tmp_path):
    root, config = fixture(tmp_path, samples=6)
    raw = config.to_dict()
    raw['trainer'].pop('checkpoint_every')
    raw['trainer'].update(cycles=5, output_dir=str(root / 'default'))
    default = train(TrainingConfig.from_dict(raw, source=config.source))
    assert sorted(p.name for p in (root / 'default/checkpoints').iterdir()) == ['cycle-000005']
    raw['trainer'].update(checkpoint_every=1, output_dir=str(root / 'frequent'))
    frequent = train(TrainingConfig.from_dict(raw, source=config.source))
    assert sorted(p.name for p in (root / 'frequent/checkpoints').iterdir()) == [
        'cycle-000003', 'cycle-000004', 'cycle-000005']
    a = torch.load(Path(default['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(frequent['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b)
    rows = [json.loads(line) for line in (root / 'frequent/metrics.jsonl').read_text().splitlines()]
    for row in rows:
        assert row['prepublication_wall_s'] >= row['wall_s'] >= 0
        assert row['cycle_timing_scope'] == 'CYCLE_PREPUBLICATION_BEFORE_METRICS_CHECKPOINT_AND_POLICY_PUBLICATION'
        assert 'checkpoint' not in row['timings_s'] and 'publication' not in row['timings_s']
        assert row['timings_s']['finite_check'] >= 0
    assert frequent['timings_s']['checkpoint'] > 0
    assert frequent['job_wall_s'] >= frequent['cycles_wall_s'] >= sum(r['prepublication_wall_s'] for r in rows)
    assert frequent['timings_s']['logging'] >= 0


def test_prepublication_metrics_keep_full_cycle_and_job_costs_in_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from distillschedule.training import session

    root, config = fixture(tmp_path)
    raw = config.to_dict()
    raw['trainer'].update(cycles=2, checkpoint_every=1, checkpoint_keep_last=1, output_dir=str(root / 'timed'))
    clock = [0.]
    monkeypatch.setattr(session, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    original_run = session.CyclePolicy.run
    original_append, original_save = session._append_json, session.save_checkpoint
    original_prune, original_json = session.prune_checkpoints, session._json

    def run(*args, **kwargs):
        value = original_run(*args, **kwargs)
        clock[0] += 2.
        return value

    def append(path, value):
        original_append(path, value)
        if path.name == 'metrics.jsonl':
            clock[0] += 7.

    def save(*args, **kwargs):
        value = original_save(*args, **kwargs)
        clock[0] += 11.
        return value

    def prune(*args, **kwargs):
        value = original_prune(*args, **kwargs)
        clock[0] += 13.
        return value

    def publish(path, value):
        original_json(path, value)
        if path.name == 'effective-policy.json':
            clock[0] += 17.

    monkeypatch.setattr(session.CyclePolicy, 'run', run)
    monkeypatch.setattr(session, '_append_json', append)
    monkeypatch.setattr(session, 'save_checkpoint', save)
    monkeypatch.setattr(session, 'prune_checkpoints', prune)
    monkeypatch.setattr(session, '_json', publish)
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    rows = [json.loads(line) for line in (root / 'timed/metrics.jsonl').read_text().splitlines()]
    assert [row['wall_s'] for row in rows] == [2., 2.]
    assert [row['prepublication_wall_s'] for row in rows] == [2., 2.]
    assert all(row['cycle_timing_scope'] == 'CYCLE_PREPUBLICATION_BEFORE_METRICS_CHECKPOINT_AND_POLICY_PUBLICATION'
               and 'cycle_wall_s' not in row and 'checkpoint' not in row['timings_s']
               and 'publication' not in row['timings_s'] for row in rows)
    assert result['timings_s']['execution'] == 4.
    assert result['timings_s']['logging'] == 14.
    assert result['timings_s']['checkpoint'] == 48.
    assert result['timings_s']['publication'] == 17.
    assert result['cycles_this_process'] == 2
    assert result['cycles_wall_s'] == result['job_wall_s'] == 83.
    assert result['cycles_timing_scope'] == 'ALL_COMPLETED_CYCLES_THROUGH_METRICS_CHECKPOINT_AND_POLICY_PUBLICATION'
    assert json.loads((root / 'timed/summary.json').read_text()) == result


def test_output_conflict_is_rejected_before_model_factory(tmp_path, monkeypatch):
    root, config = fixture(tmp_path)
    Path(config.trainer['output_dir']).mkdir()
    from distillschedule.training import session
    def forbidden(*a, **k):
        raise AssertionError('model construction reached')
    monkeypatch.setattr(session, '_construct_model', forbidden)
    with pytest.raises(FileExistsError):
        train(config)


class _SharedCheckpointRole(torch.nn.Module):
    def __init__(self, frozen):
        super().__init__()
        self.frozen = torch.nn.Parameter(frozen, requires_grad=False)
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.weight_alias = self.weight

    def forward(self, value):
        return value @ self.frozen + self.weight


def _shared_checkpoint_state():
    frozen = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    roles = {name: _SharedCheckpointRole(frozen) for name in ('student', 'fake', 'teacher')}
    return {**roles, **{name + '_optimizer': torch.optim.AdamW([role.weight], lr=.01)
                       for name, role in roles.items()},
            'ema': {name: role.weight.detach().clone() for name, role in roles.items()}}


def _checkpoint_update(state):
    for name in ('student', 'fake', 'teacher'):
        optimizer = state[name + '_optimizer']
        state[name](torch.ones(1, 2)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state['ema'][name].lerp_(state[name].weight.detach(), .1)


def test_shared_role_snapshot_safe_roundtrip_preserves_optimizer_updates(tmp_path):
    from copy import deepcopy
    from distillschedule.training.checkpoint import (
        SCHEMA, pack_state, read_checkpoint, read_manifest, save_checkpoint, unpack_state,
    )
    state = _shared_checkpoint_state()
    _checkpoint_update(state)
    path = tmp_path / 'shared'
    identity = {'fixture': 'shared3roles'}
    save_checkpoint(path, state, identity=identity, cycle=1, cursor=1, versions={})
    assert read_manifest(path, identity=identity)['schema_version'] == SCHEMA == 'distillschedule.training_checkpoint.v2'
    payload = read_checkpoint(path, identity=identity)
    safe = torch.load(path / 'state.pt', weights_only=True)
    assert equal(safe, payload)
    restored = _shared_checkpoint_state()
    parameters = {name: restored[name].weight for name in ('student', 'fake', 'teacher')}
    unpack_state(payload['state'], restored)
    assert equal(pack_state(state), pack_state(restored))
    for name, parameter in parameters.items():
        assert restored[name].weight is restored[name].weight_alias is parameter
        assert restored[name + '_optimizer'].param_groups[0]['params'][0] is parameter
        assert parameter in restored[name + '_optimizer'].state
    # Pre-sharing v2 files remain ordinary full state dictionaries, with no
    # reference markers or dropped frozen keys. Their reader is unchanged.
    old = deepcopy(payload['state'])
    for name in parameters:
        old[name]['value']['frozen'] = old[name]['value']['frozen'].clone()
    unpack_state(old, restored)
    malformed = deepcopy(old)
    malformed['student_optimizer']['parameters'] = [['fake.weight']]
    with pytest.raises(ValueError, match='parameter ownership'):
        unpack_state(malformed, restored)
    _checkpoint_update(state)
    _checkpoint_update(restored)
    assert equal(pack_state(state), pack_state(restored))
    snapshots = [payload['state'][name]['value']['frozen'] for name in parameters]
    assert len({tensor.untyped_storage()._cdata for tensor in snapshots}) == 1
    assert snapshots[0].untyped_storage()._cdata != state['student'].frozen.untyped_storage()._cdata


def test_snapshot_reuses_only_exact_live_storage_views():
    from distillschedule.training.checkpoint import pack_state
    base = torch.arange(16).reshape(4, 4)
    complex_base = torch.tensor([1 + 2j, 3 + 4j])
    empty_a, empty_b = torch.empty(0), torch.empty(0)
    values = dict(base=base, detached=base.detach(), independent=base.clone(),
                  transposed=base.t(), offset=base[1:], reshaped=base.reshape(2, 8),
                  first_row=base[0], next_row=base[1], dtype_view=base.view(torch.float64),
                  negative=base._neg_view(), complex=complex_base, conjugate=complex_base.conj(),
                  empty=empty_a, empty_alias=empty_a.detach(), empty_independent=empty_b)
    saved = pack_state({'views': values})['views']['value']
    store = lambda tensor: tensor.untyped_storage()._cdata
    assert store(saved['base']) == store(saved['detached'])
    distinct = ('base', 'independent', 'transposed', 'offset', 'reshaped',
                'first_row', 'next_row', 'dtype_view', 'negative', 'complex', 'conjugate')
    assert len({store(saved[name]) for name in distinct}) == len(distinct)
    assert store(saved['empty']) == store(saved['empty_alias']) != store(saved['empty_independent'])
    for name, tensor in values.items():
        assert torch.equal(saved[name], tensor)
        assert store(saved[name]) != store(tensor)
    again = pack_state({'views': values})['views']['value']
    assert store(again['base']) != store(saved['base'])
    base.add_(1)
    fresh = pack_state({'views': values})['views']['value']
    assert torch.equal(fresh['base'], base) and not torch.equal(saved['base'], base)


def test_snapshot_does_not_retain_temporary_state_or_reuse_mutated_views():
    import gc
    import weakref
    from torch.multiprocessing.reductions import StorageWeakRef
    from distillschedule.training.checkpoint import pack_state
    base = torch.zeros(3)
    tensor_refs, storage_refs, wrapper_refs = [], [], []
    module = torch.nn.Linear(2, 2)
    def observe(module, state_dict, prefix, metadata):
        wrapper_refs.extend(weakref.ref(value) for value in state_dict.values())
    module.register_state_dict_post_hook(observe)

    class TemporaryState:
        def state_dict(self):
            value = base.clone()
            tensor_refs.append(weakref.ref(value))
            storage_refs.append(StorageWeakRef(value.untyped_storage()))
            base.add_(1)
            return {'value': value}

        def load_state_dict(self, value):
            pass

    saved = pack_state({'module': module, 'before': base, 'temporary_a': TemporaryState(),
                        'temporary_b': TemporaryState(), 'after': base})
    gc.collect()
    assert all(ref() is None for ref in tensor_refs)
    assert wrapper_refs and all(ref() is None for ref in wrapper_refs)
    assert all(ref.expired() for ref in storage_refs)
    assert torch.equal(saved['before']['value'], torch.zeros(3))
    assert torch.equal(saved['after']['value'], torch.full((3,), 2.))
    assert torch.equal(saved['temporary_a']['value']['value'], torch.zeros(3))
    assert torch.equal(saved['temporary_b']['value']['value'], torch.ones(3))


def test_cpu_tree_uses_one_independent_copy_without_an_extra_clone():
    from torch.utils._python_dispatch import TorchDispatchMode
    from distillschedule.training.checkpoint import cpu_tree

    class Copies(TorchDispatchMode):
        def __init__(self):
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls.append(func)
            return func(*args, **(kwargs or {}))

    for tensor in (torch.arange(12.).reshape(3, 4).t(),
                   torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)[:, ::2],
                   torch.tensor([1 + 2j, 3 + 4j]).conj()):
        expected = tensor.detach().cpu().clone()
        with Copies() as copies:
            saved = cpu_tree(tensor)
        assert torch.equal(saved, expected) and saved.stride() == expected.stride()
        assert not saved.requires_grad and saved.untyped_storage()._cdata != tensor.untyped_storage()._cdata
        assert copies.calls.count(torch.ops.aten._to_copy.default) == 1
        if not tensor.is_conj():  # Torch may resolve a conjugate bit inside to().
            assert torch.ops.aten.clone.default not in copies.calls
        tensor.add_(1)
        assert torch.equal(saved, expected)


def _native_restore_fault_worker(rank, rendezvous, fault):
    """SYNTHETIC_TEST_ONLY: actual native boundary, injected CPU engine."""
    from datetime import timedelta
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch.distributed as dist
    from distillschedule.training.h3_ar_state import NativeCheckpoint, _ENGINE_CLIENT_METADATA
    from distillschedule.strategies.h3_ar_session import H3ARSession
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=5))
    try:
        model = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        calls = []
        def load(*args, **kwargs):
            calls.append(True)
            assert model.weight.requires_grad
            if rank == 1 and fault == 'load':
                raise RuntimeError('native-install-sentinel')
            with torch.no_grad():
                model.weight.fill_(3)
            engine.global_steps = 2
            client = dict.fromkeys(_ENGINE_CLIENT_METADATA)
            client.update(cycle=1, native_global_steps=2, frozen_param_shapes=None,
                          frozen_param_fragments=None, param_shapes=[{'weight': [1, 1]}])
            if fault == 'result':
                client['cycle'] = 2
            return 'SYNTHETIC_TEST_ONLY', client
        engine = SimpleNamespace(module=model, global_steps=0, load_checkpoint=load)
        owner = object.__new__(NativeCheckpoint)
        owner.session = SimpleNamespace(engine=engine)
        owner.parameters = dict(model.named_parameters())
        owner.permanent = {id(model.weight)}
        owner.partition_layout = [[['weight', [1, 1]]]]
        def validate(*args):
            if rank == 1 and fault == 'admission':
                raise RuntimeError('native-admission-sentinel')
        owner.validate_restore = validate
        registered_trainables = owner.registered_trainables
        @contextmanager
        def trainables():
            with registered_trainables():
                if rank == 1 and fault == 'preparation':
                    raise RuntimeError('native-preparation-sentinel')
                yield
            if rank == 1 and fault == 'cleanup':
                raise RuntimeError('native-cleanup-sentinel')
        owner.registered_trainables = trainables
        metadata = {'client_state': {'cycle': 1, 'native_global_steps': 2}}
        if fault is None:
            owner.load('.', metadata)
            assert model.weight.item() == 3 and not model.weight.requires_grad
            return
        message = ('native engine did not restore exact' if fault == 'result'
                   else 'native-' + ('install' if fault == 'load' else fault) + '-sentinel')
        with pytest.raises(ValueError, match=message) as error:
            owner.load('.', metadata)
        errors = [None, None]
        dist.all_gather_object(errors, str(error.value))
        assert errors[0] == errors[1] and 'Timed out' not in errors[0]
        assert not model.weight.requires_grad and owner.session._checkpoint_restore_failed
        assert len(calls) == (1 if fault in ('load', 'cleanup', 'result') else 0)
        with pytest.raises(ValueError, match='failed'):
            owner.load('.', metadata)
        with pytest.raises(ValueError, match='failed'):
            H3ARSession.prepare(owner.session, 1)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fault', ['load', 'admission', 'preparation', 'cleanup', 'result', None])
def test_native_restore_exchanges_local_errors_and_invalidates_partial_owner(tmp_path, fault):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_native_restore_fault_worker,
        args=(str(tmp_path / 'native-init'), fault), nprocs=2, join=True)


@pytest.mark.parametrize('fault', ['state', 'rng'])
def test_native_owner_is_invalidated_if_common_restore_fails_after_native_install(monkeypatch, fault):
    """SYNTHETIC_TEST_ONLY: shared restore interface, no engine or GPU."""
    from distillschedule.training import checkpoint as checkpoint_module
    from distillschedule.training.checkpoint import pack_state, rng_state, restore_checkpoint
    state = {'value': torch.ones(1)}
    payload = dict(state=pack_state(state), rng=rng_state(), cycle=1, cursor=1, versions={})
    monkeypatch.setattr(checkpoint_module, 'read_checkpoint', lambda *args, **kwargs: payload)
    monkeypatch.setattr(checkpoint_module, 'read_manifest', lambda *args, **kwargs:
                        {'native_checkpoint': {'owner': {'fixture': 'SYNTHETIC_TEST_ONLY'}}})
    class Owner:
        installed = False
        invalidated = False
        def validate_restore(self, *args): pass
        def load(self, *args): self.installed = True
        def invalidate_restore(self): self.invalidated = True
    owner = Owner()
    def fail(*args, **kwargs):
        assert owner.installed
        raise RuntimeError('common-' + fault + '-sentinel')
    monkeypatch.setattr(checkpoint_module, 'unpack_state' if fault == 'state' else 'restore_rng', fail)
    with pytest.raises(RuntimeError, match='common-' + fault + '-sentinel'):
        restore_checkpoint('SYNTHETIC_TEST_ONLY', state, identity={}, native_checkpoint=owner)
    assert owner.installed and owner.invalidated
