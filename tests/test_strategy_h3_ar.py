"""W6 is the original previous/current AV protocol, never Grid1000."""
import pytest
import json
from contextlib import nullcontext
from types import SimpleNamespace
from distillschedule.strategies.h3_ar_task import H3ARTask, H3ARImplementation, native_graph
from distillschedule.strategies.h3_ar_planning import assemble, plan, validate_plan, source_identity
from distillschedule.strategies.h3_ar_worker import capacity_ok, execute_cycles, topology
from distillschedule.workloads.best_known.registry import fingerprint, write_new_json

POLICY = dict(gpu_limit_bytes=1000, gpu_reserve_bytes=10, host_limit_bytes=3000, host_reserve_bytes=100)
MEMORY = dict(peak_allocated_bytes=400, driver_free_bytes=20, host_peak_bytes=1000,
              oom_event_delta=dict(oom=0, oom_kill=0))


def test_contract_is_original_and_frozen():
    task = H3ARTask()
    assert task.video_frames == task.previous_frames == 56
    assert (task.video_height, task.video_width) == (768, 1344)
    assert task.input_shapes['video'] == [1, 24, 17, 48, 84]
    assert task.census == dict(student=9, fake_score=2, teacher=1, student_grad=1,
                              fake_score_grad=1, student_backward=1, critic_backward=1)
    assert H3ARTask.from_dict(task.to_dict()) == task
    for key, value in [('video_frames', 97), ('denoising_steps', 4), ('lora_rank', 256),
                       ('fake_updates', 5), ('ema_updates', 1), ('protocol', 'grid1000')]:
        with pytest.raises(ValueError):
            H3ARTask.from_dict({**task.to_dict(), key: value})
    with pytest.raises(ValueError):
        H3ARTask.from_dict({'ignored_flag': True})


def test_only_native_baseline_and_accepted_bundle_are_candidates():
    assert H3ARImplementation('zero3-native').to_dict()['selected_cache_prefetch'] is False
    assert H3ARImplementation('zero3-ap').to_dict()['selected_cache_prefetch'] is True
    assert H3ARImplementation('zero3-ap').to_dict()['attention_replay_cache_storage'] == 'cpu_staged'
    assert H3ARImplementation('zero3-ap').to_dict()['checkpoint_boundary_offload'] is False
    with pytest.raises(ValueError):
        H3ARImplementation('grid1000')
    graph = native_graph(H3ARTask(), cycle=2)
    assert graph.topological_order() == ('native_update',)
    assert graph.operations[0].required_versions == dict(student=1, fake=1, ema=0)
    assert graph.operations[0].produced_versions == dict(student=2, fake=2)


def calibration_rows(tmp_path, implementation, duration, *, memory_peak=400):
    """Synthetic receipt fixtures stay in pytest tmpdir; never GPU evidence."""
    task = H3ARTask(); impl = H3ARImplementation(implementation)
    result = []
    for rank in range(16):
        cycles = [dict(cycle=i, census=task.census, updates=dict(student=1, critic=1),
            adam_updates=dict(student=1, critic=1), final_versions=dict(student=i, fake=i, ema=0),
            prepared=dict(status='PASS', executed_operations=['native_update']),
            RNG_reseeded_between_cycles=False, duration_ns=duration+rank+i,
            capacity_pass=True, memory={**MEMORY, 'peak_allocated_bytes': memory_peak},
            selected_cache=dict(cache_epoch=i-1, source_student_version=i-1, intervening_fake_version=i,
                same_process_pinned_pool_reuse=True, stale_result_reuse=False) if impl.optimized else None,
            backend_effects=dict(actual_compute_observed=True, implementation=impl.to_dict()),
            state={'synthetic_fixture': True}) for i in (1, 2)]
        raw = dict(schema_version='distillschedule.original_h3_ar_execution.v1',
            status='COMPLETED_ORIGINAL_H3_AR_WORK', mode='calibrate', rank=rank,
            task=task.to_dict(), implementation=impl.to_dict(),
            identity=dict(source=source_identity(), assets_sha256='a'*64, task_sha256=task.digest),
            gpu_execution='NATIVE_CUDA_RUN', profiled=False, finite=True, capacity_policy=POLICY,
            build=dict(native_implementation_installed=True, implementation=impl.to_dict(), input_shapes=task.input_shapes),
            input_identity=dict(sha256=f'{rank:064x}'),
            cycles=cycles, static_bytes=100, initial_versions=dict(student=0, fake=0, ema=0),
            final_versions=dict(student=2, fake=2, ema=0), runtime={'synthetic_fixture': True},
            hardware=dict(name='synthetic_fixture', total_memory=2000))
        path = tmp_path/f'{implementation}-{rank}.json'; write_new_json(path, raw); result.append(path)
    return result


def collection(tmp_path, implementation, duration, **kwargs):
    paths = calibration_rows(tmp_path, implementation, duration, **kwargs)
    output = tmp_path/implementation
    assemble(H3ARTask(), paths, output)
    return output/'collection.json'


def test_plan_ranks_only_actual_observations_and_roundtrips(tmp_path):
    native = collection(tmp_path, 'zero3-native', 200)
    accepted = collection(tmp_path, 'zero3-ap', 100)
    task = H3ARTask(); document = plan(task, [native, accepted])
    assert document['selected']['implementation'] == H3ARImplementation('zero3-ap').to_dict()
    assert document['selected']['order'] == ['native_update']
    assert document['selected']['predicted_makespan_ns'] == 117
    path = tmp_path/'saved.json'; write_new_json(path, document)
    validated, _ = validate_plan(task, path)
    assert validated == document
    # No model-name winner map: faster native observations must win too.
    assert plan(task, [native])['selected']['candidate_id'] == 'zero3-native'
    assert document['claimable_speedup'] is False


def test_capacity_rejects_faster_candidate(tmp_path):
    native = collection(tmp_path, 'zero3-native', 200, memory_peak=400)
    accepted = collection(tmp_path, 'zero3-ap', 100, memory_peak=800)
    result = plan(H3ARTask(), [native, accepted], memory_limit_bytes=500)
    assert result['selected']['candidate_id'] == 'zero3-native'
    with pytest.raises(ValueError, match='no feasible'):
        plan(H3ARTask(), [native], memory_limit_bytes=50)


@pytest.mark.parametrize('change', ['source', 'task', 'version', 'census', 'capacity', 'profile', 'branch'])
def test_calibration_rejects_wrong_native_contract(tmp_path, change):
    paths = calibration_rows(tmp_path, 'zero3-ap', 100)
    row = json.loads(paths[-1].read_text())
    if change == 'source': row['identity']['source']['source_head'] = 'b'*40
    elif change == 'task': row['task']['video_frames'] = 97
    elif change == 'version': row['cycles'][1]['final_versions']['fake'] = 3
    elif change == 'census': row['cycles'][1]['census']['student'] = 8
    elif change == 'capacity': row['cycles'][1]['memory']['peak_allocated_bytes'] = 2000
    elif change == 'profile': row['profiled'] = True
    elif change == 'branch': row['cycles'][1]['selected_cache'] = None
    paths[-1].write_text(json.dumps(row))
    with pytest.raises(ValueError):
        assemble(H3ARTask(), paths, tmp_path/'bad')


@pytest.mark.parametrize('key', ['order', 'memory', 'implementation', 'timeline'])
def test_rehashed_plan_tampering_is_rejected(tmp_path, key):
    observed = collection(tmp_path, 'zero3-ap', 100)
    document = plan(H3ARTask(), [observed])
    if key == 'order': document['selected']['order'] = ['ignored_old_shell']
    elif key == 'memory': document['selected']['memory'][0]['peak_bytes'] = 1
    elif key == 'implementation': document['selected']['implementation']['selected_cache_prefetch'] = False
    elif key == 'timeline': document['selected']['timeline'][0]['end_ns'] = 1
    document.pop('plan_sha256'); document['plan_sha256'] = fingerprint(document)
    path = tmp_path/'tampered.json'; write_new_json(path, document)
    with pytest.raises(ValueError):
        validate_plan(H3ARTask(), path)


def test_saved_plan_rechecks_source_and_receipt_bytes(tmp_path, monkeypatch):
    observed = collection(tmp_path, 'zero3-ap', 100)
    document = plan(H3ARTask(), [observed]); path = tmp_path/'saved.json'; write_new_json(path, document)
    import distillschedule.strategies.h3_ar_planning as planning
    monkeypatch.setattr(planning, 'source_identity', lambda: dict(source_head='f'*40, source_fingerprint='f'*64))
    with pytest.raises(ValueError, match='source changed'):
        validate_plan(H3ARTask(), path)
    monkeypatch.undo()
    (tmp_path/'zero3-ap-0.json').write_text('{}')
    with pytest.raises(ValueError, match='file changed'):
        validate_plan(H3ARTask(), path)


def test_native_session_prepared_handler_consumes_saved_order_and_versions(tmp_path):
    from distillschedule.strategies.h3_ar_session import H3ARSession
    from distillschedule.runtime.version_sync import VersionLedger
    document = plan(H3ARTask(), [collection(tmp_path, 'zero3-ap', 100)])
    session = H3ARSession.__new__(H3ARSession)
    session.task = H3ARTask(); session.ledger = VersionLedger(dict(student=0, fake=0, ema=0))
    session.census = session.updates = session.adam = __import__('collections').Counter()
    session.pool = object(); session.view = object(); session.leaf = None
    session.real = session.positive = session.negative = object()
    autocasts = []; updates = []; cache_epochs = []
    session.torch = SimpleNamespace(bfloat16='bf16', autocast=lambda **kw: (autocasts.append(kw), nullcontext())[1])
    session.epoch = SimpleNamespace(begin_next_cycle=lambda p, v: cache_epochs.append(1))
    trainer = SimpleNamespace(generator_step=0, critic_step=0)
    def native_update(real, positive, negative):
        updates.append((real, positive, negative))
        trainer.generator_step += 1; trainer.critic_step += 1
        return dict(gan_active=False)
    trainer.update = native_update; session.trainer = trainer
    for i in (1, 2):
        prepared = session.prepare(i, tuple(document['selected']['order']))
        prepared.run()
        assert prepared.receipt()['executed_operations'] == ['native_update']
        assert prepared.receipt()['final_versions'] == dict(student=i, fake=i, ema=0)
    assert len(updates) == 2 and cache_epochs == [1]
    assert autocasts == [dict(device_type='cuda', dtype='bf16', cache_enabled=False)]*2
    with pytest.raises(ValueError, match='versions diverged'):
        session.prepare(2, ('native_update',))


def test_execute_cycles_uses_common_capacity_and_native_boundaries():
    from distillschedule.strategies.h3_ar_session import H3ARSession
    from distillschedule.runtime.version_sync import VersionLedger
    session = H3ARSession.__new__(H3ARSession); task = session.task = H3ARTask()
    session.ledger = VersionLedger(dict(student=0, fake=0, ema=0))
    session.census = session.updates = session.adam = __import__('collections').Counter()
    session.pool = session.leaf = None; session.real = session.positive = session.negative = object()
    session.device = 'synthetic-fixture'
    session.torch = SimpleNamespace(bfloat16='bf16', autocast=lambda **kw: nullcontext())
    trainer = SimpleNamespace(generator_step=0, critic_step=0)
    def update(*_):
        trainer.generator_step += 1; trainer.critic_step += 1; return {'loss': 1.0}
    trainer.update = update; session.trainer = trainer
    session.finish = lambda c: dict(census=task.census, updates=dict(student=1, critic=1),
        adam_updates=dict(student=1, critic=1), final_versions=session.versions(), metrics=dict(loss=1.0))
    session.state_metadata = lambda c: dict(outside_timer=True)
    syncs = []; barriers = []
    clock = iter([0, 10, 20, 40]).__next__
    result = execute_cycles(session, mode='run', torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: syncs.append(d))),
        distributed=SimpleNamespace(barrier=lambda: barriers.append(1)), capacity_policy=POLICY,
        memory=lambda: MEMORY, clock=clock)
    assert [c['duration_ns'] for c in result['cycles']] == [10, 20]
    assert len(syncs) == len(barriers) == 4
    assert result['final_versions'] == dict(student=2, fake=2, ema=0)
    assert not capacity_ok({**MEMORY, 'oom_event_delta': {'oom': 1}}, POLICY)


def test_cli_routing_is_opt_in_and_does_not_import_torch(tmp_path, capsys):
    import sys
    from distillschedule.native.cli import main
    path = tmp_path/'task.json'; write_new_json(path, H3ARTask().to_dict())
    before = 'torch' in sys.modules
    assert main(['validate', '--task', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['task']['schema_version'] == 'distillschedule.original_h3_ar_task.v1'
    assert result['gpu_execution'] == 'NOT_RUN'
    assert ('torch' in sys.modules) == before


def test_topology_has_no_world1_or_independent_world8_fallback():
    assert topology(dict(RANK='9', LOCAL_RANK='1', WORLD_SIZE='16', LOCAL_WORLD_SIZE='8')) == (9, 1)
    for env in ({}, dict(RANK='1', LOCAL_RANK='1', WORLD_SIZE='8', LOCAL_WORLD_SIZE='8')):
        with pytest.raises(ValueError): topology(env)


def test_full_native_source_receipt_inventory_is_fail_closed(tmp_path, monkeypatch):
    import distillschedule.strategies.h3_ar_assets as assets
    from distillschedule.workloads.best_known.registry import file_sha256
    source = tmp_path/'source'; source.mkdir(); item = source/'math.py'; item.write_text('NATIVE = True\n')
    sha = file_sha256(item)
    receipt = tmp_path/'source_receipt.json'
    write_new_json(receipt, dict(status='PASS_EXACT_SOURCE_SNAPSHOT_ONLY', objects=[dict(relative_path='math.py', sha256=sha)]))
    ref = dict(path=str(receipt), sha256=file_sha256(receipt))
    monkeypatch.setattr(assets, 'SOURCE_RECEIPT', ref['sha256'])
    monkeypatch.setattr(assets, 'SOURCE_PINS', {'math.py': sha})
    assets.validate_source_tree(source, ref, {'math.py': sha})
    with pytest.raises(ValueError, match='inventory differs'):
        assets.validate_source_tree(source, ref, {})
    (source/'unbound.py').write_text('EXTRA = True\n')
    with pytest.raises(ValueError, match='tree changed'):
        assets.validate_source_tree(source, ref, {'math.py': sha})
    (source/'unbound.py').unlink(); item.write_text('NATIVE = False\n')
    with pytest.raises(ValueError, match='tree changed'):
        assets.validate_source_tree(source, ref, {'math.py': sha})


def test_gpu_run_requires_saved_plan_not_a_recipe_override(tmp_path):
    from distillschedule.strategies.h3_ar_worker import run_worker
    with pytest.MonkeyPatch.context() as patch:
        for key, value in dict(RANK='0', LOCAL_RANK='0', WORLD_SIZE='16', LOCAL_WORLD_SIZE='8').items():
            patch.setenv(key, value)
        with pytest.raises(ValueError, match='consume one saved plan'):
            run_worker(H3ARTask(), None, tmp_path, mode='run', capacity_policy=POLICY,
                       implementation=H3ARImplementation('zero3-ap'))


def test_checkpoint_weights_hash_once_not_per_rank_and_crosshost_stat(tmp_path, monkeypatch):
    import distillschedule.strategies.h3_ar_assets as assets
    root = tmp_path/'model'; root.mkdir()
    (root/'config.json').write_text('{}')
    weight = root/'model.safetensors'; weight.write_bytes(b'synthetic-payload-not-a-model')
    audit = tmp_path/'checkpoint-audit.json'
    expected = assets.admit_checkpoint_once(root, audit)
    reference = dict(path=str(audit), sha256=assets.file_sha256(audit))
    original_hash = assets.file_sha256
    def no_large_rehash(path):
        assert not str(path).endswith('.safetensors'), 'worker must not rescan checkpoint payload'
        return original_hash(path)
    monkeypatch.setattr(assets, 'file_sha256', no_large_rehash)
    witness = assets.validate_checkpoint_audit(root, reference, expected)['worker_witness']
    assert witness['full_payload_rehashed'] is False
    assert assets.admit_checkpoint_once(root, audit) == expected
    original_stat = assets.payload_stat
    monkeypatch.setattr(assets.socket, 'gethostname', lambda: 'another-test-node')
    def other_mount_stat(path):
        observed = original_stat(path)
        return {**observed, 'device': observed['device']+1, 'inode': observed['inode']+1}
    monkeypatch.setattr(assets, 'payload_stat', other_mount_stat)
    assert assets.validate_checkpoint_audit(root, reference, expected)['worker_witness']['stat_scope'].startswith('cross_host')
    weight.write_bytes(b'changed payload')
    with pytest.raises(ValueError, match='stat changed'):
        assets.validate_checkpoint_audit(root, reference, expected)


def test_original_startup_reserve_is_separate_from_sustained_policy(tmp_path):
    from distillschedule.strategies.h3_ar_worker import host_admission
    (tmp_path/'memory.max').write_text(str(512 << 30))
    (tmp_path/'memory.current').write_text(str(400 << 30))
    policy = {**POLICY, 'host_limit_bytes': 512 << 30, 'host_reserve_bytes': 0}
    observed = host_admission(policy, tmp_path)
    assert observed['startup_reserve_bytes'] == 64 << 30
    assert policy['host_reserve_bytes'] == 0
    (tmp_path/'memory.current').write_text(str(460 << 30))
    with pytest.raises(ValueError, match='startup cgroup reserve64GiB'):
        host_admission(policy, tmp_path)


def test_wrong_torch_runtime_is_rejected_before_model_work():
    from distillschedule.strategies.h3_ar_worker import runtime_identity
    with pytest.raises(ValueError, match='real Torch 2.10.0'):
        runtime_identity(SimpleNamespace(__version__='2.9.0', version=SimpleNamespace(cuda='12.8')))


def test_native_adapter_discovery_keeps_canonical_five(tmp_path, capsys):
    from distillschedule.native.cli import main
    assert main(['list-adapters']) == 0
    discovery = json.loads(capsys.readouterr().out)
    assert discovery['canonical_profiles'] == 'UNCHANGED_FIVE_FROZEN_PROFILES'
    assert any(row['schema_version'] == H3ARTask().schema_version for row in discovery['adapters'])


@pytest.mark.parametrize('candidate', ['zero3-native', 'zero3-ap'])
def test_saved_selection_changes_actual_native_config_and_cache_install_branch(tmp_path, candidate):
    from distillschedule.strategies.h3_ar_session import configure_native_execution, install_native_attention_cache
    saved = plan(H3ARTask(), [collection(tmp_path, candidate, 100)])
    path = tmp_path/'saved.json'; write_new_json(path, saved)
    verified, _ = validate_plan(H3ARTask(), path)
    impl = H3ARImplementation.from_dict(verified['selected']['implementation'])
    native_config = dict(zero_optimization=dict(stage=3, overlap_comm=False, stage3_prefetch_bucket_size=50000000))
    assert configure_native_execution(impl, native_config) is native_config
    assert native_config['zero_optimization']['overlap_comm'] == impl.optimized
    assert native_config['zero_optimization']['stage3_prefetch_bucket_size'] == (657499392 if impl.optimized else 50000000)
    calls = []; native_objects = (object(), object(), object(), object())
    epoch = SimpleNamespace(install=lambda *args: (calls.append(args), ('pool', 'view'))[1])
    assets = SimpleNamespace(helper=lambda name: (calls.append(name), epoch)[1])
    installed, pool, view = install_native_attention_cache(impl, assets, *native_objects)
    if impl.optimized:
        assert calls == ['native_selected_cycle_scope_r1', native_objects]
        assert (installed, pool, view) == (epoch, 'pool', 'view')
    else:
        assert calls == [] and (installed, pool, view) == (None, None, None)
