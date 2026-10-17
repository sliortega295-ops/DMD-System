"""W6 is the original previous/current AV protocol, never Grid1000."""
import pytest
import json
from contextlib import nullcontext
from types import SimpleNamespace
from distillschedule.strategies.h3_ar_task import H3ARTask, H3ARImplementation, native_graph
from distillschedule.strategies.h3_ar_planning import assemble, plan, validate_plan, source_identity
from distillschedule.strategies.h3_ar_worker import capacity_ok, execute_cycles, topology
from distillschedule.strategies.h3_ar_inputs import expected_validation
from distillschedule.workloads.best_known.registry import fingerprint, write_new_json

POLICY = dict(gpu_limit_bytes=1000, gpu_reserve_bytes=10, host_limit_bytes=3000, host_reserve_bytes=100)
MEMORY = dict(peak_allocated_bytes=400, driver_free_bytes=20, host_current_bytes=900, host_peak_bytes=1000,
    host_peak_status='OBSERVED', host_peak_scope='CGROUP_LIFETIME_NOT_RESET_BY_OBSERVER',
    host_capacity_evidence='OBSERVED_CGROUP_LIFETIME_PEAK', cgroup_events=dict(oom=0, oom_kill=0),
    capacity_observation=dict(cgroup_path='/cpu_test_fixture', physical_host_bytes=4000, cgroup_hard_limit_bytes=3000,
        effective_host_limit_bytes=3000, scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL'),
    oom_event_delta=dict(oom=0, oom_kill=0))


def test_contract_is_original_and_frozen():
    task = H3ARTask()
    assert task.video_frames == task.previous_frames == 56
    assert (task.video_height, task.video_width) == (768, 1344)
    assert task.input_shapes(0)['video'] == [1, 24, 17, 48, 84]
    assert task.census == dict(student=9, fake_score=2, teacher=1, student_grad=1,
                              fake_score_grad=1, student_backward=1, critic_backward=1)
    assert H3ARTask.from_dict(task.to_dict()) == task
    for key, value in [('video_frames', 97), ('denoising_steps', 4), ('lora_rank', 256),
                       ('fake_updates', 5), ('ema_updates', 1), ('protocol', 'grid1000')]:
        with pytest.raises(ValueError):
            H3ARTask.from_dict({**task.to_dict(), key: value})
    with pytest.raises(ValueError):
        H3ARTask.from_dict({'ignored_flag': True})


def test_rank6_original_audio94_is_bound_to_its_sample_not_a_global_shape():
    """The real original parser has one longer audio sample; never pad/crop it."""
    task = H3ARTask()
    assert task.schema_version == 'distillschedule.original_h3_ar_task.v2'
    assert task.input_shapes(6)['audio'] == [2, 32, 94]
    assert task.input_shapes(6)['previous_audio'] == [188, 32]
    for rank in set(range(16)) - {6}:
        assert task.input_shapes(rank)['audio'] == [2, 32, 93]
        assert task.input_shapes(rank)['previous_audio'] == [186, 32]
    assert len({task.expected_input(r)['input_identity']['sha256'] for r in range(16)}) == 16
    with pytest.raises(ValueError):
        H3ARTask.from_dict({**task.to_dict(), 'schema_version': 'distillschedule.original_h3_ar_task.v1'})


def input_tensor_fixture(metadata, torch):
    """Shape-only meta tensors: no original values, no CUDA, no GPU evidence."""
    if isinstance(metadata, dict):
        if set(metadata) == {'shape', 'dtype'}:
            return torch.empty(metadata['shape'], dtype=getattr(torch, metadata['dtype'].split('.')[1]), device='meta')
        return {k: input_tensor_fixture(v, torch) for k, v in metadata.items()}
    if isinstance(metadata, list):
        return [input_tensor_fixture(v, torch) for v in metadata]
    assert metadata is None


@pytest.mark.parametrize('rank', range(16))
def test_same_session_validator_accepts_exact_original_per_rank_tensor_metadata(rank):
    import torch
    from distillschedule.strategies.h3_ar_inputs import validate_prepared_input
    task = H3ARTask(); row = task.expected_input(rank)
    raw = input_tensor_fixture(row['raw_tensors'], torch)
    positive = {**input_tensor_fixture(row['positive_tensors'], torch), **row['positive_scalars']}
    assert validate_prepared_input(task, rank, row['input_identity'], raw, positive, None, torch) == expected_validation(task, rank)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change', ['wrong_rank', 'sample_path', 'sample_sha', 'audio93_on_rank6',
    'audio94_on_rank5', 'packed_dtype', 'packed_length', 'positive_dtype', 'raw_dtype', 'negative', 'conditioning'])
def test_rank_input_validator_rejects_sample_shape_packing_and_dtype_substitution(change):
    import torch
    from distillschedule.strategies.h3_ar_inputs import validate_prepared_input
    task = H3ARTask(); rank = 5 if change == 'audio94_on_rank5' else 6
    row = task.expected_input(rank); identity = dict(row['input_identity'])
    raw = input_tensor_fixture(row['raw_tensors'], torch)
    positive = {**input_tensor_fixture(row['positive_tensors'], torch), **row['positive_scalars']}; negative = None
    if change == 'wrong_rank': rank = 5
    elif change == 'sample_path': identity['path'] = task.expected_input(5)['input_identity']['path']
    elif change == 'sample_sha': identity['sha256'] = 'f'*64
    elif change in ('audio93_on_rank6', 'audio94_on_rank5'):
        positive['audio_input_latents'] = torch.empty((2, 32, 93 if rank == 6 else 94), device='meta', dtype=torch.bfloat16)
    elif change == 'packed_dtype': positive['packed']['cu_seqlens'] = positive['packed']['cu_seqlens'].to(torch.int64)
    elif change == 'packed_length': positive['packed']['audio_pos'] = torch.empty(372, device='meta', dtype=torch.int64)
    elif change == 'positive_dtype': positive['input_latents'] = positive['input_latents'].float()
    elif change == 'raw_dtype': raw[1]['text_token_tags'] = raw[1]['text_token_tags'].int()
    elif change == 'negative': negative = {}
    elif change == 'conditioning': positive['audio_cond_noise_aug'] = 0.0
    with pytest.raises(ValueError, match='rank-qualified|conditioning'):
        validate_prepared_input(task, rank, identity, raw, positive, negative, torch)


def test_actual_session_constructor_enters_shared_rank6_validator_before_model(tmp_path, monkeypatch):
    """Actual constructor/parser handoff, using explicitly synthetic meta inputs."""
    import torch
    import distillschedule.strategies.h3_ar_inputs as inputs
    from distillschedule.strategies.h3_ar_session import H3ARSession
    task = H3ARTask(); row = task.expected_input(6); contract = inputs.input_contract()
    raw = input_tensor_fixture(row['raw_tensors'], torch)
    positive = {**input_tensor_fixture(row['positive_tensors'], torch), **row['positive_scalars']}
    manifest = tmp_path/'cache.json'; write_new_json(manifest, dict(status='PASS_CACHE_SOURCE_AND_SCHEMA',
        rows=[task.expected_input(r)['input_identity'] for r in range(16)]))
    payload = tmp_path/'synthetic-shape-only.pth'; payload.write_text('NOT_NATIVE_CACHE_VALUES')
    calls = []
    def load(path, **kwargs):
        assert path == payload and kwargs == dict(map_location='cpu', weights_only=True)
        calls.append('load'); return raw
    def prepare(value, cfg):
        assert value is raw and cfg == 1.0
        calls.append('original_parser_handoff_fixture'); return positive, None
    class BeforeModel(Exception): pass
    def helper(name):
        assert name == 'native_mechanisms_r3'; calls.append('validated_before_helper_or_model'); raise BeforeModel
    native = SimpleNamespace(DMD2Config=lambda **kwargs: SimpleNamespace(teacher_cfg_scale=kwargs['teacher_cfg_scale']), prepare_batch=prepare)
    assets = SimpleNamespace(cache_manifest=contract['cache_manifest'], source_module=lambda name: native, helper=helper)
    monkeypatch.setattr(inputs, 'checked', lambda ref: manifest if ref == contract['cache_manifest'] else payload)
    monkeypatch.setattr(inputs, 'validate_parser', lambda module: calls.append('source_pins_fixture'))
    monkeypatch.setattr(torch, 'load', load)
    with pytest.raises(BeforeModel):
        H3ARSession(task, assets, H3ARImplementation('zero3-native'), rank=6, local_rank=6, torch=torch, accelerator=None)
    assert calls == ['source_pins_fixture', 'load', 'original_parser_handoff_fixture', 'validated_before_helper_or_model']
    assert positive['audio_input_latents'].shape[-1] == 94 and not torch.cuda.is_initialized()


@pytest.mark.parametrize('change', ['wrong_identity', 'old_global93', 'missing_witness', 'wrong_dtype', 'profile_sha'])
def test_assembler_and_saved_plan_recheck_exact_rank_input_witness(tmp_path, change):
    paths = calibration_rows(tmp_path, 'zero3-ap', 100)
    row = json.loads(paths[6].read_text())
    if change == 'wrong_identity': row['input_identity'] = H3ARTask().expected_input(5)['input_identity']
    elif change == 'old_global93': row['build']['input_shapes']['audio'][-1] = 93
    elif change == 'missing_witness': del row['build']['input_validation']
    elif change == 'wrong_dtype': row['build']['input_validation']['positive_tensors']['packed']['cu_seqlens']['dtype'] = 'torch.int64'
    elif change == 'profile_sha': row['build']['input_validation']['input_contract_sha256'] = 'f'*64
    paths[6].write_text(json.dumps(row))
    with pytest.raises(ValueError):
        assemble(H3ARTask(), paths, tmp_path/'bad')


def test_input_profile_file_itself_is_strongly_bound_and_old_task_refused(tmp_path, monkeypatch):
    import distillschedule.strategies.h3_ar_inputs as inputs
    task = H3ARTask(); changed = task.to_dict(); changed['input_contract_sha256'] = 'f'*64
    with pytest.raises(ValueError): H3ARTask.from_dict(changed)
    profile = tmp_path/inputs.CONTRACT_PATH; profile.parent.mkdir(parents=True); profile.write_text('{}')
    monkeypatch.setattr(inputs, 'repository_root', lambda: tmp_path)
    with pytest.raises(ValueError, match='contract file changed'): H3ARTask()


def test_parser_pins_cover_both_original_prepare_functions():
    import ast
    from pathlib import Path
    from distillschedule.strategies.h3_ar_inputs import validate_parser
    original = Path('/home/lyy/Helios/research_artifacts/DMD-System/h3_diffsynth_dmd2_world16_20261010/artifacts/reference_source_snapshot/examples/minimax_h3/model_training')
    functions = {}
    for filename, name in [('h3_dmd2.py', 'prepare_batch'), ('train_ours.py', 'prepare_cached_inputs')]:
        path = original/filename
        definition = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {}
        # Definition only; original body is not invoked or reimplemented here.
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
        functions[name] = namespace[name]
    validate_parser(SimpleNamespace(**functions))
    functions['prepare_cached_inputs'] = lambda value: value
    with pytest.raises(ValueError, match='parser source changed'):
        validate_parser(SimpleNamespace(**functions))


@pytest.mark.parametrize('field,value', [('schema_version', 'distillschedule.original_h3_ar_task.v1'),
    ('input_contract_sha256', 'f'*64)])
def test_saved_plan_cannot_rehash_away_rank_input_contract(tmp_path, field, value):
    task = H3ARTask(); observed = collection(tmp_path, 'zero3-ap', 100)
    document = plan(task, [observed]); document['task'][field] = value
    document.pop('plan_sha256'); document['plan_sha256'] = fingerprint(document)
    path = tmp_path/'bad-task-plan.json'; write_new_json(path, document)
    with pytest.raises(ValueError, match='saved W6 plan/task changed'):
        validate_plan(task, path)


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
            build=dict(native_implementation_installed=True, implementation=impl.to_dict(),
                       input_shapes=task.input_shapes(rank), input_validation=expected_validation(task, rank)),
            input_identity=task.expected_input(rank)['input_identity'],
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
    assert result['task']['schema_version'] == 'distillschedule.original_h3_ar_task.v2'
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
    (tmp_path/'memory.events').write_text('oom 0\noom_kill 0\n')
    policy = {**POLICY, 'host_limit_bytes': 512 << 30, 'host_reserve_bytes': 0}
    capacity = dict(cgroup_path=str(tmp_path), physical_host_bytes=512 << 30, cgroup_hard_limit_bytes=512 << 30,
        effective_host_limit_bytes=512 << 30, scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
    observed = host_admission(policy, tmp_path, capacity=capacity)
    assert observed['startup_reserve_bytes'] == 64 << 30
    assert policy['host_reserve_bytes'] == 0
    (tmp_path/'memory.current').write_text(str(460 << 30))
    with pytest.raises(ValueError, match='startup cgroup reserve64GiB'):
        host_admission(policy, tmp_path, capacity=capacity)


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


@pytest.mark.parametrize('change_capacity', [False, True])
@pytest.mark.parametrize('node_rank,use_v2,mode', [(0, False, 'calibrate'), (0, True, 'calibrate'),
    (1, True, 'calibrate'), (0, True, 'run'), (1, True, 'run')])
def test_worker_completes_without_memory_peak_using_exact_hardcap_not_current_as_peak(tmp_path, monkeypatch, change_capacity, node_rank, use_v2, mode):
    """Execute run_worker's actual observation branch with CPU-only native stand-ins."""
    import sys
    from types import ModuleType
    import distillschedule.strategies.h3_ar_worker as worker
    import distillschedule.strategies.h3_ar_session as native_session
    from distillschedule.runtime.version_sync import VersionLedger
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.strategies.h3_ar_planning import CAPACITY_V2
    limits = {'0': 512 << 30, '1': 500 << 30}; limit = limits[str(node_rank)]
    policy = dict(schema_version=CAPACITY_V2, gpu_limit_bytes=1000, gpu_reserve_bytes=10,
        host_limits=limits, host_reserve_bytes=0) if use_v2 else {**POLICY, 'host_limit_bytes': limit, 'host_reserve_bytes': 0}
    cgroup = tmp_path/'cgroup'; cgroup.mkdir()
    (cgroup/'memory.current').write_text('1000')
    (cgroup/'memory.max').write_text(str(limit))
    (cgroup/'memory.events').write_text('oom 7\noom_kill 3\n')
    assert not (cgroup/'memory.peak').exists()
    capacity = dict(cgroup_path=str(cgroup), cgroup_limits=[dict(path=str(cgroup), memory_max=str(limit), limit_bytes=limit)],
        physical_host_bytes=600 << 30, cgroup_hard_limit_bytes=limit, effective_host_limit_bytes=limit,
        scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
    observations = []
    def actual_capacity():
        observations.append(True)
        return cgroup, {**capacity, 'physical_host_bytes': (601 if change_capacity and len(observations) > 1 else 600) << 30}
    monkeypatch.setattr(worker, '_capacity_observation', actual_capacity)
    monkeypatch.setattr(worker, 'runtime_identity', lambda torch: {'cpu_test_fixture': True})
    order = []; visible = [f'GPU-test-{i}' for i in range(8)]
    startup = tmp_path/'startup.json'; write_new_json(startup, dict(ordered_gpu_uuids=visible))
    startup_ref = dict(path=str(startup), sha256=file_sha256(startup))
    environment = ModuleType('accelerate.utils.environment')
    environment.patch_installed = False
    def install_gpu_probe(module, path):
        assert module is environment; module.patch_installed = True; order.append('install_gpu_probe')
    assets = SimpleNamespace(validate=lambda: {}, digest='a'*64, install_import_paths=lambda: None,
        startup_gpu_info={'0': startup_ref, '1': startup_ref}, audit_imports=lambda: None,
        helper=lambda name: SimpleNamespace(install=install_gpu_probe if name == 'native_accelerate_gpu_info_cache_r1' else lambda **kw: None))
    torch = ModuleType('torch')
    def set_device(local):
        assert environment.patch_installed; order.append('set_device')
    torch.cuda = SimpleNamespace(set_device=set_device, get_device_properties=lambda local: SimpleNamespace(
        name='cpu_test_fixture', total_memory=2000, major=8, minor=0, uuid='GPU-test-0'),
        mem_get_info=lambda local: (1500, 2000), max_memory_allocated=lambda local: 400,
        max_memory_reserved=lambda local: 500, synchronize=lambda device: None)
    dist = ModuleType('torch.distributed')
    dist.init_process_group = lambda *a, **kw: None
    dist.destroy_process_group = lambda: None
    dist.barrier = lambda: None
    torch.distributed = dist
    accelerate = ModuleType('accelerate'); accelerate.__path__ = []
    def accelerator(**kwargs):
        assert environment.patch_installed; order.append('Accelerator')
        return SimpleNamespace(distributed_type='deepspeed')
    accelerate.Accelerator = accelerator; accelerate.DeepSpeedPlugin = lambda **kw: None
    utils = ModuleType('accelerate.utils'); utils.__path__ = []
    utils.DistributedType = SimpleNamespace(DEEPSPEED='deepspeed'); utils.environment = environment
    accelerate.utils = utils
    for name, module in {'torch': torch, 'torch.distributed': dist, 'accelerate': accelerate,
                         'accelerate.utils': utils, 'accelerate.utils.environment': environment}.items():
        monkeypatch.setitem(sys.modules, name, module)
    class NativeFixture(native_session.H3ARSession):
        def __init__(self, task, assets, implementation, **kwargs):
            self.task = task; self.device = 'cpu_test_fixture'; self.static_bytes = 100
            self.rank = kwargs['rank']; self.build = {}; self.input_identity = {'cpu_test_fixture': True, 'rank': self.rank}
            self.ledger = VersionLedger(dict(student=0, fake=0, ema=0))
            self.census = self.updates = self.adam = __import__('collections').Counter()
            self.pool = self.leaf = None; self.real = self.positive = self.negative = object()
            self.torch = SimpleNamespace(bfloat16='bf16', autocast=lambda **kw: nullcontext())
            self.trainer = SimpleNamespace(generator_step=0, critic_step=0, update=self.update)
        def update(self, *_):
            self.trainer.generator_step += 1; self.trainer.critic_step += 1
            return dict(loss=1.0)
        def finish(self, cycle):
            return dict(census=self.task.census, updates=dict(student=1, critic=1),
                adam_updates=dict(student=1, critic=1), final_versions=self.versions(), metrics=dict(loss=1.0))
        def state_metadata(self, cycle):
            return dict(cpu_test_fixture=True)
    monkeypatch.setattr(native_session, 'H3ARSession', NativeFixture)
    saved_path = None
    if mode == 'run':
        import socket
        rows_root = tmp_path/'calibration'; rows_root.mkdir()
        rank_paths = calibration_rows(rows_root, 'zero3-native', 100)
        for index, path in enumerate(rank_paths):
            row = json.loads(path.read_text()); node = index//8; cap = dict(capacity)
            if node != node_rank:
                location = str(tmp_path/('other-node-'+str(node)))
                cap.update(cgroup_path=location, cgroup_hard_limit_bytes=limits[str(node)], effective_host_limit_bytes=limits[str(node)],
                    cgroup_limits=[dict(path=location, memory_max=str(limits[str(node)]), limit_bytes=limits[str(node)])])
            row.update(capacity_policy=policy, runtime={'cpu_test_fixture': True}, hardware=dict(name='cpu_test_fixture',
                total_memory=2000, capability=[8, 0], uuid='GPU-test-0', hostname=socket.gethostname()))
            for cycle in row['cycles']:
                cycle['memory'].update(host_current_bytes=1000, host_peak_bytes=None, host_peak_status='UNAVAILABLE',
                    host_peak_scope='UNAVAILABLE_CGROUP_V2_MEMORY_PEAK',
                    host_capacity_evidence='FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK',
                    cgroup_events=dict(oom=7, oom_kill=3), capacity_observation=cap,
                    policy_node_rank=node, policy_host_limit_bytes=limits[str(node)])
            path.write_text(json.dumps(row))
        output = tmp_path/'assembled'; assemble(H3ARTask(), rank_paths, output)
        saved_path = tmp_path/'selected-plan.json'; write_new_json(saved_path, plan(H3ARTask(), [output/'collection.json']))
    for key, value in dict(RANK=str(node_rank*8), LOCAL_RANK='0', WORLD_SIZE='16', LOCAL_WORLD_SIZE='8',
        CUDA_VISIBLE_DEVICES=','.join(visible), DIFFSYNTH_ATTENTION_IMPLEMENTATION='torch',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTORCH_ALLOC_CONF='expandable_segments:True').items():
        monkeypatch.setenv(key, value)
    def execute():
        return worker.run_worker(H3ARTask(), assets, tmp_path/'execution', mode=mode,
            capacity_policy=policy, implementation=H3ARImplementation('zero3-native') if mode != 'run' else None,
            plan_path=saved_path)
    if change_capacity:
        with pytest.raises(ValueError, match='capacity/capability changed during native execution'):
            execute()
        assert (tmp_path/'execution'/f'failure_{node_rank*8:03d}.json').is_file()
        return
    result = execute()
    assert result['capacity_policy'] == policy and result['input_identity']['rank'] == node_rank*8
    if mode == 'run': assert result['plan_sha256'] == json.loads(saved_path.read_text())['plan_sha256']
    assert order == ['install_gpu_probe', 'set_device', 'Accelerator']
    for cycle in result['cycles']:
        assert cycle['memory']['host_peak_bytes'] is None
        assert cycle['memory']['host_peak_status'] == 'UNAVAILABLE'
        assert cycle['memory']['host_current_bytes'] == 1000
        assert cycle['memory']['policy_node_rank'] == node_rank and cycle['memory']['policy_host_limit_bytes'] == limit
        assert cycle['memory']['host_capacity_evidence'] == 'FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK'


@pytest.mark.parametrize('invalid', ['reserve', 'stricter_budget', 'no_finite_cap', 'missing_oom', 'current_over_budget'])
def test_unavailable_host_peak_never_relaxes_capacity_policy(tmp_path, invalid):
    from distillschedule.strategies.h3_ar_worker import observe_host_memory
    (tmp_path/'memory.current').write_text('1000')
    (tmp_path/'memory.events').write_text('oom 0\noom_kill 0\n')
    capacity = dict(cgroup_path=str(tmp_path), physical_host_bytes=4000, cgroup_hard_limit_bytes=3000,
        effective_host_limit_bytes=3000, scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
    policy = {**POLICY, 'host_reserve_bytes': 0}
    if invalid == 'reserve': policy['host_reserve_bytes'] = 1
    elif invalid == 'stricter_budget': policy['host_limit_bytes'] = 2999
    elif invalid == 'no_finite_cap':
        capacity['cgroup_hard_limit_bytes'] = None; capacity['effective_host_limit_bytes'] = 4000
    elif invalid == 'missing_oom': (tmp_path/'memory.events').write_text('oom 0\n')
    elif invalid == 'current_over_budget': (tmp_path/'memory.current').write_text('3001')
    with pytest.raises(ValueError, match='host capacity unproved'):
        observe_host_memory(policy, root=tmp_path, capacity=capacity)


def test_missing_peak_capacity_witness_rejects_oom_and_peak_substitution(tmp_path):
    from distillschedule.strategies.h3_ar_worker import observe_host_memory
    (tmp_path/'memory.current').write_text('1000')
    (tmp_path/'memory.events').write_text('oom 7\noom_kill 3\n')
    capacity = dict(cgroup_path=str(tmp_path), physical_host_bytes=4000, cgroup_hard_limit_bytes=3000,
        effective_host_limit_bytes=3000, scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
    policy = {**POLICY, 'host_reserve_bytes': 0}
    memory = {**MEMORY, **observe_host_memory(policy, root=tmp_path, capacity=capacity)}
    assert capacity_ok(memory, policy)
    assert not capacity_ok({**memory, 'host_peak_bytes': memory['host_current_bytes']}, policy)
    assert not capacity_ok({**memory, 'oom_event_delta': dict(oom=1, oom_kill=0)}, policy)
    assert not capacity_ok({**memory, 'oom_event_delta': dict(oom=0)}, policy)


def test_observed_host_peak_still_enforces_declared_reserve_and_peak_validity(tmp_path):
    from distillschedule.strategies.h3_ar_worker import observe_host_memory
    (tmp_path/'memory.current').write_text('1000')
    (tmp_path/'memory.events').write_text('oom 0\noom_kill 0\n')
    (tmp_path/'memory.peak').write_text('1200')
    capacity = dict(cgroup_path=str(tmp_path), physical_host_bytes=4000, cgroup_hard_limit_bytes=3000,
        effective_host_limit_bytes=3000, scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
    observed = observe_host_memory(POLICY, root=tmp_path, capacity=capacity)
    assert observed['host_peak_bytes'] == 1200 and observed['host_peak_status'] == 'OBSERVED'
    for invalid in ('900', '2901'):
        (tmp_path/'memory.peak').write_text(invalid)
        with pytest.raises(ValueError):
            observe_host_memory(POLICY, root=tmp_path, capacity=capacity)


def test_assembler_and_saved_plan_bind_unavailable_peak_capability(tmp_path):
    paths = calibration_rows(tmp_path, 'zero3-native', 100)
    for path in paths:
        row = json.loads(path.read_text()); row['capacity_policy']['host_reserve_bytes'] = 0
        for cycle in row['cycles']:
            cycle['memory'].update(host_peak_bytes=None, host_peak_status='UNAVAILABLE',
                host_peak_scope='UNAVAILABLE_CGROUP_V2_MEMORY_PEAK',
                host_capacity_evidence='FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK')
        path.write_text(json.dumps(row))
    output = tmp_path/'assembled'
    collection = assemble(H3ARTask(), paths, output)
    assert collection['host_capacity_bindings'][0][0]['host_peak_status'] == 'UNAVAILABLE'
    document = plan(H3ARTask(), [output/'collection.json'])
    saved = tmp_path/'saved.json'; write_new_json(saved, document)
    assert validate_plan(H3ARTask(), saved)[1]['host_capacity_bindings'] == collection['host_capacity_bindings']
    collection['host_capacity_bindings'][0][0]['host_peak_status'] = 'OBSERVED'
    (output/'collection.json').write_text(json.dumps(collection))
    with pytest.raises(ValueError, match='capacity/capability changed'):
        plan(H3ARTask(), [output/'collection.json'])


def test_v2_exact_distinct_node_host_limits_keep_unavailable_peak_and_node_binding(tmp_path):
    from distillschedule.strategies.h3_ar_planning import validate_policy
    from distillschedule.strategies.h3_ar_worker import observe_host_memory
    policy=dict(schema_version='distillschedule.original_h3_ar_capacity.v2',gpu_limit_bytes=1000,
        gpu_reserve_bytes=10,host_limits={'0':3000,'1':4000},host_reserve_bytes=0)
    validate_policy(policy)
    for node,limit in ((0,3000),(1,4000)):
        root=tmp_path/str(node);root.mkdir();(root/'memory.current').write_text('1000')
        (root/'memory.events').write_text('oom 7\noom_kill 3\n')
        capacity=dict(cgroup_path=str(root),physical_host_bytes=5000,cgroup_hard_limit_bytes=limit,
            effective_host_limit_bytes=limit,scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
        observed=observe_host_memory(policy,node_rank=node,root=root,capacity=capacity)
        memory={**MEMORY,**observed}
        assert memory['host_peak_bytes'] is None and memory['policy_node_rank']==node
        assert memory['policy_host_limit_bytes']==limit
        assert capacity_ok(memory,policy,node_rank=node)
        assert not capacity_ok(memory,policy,node_rank=1-node)
        with pytest.raises(ValueError):observe_host_memory(policy,node_rank=1-node,root=root,capacity=capacity)


def v2_calibration_rows(tmp_path):
    from distillschedule.strategies.h3_ar_planning import CAPACITY_V2
    paths=calibration_rows(tmp_path,'zero3-native',100)
    policy=dict(schema_version=CAPACITY_V2,gpu_limit_bytes=1000,gpu_reserve_bytes=10,
        host_limits={'0':3000,'1':4000},host_reserve_bytes=0)
    for rank,path in enumerate(paths):
        row=json.loads(path.read_text());row['capacity_policy']=policy;node=rank//8;limit=policy['host_limits'][str(node)]
        for cycle in row['cycles']:
            cycle['memory'].update(host_peak_bytes=None,host_peak_status='UNAVAILABLE',
                host_peak_scope='UNAVAILABLE_CGROUP_V2_MEMORY_PEAK',
                host_capacity_evidence='FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK',
                capacity_observation=dict(cgroup_path='/cpu_fixture/node'+str(node),physical_host_bytes=5000,
                    cgroup_hard_limit_bytes=limit,effective_host_limit_bytes=limit,
                    scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL'),
                policy_node_rank=node,policy_host_limit_bytes=limit)
        path.write_text(json.dumps(row))
    return paths,policy


def test_v2_full_policy_and_node_bindings_survive_assembly_plan_and_tamper_rejection(tmp_path):
    paths,policy=v2_calibration_rows(tmp_path);output=tmp_path/'assembled'
    collection=assemble(H3ARTask(),paths,output)
    assert collection['capacity_policy']==policy
    assert collection['host_capacity_bindings'][8][0]['policy_node_rank']==1
    assert collection['host_capacity_bindings'][8][0]['policy_host_limit_bytes']==4000
    saved=plan(H3ARTask(),[output/'collection.json']);path=tmp_path/'saved.json';write_new_json(path,saved)
    assert validate_plan(H3ARTask(),path)[1]['capacity_policy']==policy
    saved['capacity_policy']['host_limits']['1']=3000
    saved['plan_sha256']=fingerprint({key:value for key,value in saved.items() if key!='plan_sha256'})
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):validate_plan(H3ARTask(),path)


def test_v2_assembler_rejects_wrong_node_even_when_host_limit_numbers_match(tmp_path):
    paths,policy=v2_calibration_rows(tmp_path)
    for path in paths:
        value=json.loads(path.read_text());value['capacity_policy']['host_limits']['1']=3000
        for cycle in value['cycles']:
            cycle['memory']['policy_host_limit_bytes']=3000
            cycle['memory']['capacity_observation'].update(cgroup_hard_limit_bytes=3000,effective_host_limit_bytes=3000)
        path.write_text(json.dumps(value))
    row=json.loads(paths[0].read_text())
    for cycle in row['cycles']:cycle['memory']['policy_node_rank']=1
    paths[0].write_text(json.dumps(row))
    with pytest.raises(ValueError,match='capacity policy failed'):assemble(H3ARTask(),paths,tmp_path/'rejected')


@pytest.mark.parametrize('invalid',['version','missing_node','extra_node','integer_keys','boolean_limit','scalar_mix','missing_context'])
def test_v2_capacity_schema_is_explicit_and_v1_scalar_policy_is_unchanged(invalid):
    from distillschedule.strategies.h3_ar_planning import CAPACITY_V2,validate_policy,policy_for_node
    assert policy_for_node(POLICY)==POLICY and policy_for_node(POLICY,1)==POLICY
    policy=dict(schema_version=CAPACITY_V2,gpu_limit_bytes=1000,gpu_reserve_bytes=10,
        host_limits={'0':3000,'1':4000},host_reserve_bytes=0)
    if invalid=='version':policy['schema_version']='distillschedule.original_h3_ar_capacity.v3'
    elif invalid=='missing_node':policy['host_limits']={'0':3000}
    elif invalid=='extra_node':policy['host_limits']['2']=4000
    elif invalid=='integer_keys':policy['host_limits']={0:3000,1:4000}
    elif invalid=='boolean_limit':policy['host_limits']['1']=True
    elif invalid=='scalar_mix':policy['host_limit_bytes']=3000
    if invalid=='missing_context':
        validate_policy(policy)
        with pytest.raises(ValueError,match='explicit node'):policy_for_node(policy)
    else:
        with pytest.raises(ValueError):validate_policy(policy)


def test_v2_wrong_actual_node_limit_rejects_in_worker_before_asset_or_model_construction(tmp_path,monkeypatch):
    import distillschedule.strategies.h3_ar_worker as worker
    from distillschedule.strategies.h3_ar_planning import CAPACITY_V2
    cgroup=tmp_path/'cgroup';cgroup.mkdir();(cgroup/'memory.current').write_text('1000')
    (cgroup/'memory.events').write_text('oom 0\noom_kill 0\n')
    observed=dict(cgroup_path=str(cgroup),physical_host_bytes=600<<30,cgroup_hard_limit_bytes=512<<30,
        effective_host_limit_bytes=512<<30,scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')
    monkeypatch.setattr(worker,'_capacity_observation',lambda:(cgroup,observed))
    for key,value in dict(RANK='8',LOCAL_RANK='0',WORLD_SIZE='16',LOCAL_WORLD_SIZE='8').items():monkeypatch.setenv(key,value)
    assets=SimpleNamespace(validate=lambda:pytest.fail('wrong-node capacity reached asset/model construction'))
    policy=dict(schema_version=CAPACITY_V2,gpu_limit_bytes=1000,gpu_reserve_bytes=10,
        host_limits={'0':512<<30,'1':500<<30},host_reserve_bytes=0)
    with pytest.raises(ValueError,match='host capacity unproved'):
        worker.run_worker(H3ARTask(),assets,tmp_path/'not-created',mode='calibrate',capacity_policy=policy,
            implementation=H3ARImplementation('zero3-native'))
    assert not (tmp_path/'not-created').exists()
