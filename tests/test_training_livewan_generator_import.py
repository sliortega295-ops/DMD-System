"""Explicit W4 full-Student initialization; never training-state continuation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from distillschedule.training import livewan_generator_import as importer


def test_w4_import_contract_is_explicit_initialization_not_resume():
    assert importer.SCHEMA == 'distillschedule.livewan_generator_weights_initialization.v1'
    assert importer.KIND == 'GENERATOR_WEIGHTS_INITIALIZATION_ONLY'


def pin(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=importer._sha(path))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return path


def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def fixture_request(root, *, qualified_runtime=False):
    """Real ORIGINAL tiny Student and real codec, synthetic OLD metadata only."""
    import torch
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.training import livewan_native_inference as native
    from distillschedule.training.wan_native_inference import _load_codec
    from test_livewan_reference_cpu import fixture_model
    from test_training_wan_native_outputs import codec_request
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    student = fixture_model(687)
    codec_spec = codec_request(root / 'codec')['codec']
    budget = dict(max_parameters=2_000_000, max_loaded_bytes=2**30,
        max_primary_export_bytes=2**30, max_attention_elements=8_000_000,
        max_input_elements=1_000_000, host_reserve_bytes=2**20)
    codec, codec_parameters = _load_codec(codec_spec, budget, native.FIXTURE)
    p = native.plain(dict(schema_version=native.SCHEMA, source=native.inference_source(),
        runtime=native.runtime_identity() if qualified_runtime else {'UNQUALIFIED_LOCAL_HELPER_ONLY': True},
        architecture=native.architecture(student), geometry=dict(resolution=[32, 32], frames=81, fps=16),
        precision=native.PRECISION, seed=822, evidence_kind=native.FIXTURE, budget=budget,
        codec=codec_parameters))
    binding = dict(path=str(root / 'OLD-inputs-not-read.json'), sha256='c'*64)
    cycle, head, world = 100, 'd'*40, 2
    payload = root / 'student.pt'
    torch.save(dict(model={n: t.detach().clone().contiguous() for n, t in sorted(student.named_parameters())},
        step=500, generator_updates=100, source_head=head, input_binding=binding, scope=importer.PAYLOAD_SCOPE), payload)
    assembly = dict(status='ASSEMBLED_CPU_WEIGHTS_NOT_GENERATION_QUALITY', **pin(payload),
        numel=sum(t.numel() for t in student.parameters()), parameter_tensors=len(list(student.parameters())),
        source_head=head, cycle=cycle, input_binding=binding, replicas_averaged=False,
        missing_parameters_replaced=False, CUDA_initialized=False,
        parameter_namespace='ORIGINAL_NATIVE_CausalBlock_AND_CausalHead_CANONICALIZER')
    docs = {}
    for role in ('selected', 'reference'):
        run_root = str(root / ('historical-' + role))
        summary = dict(schema_version='distillschedule.livewan_trajectory_summary.v1',
            status='PASS', errors=[], head=head, expected_head=head, heads={'owner': head},
            cycles=cycle, final_versions=dict(generator=cycle, critic=5*cycle, ema=0),
            rank_cycle_receipts=cycle*world, expected_rank_cycle_receipts=cycle*world,
            run_root=run_root, sample_contract='SYNTHETIC_HISTORICAL_METADATA_NOT_TRAINED_100',
            identity={'source_receipt_sha256': 'e'*64})
        docs[role + '_summary'] = pin(write_json(root / (role + '.json'), summary))
        if role == 'selected':
            assembly['checkpoint_sources'] = [dict(path=str(Path(run_root) / 'training_checkpoints' / 'cycle_100' / f'rank_{r:02d}.pt'),
                sha256=str(r)*64) for r in range(world)]
    comparison = dict(status='FAIL_EXTREME_NUMERICAL_DRIFT', source_head=head, cycles=cycle,
        rank_cycle_receipts_per_arm=cycle*world, thresholds_changed=False,
        state_scope='ORIGINAL_CANONICAL_NUMERIC_STATE_SKETCHES_NOT_FULL_STATE_BITWISE_EQUALITY',
        same_HEAD_self_repeat_noise_floor='NOT_RUN', extreme_violations=[dict(component='generator_adam', metric='cosine', value=.58)])
    docs['assembly'] = pin(write_json(root / 'assembly.json', assembly))
    docs['numerical_comparison'] = pin(write_json(root / 'comparison.json', comparison))
    source = root / 'old-WanModel.py'
    shutil.copyfile(Path(provider.__file__).parent / 'wan/modules/model.py', source)
    docs['model_source'] = pin(source)
    request = dict(schema_version=importer.SCHEMA, kind=importer.KIND, source=importer.source_identity(),
        parameters=p, student=dict(payload=pin(payload)), codec=codec_spec,
        origin=dict(contract=importer.ORIGIN_CONTRACT, **docs))
    path = write_json(root / 'request.json', request)
    return dict(root=root, request=request, path=path, student=student, codec=codec, comparison=comparison)


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    return fixture_request(tmp_path_factory.mktemp('actual-original-tiny-W4-init'))


def test_discovery_and_current_source_identity_remain_stdlib_only():
    overlay = Path(importer.__file__).parent
    script = ('import distillschedule.training,sys; '
        f'distillschedule.training.__path__.insert(0,{str(overlay)!r}); '
        'from distillschedule.training import livewan_generator_import as x; '
        'assert x.source_identity()["importer_sha256"]; '
        'assert not any(n in sys.modules for n in ("torch","diffusers","peft","wan","wanstreamer"))')
    result = subprocess.run([sys.executable, '-B', '-c', script], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_origin_preserves_whole_pair_fail_and_separate_execution_pass_at_new_cycle_zero(case):
    request, provenance = importer.validate_request(importer.read_request(case['path']))
    assert request == case['request']
    assert provenance['origin']['numerical_comparison'] == case['comparison']
    assert provenance['origin_numerical_status'] == 'FAIL_EXTREME_NUMERICAL_DRIFT'
    assert provenance['origin_execution_status'] == {'selected': 'PASS', 'reference': 'PASS'}
    assert provenance['origin_cycle'] == 100 and provenance['origin_versions'] == {'generator': 100, 'critic': 500, 'ema': 0}
    assert provenance['origin_input_binding'] == provenance['origin']['assembly']['input_binding']
    assert provenance['current_cycle'] == provenance['training_cycles_executed'] == 0
    assert provenance['positional_state']['policy'] == importer.ROPE_POLICY
    assert provenance['positional_state']['old_runtime_values_captured'] is False
    assert not provenance['default_promotion'] and provenance['current_numerical_validation'] == 'NOT_RUN'


@pytest.mark.parametrize('fault', ['kind', 'source', 'origin_contract', 'payload_pin', 'source_model',
    'selected_fail', 'source_head', 'versions', 'census', 'checkpoint_ref', 'comparison_source',
    'comparison_census', 'averaged', 'fill', 'namespace', 'input_binding', 'inventory_budget', 'old_checkpoint'])
def test_request_refuses_changed_original_source_receipts_and_implicit_resume(case, tmp_path, fault):
    value = deepcopy(case['request'])
    def doc(name, change):
        raw = json.loads(Path(value['origin'][name]['path']).read_text())
        change(raw)
        value['origin'][name] = pin(write_json(tmp_path / (name + '.json'), raw))
    if fault == 'kind': value['kind'] = 'RESUME'
    elif fault == 'source': value['source']['importer_sha256'] = '0'*64
    elif fault == 'origin_contract': value['origin']['contract'] = 'resume-v2'
    elif fault == 'payload_pin': value['student']['payload']['sha256'] = '0'*64
    elif fault == 'source_model':
        path = tmp_path / 'wrong-model.py'; path.write_text('class WanModel: pass\n')
        value['origin']['model_source'] = pin(path)
    elif fault == 'selected_fail': doc('selected_summary', lambda x: x.update(status='FAIL'))
    elif fault == 'source_head': doc('selected_summary', lambda x: x.update(head='a'*40))
    elif fault == 'versions': doc('selected_summary', lambda x: x['final_versions'].update(ema=100))
    elif fault == 'census': doc('selected_summary', lambda x: x.update(rank_cycle_receipts=100))
    elif fault == 'checkpoint_ref': doc('assembly', lambda x: x['checkpoint_sources'].reverse())
    elif fault == 'comparison_source': doc('numerical_comparison', lambda x: x.update(source_head='a'*40))
    elif fault == 'comparison_census': doc('numerical_comparison', lambda x: x.update(rank_cycle_receipts_per_arm=1))
    elif fault == 'averaged': doc('assembly', lambda x: x.update(replicas_averaged=True))
    elif fault == 'fill': doc('assembly', lambda x: x.update(missing_parameters_replaced=True))
    elif fault == 'namespace': doc('assembly', lambda x: x.update(parameter_namespace='EMA'))
    elif fault == 'input_binding': doc('assembly', lambda x: x.update(input_binding={}))
    elif fault == 'inventory_budget': value['parameters']['budget']['max_loaded_bytes'] = 1
    else: value['resume_from'] = '/old/cycle-100'
    with pytest.raises(ValueError): importer.validate_request(value)


def test_read_request_rejects_duplicate_keys_and_nan_and_noncanonical_path(tmp_path):
    path = tmp_path / 'duplicate.json'; path.write_text('{"kind":"one","kind":"two"}')
    with pytest.raises(ValueError): importer.read_request(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError): importer.read_request(path)
    link = tmp_path / 'link.json'; link.symlink_to(path)
    with pytest.raises(ValueError): importer.read_request(link)


@pytest.mark.parametrize('fault', ['boolean_old_counter', 'nonmapping_budget'])
def test_request_uses_exact_counter_types_and_cleanly_refuses_malformed_budget(case, tmp_path, fault):
    value = deepcopy(case['request'])
    if fault == 'boolean_old_counter':
        summary = json.loads(Path(value['origin']['selected_summary']['path']).read_text())
        summary['final_versions']['ema'] = False  # equal to 0 in Python, but not an old integer clock
        value['origin']['selected_summary'] = pin(write_json(tmp_path / 'boolean.json', summary))
    else:
        value['parameters']['budget'] = None
    with pytest.raises(ValueError): importer.validate_request(value)


def test_real_tiny_original_complete_fp32_header_values_rope_and_rng(case):
    import torch
    from distillschedule.training import livewan_native_inference as native
    from distillschedule.training.checkpoint import rng_state
    from test_livewan_reference_cpu import fixture_model
    before = rng_state()
    values = importer.stage_student(case['student'], case['request'])
    student = fixture_model(688)
    with torch.no_grad(): student.freqs[1, 0] *= complex(.9, .3)
    view = native.current_view(student)
    importer.restore_student(view, values)
    assert same(before, rng_state())
    assert same(student.state_dict(), case['student'].state_dict())
    assert torch.equal(student.freqs, case['student'].freqs)
    assert student.freqs is view.rope_frequencies and student.freqs.dtype == torch.complex128
    assert not student.training and all(not p.requires_grad for p in student.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['missing', 'extra', 'prefix', 'shape', 'dtype', 'nan', 'grad',
    'scope', 'source', 'step', 'gcount', 'input_binding', 'ema', 'non_zip'])
def test_payload_preflight_rejects_whole_bad_inventory_before_student_mutation(case, tmp_path, fault):
    import torch
    raw = torch.load(case['request']['student']['payload']['path'], weights_only=True)
    state = raw['model']; name = next(iter(state))
    if fault == 'missing': state.pop(name)
    elif fault == 'extra': state['invented'] = torch.zeros(1)
    elif fault == 'prefix': state['module.'+name] = state.pop(name)
    elif fault == 'shape': state[name] = torch.zeros(1)
    elif fault == 'dtype': state[name] = state[name].bfloat16()
    elif fault == 'nan': state[name].reshape(-1)[0] = float('nan')
    elif fault == 'grad': state[name].requires_grad_(True)
    elif fault == 'scope': raw['scope'] = 'CURRENT_EMA'
    elif fault == 'source': raw['source_head'] = '0'*40
    elif fault == 'step': raw['step'] = 100
    elif fault == 'gcount': raw['generator_updates'] = 500
    elif fault == 'input_binding': raw['input_binding']['sha256'] = '0'*64
    elif fault == 'ema': raw['ema'] = state
    payload = tmp_path / 'bad.pt'
    if fault == 'non_zip': payload.write_text('not a torch archive')
    else: torch.save(raw, payload)
    request = deepcopy(case['request']); request['student']['payload'] = pin(payload)
    assembly = json.loads(Path(request['origin']['assembly']['path']).read_text())
    assembly.update(pin(payload))
    request['origin']['assembly'] = pin(write_json(tmp_path / 'assembly.json', assembly))
    before = {n: t.clone() for n, t in case['student'].state_dict().items()}
    with pytest.raises(ValueError): importer.stage_student(case['student'], request)
    assert same(before, case['student'].state_dict())


def test_staged_student_owns_storage_after_input_mmap_changes(case, tmp_path):
    import torch
    request = deepcopy(case['request'])
    payload = tmp_path / 'mutable.pt'; shutil.copyfile(request['student']['payload']['path'], payload)
    request['student']['payload'] = pin(payload)
    assembly = json.loads(Path(request['origin']['assembly']['path']).read_text()); assembly.update(pin(payload))
    request['origin']['assembly'] = pin(write_json(tmp_path / 'assembly.json', assembly))
    values = importer.stage_student(case['student'], request)
    expected = {n: t.clone() for n, t in values.items()}
    raw = torch.load(payload, weights_only=True)
    for t in raw['model'].values(): t.add_(1)
    torch.save(raw, payload)
    assert same(values, expected)


def test_real_original_view_codec_shared_publisher_and_fresh_strict_restore(case, tmp_path):
    """Local real-class/shared-publisher proof, NOT public runtime qualification."""
    import torch
    from distillschedule.training import livewan_native_inference as native
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.inference import publish_export, read_export, restore_modules
    from distillschedule.training.plugins import load_callable
    from test_livewan_reference_cpu import fixture_model
    values = importer.stage_student(case['student'], case['request'])
    view = native.current_view(fixture_model(702)); importer.restore_student(view, values)
    components = ModelComponents(state=dict(generator=view, codec=case['codec']))
    _, source = load_callable(native.FACTORY)
    _, provenance = importer.validate_request(case['request'])
    model = dict(factory=native.FACTORY, source=source, parameters=case['request']['parameters'], metadata={})
    manifest = publish_export(components, tmp_path / 'export', model=model,
        identity=dict(model=source, model_metadata={}, weights_initialization=provenance), cycle=0)
    assert read_export(tmp_path / 'export')[1] == manifest
    target = ModelComponents(state=dict(generator=native.current_view(fixture_model(703)), codec=deepcopy(case['codec'])))
    restore_modules(target, tmp_path / 'export', manifest)
    assert same(target.state['generator'].state_dict(), view.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        publish_export(components, tmp_path / 'export', model=model, identity={}, cycle=0)
    script = '''
from pathlib import Path
import sys,torch
from distillschedule.training import livewan_native_inference as native
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.inference import read_export,restore_modules
from test_livewan_reference_cpu import fixture_model
from distillschedule.training.wan_native_inference import _codec_class
def forbidden(*a,**k):raise AssertionError('optimizer/trainer constructed')
torch.optim.AdamW.__init__=forbidden
root,manifest=read_export(sys.argv[1]);p=manifest['model']['parameters']
codec=_codec_class()(**p['codec']['architecture'])
state=dict(generator=native.current_view(fixture_model(704)),codec=codec)
components=ModelComponents(state=state)
restore_modules(components,root,manifest)
expected=torch.load(root/'weights.pt',weights_only=True)
assert torch.equal(state['generator'].model.head.head.weight,expected['generator']['model.head.head.weight'])
assert state['generator'].model.freqs is state['generator'].rope_frequencies
assert not torch.cuda.is_initialized()
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path / 'export')],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(__file__).resolve().parent),
                                                       os.environ.get('PYTHONPATH', '')))),
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_importer_uses_current_public_factory_and_never_training_checkpoint_owner():
    import ast
    tree = ast.parse(Path(importer.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and any(token in name for token in ('livewan_c51_training', 'livewan_c51_state', 'native_state')) for name in imports)
    assert 'livewan_native_inference' in Path(importer.__file__).read_text()
    assert 'factory(p)' in Path(importer.__file__).read_text()
    assert not any(isinstance(node, ast.Attribute) and node.attr in ('backward', 'step', 'set_rng_state') for node in ast.walk(tree))
