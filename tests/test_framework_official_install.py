"""Pure admission/provenance checks plus an explicit real installed-native lane.

The opt-in lane must run in the already-qualified original-model environment;
neither these tests nor the verifier installs any numerical dependencies.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tool():
    spec = importlib.util.spec_from_file_location('official_install_tool', ROOT/'scripts/check_framework_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def witness():
    namespace = {}
    exec(tool().OFFICIAL_INPUT_WITNESS, namespace)
    return SimpleNamespace(**namespace)


def test_sd_input_witness_rejects_tensor_aliases_and_preserves_rng():
    import torch
    api = witness()
    rows = [{'text_input_ids_one': torch.tensor([[[1, index, 2]]])} for index in range(4)]
    before = torch.get_rng_state().clone()
    api.require_distinct_sd_conditions(rows)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(AssertionError, match='distinguishable SD conditions'):
        api.require_distinct_sd_conditions([rows[0], rows[0]])
    assert api.feed_digest(rows[0]) != api.feed_digest(rows[1])
    assert api.feed_digest(rows[0]) == api.feed_digest({'text_input_ids_one': rows[0]['text_input_ids_one'].clone()})


@pytest.mark.parametrize('wrong_feed', [False, True])
def test_sd_actual_iteration_feed_witness_rejects_wrong_condition_with_same_ids(monkeypatch, wrong_feed):
    """Real macrocycle selection, tiny feed sink; not upstream-model qualification."""
    import contextlib
    import torch
    from distillschedule.training.dmd2_owner import DMD2Owner
    api = witness()
    owner = object.__new__(DMD2Owner)
    owner.ratio = 2; owner.native_step = 0
    owner.generator_updates = owner.guidance_updates = 0
    owner.args = SimpleNamespace(denoising=True, cls_on_clean_image=False, dfake_gen_update_ratio=2)
    owner.schedulers = {}
    owner._validate_live = lambda: None
    owner.validate_feeds = lambda feeds: None
    owner.random_scope = contextlib.nullcontext
    def native_sink(self, prompt, denoising_dict, real_train_dict):
        self.generator_updates += int(self.native_step % self.ratio == 0)
        self.guidance_updates += 1; self.native_step += 1
        return {'test_only': True}
    monkeypatch.setattr(DMD2Owner, '_native_iteration', native_sink)
    row = lambda index: {'text_input_ids_one': torch.tensor([[[index]]])}
    expected = {'generator': [row(1)], 'guidance': [row(2)], 'denoising': [row(3), row(4)]}
    actual = {key: list(value) for key, value in expected.items()}
    # IDs/progress are deliberately unchanged; only the consumed condition differs.
    bookkeeping = {'sample_id': {'generator': ['g'], 'guidance': ['c']}, 'progress': 1}
    if wrong_feed:
        actual['guidance'] = actual['generator']
    before = torch.get_rng_state().clone()
    with api.observe_sd_inputs(DMD2Owner) as observed:
        result = owner.run(actual, cycle=0)
    assert DMD2Owner._native_iteration is native_sink
    assert torch.equal(before, torch.get_rng_state())
    assert bookkeeping == {'sample_id': {'generator': ['g'], 'guidance': ['c']}, 'progress': 1}
    assert result['native_step'] == 2
    if wrong_feed:
        with pytest.raises(AssertionError, match='actual native input consumption'):
            api.validate_sd_consumption(observed, [expected], ratio=2)
    else:
        api.validate_sd_consumption(observed, [expected], ratio=2)


def test_sd_feed_witness_restores_original_method_after_failure():
    api = witness()
    class Broken:
        native_step = 4
        def _native_iteration(self, prompt, denoising_dict, real_train_dict):
            raise RuntimeError('native failure')
    original = Broken._native_iteration
    with pytest.raises(RuntimeError, match='native failure'):
        with api.observe_sd_inputs(Broken) as observed:
            Broken()._native_iteration(None, None, None)
    assert Broken._native_iteration is original and observed[0]['native_step'] == 4


@pytest.mark.parametrize('arguments,message', [
    (['--official-workflows'], 'requires both --dmd2-source and --self-forcing-source'),
    (['--official-workflows', '--dmd2-source', 'missing'], 'requires both'),
    (['--dmd2-source', 'missing'], 'require --official-workflows'),
    (['--self-forcing-source', 'missing'], 'require --official-workflows'),
    (['--official-workflows', '--modern-flow'], 'not allowed with argument'),
])
def test_official_arguments_reject_before_dependency_or_install_work(monkeypatch, capsys, arguments, message):
    module = tool()
    def forbidden(*args, **kwargs):
        pytest.fail('invalid arguments reached dependency/build/install/model work')
    monkeypatch.setattr(module, 'dependency_roots', forbidden)
    monkeypatch.setattr(module, 'qualified_dependencies', forbidden)
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(sys, 'argv', ['check_framework_install.py', *arguments])
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('kind', ['missing', 'file'])
def test_official_source_paths_are_explicit_local_directories(tmp_path, monkeypatch, capsys, kind):
    module = tool()
    first = tmp_path/'dmd2'; first.mkdir()
    second = tmp_path/'self-forcing'
    if kind == 'file':
        second.write_text('not a source tree')
    monkeypatch.setattr(module, 'dependency_roots', lambda *args: pytest.fail('early source admission was skipped'))
    monkeypatch.setattr(sys, 'argv', ['tool', '--official-workflows', '--dmd2-source', str(first),
                                     '--self-forcing-source', str(second)])
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 2
    assert '--self-forcing-source must be an existing local directory' in capsys.readouterr().err


def test_help_does_not_import_numerical_models_sources_or_open_weights(tmp_path):
    script = '''
import runpy, sys
def audit(event, args):
    if event == 'import':
        assert args[0].split('.')[0] not in ('torch','diffusers','transformers','peft','distillschedule','runtime'), args[0]
    if event == 'open' and isinstance(args[0], str):
        assert not args[0].endswith(('.pt','.bin','.safetensors')), args[0]
    assert event not in ('socket.connect','socket.getaddrinfo'), event
sys.addaudithook(audit)
path = sys.argv[1]; sys.argv = [path, '--help']
runpy.run_path(path, run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT/'scripts/check_framework_install.py')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--official-workflows' in result.stdout
    assert '--dmd2-source' in result.stdout and '--self-forcing-source' in result.stdout


def test_official_dependency_qualification_keeps_the_original_tf4_stack(tmp_path, monkeypatch):
    import importlib
    versions = dict(torch='2.10.0+cpu', numpy='2.5.3', PIL='12.3.0', safetensors='0.8.0',
                    diffusers='0.37.1', transformers='4.51.3', peft='0.17.1', accelerate='1.14.0',
                    ftfy='6.3.1', einops='0.8.1', omegaconf='2.3.0', regex='2026.1.15', sentencepiece='0.2.2')
    monkeypatch.setattr(importlib, 'import_module', lambda name: SimpleNamespace(
        __file__=str(tmp_path/name/'__init__.py'), __version__=versions[name]))
    module = tool()
    qualified = module.qualified_dependencies([str(tmp_path)], official=True)
    assert set(qualified) == set(versions)
    versions['transformers'] = '5.14.1'
    with pytest.raises(RuntimeError, match='official-workflows.*transformers'):
        module.qualified_dependencies([str(tmp_path)], official=True)


def fake_installed_child(tmp_path, *, mode='query', leak=False, bad_source=False, query_import=False):
    """Execute the real child bootstrap against disposable installed-like stubs.

    Only this test harness admits its fake package root; the production child
    never changes sys.path or imports from the checkout.
    """
    prefix = tmp_path/'isolated'; site = prefix/'site-packages'; site.mkdir(parents=True)
    recipes = prefix/'share/distillschedule/recipes'; recipes.mkdir(parents=True)
    package = site/'distillschedule'; package.mkdir(); (package/'__init__.py').write_text('')
    (site/'runtime.py').write_text('')
    (package/'recipes.py').write_text('from pathlib import Path\ndef directory(): return Path('+repr(str(recipes))+')\n')
    query_body = "__import__('torch')" if query_import else "print('{\"query\":true}')"
    (package/'cli.py').write_text('def main(args):\n    '+query_body+'\n    return 0\n')
    if leak:
        (package/'cli.py').write_text('import types, sys\n'
            'm = types.ModuleType("distillschedule.leaked"); m.__file__ = '+repr(str(ROOT/'src/distillschedule/leaked.py'))+'\n'
            'sys.modules[m.__name__] = m\ndef main(args): return 0\n')
    training = package/'training'; training.mkdir(); (training/'__init__.py').write_text('')
    for name in ('sd', 'self_forcing_source'):
        (training/(name+'.py')).write_text("SOURCE_FILES = {'native.py':'expected'}\nREVISION='locked'\nREPOSITORY='https://example.invalid/native'\n")
    (training/'upstream.py').write_text('''
from pathlib import Path
class PinnedSource:
    def __init__(self, root, files):
        self.root = Path(root)
        if (self.root/'native.py').read_text() != files['native.py']:
            raise ValueError('pinned source SHA256 mismatch: native.py')
        self.identity = {'files':files}
    def validate(self): return self.identity
''')
    sources = []
    for name in ('dmd2', 'self-forcing'):
        source = tmp_path/name; source.mkdir()
        (source/'native.py').write_text('corrupt' if bad_source else 'expected')
        sources.append(str(source))
    arguments = sources if mode == 'sources' else ['doctor']
    if mode == 'fixture':
        helper = recipes/'dmd2_sd15/create_cpu_fixture.py'; helper.parent.mkdir()
        helper.write_text('''
import json
from pathlib import Path
def build_fixture(output, source_root, **options):
    root = Path(output); root.mkdir()
    (root/'training.json').write_text(json.dumps({'TEST_ONLY': True}))
    (root/'prompts.json').write_text(json.dumps({'records': [
        {'id':str(index),'prompt':'a different image '+str(index)} for index in range(8)]}))
    return {'config':str(root/'training.json')}
''')
        arguments = ['dmd2_sd15',str(tmp_path/'fixture'),sources[0]]
    receipt = tmp_path/'receipt.json'
    bootstrap = tool().OFFICIAL_CHILD_BOOTSTRAP
    setup = ('import sys,sysconfig; '
             'sys.path.insert(0,'+repr(str(site))+'); '
             'sys.prefix='+repr(str(prefix))+'; '
             'sysconfig.get_paths=lambda: {"purelib":'+repr(str(site))+'}; '
             'sys.argv=["-c",'+repr(str(ROOT))+','+repr(str(receipt))+','+repr(mode)+','+
             ','.join(map(repr, arguments))+']; '
             'exec('+repr(bootstrap)+')')
    result = subprocess.run([sys.executable, '-I', '-c', setup], cwd=tmp_path, capture_output=True, text=True)
    return result, receipt, site, recipes


def test_real_fixture_bootstrap_rewrites_only_opt_in_sd_training_prompts(tmp_path):
    result, receipt, _, _ = fake_installed_child(tmp_path, mode='fixture')
    assert result.returncode == 0, result.stdout+result.stderr
    assert receipt.exists()
    prompts = json.loads((tmp_path/'fixture/prompts.json').read_text())['records']
    assert [row['id'] for row in prompts] == list(map(str,range(8)))
    assert [row['prompt'] for row in prompts] == ['p'+str(index) for index in range(8)]


def test_distinct_opt_in_prompts_survive_actual_eight_token_clip_limit(tmp_path):
    # No model weights: only the same real tokenizer class as the old fixture.
    pytest.importorskip('ftfy',reason='actual CLIP tiny tokenizer requires its pinned optional extra')
    import torch
    from transformers import CLIPTokenizer
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    vocabulary = {'<|startoftext|>':0,'<|endoftext|>':1}
    for token in bytes_to_unicode().values():
        for suffix in ('','</w>'):
            vocabulary[token+suffix] = len(vocabulary)
    vocab = tmp_path/'vocab.json'; vocab.write_text(json.dumps(vocabulary))
    merges = tmp_path/'merges.txt'; merges.write_text('#version: 0.2\n')
    tokenizer = CLIPTokenizer(str(vocab),str(merges),model_max_length=8)
    def rows(prompts):
        tokens = tokenizer(prompts,padding='max_length',max_length=8,truncation=True,return_tensors='pt').input_ids
        return [{'text_input_ids_one':row.reshape(1,1,-1)} for row in tokens]
    api = witness()
    with pytest.raises(AssertionError,match='distinguishable SD conditions'):
        api.require_distinct_sd_conditions(rows(['a different image '+str(index) for index in range(8)]))
    before = torch.get_rng_state().clone()
    api.require_distinct_sd_conditions(rows(['p'+str(index) for index in range(8)]))
    assert torch.equal(before,torch.get_rng_state())


def test_fresh_child_asserts_actual_installed_origins_and_writes_receipt(tmp_path):
    result, receipt, site, recipes = fake_installed_child(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(receipt.read_text())
    assert report['isolated'] is True and report['working_directory_outside_checkout'] is True
    assert report['queries_without_torch_cuda_or_weights'] is True
    assert report['recipes'] == str(recipes)
    assert all(Path(path).is_relative_to(site) for path in report['module_paths'].values())


def test_fresh_child_rejects_transitive_checkout_import_not_just_top_level(tmp_path):
    result, receipt, _, _ = fake_installed_child(tmp_path, leak=True)
    assert result.returncode != 0
    assert 'distillschedule.leaked' in result.stderr
    assert not receipt.exists()


def test_query_child_forbids_numerical_initialization(tmp_path):
    result, receipt, _, _ = fake_installed_child(tmp_path, query_import=True)
    assert result.returncode != 0
    assert 'torch' in result.stderr
    assert not receipt.exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_installed_source_preflight_uses_existing_pinned_loader_before_fixture(tmp_path, corrupt):
    result, receipt, _, _ = fake_installed_child(tmp_path, mode='sources', bad_source=corrupt)
    if corrupt:
        assert result.returncode != 0 and 'pinned source SHA256 mismatch' in result.stderr
        assert not receipt.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert set(json.loads(result.stdout)) == {'dmd2', 'self_forcing'}
        assert receipt.exists()


def test_embedded_acceptance_is_valid_and_children_share_the_provenance_bootstrap():
    module = tool()
    compile(module.OFFICIAL_WORKFLOWS_ACCEPTANCE, '<installed-official-acceptance>', 'exec')
    # The real command builder is tested without importing Torch/models or
    # constructing a wheel/venv: execute only its definitions before main work.
    prefix = module.OFFICIAL_WORKFLOWS_ACCEPTANCE.split('# BEGIN OFFICIAL ACCEPTANCE')[0]
    namespace = {'__name__':'acceptance_definition_test'}
    exec(prefix, namespace)
    argv = namespace['child_argv']('/installed/bin/python', '/checkout', '/receipt.json', 'cli', ['train','config.json'])
    assert argv == ['/installed/bin/python', '-I', '-c', module.OFFICIAL_CHILD_BOOTSTRAP,
                    '/checkout', '/receipt.json', 'cli', 'train', 'config.json']


def test_official_dispatch_preserves_offline_installed_invocation_without_installing(tmp_path, monkeypatch):
    module = tool()
    sources = [tmp_path/'dmd2', tmp_path/'self-forcing']
    for source in sources:
        source.mkdir()
    wheel = tmp_path/'distillschedule-0.3.0-py3-none-any.whl'; wheel.write_bytes(b'test-only-wheel')
    work = tmp_path/'acceptance'; site = work/'isolated/site-packages'
    qualified = {'torch':{'version':'2.10.0+cpu','origin':'/dependencies/torch/__init__.py'}}
    roots = [str(tmp_path/'dependency-site')]
    monkeypatch.setattr(module, 'dependency_roots', lambda checkout: roots)
    def qualify(actual_roots, *, official):
        assert actual_roots == roots and official is True
        return qualified
    monkeypatch.setattr(module, 'qualified_dependencies', qualify)
    class NoEnvironmentCreation:
        def __init__(self, *, with_pip):
            assert with_pip is True
        def create(self, target):
            assert target == work/'isolated'
            site.mkdir(parents=True)
    monkeypatch.setattr(module.venv, 'EnvBuilder', NoEnvironmentCreation)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: str(site)+'\n')
    calls = []
    def no_subprocess(argv, **kwargs):
        calls.append((argv,kwargs))
        assert kwargs['cwd'] == work
        assert 'PYTHONPATH' not in kwargs['env'] and 'PYTHONHOME' not in kwargs['env']
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['HF_HUB_OFFLINE'] == kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
        if '-c' in argv:
            assert argv[:3] == [str(work/'isolated/bin/python'),'-I','-c']
            assert argv[3] == module.OFFICIAL_WORKFLOWS_ACCEPTANCE
            assert argv[4] == str(ROOT) and json.loads(argv[5]) == qualified
            assert json.loads(argv[6]) == dict(dmd2=str(sources[0]), self_forcing=str(sources[1]))
            return SimpleNamespace(returncode=0,stdout=json.dumps(dict(source={},status='TEST_DISPATCH_ONLY')),stderr='')
        assert argv[1:7] == ['-I','-m','pip','install','--no-index','--no-deps']
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess, 'run', no_subprocess)
    monkeypatch.setenv('PYTHONPATH', str(ROOT/'src'))
    monkeypatch.setenv('PYTHONHOME', '/must-not-reach-child')
    monkeypatch.setattr(sys, 'argv', ['tool','--official-workflows','--dmd2-source',str(sources[0]),
        '--self-forcing-source',str(sources[1]),'--wheel',str(wheel),'--work-dir',str(work)])
    assert module.main() == 0
    assert len(calls) == 2
    assert (site/'reused-dependencies.pth').read_text() == '\n'.join(roots)+'\n'


@pytest.mark.skipif(os.environ.get('DMD_OFFICIAL_INSTALLED_CPU') != '1',
                    reason='explicit installed official native CPU lane; requires both pinned local source trees')
def test_actual_installed_official_sd15_sdxl_self_forcing_resume_export_new_inputs(tmp_path):
    sources = {key: os.environ.get(key) for key in ('DMD2_SOURCE_ROOT','SELF_FORCING_SOURCE_ROOT')}
    assert all(sources.values()), 'opt-in installed official acceptance requires BOTH explicit source roots'
    output = tmp_path/'installed-official.json'
    arguments = [sys.executable, str(ROOT/'scripts/check_framework_install.py'), '--official-workflows',
                 '--dmd2-source', sources['DMD2_SOURCE_ROOT'], '--self-forcing-source', sources['SELF_FORCING_SOURCE_ROOT'],
                 '--work-dir', str(tmp_path/'acceptance'), '--output', str(output)]
    if os.environ.get('DMD_OFFICIAL_INSTALLED_WHEEL'):
        arguments += ['--wheel', os.environ['DMD_OFFICIAL_INSTALLED_WHEEL']]
    result = subprocess.run(arguments, cwd=tmp_path, env={**os.environ,'PIP_NO_INDEX':'1'},
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'PASS_INSTALLED_OFFICIAL_WORKFLOWS_CPU'
    assert report['queries_without_torch_cuda_or_weights'] is True
    assert set(report['workflows']) == {'dmd2_sd15','dmd2_sdxl','self_forcing_official'}
    for name, proof in report['workflows'].items():
        for key in ('complete_packed_state_equal','fresh_process_resume','independent_stream_sampling_equal',
                    'real_native_classes','native_optimizer_roles_updated','teacher_unchanged',
                    'training_assets_checkpoints_deleted','new_conditions_differ_from_training',
                    'fresh_process_generation','exact_current_student_output','source_identity_equal'):
            assert proof[key] is True, (name,key)
        assert proof['complete_cycles'] == 3 and proof['native_iterations'] == 6
        assert proof['native_ratio'] == 2
        assert proof['evidence_kind'] == 'RANDOM_TINY_ACTUAL_NATIVE_CPU_ONLY'
        assert len(proof['sample_ids']) == 3
        assert proof['generated']['kind'] == 'INFERENCE_FROM_EXPORTED_WEIGHTS'
    assert report['workflows']['self_forcing_official']['ema'] == dict(initialized_step=2, updates=2)
    for name in ('dmd2_sd15','dmd2_sdxl'):
        assert report['workflows'][name]['schedulers'] == dict(generator=6,guidance=6)
        assert report['workflows'][name]['ema'] == 'NOT_CONFIGURED'
    commands = report['commands']
    assert commands and all(row['argv'][1:3] == ['-I','-c'] for row in commands)
    for command in commands:
        origin = command['origin']
        assert command['returncode'] == 0 and origin['isolated'] is True
        assert origin['working_directory_outside_checkout'] is True
        assert origin['module_paths']
        assert all(Path(path).is_relative_to(origin['site_packages']) for path in origin['module_paths'].values())
    assert report['gpu_execution'] == report['pretrained_quality'] == report['quality_validation'] == 'NOT_RUN'


def test_cpu_export_reference_storage_has_exact_values_parameters_and_rng():
    import torch
    api = witness()
    model = torch.nn.Linear(8, 32)
    model.weight = torch.nn.Parameter(torch.arange(257, dtype=torch.float32)[1:].reshape(32, 8))
    model.bias = torch.nn.Parameter(torch.arange(33, dtype=torch.float32)[1:], requires_grad=False)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    owners = {name: id(value) for name, value in model.named_parameters()}
    old_storage = {name: value.data_ptr() for name, value in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    api.materialize_cpu_reference_storage(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.weight.requires_grad is True and model.bias.requires_grad is False
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name]) and id(value) == owners[name]
        assert value.storage_offset() == 0 and value.data_ptr() != old_storage[name]
    aligned = torch.nn.Linear(8, 32)
    aligned.load_state_dict(model.state_dict())
    x = torch.arange(8, dtype=torch.float32).reshape(1,8) / 7
    assert torch.equal(model(x), aligned(x))
