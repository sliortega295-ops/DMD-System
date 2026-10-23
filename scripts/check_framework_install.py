#!/usr/bin/env python3
"""Exercise the wheel outside the checkout, reusing installed CPU dependencies.

No downloads or model assets. The isolated package installation is new; Torch,
NumPy and Pillow are reused from the invoking validated environment through a
dependency-only .pth path. Actual framework imports must be in the new venv.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import site
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def dependency_roots(checkout, *, paths=None):
    """Keep dependency-only import roots in the invoking interpreter's order.

    An overlay may intentionally provide Torch 2.10 before a base environment's
    Torch 2.14. site.getsitepackages() alone discards that qualification. Never
    expose the checkout or another framework/runtime tree to an installed-wheel
    acceptance, and never emit executable or multi-line .pth directives.
    """
    checkout = Path(checkout).resolve()
    ordered = list(sys.path if paths is None else paths)
    if paths is None:
        ordered.extend(site.getsitepackages())
    result = []
    for item in ordered:
        if not isinstance(item, str) or not item or any(c in item for c in ('\n', '\r')):
            continue
        root = Path(item).resolve()
        if (not root.is_dir() or root.is_relative_to(checkout) or
                (root / 'distillschedule').exists() or (root / 'runtime').exists()):
            continue
        if not (root.name in ('site-packages', 'dist-packages') or
                next(root.glob('*.dist-info'), None) is not None):
            continue
        if str(root) not in result:
            result.append(str(root))
    return result


def qualified_dependencies(roots, *, modern=False):
    """Record the real imported numerical stack, rather than guessing versions."""
    import importlib
    import importlib.metadata
    packages = {'torch':'torch', 'numpy':'numpy', 'Pillow':'PIL', 'safetensors':'safetensors'}
    if modern:
        packages.update({'diffusers':'diffusers', 'transformers':'transformers', 'peft':'peft',
                         'tokenizers':'tokenizers', 'accelerate':'accelerate', 'huggingface-hub':'huggingface_hub'})
    result = {}
    modern_pins = dict(torch='2.10.0', diffusers='0.37.1', transformers='5.14.1', peft='0.18.1',
                       tokenizers='0.22.2', accelerate='1.14.0', huggingface_hub='1.7.1', safetensors='0.8.0')
    for distribution, name in packages.items():
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not any(origin.is_relative_to(Path(root)) for root in roots):
            raise RuntimeError('dependency origin is not a reusable package root: ' + str(origin))
        version = str(module.__version__) if hasattr(module, '__version__') else importlib.metadata.version(distribution)
        if modern and name in modern_pins and version.split('+')[0] != modern_pins[name]:
            raise RuntimeError('modern-flow requires ' + name + '==' + modern_pins[name] + ', found ' + version)
        result[name] = {'version':version, 'origin':str(origin)}
    return result


MODERN_FLOW_ACCEPTANCE = r'''
import contextlib, hashlib, importlib, io, json, shutil, subprocess, sys, sysconfig
from pathlib import Path
import distillschedule, runtime
site = Path(sysconfig.get_paths()['purelib']).resolve()
checkout = Path(sys.argv[1]).resolve()
assert not Path.cwd().resolve().is_relative_to(checkout)
assert Path(distillschedule.__file__).resolve().is_relative_to(site)
assert Path(runtime.__file__).resolve().is_relative_to(site)
assert all(not Path(path).resolve().is_relative_to(checkout) for path in sys.path if path)
from distillschedule.recipes import directory
recipes = directory()
assert recipes.is_relative_to(Path(sys.prefix)) and not recipes.is_relative_to(checkout)
queries = []
query_phase = True
def audit(event, args):
    if query_phase and event == 'open' and isinstance(args[0], (str, bytes)):
        assert Path(str(args[0])).suffix not in ('.pt', '.bin', '.safetensors'), args[0]
sys.addaudithook(audit)
from distillschedule.cli import main
for arguments in (['doctor'], ['workloads','list'], ['models','list'], ['capabilities','list'],
                  ['config','validate',str(recipes/'flux_dmd/training.template.json')]):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert main(arguments) == 0
    queries.append(dict(arguments=arguments, output=output.getvalue()))
assert not any(name in sys.modules for name in ('torch','diffusers','transformers','peft'))
query_phase = False
import numpy as np
import torch
from PIL import Image
from distillschedule.data import PreparedDataset
from distillschedule.data.manifest import source_identity
from distillschedule.training.config import load_training_config
from distillschedule.training.session import _load_for_output
expected_dependencies = json.loads(sys.argv[2])
for name, expected in expected_dependencies.items():
    module = importlib.import_module(name)
    assert str(module.__version__) == expected['version'], (name, module.__version__, expected)
    assert str(Path(module.__file__).resolve()) == expected['origin'], (name, module.__file__, expected)
assert not torch.cuda.is_initialized()
commands = []
def command(arguments):
    argv = [sys.executable, '-I', *map(str, arguments)]
    result = subprocess.run(argv, text=True, capture_output=True)
    commands.append(dict(argv=argv, cwd=str(Path.cwd()), returncode=result.returncode))
    with Path('subprocess.log').open('a') as stream:
        stream.write(json.dumps(commands[-1])+'\n'+result.stdout+result.stderr+'\n')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)
def cli(*arguments):
    return command(['-m','distillschedule',*arguments])
def equal(x,y):
    if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and torch.equal(x,y)
    if isinstance(x,dict): return isinstance(y,dict) and x.keys()==y.keys() and all(equal(v,y[k]) for k,v in x.items())
    if isinstance(x,(tuple,list)): return isinstance(y,type(x)) and len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x == y
families = {}
for family, recipe, shape, kind, size, preparer in (
        ('flux2-klein-base','flow_image',[1,4,4,4],'image',8,'flow_image_inputs'),
        ('wan22-ti2v5b-t2v','wan22_dmd',[1,4,2,2,2],'video',32,'wan22_inputs')):
    root = Path(family).resolve(); root.mkdir()
    helper = recipes/recipe/'create_cpu_fixture.py'
    arguments = [helper,root/'assets','--cycles',3]
    if kind == 'image': arguments += ['--family',family]
    assets = command(arguments)
    config = Path(assets['config'])
    prepared = cli('data','prepare',root/'assets/raw.json','--output',root/'assets/prepared-cli',
        '--preparer','distillschedule.data.'+preparer+':prepare_inputs')
    assets['prepared'] = prepared['manifest']
    request = json.loads(config.read_text()); request['data']['prepared'] = assets['prepared']
    config.write_text(json.dumps(request))
    cli('data','validate',assets['prepared'],'--sources')
    first = cli('train',config,'--cycles',1,'--output',root/'partial')
    resumed = cli('train',config,'--cycles',3,'--resume',first['checkpoint'],'--output',root/'resumed')
    whole = cli('train',config,'--cycles',3,'--output',root/'whole')
    a = torch.load(Path(resumed['checkpoint'])/'state.pt',weights_only=True)
    b = torch.load(Path(whole['checkpoint'])/'state.pt',weights_only=True)
    assert equal(a,b), 'complete state/RNG/Adam/schedulers/EMA/data/versions must match'
    assert a['state']['dmd_counters']['value']['counters'].tolist() == [3,3,15]
    rows = [json.loads(line) for line in (root/'whole/metrics.jsonl').read_text().splitlines()]
    continued = [json.loads(line) for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
    assert [row['sample_id'] for row in continued] == [row['sample_id'] for row in rows][1:]
    current = _load_for_output(load_training_config(config),resumed['checkpoint'])[0]
    training_feeds = PreparedDataset(assets['prepared']).sample(0)
    with torch.no_grad():
        sigma = torch.tensor([.5])
        student = current.state['student'].denoise(training_feeds['latents'],sigma,training_feeds,conditional=True)
        teacher = current.state['teacher'].denoise(training_feeds['latents'],sigma,training_feeds,conditional=True)
    assert not equal(student,teacher), 'current Student must be an actual updated model'
    exported = cli('export',config,'--checkpoint',resumed['checkpoint'],'--output',root/'export')
    assert set(exported['model']['modules']) == {'student','media'}
    native_class = type(current.state['student'].native)
    source_path = Path(importlib.import_module(native_class.__module__).__file__).resolve()
    for folder in ('partial','resumed','whole'):
        shutil.rmtree(root/folder)
    shutil.rmtree(root/'assets')
    assert not list(root.rglob('state.pt')) and not config.exists()
    source = dict(schema_version='distillschedule.export_inputs.v1', weights='export',mode='prompts',
        records=[dict(id='NEW_AFTER_DELETION',prompt='a violet kite',latent_shape=shape)])
    (root/'new.json').write_text(json.dumps(source))
    prepared_result = cli('data','prepare',root/'new.json','--output',root/'new-prepared',
        '--preparer','distillschedule.data.export_inputs:prepare_inputs')
    dataset = PreparedDataset(prepared_result['manifest']); assert dataset.validate(sources=True)['status']=='PASS'
    feeds = dataset.sample(0)
    assert not equal(feeds['context'],training_feeds['context'])
    assert equal(tuple(feeds[key] for key in ('context','context_mask')),current.state['media'].encode_text(['a violet kite']))
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'],parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'),data=dict(prepared=str(dataset.path)),execution=dict(mode='reference'),
        trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(root/'unused')))
    (root/'inference.json').write_text(json.dumps(inference))
    actual = cli('generate',root/'inference.json','--weights',root/'export','--prepared',prepared_result['manifest'],
        '--output',root/'generation','--output-kind',kind)
    for value in current.state.values():
        if isinstance(value,torch.nn.Module): value.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        generated=current.generate(current.state,feeds,{})
        expected=dict(generated=generated,decoded=current.decode(generated,{}))
    artifact=torch.load(actual['artifact'],weights_only=True)
    assert equal(expected,artifact), 'fresh portable output must equal native CURRENT Student'
    pixels=((np.arange(size*size*3).reshape(size,size,3)*11+37)%256).astype('uint8')
    if kind=='image':
        media=root/'new-image.png'; Image.fromarray(pixels).save(media)
        assert Path(actual['image']['path']).is_file() and artifact['decoded']['image'].shape==(8,8,3)
    else:
        media=root/'new-frames'; media.mkdir()
        for i in range(5): Image.fromarray(np.roll(pixels,i,axis=0)).save(media/f'{i:04d}.png')
        assert len(list(Path(actual['video']['frames']).glob('*.png')))==5
        assert artifact['decoded']['video'].shape==(5,32,32,3)
    source.update(mode='raw',records=[dict(id='NEW_RAW_AFTER_DELETION',prompt='a teal bird',
        **{kind:dict(path=media.name,sha256=source_identity(media,frames=kind=='video')['sha256'])})])
    (root/'raw.json').write_text(json.dumps(source))
    raw=cli('data','prepare',root/'raw.json','--output',root/'raw-prepared',
        '--preparer','distillschedule.data.export_inputs:prepare_inputs')
    raw_actual=cli('generate',root/'inference.json','--weights',root/'export','--prepared',raw['manifest'],
        '--output',root/'raw-generation','--output-kind',kind)
    assert raw_actual['sample_id']=='NEW_RAW_AFTER_DELETION'
    families[family]=dict(evidence_kind='SYNTHETIC_TEST_ONLY', native_class=native_class.__name__,
        native_source=str(source_path),native_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        recipe=str(helper),recipe_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        complete_state_rng_data_equal=True,fresh_process_resume=True,clocks=[3,3,15],
        sample_ids=[row['sample_id'] for row in rows],training_assets_checkpoints_deleted=True,
        current_student_differs_from_teacher=True,public_export_only_new_prompt=True,
        public_export_only_new_raw_media=True,exact_current_student_decode=True,
        generated=actual,raw_generated=raw_actual,export_manifest_sha256=actual['export_manifest_sha256'])
for name,module in tuple(sys.modules.items()):
    if (name=='distillschedule' or name.startswith('distillschedule.') or name=='runtime' or name.startswith('runtime.')) and getattr(module,'__file__',None):
        assert Path(module.__file__).resolve().is_relative_to(site), (name,module.__file__)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INSTALLED_MODERN_FLOW_CPU',families=families,commands=commands,
    queries_without_torch_cuda_or_weights=True,queries=queries,scope='SYNTHETIC_TEST_ONLY',
    source=dict(working_directory_outside_checkout=True,site_packages=str(site),recipes=str(recipes),
        python_search_path=sys.path,
        module_paths=dict(distillschedule=str(Path(distillschedule.__file__).resolve()),runtime=str(Path(runtime.__file__).resolve())),
        python_version=sys.version.split()[0],qualified_dependencies=expected_dependencies),
    gpu_execution='NOT_RUN',pretrained_quality='NOT_RUN',capacity_validation='NOT_RUN',quality_validation='NOT_RUN')))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modern-flow", action="store_true", help="actual tiny native image and video CPU acceptance")
    parser.add_argument("--wheel", type=Path, help="validate this already-built wheel without rebuilding")
    parser.add_argument("--work-dir", type=Path, help="preserve installation/media/logs in a NEW directory outside checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dependency_sites = dependency_roots(root)
    qualified = qualified_dependencies(dependency_sites, modern=args.modern_flow)
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.work_dir:
        args.work_dir = args.work_dir.resolve()
        if args.work_dir.is_relative_to(root):
            raise ValueError('installed acceptance work directory must be outside the checkout')
        args.work_dir.mkdir(parents=True, exist_ok=False)
    context = nullcontext(str(args.work_dir)) if args.work_dir else tempfile.TemporaryDirectory(prefix="dmd-framework-wheel-")
    with context as scratch:
        work = Path(scratch)
        wheels = work / "wheels"
        wheels.mkdir()
        # Invoke the already installed PEP 517 backend directly. Validated uv
        # environments need not carry pip, and this must not modify them just
        # to build a wheel or fetch isolated build dependencies.
        if args.wheel:
            import shutil
            shutil.copy2(args.wheel.resolve(), wheels / args.wheel.name)
        else:
            subprocess.run([sys.executable, "-c", ("import sys; from setuptools.build_meta import build_wheel; "
                        "build_wheel(sys.argv[1])"), str(wheels)],
                       cwd=root, env=env, check=True, stdout=subprocess.PIPE, text=True)
        files = list(wheels.glob("distillschedule-*.whl"))
        if len(files) != 1:
            raise RuntimeError("expected one framework wheel")
        target = work / "isolated"
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / "bin/python"
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(files[0])],
                       cwd=work, env=env, check=True, stdout=subprocess.PIPE, text=True)
        package_site = Path(subprocess.check_output([str(python), "-I", "-c",
                            "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip())
        (package_site / "reused-dependencies.pth").write_text("\n".join(dependency_sites) + "\n")
        script = r'''
import json, os, shutil, subprocess, sys, sysconfig
from pathlib import Path
import distillschedule, runtime
site = Path(sysconfig.get_paths()['purelib']).resolve()
checkout = Path(sys.argv[1]).resolve()
assert not Path.cwd().resolve().is_relative_to(checkout)
assert Path(distillschedule.__file__).resolve().is_relative_to(site)
assert Path(runtime.__file__).resolve().is_relative_to(site)
from distillschedule.cli import main
assert main(['doctor']) == 0
assert 'torch' not in sys.modules
from distillschedule.examples.tiny_av import QUALITY_STATUS, build_fixture
from distillschedule.data import PreparedDataset, prepare
from distillschedule.training.config import digest, load_training_config, TrainingConfig
from distillschedule.training.session import train, evaluate, export
import torch
expected_dependencies = json.loads(sys.argv[2])
import importlib
for name, expected in expected_dependencies.items():
    module = importlib.import_module(name)
    assert str(module.__version__) == expected['version'], (name, module.__version__, expected)
    assert str(Path(module.__file__).resolve()) == expected['origin'], (name, module.__file__, expected)
assert not torch.cuda.is_initialized()
def run_cli(*args):
    result = subprocess.run([sys.executable, '-I', '-m', 'distillschedule', *map(str, args)],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)
root = Path('demo').resolve()
build_fixture(root, samples=4)
prepare(root/'raw/manifest.json', root/'prepared')
training_inputs = PreparedDataset(root/'prepared/manifest.json')
config = load_training_config(root/'train.json')
raw = config.to_dict(); raw['trainer']['cycles'] = 2
first = train(TrainingConfig.from_dict(raw, source=config.source))
resumed = run_cli('train', root/'train.json', '--resume', first['checkpoint'],
                  '--cycles', 4, '--output', root/'resumed')
assert resumed['resume_cycle'] == 2 and resumed['cycles_this_process'] == 2
full = train(TrainingConfig.from_dict({**config.to_dict(),
    'trainer': {**dict(config.trainer), 'output_dir':str(root/'full')}}, source=config.source))
a = torch.load(Path(resumed['checkpoint'])/'state.pt', weights_only=True)
b = torch.load(Path(full['checkpoint'])/'state.pt', weights_only=True)
def equal(x,y):
    if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and torch.equal(x,y)
    if isinstance(x,dict): return x.keys()==y.keys() and all(equal(v,y[k]) for k,v in x.items())
    if isinstance(x,(tuple,list)): return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
    return x==y
assert all(equal(a[k],b[k]) for k in ('state','rng','cycle','cursor','versions','policy'))
evaluate(config, first['checkpoint'], root/'generated')
weights = Path('exported').resolve()
exported = export(config, first['checkpoint'], weights)
# Installed-source costs are measured with the same existing planner too.
auto = config.to_dict(); auto['trainer'].update(cycles=2, output_dir=str(root/'auto'))
auto['execution'] = dict(mode='auto', memory_limit_bytes=100000000,
    max_candidates=2, samples=1, max_program_runs=8)
planned = train(TrainingConfig.from_dict(auto, source=config.source))
assert planned['policy']['plan_sha256']
assert planned['policy']['initial_reference_gate']['status']=='PASS'
plan_path = root/'auto/planning/selected-plan.json'
saved = train(TrainingConfig.from_dict({**auto,
    'trainer': {**auto['trainer'], 'output_dir':str(root/'saved-plan')},
    'execution': {'mode':'plan', 'plan':str(plan_path)}}, source=config.source))
assert saved['policy']['plan_sha256'] == planned['policy']['plan_sha256']
assert saved['policy']['implementation'] == planned['policy']['implementation']
assert saved['final_versions'] == planned['final_versions']
assert not torch.cuda.is_initialized()
# A wheel installation must not require the development Git executable.
auto['trainer']['output_dir'] = str(root/'auto-no-git')
os.environ['PATH'] = str(root/'no-system-executables')
without_git = train(TrainingConfig.from_dict(auto, source=config.source))
assert without_git['policy']['initial_reference_gate']['status']=='PASS'
# The new record has genuinely different video/audio/text, not a deterministic
# copy of training records 0..3 at another path. Discard those unused copies.
new = Path('new-inputs').resolve()
build_fixture(new, samples=5)
raw_path = new/'raw/manifest.json'
document = json.loads(raw_path.read_text())
for record in document['records'][:-1]:
    shutil.rmtree(new/'raw'/record['id'])
record = document['records'][-1]
record['id'] = 'new-input-000004'
document['records'] = [record]
raw_path.write_text(json.dumps(document) + '\n')
(new/'train.json').unlink()
prepare(raw_path, new/'prepared')
new_inputs = PreparedDataset(new/'prepared/manifest.json')
training_hashes = {row['features']['sha256'] for row in training_inputs.manifest['records']}
new_hashes = {row['features']['sha256'] for row in new_inputs.manifest['records']}
assert training_hashes.isdisjoint(new_hashes)
assert all(not equal(new_inputs.sample(0), training_inputs.sample(index))
           for index in range(len(training_inputs)))
request = Path('inference-request.json').resolve()
request.write_text(json.dumps(config.to_dict()) + '\n')
# Keep only the resolved inference request and portable weights. A fresh Python
# process cannot recover any original raw/prepared data or checkpoint in memory.
shutil.rmtree(root)
assert not root.exists() and not Path(config.data['prepared']).exists()
generated = run_cli('generate', request, '--weights', weights,
                    '--prepared', new/'prepared/manifest.json', '--output', Path('new-generated').resolve())
assert generated['kind'] == 'INFERENCE_FROM_EXPORTED_WEIGHTS'
assert generated['sample_id'] == 'new-input-000004'
assert generated['input_identity'] == new_inputs.identity != training_inputs.identity
assert generated['model']['source'] == exported['model']['source']
assert generated['loaded_factory'] == exported['model']['source']
assert generated['model']['modules'] == exported['model']['modules']
assert generated['quality_validation'] == exported['quality_validation'] == 'NOT_RUN'
values = torch.load(generated['artifact'], weights_only=True)
video = values['decoded']['video']
assert values['decoded']['quality_status'] == QUALITY_STATUS
assert generated['video']['layout'] == 'THWC' and video.dtype == torch.uint8
assert list(video.shape) == generated['video']['shape']
import numpy as np
from PIL import Image
frames = sorted(Path(generated['video']['frames']).glob('*.png'))
assert len(frames) == len(video)
for index, path in enumerate(frames):
    with Image.open(path) as image:
        assert torch.equal(torch.from_numpy(np.array(image)), video[index])
for name, module in tuple(sys.modules.items()):
    if (name == 'distillschedule' or name.startswith('distillschedule.')
            or name == 'runtime' or name.startswith('runtime.')) and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(site), name
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INSTALLED_FRAMEWORK_CPU',
    raw_media=True, real_autograd=True, restart_full_state_equal=True,
    generate=True, export=True, measured_plan_executed=True, git_optional=True,
    fresh_process_resume=True, saved_plan_executed=True,
    exported_weights_new_inputs=True, training_artifacts_unavailable=not root.exists(),
    fresh_process_generation=True, scope='LOCAL_RANDOM_TINY_MEDIA_FIXTURE',
    source=dict(working_directory_outside_checkout=True, site_packages=str(site),
        module_paths={'distillschedule':str(Path(distillschedule.__file__).resolve()),
                      'runtime':str(Path(runtime.__file__).resolve())},
        python_version=sys.version.split()[0], torch_version=torch.__version__,
        qualified_dependencies=expected_dependencies),
    inference=dict(kind=generated['kind'], sample_id=generated['sample_id'],
        input_identity=generated['input_identity'], training_input_identity=training_inputs.identity,
        prepared_features_disjoint=True, encoded_feeds_disjoint=True,
        video_layout=generated['video']['layout'], source_bound=True, module_layout_bound=True,
        model_source_sha256=generated['loaded_factory']['sha256'],
        module_layout_sha256=digest(generated['model']['modules']),
        export_manifest_sha256=generated['export_manifest_sha256'],
        quality_validation=generated['quality_validation']),
    gpu_execution='NOT_RUN', pretrained_quality='NOT_RUN', quality_validation='NOT_RUN',
    native_pretrained_lifecycles='NOT_RUN')))
'''
        if args.modern_flow:
            script = MODERN_FLOW_ACCEPTANCE
        result = subprocess.run([str(python), "-I", "-c", script, str(root), json.dumps(qualified)], cwd=work, env=env,
                                text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("installed framework acceptance failed:\n" + result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        report["source"]["wheel_sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
