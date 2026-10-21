#!/usr/bin/env python3
"""Exercise the wheel outside the checkout, reusing installed CPU dependencies.

No downloads or model assets. The isolated package installation is new; Torch,
NumPy and Pillow are reused from the invoking validated environment through a
dependency-only .pth path. Actual framework imports must be in the new venv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import site
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="dmd-framework-wheel-") as scratch:
        work = Path(scratch)
        wheels = work / "wheels"
        wheels.mkdir()
        # Invoke the already installed PEP 517 backend directly. Validated uv
        # environments need not carry pip, and this must not modify them just
        # to build a wheel or fetch isolated build dependencies.
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
        dependency_sites = [path for path in site.getsitepackages() if Path(path).is_dir()]
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
        python_version=sys.version.split()[0], torch_version=torch.__version__),
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
        result = subprocess.run([str(python), "-I", "-c", script, str(root)], cwd=work, env=env,
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
