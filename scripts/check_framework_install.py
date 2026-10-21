#!/usr/bin/env python3
"""Exercise the wheel outside the checkout, reusing installed CPU dependencies.

No downloads or model assets. The isolated package installation is new; Torch,
NumPy and Pillow are reused from the invoking validated environment through a
dependency-only .pth path. Actual framework imports must be in the new venv.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="dmd-framework-wheel-") as scratch:
        work = Path(scratch)
        wheels = work / "wheels"
        wheels.mkdir()
        # Invoke the already installed PEP 517 backend directly. Validated uv
        # environments need not carry pip, and this must not modify them just
        # to build a wheel or fetch isolated build dependencies.
        subprocess.run([sys.executable, "-c", "import sys; from setuptools.build_meta import build_wheel; "
                        "build_wheel(sys.argv[1])", str(wheels)],
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
import json, os, sys, sysconfig
from pathlib import Path
import distillschedule, runtime
site = Path(sysconfig.get_paths()['purelib']).resolve()
assert Path(distillschedule.__file__).resolve().is_relative_to(site)
assert Path(runtime.__file__).resolve().is_relative_to(site)
from distillschedule.cli import main
assert main(['doctor']) == 0
assert 'torch' not in sys.modules
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.data import prepare
from distillschedule.training.config import load_training_config, TrainingConfig
from distillschedule.training.session import train, evaluate, export
import torch
assert not torch.cuda.is_initialized()
root = Path('demo').resolve()
build_fixture(root, samples=4)
prepare(root/'raw/manifest.json', root/'prepared')
config = load_training_config(root/'train.json')
raw = config.to_dict(); raw['trainer']['cycles'] = 2
first = train(TrainingConfig.from_dict(raw, source=config.source))
raw['trainer'].update(cycles=4, resume_from=first['checkpoint'], output_dir=str(root/'resumed'))
resumed = train(TrainingConfig.from_dict(raw, source=config.source))
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
export(config, first['checkpoint'], root/'exported')
# Installed-source costs are measured with the same existing planner too.
auto = config.to_dict(); auto['trainer'].update(cycles=2, output_dir=str(root/'auto'))
auto['execution'] = dict(mode='auto', memory_limit_bytes=100000000,
    max_candidates=2, samples=1, max_program_runs=8)
planned = train(TrainingConfig.from_dict(auto, source=config.source))
assert planned['policy']['plan_sha256']
assert planned['policy']['initial_reference_gate']['status']=='PASS'
assert not torch.cuda.is_initialized()
# A wheel installation must not require the development Git executable.
auto['trainer']['output_dir'] = str(root/'auto-no-git')
os.environ['PATH'] = str(root/'no-system-executables')
without_git = train(TrainingConfig.from_dict(auto, source=config.source))
assert without_git['policy']['initial_reference_gate']['status']=='PASS'
print(json.dumps(dict(status='PASS_INSTALLED_FRAMEWORK_CPU',
    raw_media=True, real_autograd=True, restart_full_state_equal=True,
    generate=True, export=True, measured_plan_executed=True, git_optional=True,
    gpu_execution='NOT_RUN', pretrained_quality='NOT_RUN')))
'''
        result = subprocess.run([str(python), "-I", "-c", script], cwd=work, env=env,
                                text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("installed framework acceptance failed:\n" + result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
