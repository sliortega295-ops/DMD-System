"""Revalidate existing remote H3 assets and private imports; no training/GPU."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
BASE = ROOT / 'research_artifacts/five_workload_h100_20260930'
OUT = BASE / 'h100_resident_baselines_20261014_r1/artifacts/h3_preparation'
PY = ROOT / 'envs/five-workload-h100-20260930/bin/python'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior_path = ROOT / 'research_artifacts/minimax_h3_full_assets_20260906/manifests/asset_sha256_receipt.json'
    prior = json.loads(prior_path.read_text())
    assert prior['status'] == 'PASS' and prior['repo_id'] == 'MiniMaxAI/MiniMax-H3'
    assert prior['revision'] == '42ed227ee7df40d41602854ae760620d6eb651fe'
    rows = []
    for spec in prior['files']:
        path = ROOT / 'assets/MiniMax-H3' / spec['path']
        actual = sha(path)
        expected = spec.get('expected_sha256') or spec['actual_sha256']
        row = dict(path=str(path), bytes=path.stat().st_size, sha256=actual,
                   expected_sha256=expected,
                   identity_source='upstream LFS SHA' if spec.get('expected_sha256') else 'pinned prior source receipt',
                   status='PASS' if actual == expected and path.stat().st_size == spec['expected_size'] else 'FAIL')
        rows.append(row)
        print('raw_H3_hash', len(rows), '/', len(prior['files']), row['status'], flush=True)
    save('raw_model_hash_receipt.json', dict(status='PASS' if all(x['status'] == 'PASS' for x in rows) else 'FAIL',
         prior_receipt_sha256=sha(prior_path), repo_id=prior['repo_id'], revision=prior['revision'], files=rows))
    assert all(x['status'] == 'PASS' for x in rows)
    loras = []
    for step in (4, 8):
        p = ROOT / f'assets/MiniMax-H3-Turbo-{step}step-v1.0-768p/minimax_h3_fl2v_turbo_{step}step_v1.0_768p_bf16.safetensors'
        x = dict(step=step, path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        assert x['bytes'] == 1383677808
        if step == 4:
            assert x['sha256'] == '1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c'
        x['use'] = 'inference asset; NOT applied to DMD bootstrap'
        loras.append(x)
    save('few_step_lora_receipt.json', loras)
    verifier = Path(__file__).parent / 'verify_h3_assets_inherited_r1.py'
    index = BASE / 'h3_common_admission_prepare_r1/manifests/h3_transfer_index_r2.json'
    assert sha(index) == '5886a6f81e65b2393d7aec8c9021dee154d7e40967ae95899744adae98e2091f'
    command = [str(PY), str(verifier), '--index', str(index), '--index-sha256', sha(index),
               '--assets-root', str(ROOT / 'assets/five_workload_h100_20260930/h3_grid_frozen_r1'),
               '--output-dir', str(OUT / 'converted_assets'), '--full-hash']
    save('converted_asset_command.json', dict(argv=command, verifier_sha256=sha(verifier)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    with (OUT / 'converted_asset_gate.log').open('x') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1800, pass_fds=(7,8,9))
    env['PYTHONPATH'] = ':'.join(map(str, [ROOT / 'repos/H3-DMD-for-A100-h100-b16-20261004',
        ROOT / 'envs/h3-data-overlay-20261004-r1/site-packages',
        ROOT / 'envs/h3-grid-overlay-20261004/site-packages',
        ROOT / 'repos/h3-diffusers-92846072-20261004-r2/src',
        ROOT / 'repos/h3-lightx2v-d034a6b0-20261004-r2',
        ROOT / 'repos/h3-lightx2v-d034a6b0-20261004-r2/lightx2v_train']))
    code = '''import torch,diffusers,transformers,peft,inspect,hashlib,json
from diffusers import MiniMaxH3Transformer3DModel
from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
from torch.distributed.fsdp import fully_shard
from h3_a100.model import MiniMaxH3A100Model
from h3_a100 import grid_entrypoint
assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
print(json.dumps(dict(status="PASS_CPU_IMPORTS",torch=torch.__version__,cuda=torch.version.cuda,diffusers=diffusers.__version__,diffusers_path=diffusers.__file__,transformers=transformers.__version__,peft=peft.__version__,forward_source_sha256=hashlib.sha256(inspect.getsource(MiniMaxH3Transformer3DModel.forward).encode()).hexdigest(),GPU="NOT_RUN",weights="NOT_LOADED"),indent=2))'''
    with (OUT / 'private_environment_imports.json').open('x') as f:
        subprocess.run([str(PY), '-c', code], env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)
    sources = {}
    for name in ('H3-DMD-for-A100-h100-b16-20261004', 'h3-diffusers-92846072-20261004-r2', 'h3-lightx2v-d034a6b0-20261004-r2'):
        p = ROOT / 'repos' / name
        probe = subprocess.run(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True, capture_output=True)
        if probe.returncode == 0:
            sources[name] = dict(path=str(p), head=probe.stdout.strip(),
                                status=subprocess.check_output(['git', '-C', str(p), 'status', '--porcelain'], text=True).strip())
        else:
            sources[name] = dict(path=str(p), head='NON_GIT_PINNED_SOURCE_SNAPSHOT',
                                identity_scope='actual imported module source SHA recorded; immutable snapshot receipt retained',
                                git_probe_status='NOT_A_GIT_WORKTREE')
    save('summary.json', dict(status='PASS_EXISTING_ASSETS_AND_PRIVATE_CPU_IMPORTS', sources=sources,
         elapsed_seconds=time.monotonic()-started, model_files=len(rows), raw_model_bytes=sum(x['bytes'] for x in rows),
         downloads='NOT_REQUIRED_ALL_PINNED_PAYLOADS_PRESENT_AND_REHASHED', new_model_bytes_downloaded=0,
         approved_missing_asset_route='BAAI institution proxy on H100 only; no local payload or TUN bulk traffic',
         training='NOT_RUN', W5_GPU='NOT_RUN', W6_specific_training_environment='NOT_RUN_SEPARATE_SOURCE_GATE_REQUIRED'))
    print('PASS_H3_PREPARATION_NO_GPU_TRAINING', flush=True)


if __name__ == '__main__':
    main()
