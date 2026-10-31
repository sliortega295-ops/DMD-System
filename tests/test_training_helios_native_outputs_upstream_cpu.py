"""Actual tiny original W2 G/native forward/Wan2.1 VAE; random, not quality.

The owner protocol injects only bootstrap/source/CUDA transport. This is not
native world16/HSDP or a pretrained 14B capacity/performance measurement.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from test_training_helios_public_cpu import STACK_AVAILABLE
from test_training_helios_native_outputs import _owned_export
from test_training_wan_native_outputs import codec_request, pinned
from test_training_wan_native_outputs_upstream_cpu import engines
from distillschedule.workloads.best_known.registry import repository_root

pytestmark = pytest.mark.skipif(not STACK_AVAILABLE, reason='requires actual original optional Helios stack')


def marker_engines(current=False):
    g, c, ema = engines(NS(lora_rank=1), current=current)
    for e, step in ((g, 102 if current else 2), (c, 510 if current else 10)):
        e.global_steps, e.global_samples = step, step * e.train_batch_size()
        e.lr_scheduler.last_epoch, e.lr_scheduler._step_count = step, step + 1
        for slots in e.optimizer.state.values():
            slots['step'].fill_(step)
    ema.optimization_step = 510 if current else 10
    return g, c, ema


def test_public_native_w2_export_restores_current_g_then_new_process_decodes_new_prompt(tmp_path, monkeypatch):
    from distillschedule.training import helios_native_inference as output
    from distillschedule.training.inference import read_export
    from distillschedule.data import import_preencoded
    torch.set_num_threads(1)
    request = codec_request(tmp_path / 'old-codec')
    request['schema_version'] = output.EXPORT_SCHEMA
    request['budget'] = dict(max_parameters=2_000_000, max_loaded_bytes=2**31,
        max_primary_export_bytes=2**31, max_transient_tensor_bytes=2**28,
        max_activation_elements=100_000_000, max_attention_elements=512_000_000,
        host_reserve_bytes=2**20, device_reserve_bytes=2**20)
    exported, receipt, old = _owned_export(tmp_path, monkeypatch, optimized=False,
        engines_factory=marker_engines, request=request, callback_type=output.NativeOutputRequest)
    root, manifest = read_export(tmp_path / 'portable')
    assert manifest['checkpoint_cycle'] == 100
    assert manifest['model']['factory'] == output.FACTORY
    p = manifest['model']['parameters']
    assert p['native']['args'] == dict(base_seq_len=256, max_seq_len=4096, base_shift=.5, max_shift=1.15)
    assert p['task']['model_family'] == 'helios14b' and p['task']['variant'] == 'no-gan'
    assert p['task']['video_height'] == 384 and p['task']['video_width'] == 640 and p['task']['video_frames'] == 33
    assert p['geometry']['noise_shape'] == [1, 16, 9, 48, 80]
    assert exported['output_transport'] == 'PRIMARY_ONLY_CPU_STAGING_ALL_OWNERS_FULL_TENSOR'
    weights = torch.load(root / 'weights.pt', weights_only=True)
    assert set(weights) == {'generator', 'codec'}
    assert any('lora_B' in k and bool(torch.count_nonzero(v)) for k, v in weights['generator'].items())
    assert not any('optimizer' in k or 'critic' in k or 'ema' in k for k in weights)
    assert str(old) not in json.dumps(p) and str(tmp_path / 'old-codec') not in json.dumps(p)
    assert receipt['output_only']['cycles'] == 0
    new = tmp_path / 'NEW-input'; new.mkdir()
    prompt = torch.randn(1, 2, 4096, generator=torch.Generator().manual_seed(981)).to(torch.bfloat16)
    torch.save(dict(prompt_embeds=prompt), new / 'prompt.pt')
    (new / 'inputs.json').write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-native-W2-prompt', features=dict(path='prompt.pt', sha256=pinned(new / 'prompt.pt')['sha256']))],
        metadata=dict(evidence_kind=request['evidence_kind']))))
    import_preencoded(new / 'inputs.json', new / 'prepared')
    shutil.rmtree(old); shutil.rmtree(tmp_path / 'old-codec')
    script = r'''
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1])
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  assert not any(str(root/p) in str(args[0]) for p in ('old-training','old-codec')),'read old training/codec asset'
sys.addaudithook(audit)
def forbidden_optimizer(*a,**kw):raise AssertionError('inference constructed optimizer')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=forbidden_optimizer
class NoTraining:
 def find_spec(self,name,path=None,target=None):
  assert name not in ('distillschedule.training.native','distillschedule.training.helios_worker',
   'distillschedule.training.helios_lifecycle'),name
sys.meta_path.insert(0,NoTraining())
torch.set_num_threads(1)
from distillschedule.training.cli import main
from distillschedule.training import helios_native_inference as output
from distillschedule.strategies.wan_capture import native_primitives
from distillschedule.adapters.helios.reference_model import load_reference_model_class
from runtime.official_baseline import stable_block_noise
native=native_primitives(source_only=True);cls=load_reference_model_class(backend='torch-sdpa')
bindings=(native.post.sample_block_noise,native.post.run_generator,native.post.inference_with_trajectory_stage2)
counts=dict(native=0,pyramid=0,old_cholesky=0,stable_psd=0,forward=0)
codes={native.post.run_generator.__code__:'native',native.post.inference_with_trajectory_stage2.__code__:'pyramid',
 native.post.sample_block_noise.__code__:'old_cholesky',stable_block_noise.__code__:'stable_psd',cls.forward.__code__:'forward'}
def profile(frame,event,arg):
 if event=='call' and frame.f_code in codes:counts[codes[frame.f_code]]+=1
try:
 sys.setprofile(profile)
 assert main(['generate',str(root/'portable.inference.json'),'--weights',str(root/'portable'),
  '--prepared',str(root/'NEW-input/prepared/manifest.json'),'--output-kind','video','--output',str(root/'video')]) == 0
finally:sys.setprofile(None)
assert bindings==(native.post.sample_block_noise,native.post.run_generator,native.post.inference_with_trajectory_stage2)
assert counts==dict(native=1,pyramid=1,old_cholesky=0,stable_psd=2,forward=3),counts
values=torch.load(root/'video/generation.pt',weights_only=True)
assert values['generated']['latents'].shape == (1,16,9,48,80)
assert values['decoded']['video'].shape == (33,384,640,3)
assert values['decoded']['decoder']['latent_clamp'] is False
assert len(list((root/'video/frames').glob('*.png'))) == 33
assert not torch.cuda.is_initialized()
print('ACTUAL_TINY_W2_NATIVE_CURRENT_G_VAE_NEW_PROMPT_PASS_NOT_PRETRAINED_GPU_OR_QUALITY')
'''
    env = dict(os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1', GROUP_RANK='0', CUDA_VISIBLE_DEVICES='')
    completed = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=repository_root(),
                               env=env, text=True, capture_output=True, timeout=360)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'ACTUAL_TINY_W2_NATIVE_CURRENT_G_VAE_NEW_PROMPT_PASS' in completed.stdout
    assert not torch.cuda.is_initialized()
