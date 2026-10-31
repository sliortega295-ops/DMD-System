"""Current native W3 G/codec fresh-process proof; not pretrained/GPU quality."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_helios_public_cpu import STACK_AVAILABLE
from test_training_helios_gan_native_outputs import _owned_gan_export, with_gan_head
from test_training_helios_native_outputs_upstream_cpu import marker_engines
from test_training_wan_native_outputs import codec_request, pinned
from distillschedule.workloads.best_known.registry import repository_root

pytestmark = pytest.mark.skipif(not STACK_AVAILABLE, reason='requires actual original optional Helios stack')


def test_native_gan_current_student_and_real_vae_generate_without_training_payloads(tmp_path, monkeypatch):
    from distillschedule.training import helios_gan_native_inference as output
    from distillschedule.training.inference import read_export
    from distillschedule.data import import_preencoded
    torch.set_num_threads(1)
    request = codec_request(tmp_path / 'old-codec')
    request['schema_version'] = output.EXPORT_SCHEMA
    request['budget'] = dict(max_parameters=2_000_000, max_loaded_bytes=2**31,
        max_primary_export_bytes=2**31, max_transient_tensor_bytes=2**28,
        max_activation_elements=100_000_000, max_attention_elements=512_000_000,
        host_reserve_bytes=2**20, device_reserve_bytes=2**20)
    result, receipt, old = _owned_gan_export(tmp_path, monkeypatch,
        engines_factory=lambda **kw: with_gan_head(marker_engines, **kw), request=request,
        callback_type=output.NativeOutputRequest)
    root, manifest = read_export(tmp_path / 'portable')
    assert manifest['checkpoint_cycle'] == 100 and manifest['model']['factory'] == output.FACTORY
    p = manifest['model']['parameters']
    assert p['native']['args'] == dict(base_seq_len=256, max_seq_len=4096, base_shift=.5, max_shift=1.15)
    assert p['schema_version'] == output.SCHEMA and p['task']['variant'] == 'gan'
    assert p['task']['model_family'] == 'helios14b' and p['task']['stage2_inference_steps'] == [2, 2, 2]
    assert manifest['identity']['training_provenance']['schema_version'] == 'distillschedule.helios_gan_training_identity.v1'
    assert receipt['final_versions'] == dict(generator=102, critic=510, ema=900)
    assert receipt['output_only']['merge_count'] == 0
    weights = torch.load(root / 'weights.pt', weights_only=True)
    assert set(weights) == {'generator', 'codec'}
    assert any('lora_B' in k and bool(torch.count_nonzero(v)) for k, v in weights['generator'].items())
    assert str(old) not in json.dumps(p) and str(tmp_path / 'old-codec') not in json.dumps(p)
    new = tmp_path / 'NEW-input'; new.mkdir()
    torch.save(dict(prompt_embeds=torch.randn(1, 2, 4096, generator=torch.Generator().manual_seed(986)).to(torch.bfloat16)), new / 'prompt.pt')
    (new / 'inputs.json').write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-native-W3-prompt', features=dict(path='prompt.pt', sha256=pinned(new / 'prompt.pt')['sha256']))],
        metadata=dict(evidence_kind=request['evidence_kind']))))
    import_preencoded(new / 'inputs.json', new / 'prepared')
    shutil.rmtree(old); shutil.rmtree(tmp_path / 'old-codec')
    script = r'''
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1])
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  assert not any(str(root/p) in str(args[0]) for p in ('old-training','old-codec')),'read unavailable training/codec'
sys.addaudithook(audit)
def no_optimizer(*a,**kw):raise AssertionError('inference constructed optimizer')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
class NoTraining:
 def find_spec(self,name,path=None,target=None):
  assert name not in ('distillschedule.training.native','distillschedule.training.helios_gan_training',
   'distillschedule.training.helios_worker','distillschedule.training.helios_lifecycle'),name
sys.meta_path.insert(0,NoTraining())
torch.set_num_threads(1)
from distillschedule.training.cli import main
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
assert counts==dict(native=1,pyramid=1,old_cholesky=0,stable_psd=2,forward=6),counts
values=torch.load(root/'video/generation.pt',weights_only=True)
assert values['generated']['latents'].shape == (1,16,9,48,80)
assert values['decoded']['video'].shape == (33,384,640,3)
assert values['decoded']['decoder']['latent_clamp'] is False
assert len(list((root/'video/frames').glob('*.png'))) == 33
assert not torch.cuda.is_initialized()
print('ACTUAL_TINY_W3_CURRENT_G_NATIVE_SAMPLER_VAE_NEW_PROMPT_PASS_NOT_PRETRAINED_GPU_OR_QUALITY')
'''
    env = dict(os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1', GROUP_RANK='0', CUDA_VISIBLE_DEVICES='')
    child = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=repository_root(), env=env,
                           text=True, capture_output=True, timeout=360)
    assert child.returncode == 0, child.stdout + child.stderr
    assert 'ACTUAL_TINY_W3_CURRENT_G_NATIVE_SAMPLER_VAE_NEW_PROMPT_PASS' in child.stdout
    assert not torch.cuda.is_initialized()
