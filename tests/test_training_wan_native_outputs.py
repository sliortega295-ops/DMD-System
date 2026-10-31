"""Native W1 output-only seam; CPU proofs are not pretrained/world16 evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import file_sha256

# Existing test-only transport/CPU scalar owners. Importing fixtures does not
# run any historical training/model/installed suite. Their scope stays explicit.
from test_strategy_wan_bundle_stream import native_cpu_bundle  # noqa: F401
from test_strategy_wan_bundle_worker import cpu_worker, _state, _owners  # noqa: F401
from test_training_end_to_end import equal


def test_public_native_export_enters_owned_export_not_training(tmp_path, monkeypatch, capsys):
    from distillschedule.training import cli, native
    config = dict(schema_version='distillschedule.training.v1', model=dict(workload='W1'),
        algorithm={}, data={}, execution=dict(adapter='native', mode='reference'),
        trainer=dict(cycles=100, output_dir=str(tmp_path/'training')))
    path = tmp_path/'training.json'; path.write_text(json.dumps(config))
    calls = []
    def export(config, checkpoint, output):
        calls.append((config, checkpoint, output))
        return dict(status='EXPORTED', gpu_execution='NOT_RUN_SYNTHETIC_DISPATCH')
    monkeypatch.setattr(native, 'export_native', export, raising=False)
    assert cli.main(['export',str(path),'--checkpoint',str(tmp_path/'cycle-100'),
                     '--output',str(tmp_path/'portable')]) == 0
    assert len(calls) == 1 and calls[0][0].trainer['cycles'] == 100
    assert calls[0][1:] == (tmp_path/'cycle-100',tmp_path/'portable')
    assert json.loads(capsys.readouterr().out)['status'] == 'EXPORTED'


def test_real_wan21_codec_uses_native_std_mean_once_without_latent_clamp():
    from diffusers import AutoencoderKLWan
    from distillschedule.training.wan_native_inference import codec_parameters, decode
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(374)
        vae = AutoencoderKLWan(base_dim=2, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=1)
    vae.requires_grad_(False).eval()
    codec = codec_parameters(vae, evidence_kind='FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY',
                             provenance=dict(status='RANDOM_NOT_PRETRAINED', origin='seed374', license='Apache-2.0'))
    p = dict(codec=codec, geometry=dict(noise_shape=[1,16,2,2,2], video_shape=[5,16,16,3],fps=16))
    latent = torch.linspace(-7,8,128).reshape(1,16,2,2,2)
    before = latent.clone()
    generated = dict(latents=latent,latent_schema=dict(latents=dict(layout='BCTHW',shape=list(latent.shape),dtype='torch.float32')))
    mean = torch.tensor(vae.config.latents_mean).view(1,16,1,1,1)
    std = torch.tensor(vae.config.latents_std).view(1,16,1,1,1)
    with torch.no_grad():
        expected = vae.decode(latent*std+mean,return_dict=False)[0]
        decoded = decode(dict(codec=vae,inference=p),generated)
    expected = ((expected[0].permute(1,2,3,0)+1)/2*255).clamp(0,255).to(torch.uint8)
    assert torch.equal(decoded['video'],expected) and torch.equal(latent,before)
    assert decoded['video'].shape == (5,16,16,3)
    assert decoded['decoder']['normalization'] == 'normalized_latents * config.latents_std + config.latents_mean'
    assert decoded['decoder']['latent_clamp'] is False
    assert decoded['decoder']['fps'] == 16 and not torch.cuda.is_initialized()


def budget():
    return dict(max_parameters=2_000_000,max_loaded_bytes=2**30,
        max_export_bytes_per_rank=2**30,max_export_bytes_all_ranks=16*2**30,
        max_activation_elements=100_000_000,max_attention_elements=8_000_000)


def pinned(path):
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path))


def codec_request(root):
    from diffusers import AutoencoderKLWan
    from safetensors.torch import save_file
    from distillschedule.training.wan_native_inference import EXPORT_SCHEMA
    root.mkdir(parents=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(374)
        codec = AutoencoderKLWan(base_dim=2,z_dim=16,dim_mult=[1,2,4,4],num_res_blocks=1)
    codec.save_config(root)
    weights = root/'codec.safetensors'
    save_file(codec.state_dict(),weights)
    return dict(schema_version=EXPORT_SCHEMA,seed=721,prompt_shape=[1,2,4096],fps=16,budget=budget(),
        evidence_kind='FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY',
        codec=dict(config=pinned(root/'config.json'),weights=[pinned(weights)],
            provenance=dict(status='RANDOM_NOT_PRETRAINED',origin='seed374',license='Apache-2.0')))


def test_pinned_actual_codec_admission_preserves_rng_and_carries_no_old_paths(tmp_path):
    from distillschedule.training.wan_native_inference import NativeOutputRequest, codec_parameters
    from distillschedule.training.checkpoint import rng_state
    request = codec_request(tmp_path/'old-codec')
    before = rng_state()
    output = NativeOutputRequest(request,tmp_path/'portable')
    output.admit()
    assert equal(before,rng_state())
    portable = codec_parameters(output.codec,evidence_kind=request['evidence_kind'],provenance=request['codec']['provenance'])
    assert str(tmp_path) not in json.dumps(portable)
    assert not output.codec.training and all(not p.requires_grad for p in output.codec.parameters())


@pytest.mark.parametrize('fault,match',[
    ('missing','file'),('hash','pin'),('layout','layout'),('dtype','FP32'),('nonfinite','nonfinite'),
    ('residual','codec'),('budget','capacity'),('existing','already exists'),('seed','seed')])
def test_codec_and_output_refusals_are_before_publication(tmp_path,fault,match):
    from safetensors.torch import load_file,save_file
    from distillschedule.training.wan_native_inference import NativeOutputRequest
    from distillschedule.training.checkpoint import rng_state
    request = codec_request(tmp_path/'codec')
    weights = Path(request['codec']['weights'][0]['path'])
    if fault == 'missing':
        weights.unlink()
    elif fault == 'hash':
        with weights.open('ab') as stream:stream.write(b'bad')
    elif fault in ('layout','dtype','nonfinite'):
        state = load_file(str(weights)); name = next(iter(state))
        if fault == 'layout':state.pop(name)
        elif fault == 'dtype':state[name] = state[name].to(torch.bfloat16)
        else:state[name].fill_(float('nan'))
        save_file(state,weights);request['codec']['weights'] = [pinned(weights)]
    elif fault == 'residual':
        config = Path(request['codec']['config']['path']);raw = json.loads(config.read_text())
        raw['z_dim'] = 48;config.write_text(json.dumps(raw));request['codec']['config'] = pinned(config)
    elif fault == 'budget':request['budget']['max_loaded_bytes'] = 64
    elif fault == 'existing':(tmp_path/'portable').mkdir()
    else:request['seed'] = -1
    before = rng_state()
    with pytest.raises((ValueError,OSError),match=match):
        NativeOutputRequest(request,tmp_path/'portable').admit()
    assert equal(before,rng_state())
    assert not (tmp_path/'portable/_SUCCESS').exists() and not (tmp_path/'portable.inference.json').exists()


def export_scalar_worker(h, name, checkpoint, callback):
    from distillschedule.strategies import wan_worker
    h.transport.output = h.root/name
    return wan_worker.run_worker(h.f.task,h.assets,output=h.transport.output,mode='export',strategy=h.f.strategy,
        cycles=3,stream_path=h.root/'stream/stream.json',resume_from=checkpoint,native_output=callback)


@pytest.fixture
def output_cpu_worker(cpu_worker):
    # Reuse the same synthetic assets and declared metadata-timeout ABI.
    return cpu_worker


def test_native_terminal_restore_calls_output_only_and_preserves_every_owner(output_cpu_worker,monkeypatch):
    """Actual native IO/state adapters; CUDA/world16/model kernels synthetic."""
    from distillschedule.strategies import wan_worker,wan_stream
    from distillschedule.training.checkpoint import rng_state
    h = output_cpu_worker;h.run('source')
    expected = _state(h.parts[0]);rng = rng_state();begins = len(h.begins)
    def unexpected(*a,**kw):raise AssertionError('output-only branch executed training/capture/cursor/final-writer')
    monkeypatch.setattr(wan_worker,'execute_cycles',unexpected)
    monkeypatch.setattr(wan_worker,'save_final_state',unexpected)
    monkeypatch.setattr(wan_stream,'restore_stream_cursor',unexpected)
    class Output:
        def __init__(self):self.calls=[]
        def admit(self):self.calls.append('admit')
        def publish(self,session,*,manifest,identity):
            self.calls.append('publish')
            assert manifest['cycle'] == manifest['cursor'] == 3
            assert not hasattr(session,'_stream_binding')
            before = _state(session);owners = _owners(session);before_rng = rng_state()
            assert equal(before,expected) and equal(before_rng,rng)
            assert session.task == h.f.task and session.runtime_cache == {} and not session.finite
            assert equal(before,_state(session)) and _owners(session) == owners and equal(before_rng,rng_state())
            return dict(status='EXPORTED_SYNTHETIC_TEST_ONLY',checkpoint_cycle=3)
    callback = Output()
    result = export_scalar_worker(h,'export-owner',h.root/'source/checkpoints/cycle-000003',callback)
    assert callback.calls == ['admit','publish'] and len(h.begins) == begins
    assert result['training_cycles_executed'] == 0 and result['checkpoint_cycle'] == 3
    assert equal(_state(h.parts[-1]),expected) and equal(rng,rng_state())
    assert not (h.root/'export-owner/checkpoints').exists()
    assert h.transport.events[-2:] == ['destroy_gloo','destroy_nccl']


@pytest.mark.parametrize('fault,match',[
    ('missing','complete'),('shard','hash'),('identity','identity'),('policy','policy'),('codec','codec')])
def test_native_output_refuses_checkpoint_or_codec_without_training(output_cpu_worker,monkeypatch,fault,match):
    from distillschedule.strategies import wan_worker
    h = output_cpu_worker;h.run('source')
    checkpoint = h.root/'source/checkpoints/cycle-000003'
    if fault == 'missing':(checkpoint/'_SUCCESS').unlink()
    elif fault == 'shard':
        with (checkpoint/'rank-00000.pt').open('ab') as stream:stream.write(b'bad')
    elif fault in ('identity','policy'):
        path = checkpoint/'manifest.json';value = json.loads(path.read_text())
        if fault == 'identity':value['identity']['native']['source_head'] = 'f'*40
        else:value['policy']['stream_sha256'] = 'f'*64
        path.write_text(json.dumps(value));(checkpoint/'_SUCCESS').write_text(file_sha256(path)+'\n')
    begins = len(h.begins)
    monkeypatch.setattr(wan_worker,'execute_cycles',lambda *a,**kw:pytest.fail('native output ran training'))
    class Output:
        def admit(self):
            if fault == 'codec':raise ValueError('required real codec missing')
        def publish(self,*a,**kw):pytest.fail('bad output was published')
    with pytest.raises(ValueError,match=match):export_scalar_worker(h,'rejected',checkpoint,Output())
    assert len(h.begins) == begins and not h.transport.initialized
    assert not (h.root/'rejected/checkpoints').exists()
    assert json.loads((h.root/'rejected/rank_00.json').read_text())['status'] == 'FAILED_ORIGINAL_WAN_WORK'


def test_native_gamma_one_third_noise_is_singular_psd_not_cholesky_pd(monkeypatch):
    """Execute unchanged native sampler body; no optional model imports needed."""
    import ast,math,os
    from types import SimpleNamespace as NS
    from distillschedule.workloads.best_known.registry import repository_root
    path = repository_root()/'third_party/helios_compat/source/helios/utils/utils_helios_post.py'
    node = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name == 'sample_block_noise')
    namespace = dict(torch=torch,math=math,os=os)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    monkeypatch.delenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',raising=False)
    covariance = torch.eye(4)*(1+1/3)-torch.ones(4,4)/3
    assert torch.linalg.matrix_rank(covariance) == 3
    assert float(torch.linalg.eigvalsh(covariance).min()) >= -2e-7
    factor = torch.linalg.cholesky_ex(covariance)
    if int(factor.info) == 0:
        # Torch2.14 CPU rounds the zero eigenvalue slightly positive; Torch2.10
        # target r1 rejects it. Do not change gamma or fabricate a local error.
        assert float(factor.L[-1,-1]) < 1e-3
        sampled = namespace['sample_block_noise'](NS(config=NS(gamma=1/3)),1,16,3,16,16)
        assert sampled.shape == (1,16,3,16,16) and bool(sampled.isfinite().all())
    else:
        with pytest.raises((ValueError,RuntimeError),match=r'(?i)positive[\s_-]*definite'):
            namespace['sample_block_noise'](NS(config=NS(gamma=1/3)),1,16,3,16,16)


def noise_parameters(seed=721):
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.training import wan_native_inference as output
    task = OriginalWanTask(video_height=128,video_width=128,video_frames=9,first_step=0,lora_rank=1,stage2_inference_steps=(1,2,3))
    return dict(seed=seed,task=task.to_dict(),geometry=dict(noise_shape=list(task.latent_shape),prompt_shape=[1,2,4096]),
        noise_protocol=output.noise_protocol(),native={**output.NATIVE,'rollout':output._rollout(task)})


def test_fresh_inference_noise_exact_psd_law_substreams_controls_and_rng():
    from distillschedule.training import wan_native_inference as output
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.strategies.wan_capture import input_stream
    from runtime.official_baseline import stable_block_noise
    p = noise_parameters();before = rng_state()
    trace = output.fresh_noise_inputs(p)
    assert equal(before,rng_state())
    prefix = output.NOISE_PREFIX;schema = output.NOISE_SCHEMA
    values = {}
    for name in ('entry_noise','rollout_noise','block_noise/1','block_noise/2'):
        values[name] = trace.tensor(prefix+'/'+name)
        with input_stream(p['seed'],schema,name):
            if name.endswith('noise'):
                expected = torch.randn(p['geometry']['noise_shape'],dtype=torch.bfloat16)
            else:
                h,w = ([8,8] if name.endswith('/1') else [16,16])
                expected = stable_block_noise(16*3*(h//2)*(w//2)).view(1,16,3,h//2,w//2,2,2).permute(
                    0,1,2,3,5,4,6).reshape(1,16,3,h,w)
        assert torch.equal(values[name],expected)
        assert values[name].device.type == 'cpu' and bool(values[name].isfinite().all())
    for name in ('block_noise/1','block_noise/2'):
        value = values[name];h,w = value.shape[-2:]
        blocks = value.reshape(1,16,3,h//2,2,w//2,2).permute(0,1,2,3,5,4,6).reshape(-1,4)
        assert value.dtype == torch.float32 and torch.allclose(blocks.sum(1),torch.zeros(len(blocks)),atol=1e-6,rtol=0)
    controls = trace.operation(prefix+'/rollout')['control']['pyramid_stage_call_sequence']
    assert [v['stage'] for v in controls] == [0,1,2]
    assert [v['denoising_exit'] for v in controls] == [0,1,2]
    assert [v['call_count'] for v in controls] == [1,2,3]
    trace.finish()
    repeated = output.fresh_noise_inputs(p)
    changed = output.fresh_noise_inputs(noise_parameters(722))
    assert all(torch.equal(repeated.tensor(prefix+'/'+k),v) for k,v in values.items())
    assert all(not torch.equal(changed.tensor(prefix+'/'+k),v) for k,v in values.items())
    assert equal(before,rng_state())


def test_fresh_noise_protocol_refuses_history_bad_contract_and_unconsumed_inputs():
    from distillschedule.training import wan_native_inference as output
    p = noise_parameters()
    for fault in ('protocol','shape','seed'):
        bad = deepcopy(p)
        if fault == 'protocol':bad['noise_protocol']['schema_version'] = 'OLD_CHOLESKY'
        elif fault == 'shape':bad['geometry']['noise_shape'][-1] += 1
        else:bad['seed'] = -1
        with pytest.raises(ValueError):output.fresh_noise_inputs(bad)
    trace = output.fresh_noise_inputs(p)
    with pytest.raises(ValueError,match='input'):trace.tensor('step/0/generator/entry_noise')
    with pytest.raises(ValueError,match='operation'):trace.operation('training/rollout')
    with pytest.raises(ValueError,match='consum'):trace.finish()


def test_generate_uses_explicit_fresh_operation_noise_not_default_cholesky(monkeypatch):
    """Routing-only stand-in forward; actual upstream chain is a separate gate."""
    from types import SimpleNamespace as NS
    from distillschedule.training import wan_native_inference as output
    from distillschedule.strategies import wan_capture
    from distillschedule.training.checkpoint import rng_state
    p = noise_parameters();calls=[]
    g = torch.nn.Linear(1,1).requires_grad_(False).eval();g.gradient_checkpointing = False
    def run(**kwargs):
        trace = kwargs['operation_trace'];prefix = kwargs['trace_prefix']
        assert prefix == output.NOISE_PREFIX
        for name in ('entry_noise','rollout_noise','block_noise/1','block_noise/2'):trace.tensor(prefix+'/'+name)
        control = trace.operation(prefix+'/rollout')['control']['pyramid_stage_call_sequence']
        assert [v['denoising_exit'] for v in control] == [0,1,2]
        assert kwargs['last_step_only'] is True and kwargs['is_use_gt_history'] is False
        calls.append(True)
        return (torch.zeros(p['geometry']['noise_shape'],dtype=torch.float32),)
    monkeypatch.setattr(wan_capture,'native_primitives',lambda **kw:NS(post=NS(run_generator=run),scheduler=lambda **kw:NS()))
    before = rng_state()
    values = output.generate(dict(generator=output.CurrentGenerator(g),inference=p),
        dict(prompt_embeds=torch.zeros(1,2,4096,dtype=torch.bfloat16)),{})
    assert calls == [True] and equal(before,rng_state())
    assert values['latents'].dtype == torch.float32 and not torch.cuda.is_initialized()


@pytest.mark.parametrize('forward_failure',[False,True],ids=['success','forward_failure'])
def test_generate_scopes_indexed_device_rng_transaction(monkeypatch,forward_failure):
    """CPU CUDA-placement state machine, NOT actual multi-GPU qualification.

    Keep the real checkpoint RNG snapshot/validation/restoration and fresh
    noise sampler. Only device placement, CUDA RNG storage and forward are
    synthetic, so a saved/current-device mismatch must fail in real validation.
    """
    from contextlib import contextmanager
    import random
    from types import SimpleNamespace as NS
    import numpy as np
    from distillschedule.training import wan_native_inference as output
    from distillschedule.strategies import wan_capture
    from distillschedule.training.checkpoint import rng_state,restore_rng
    p = noise_parameters();current = [0];events = []
    cuda_rng = {0:torch.arange(8,dtype=torch.uint8),1:torch.arange(8,16,dtype=torch.uint8)}
    saved_cuda = {device:value.clone() for device,value in cuda_rng.items()}
    before = rng_state(device='cpu')
    assert 'numpy' in before
    @contextmanager
    def select(device):
        previous = current[0];current[0] = torch.device(device).index
        events.append(('enter',current[0]))
        try:
            yield
        finally:
            current[0] = previous;events.append(('exit',previous))
    def read(device):
        assert device == 1, 'must not query the nonowned CUDA RNG'
        events.append(('read',device));return cuda_rng[device].clone()
    def seed(value):
        events.append(('seed',current[0],value));cuda_rng[current[0]].fill_(value % 256)
    def restore(value,device):
        assert current[0] == device
        events.append(('restore',device));cuda_rng[device] = value.clone()
    def unrelated(*args,**kwargs):
        pytest.fail('must not enumerate or query all visible CUDA devices')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:current[0])
    monkeypatch.setattr(torch.cuda,'device',select)
    monkeypatch.setattr(torch.cuda,'get_rng_state',read)
    monkeypatch.setattr(torch.cuda,'manual_seed',seed)
    monkeypatch.setattr(torch.cuda,'set_rng_state',restore)
    for name in ('get_rng_state_all','set_rng_state_all','manual_seed_all','device_count'):
        monkeypatch.setattr(torch.cuda,name,unrelated)
    g = NS(training=False,gradient_checkpointing=False,
        parameters=lambda:iter([NS(device=torch.device('cuda:1'),requires_grad=False)]),modules=lambda:[])
    prompt = torch.zeros(1,2,4096,dtype=torch.bfloat16)
    def place(device):
        assert device == torch.device('cuda:1')
        return prompt  # CPU-only placement stand-in, never a GPU tensor.
    monkeypatch.setattr(prompt,'to',place)
    def run(**kwargs):
        assert kwargs['accelerator'].device == torch.device('cuda:1')
        events.append(('forward',current[0]))
        # Exercise restoration after actual Python/CPU/NumPy draws and a
        # synthetic draw on the current CUDA generator, even on failure.
        random.random();torch.rand(3);np.random.random(3)
        cuda_rng[current[0]].add_(1)
        trace = kwargs['operation_trace'];prefix = kwargs['trace_prefix']
        for name in ('entry_noise','rollout_noise','block_noise/1','block_noise/2'):
            trace.tensor(prefix+'/'+name)
        trace.operation(prefix+'/rollout')
        if forward_failure:
            raise RuntimeError('synthetic native forward failed')
        return (torch.zeros(p['geometry']['noise_shape'],dtype=torch.float32),)
    monkeypatch.setattr(wan_capture,'native_primitives',lambda **kw:NS(post=NS(run_generator=run),scheduler=lambda **kw:NS()))
    try:
        state = dict(generator=NS(model=g),inference=p)
        if forward_failure:
            with pytest.raises(RuntimeError,match='synthetic native forward failed'):
                output.generate(state,dict(prompt_embeds=prompt),{})
        else:
            assert output.generate(state,dict(prompt_embeds=prompt),{})['latents'].dtype == torch.float32
        assert equal(before,rng_state(device='cpu')) and current == [0]
        assert all(torch.equal(cuda_rng[device],saved_cuda[device]) for device in (0,1))
        assert [e for e in events if e[0] == 'seed'] == [('seed',1,p['seed'])]
        assert [e for e in events if e[0] == 'forward'] == [('forward',1)]
        assert events[0] == ('enter',1) and events[-1] == ('exit',0)
        assert ('restore',1) in events
    finally:
        restore_rng(before)  # Isolate the intentional RED from other CPU tests.


def test_noise_metadata_matches_independent_big_endian_seed_and_draws():
    """Independent SHA/RNG/projection oracle, not input_stream vs itself."""
    import hashlib,math
    from distillschedule.training import wan_native_inference as output
    p = noise_parameters()
    assert 'first8 big-endian' in p['noise_protocol']['seed_derivation']
    trace = output.fresh_noise_inputs(p)
    for name in ('entry_noise','rollout_noise','block_noise/1','block_noise/2'):
        encoded = json.dumps([p['seed'],output.NOISE_SCHEMA,name],separators=(',',':')).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8],'big') % (2**63)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        if name.endswith('noise'):
            expected = torch.randn(p['geometry']['noise_shape'],generator=generator,dtype=torch.bfloat16)
        else:
            h,w = ((8,8) if name.endswith('/1') else (16,16))
            raw = torch.randn(16*3*(h//2)*(w//2),4,generator=generator,dtype=torch.float32)
            projected = (raw-raw.mean(dim=-1,keepdim=True))*math.sqrt(4/3)
            expected = projected.view(1,16,3,h//2,w//2,2,2).permute(0,1,2,3,5,4,6).reshape(1,16,3,h,w)
        assert torch.equal(trace.tensor(output.NOISE_PREFIX+'/'+name),expected)


def test_real_program_generation_binds_codec_callback_to_factory_state(monkeypatch):
    """Real Program callback + real VAE; only optional G construction/forward injected."""
    from types import SimpleNamespace as NS
    from diffusers import AutoencoderKLWan
    from distillschedule.training import wan_native_inference as output
    from distillschedule.training.session import _generate_values
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(374)
        vae = AutoencoderKLWan(base_dim=2,z_dim=16,dim_mult=[1,2,4,4],num_res_blocks=1)
    codec = output.codec_parameters(vae,evidence_kind='FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY',
        provenance=dict(status='RANDOM_NOT_PRETRAINED',origin='seed374',license='Apache-2.0'))
    p = dict(codec=codec,evidence_kind=codec['evidence_kind'],
        geometry=dict(noise_shape=[1,16,2,2,2],video_shape=[5,16,16,3],fps=16))
    g = torch.nn.Linear(1,1)
    g.disable_gradient_checkpointing = lambda:None
    monkeypatch.setattr(output,'_admit',lambda parameters:(deepcopy(p),None,None,None))
    monkeypatch.setattr(output,'_generator',lambda parameters,meta:g)
    components = output.build_inference_model(dict(device='cpu'))
    latents = torch.linspace(-7,8,128).reshape(1,16,2,2,2)
    generated = dict(latents=latents,latent_schema=dict(latents=dict(layout='BCTHW',shape=list(latents.shape),dtype='torch.float32')))
    components.generate = lambda state,feeds,parameters:generated
    result = _generate_values(NS(trainer=dict(device='cpu',seed=721)),components,{},dict(),video_required=True,output_kind='video')
    expected = output.decode(components.state,generated)
    assert torch.equal(result['decoded']['video'],expected['video'])
    assert result['decoded']['video'].shape == (5,16,16,3)
    assert result['decoded']['decoder']['latent_clamp'] is False
    with pytest.raises(ValueError,match='overrides'):
        components.decode(generated,dict(fps=999))


def _original_block_noise_function():
    """Exact source function, same lightweight ownership as the default canary."""
    import ast,math,os
    from distillschedule.workloads.best_known.registry import repository_root
    path=repository_root()/'third_party/helios_compat/source/helios/utils/utils_helios_post.py'
    node=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='sample_block_noise')
    namespace=dict(torch=torch,math=math,os=os)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['sample_block_noise']


def test_explicit_stable_block_noise_policy_matches_primitive_and_rng(monkeypatch):
    from types import SimpleNamespace as NS
    import runtime.official_baseline as official
    sampler=_original_block_noise_function();primitive=official.stable_block_noise;calls=[]
    def observed(blocks,**kwargs):calls.append((blocks,kwargs));return primitive(blocks,**kwargs)
    monkeypatch.setattr(official,'stable_block_noise',observed)
    monkeypatch.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE','1')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(819);expected=primitive(2*3*2*3)
        expected=expected.view(1,2,3,2,3,2,2).permute(0,1,2,3,5,4,6).reshape(1,2,3,4,6)
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(819);actual=sampler(NS(config=NS(gamma=1/3)),1,2,3,4,6)
        assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),expected_rng)
        assert actual.dtype==torch.float32 and actual.device.type=='cpu'
    assert calls==[(36,{})] and not torch.cuda.is_initialized()


@pytest.mark.parametrize('gamma',[0.,0.25,0.5])
def test_stable_block_noise_policy_rejects_other_gamma_before_rng(monkeypatch,gamma):
    from types import SimpleNamespace as NS
    monkeypatch.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE','1');before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='requires gamma=1/3'):
        _original_block_noise_function()(NS(config=NS(gamma=gamma)),1,2,3,4,6)
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('flag',['unset','0','true'])
def test_default_block_noise_still_uses_original_distribution(monkeypatch,flag):
    from types import SimpleNamespace as NS
    import runtime.official_baseline as official
    if flag=='unset':monkeypatch.delenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',raising=False)
    else:monkeypatch.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',flag)
    def forbidden(*args,**kwargs):pytest.fail('default sampler selected stable opt-in without exact flag=1')
    monkeypatch.setattr(official,'stable_block_noise',forbidden)
    original=torch.distributions.MultivariateNormal;calls=[]
    def observed(loc,**kwargs):calls.append((loc,kwargs['covariance_matrix']));return original(loc,**kwargs)
    monkeypatch.setattr(torch.distributions,'MultivariateNormal',observed)
    # Non-singular gamma isolates routing. The preexisting gamma=1/3 canary
    # separately retains the unmodified platform-dependent Cholesky behavior.
    gamma=.25
    with torch.random.fork_rng(devices=[]):
        value=_original_block_noise_function()(NS(config=NS(gamma=gamma)),1,2,3,4,6)
    assert len(calls)==1 and torch.equal(calls[0][0],torch.zeros(4))
    assert torch.equal(calls[0][1],torch.eye(4)*(1+gamma)-torch.ones(4,4)*gamma)
    assert value.shape==(1,2,3,4,6) and value.dtype==torch.float32 and not torch.cuda.is_initialized()
