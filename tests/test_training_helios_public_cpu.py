"""Public full original tiny CPU mechanism, NOT pretrained or media quality.

No surrogate losses or model/head extraction. Nontrivial tests use explicit
TEST-ONLY nonzero saved bases; production strictly loads their pinned bytes.
The unchanged original constructor's cold-zero output is tested separately.
"""
from copy import deepcopy
import json
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_helios_step_owner_cpu import _config, equal
from distillschedule.training.checkpoint import rng_state, pack_state, read_checkpoint
from distillschedule.training.config import TrainingConfig
from distillschedule.training import session
from distillschedule.workloads.best_known.registry import file_sha256, repository_root

FIXTURE = 'FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY'


def _optional_stack():
    try:
        return (all(importlib.util.find_spec(name) is not None for name in ('diffusers','peft','omegaconf','safetensors'))
                and importlib.metadata.version('diffusers')=='0.37.1'
                and importlib.metadata.version('peft')=='0.17.1')
    except importlib.metadata.PackageNotFoundError:
        return False


STACK_AVAILABLE = _optional_stack()
if os.environ.get('HELIOS_PUBLIC_REQUIRE_STACK')=='1' and not STACK_AVAILABLE:
    raise RuntimeError('required full Helios CPU stack missing: Diffusers0.37.1/PEFT0.17.1')
pytestmark = pytest.mark.skipif(not STACK_AVAILABLE,
    reason='full original Helios CPU proof requires separate Diffusers0.37.1/PEFT0.17.1 optional stack')


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return Path(path)


def fixture_files(root, *, separate=True, capacity=20, gan_start=3, ema_start=2, cold_zero=False):
    """Write pinned ORIGINAL initialized full-class bases, native-shaped sources."""
    from safetensors.torch import save_file
    from omegaconf import OmegaConf
    from distillschedule.adapters.helios.reference_model import load_reference_model_class, reference_model_identity
    from distillschedule.training.helios_public_assets import runtime_identity
    from distillschedule.data.helios_native_inputs import producer_identity
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    args = _config()
    args.training_config.history_sizes = [16, 2, 1]
    args.training_config.max_train_steps = capacity
    args.training_config.is_separate_gan_grad = separate
    args.training_config.gan_start_step = gan_start
    args.training_config.ema_start_step = ema_start
    args_path = write_json(root/'args.json', OmegaConf.to_container(args, resolve=True))
    cls = load_reference_model_class(backend='torch-sdpa')
    roles = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        for role in ('generator', 'critic'):
            model = cls(patch_size=(1,2,2), has_multi_term_memory_patch=True,
                num_attention_heads=2, attention_head_dim=16, in_channels=16, out_channels=16,
                text_dim=4, freq_dim=16, ffn_dim=64, num_layers=1, rope_dim=(4,6,6),
                is_use_gan=role=='critic', is_use_gan_final=role=='critic',
                is_use_gan_hooks=False, gan_cond_map_dim=32)
            # TEST-ONLY saved local nonzero frozen checkpoint. Production never
            # edits weights; it strictly loads these pinned, NOT_PRETRAINED bytes.
            if not cold_zero:
                with torch.no_grad():
                    model.proj_out.weight.normal_(mean=0.,std=.02)
            model.save_config(root/role)
            weights = root/role/'model.safetensors'
            save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()}, weights,
                      metadata={'evidence_kind': FIXTURE})
            roles[role] = dict(config=pin(root/role/'config.json'), weights=[pin(weights)])
    negative = root/'negative.safetensors'
    save_file({'negative_prompt_embeds':torch.zeros(1,2,4)}, negative, metadata={'evidence_kind':FIXTURE})
    geometry = dict(noise_shape=[1,16,9,16,16], prompt_shape=[1,2,4])
    assets = dict(schema_version='distillschedule.helios_cpu_reference_assets.v1',
        evidence_kind=FIXTURE, source=reference_model_identity(backend='torch-sdpa'),
        runtime=runtime_identity(), args=pin(args_path), negative=pin(negative),
        geometry=geometry, budget=dict(max_parameters=600000, max_loaded_bytes=10000000), **roles)
    asset_path = write_json(root/'assets.json', assets)
    records = []
    for index in range(capacity):
        source = root/f'feature-{index:03d}.pt'
        generator = torch.Generator().manual_seed(811+index)
        torch.save(dict(vae_latent=torch.randn(3,16,9,16,16,generator=generator),
            prompt_embed=torch.randn(2,4,generator=generator), prompt_raw=f'fixture {index}',
            evidence_kind=FIXTURE), source)
        records.append(dict(id=f'fixture-{index}', features=pin(source), seed=811, epoch=0, native_index=index))
    sources = dict(schema_version='distillschedule.helios_native_inputs.v1', evidence_kind=FIXTURE,
        producer=producer_identity(), precision='CPU_FP32_FROM_NATIVE', args=pin(args_path),
        geometry=geometry, records=records)
    source_path = write_json(root/'inputs.json', sources)
    return asset_path, source_path


def prepared_config(root, *, separate=True, capacity=20, cycles=4, gan_start=3, ema_start=2):
    from distillschedule.data.helios_native_inputs import prepare_inputs
    assets, sources = fixture_files(root/'local', separate=separate, capacity=capacity,
                                    gan_start=gan_start, ema_start=ema_start)
    prepared = prepare_inputs(sources, root/'prepared')
    value = dict(schema_version='distillschedule.training.v1',
        model=dict(factory='distillschedule.training.helios_public:build_model',
                   parameters=dict(asset_manifest=str(assets))),
        algorithm=dict(factory='distillschedule.training.helios_public:build_algorithm'),
        data=dict(prepared=str(prepared), sampling=dict(kind='finite',shuffle=False)),
        execution=dict(adapter='program',mode='reference'),
        trainer=dict(cycles=cycles, seed=773, device='cpu', output_dir=str(root/'train'),checkpoint_every=1))
    return TrainingConfig.from_dict(value, source=root/'training.json')


def run_capturing(config, monkeypatch, *, observe=True):
    owned, receipts = [], []
    original = session.construct
    observations = dict(helper_calls=[],critic_rollouts=[],parameter_ids=[],outer_rng=None,gradient_hooks=[])
    critic_active = False
    from distillschedule.training.policy import CyclePolicy
    original_run = CyclePolicy.run
    def capture_cycle(policy, *args, **kwargs):
        result = original_run(policy, *args, **kwargs)
        receipts.extend(deepcopy(result[1][f'helios.native_step.{i}']) for i in range(5))
        observations['parameter_ids'].append(owned[0]._live_ownership())
        return result
    if not observe:
        monkeypatch.setattr(CyclePolicy,'run',capture_cycle)
    def construct(*args, **kwargs):
        result = original(*args, **kwargs)
        owner = result[0].state['native_step']
        owned.append(owner)
        observations['outer_rng'] = rng_state()
        observations['parameter_ids'].append(owner._live_ownership())
        owner.generator.module.register_forward_pre_hook(
            lambda *a: observations['critic_rollouts'].append(owner.global_step) if critic_active else None)
        for engine in (owner.generator,owner.critic):
            next(iter(engine.optimizer.param_groups[0]['params'])).register_hook(
                lambda grad: observations['gradient_hooks'].append(True))
        assert owner.versions==dict(generator=0,critic=0,ema=0)
        assert not owner.generator.optimizer.state and not owner.critic.optimizer.state
        return result
    monkeypatch.setattr(session, 'construct', construct)
    # Profile actual module-level Program binding returns, not a mocked loss.
    def profile(frame,event,arg):
        nonlocal critic_active
        if not owned:
            return
        owner = owned[0]
        if frame.f_code is owner.native.post._critic_loss.__code__:
            if event=='call':
                critic_active = True
                observations['helper_calls'].append(('critic',frame.f_locals['global_step']))
                assert frame.f_locals['precomputed_generated_image_or_video'] is None
            elif event=='return':
                critic_active = False
        if event=='call' and frame.f_code is owner.native.post._generator_loss.__code__:
            observations['helper_calls'].append(('generator',frame.f_locals['global_step']))
        if event=='return' and frame.f_code.co_name=='native_step' and isinstance(arg,dict):
            receipts.append(deepcopy(arg))
            observations['parameter_ids'].append(owner._live_ownership())
    try:
        if observe:
            sys.setprofile(profile)
        result = session.train(config)
    finally:
        sys.setprofile(None)
        monkeypatch.setattr(session,'construct',original)
        monkeypatch.setattr(CyclePolicy,'run',original_run)
    owner = owned[0]
    owner._test_observations = observations
    return result, owner, receipts


@pytest.mark.parametrize('separate',[True,False])
def test_public_program_executes_g1_c5_with_real_original_owner(tmp_path, monkeypatch, separate):
    config = prepared_config(tmp_path, separate=separate, capacity=10, cycles=2)
    result, owner, receipts = run_capturing(config, monkeypatch)
    assert result['status']=='COMPLETE' and result['cursor']==2
    assert result['final_versions']==dict(native_step=10)
    assert len(receipts)==10 and [r['global_step'] for r in receipts]==list(range(10))
    assert [r['train_generator'] for r in receipts]==[i%5==0 for i in range(10)]
    assert [r['use_gan'] for r in receipts]==[i>=3 for i in range(10)]
    assert owner.versions==dict(generator=2,critic=10,ema=3)
    metrics = [json.loads(line)['metrics'] for line in
        (Path(config.trainer['output_dir'])/'metrics.jsonl').read_text().splitlines()]
    for cycle, row in enumerate(metrics):
        actual = receipts[5*cycle:5*(cycle+1)]
        assert row['generator/loss'] == actual[0]['generator_loss']
        assert [row[f'critic/loss_{i+1}'] for i in range(5)] == [r['critic_loss'] for r in actual]
    assert owner.inputs is None and owner.global_step==10
    assert set(owner.state_dict())=={'schema','identity','global_step','versions','rng','adapters','native','consumed'}
    assert any(torch.count_nonzero(v['exp_avg']) for v in owner.critic.optimizer.state.values())
    assert any(torch.count_nonzero(v['exp_avg']) for v in owner.generator.optimizer.state.values())
    observed = owner._test_observations
    assert observed['helper_calls']==[(role,t) for t in range(10)
        for role in (('generator','critic') if t%5==0 else ('critic',))]
    assert all(observed['critic_rollouts'].count(t)>=3 for t in range(10))
    assert len(set(map(str,observed['parameter_ids'])))==1 and observed['gradient_hooks']
    assert equal(observed['outer_rng'],rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('separate',[True,False])
def test_public_fresh_os_process_four_vs_three_resume_exact(tmp_path, monkeypatch, separate):
    config = prepared_config(tmp_path,separate=separate)
    full,owner,receipts = run_capturing(config,monkeypatch,observe=False)
    uninterrupted = read_checkpoint(full['checkpoint'],identity=full['identity'])
    split_value = config.to_dict()
    split_value['trainer'].update(cycles=3,output_dir=str(tmp_path/'split'))
    split = TrainingConfig.from_dict(split_value,source=config.source)
    partial = session.train(split)
    full_prefix = read_checkpoint(Path(full['checkpoint']).parent/'cycle-000003',identity=full['identity'])
    split_prefix = read_checkpoint(partial['checkpoint'],identity=partial['identity'])
    assert equal(full_prefix['state'],split_prefix['state']), 'freshly reconstructed source base/buffers diverged before restart'
    assert equal(full_prefix['rng'],split_prefix['rng'])
    expected = dict(state=uninterrupted['state'],rng=uninterrupted['rng'],receipts=receipts[-5:])
    torch.save(expected,tmp_path/'expected.pt')
    resumed_value = config.to_dict()
    resumed_value['trainer'].update(output_dir=str(tmp_path/'resumed'),resume_from=partial['checkpoint'],checkpoint_every=4)
    write_json(tmp_path/'resume.json',resumed_value)
    script = r"""
from pathlib import Path
import sys, torch, pytest
from test_training_helios_public_cpu import run_capturing, equal
from distillschedule.training.config import TrainingConfig
from distillschedule.training.checkpoint import read_checkpoint, rng_state
root = Path(sys.argv[1])
from distillschedule.training.config import load_training_config
config = load_training_config(root/'resume.json')
with pytest.MonkeyPatch.context() as patch:
    result,owner,receipts = run_capturing(config,patch,observe=False)
expected = torch.load(root/'expected.pt',weights_only=True)
payload = read_checkpoint(result['checkpoint'],identity=result['identity'])
assert equal(expected['state'],payload['state']), 'weights/Adam/LR/noise schedulers/EMA/flags/native RNG drift'
assert equal(expected['rng'],payload['rng']) and equal(payload['rng'],rng_state()), 'outer RNG drift'
assert equal(expected['receipts'],receipts), 'next five actual native loss/input/RNG receipts drift'
assert result['resume_cycle']==3 and result['cursor']==4 and result['final_versions']=={'native_step':20}
assert owner.global_step==20 and owner.versions==dict(generator=4,critic=20,ema=5)
assert len(set(map(str,owner._test_observations['parameter_ids'])))==1
assert not torch.cuda.is_initialized()
print('PUBLIC_FULL_ORIGINAL_TINY_CPU_FRESH_PROCESS_EXACT_4_VS_3_PLUS_RESUME_PASS')
"""
    process = subprocess.run([sys.executable,'-c',script,str(tmp_path)],env=dict(os.environ),
        cwd=repository_root(),text=True,capture_output=True,timeout=240)
    assert process.returncode==0, process.stdout+process.stderr
    assert 'FRESH_PROCESS_EXACT' in process.stdout
    evidence = os.environ.get('HELIOS_PUBLIC_U1_EVIDENCE')
    if evidence:
        from distillschedule.training.helios_step_owner import _content
        from distillschedule.workloads.best_known.registry import fingerprint
        write_json(Path(evidence)/f'fresh-process-separate-{str(separate).lower()}.json',
            dict(status='PASS_FULL_ORIGINAL_TINY_CPU_NOT_PRETRAINED_OR_QUALITY',separate=separate,
                 torch=str(torch.__version__),actual_source=owner.identity['source'],runtime=owner.identity['runtime'],
                 final_state_sha256=fingerprint(_content(expected['state'])),
                 actual_g_calls=4,actual_full_c_calls=20,final_cursor=4,program_versions=dict(native_step=20),
                 native_versions=owner.versions,receipts=receipts,stdout=process.stdout,
                 checkpoint_manifest_sha256=file_sha256(Path(full['checkpoint'])/'manifest.json'),
                 fresh_process_exact=True,quality_validation='NOT_RUN',gpu='NOT_RUN',claimable_speedup=False))


def test_unchanged_original_cold_zero_base_gradient_is_allowed(tmp_path, monkeypatch):
    from distillschedule.data.helios_native_inputs import prepare_inputs
    from distillschedule.training.helios_public import build_model,build_algorithm
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.data import PreparedDataset
    assets,sources = fixture_files(tmp_path/'cold',capacity=5,cold_zero=True)
    dataset = PreparedDataset(prepare_inputs(sources,tmp_path/'prepared'))
    torch.manual_seed(773)
    components = build_model(dict(asset_manifest=str(assets)))
    owner = components.state['native_step']
    workload = build_algorithm(components,dataset.sample(0),{},0,123)
    backend = ExecutionBackend(workload)
    runtime = backend.session(backend.inventory(1)['candidates'][0],copy_state=False,persistent=True)
    outputs = runtime.run()
    assert outputs['helios.native_step.0']['generator_loss']==0.
    assert owner.versions==dict(generator=1,critic=5,ema=2)
    assert all(torch.count_nonzero(v['exp_avg'])==0 for v in owner.generator.optimizer.state.values())


@pytest.mark.parametrize('change',['auto','manual','plan','epochs','dtype','world','validation','algorithm'])
def test_public_unsupported_request_refuses_before_output(tmp_path,monkeypatch,change):
    from dataclasses import replace
    config = prepared_config(tmp_path,capacity=5,cycles=1)
    if change in ('auto','manual','plan'):
        config = replace(config,execution={'mode':change})
    elif change=='epochs':
        config = replace(config,data={**config.data,'sampling':{'kind':'epochs','shuffle':False}})
    elif change=='dtype':
        config = replace(config,model={**config.model,'parameters':{**config.model['parameters'],'precision':'bf16'}})
    elif change=='world':
        monkeypatch.setenv('WORLD_SIZE','2')
    elif change=='validation':
        config = replace(config,trainer={**config.trainer,'validation':{'every_cycles':1,'samples':1}})
    elif change=='algorithm':
        config = replace(config,algorithm={**config.algorithm,'parameters':{'shortcut':True}})
    with pytest.raises((ValueError,RuntimeError)):
        session.train(config)
    assert not Path(config.trainer['output_dir']).exists()


@pytest.mark.parametrize('change',['reorder','id','nonfinite','dtype','shape','bytes'])
def test_all_five_rows_validate_before_first_native_update(tmp_path,change):
    from distillschedule.training.helios_public import build_model,build_algorithm
    from distillschedule.data import PreparedDataset
    config = prepared_config(tmp_path,capacity=5,cycles=1)
    components = build_model(config.model['parameters'])
    owner = components.state['native_step']
    dataset = PreparedDataset(config.data['prepared'])
    feeds = dataset.sample(0)
    row = feeds['native_rows'][-1]
    if change=='reorder': row['absolute_step'].fill_(1)
    elif change=='id': row['identity'].zero_()
    elif change=='nonfinite': row['inputs']['gan_vae_latents'].fill_(float('nan'))
    elif change=='dtype': row['inputs']['gan_vae_latents']=row['inputs']['gan_vae_latents'].double()
    elif change=='shape': row['inputs']['gan_vae_latents']=row['inputs']['gan_vae_latents'][:,:,:1]
    elif change=='bytes': row['inputs']['gan_vae_latents'].add_(1)
    before = pack_state(components.state);outer=rng_state()
    with pytest.raises(ValueError): build_algorithm(components,feeds,{},0,123)
    assert equal(before,pack_state(components.state)) and equal(outer,rng_state())
    assert owner.global_step==0 and not owner._poisoned


@pytest.fixture(scope='module')
def public_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp('public-original-checkpoint')
    config = prepared_config(root,capacity=10,cycles=1)
    result = session.train(config)
    return config,result


@pytest.mark.parametrize('change',['inner_step','inner_versions','consumed_count','consumed_chain',
                                  'outer_cycle','outer_cursor','outer_versions','topology',
                                  'generator_buffer_nan','generator_buffer_finite',
                                  'critic_buffer_nan','critic_buffer_finite'])
def test_rehashed_common_checkpoint_rejects_before_live_mutation(public_checkpoint,tmp_path,change):
    import shutil
    from distillschedule.training.checkpoint import restore_checkpoint
    from distillschedule.training.tensor_state import pack,decode
    config,result = public_checkpoint
    checkpoint = tmp_path/'tampered'
    shutil.copytree(result['checkpoint'],checkpoint)
    payload = torch.load(checkpoint/'state.pt',weights_only=True)
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    value = decode(payload['state']['native_step']['value'],{})
    if change=='inner_step': value['global_step']=10
    elif change=='inner_versions': value['versions']['generator']=2
    elif change=='consumed_count': value['consumed'].pop()
    elif change=='consumed_chain': value['consumed'][-1]['chain']='0'*64
    elif change=='outer_cycle': payload['cycle']=manifest['cycle']=2;payload['cursor']=manifest['cursor']=2
    elif change=='outer_cursor': payload['cursor']=manifest['cursor']=5
    elif change=='outer_versions': payload['versions']=manifest['versions']=dict(native_step=10)
    elif change=='topology': manifest['topology']={'world_size':2}
    elif '_buffer_' in change:
        role,corruption=change.split('_buffer_')
        buffer=value['native'][role+'.weights']['buffers']['rope.freqs_base_t']
        buffer.flatten()[0]=float('nan') if corruption=='nan' else buffer.flatten()[0]+1
    payload['state']['native_step']['value']=pack(value)
    torch.save(payload,checkpoint/'state.pt')
    manifest['files']['state.pt']=file_sha256(checkpoint/'state.pt')
    write_json(checkpoint/'manifest.json',manifest)
    (checkpoint/'_SUCCESS').write_text(file_sha256(checkpoint/'manifest.json')+'\n')
    components,dataset,algorithm,identity = session.construct(config)
    before=pack_state(components.state);outer=rng_state()
    with pytest.raises(ValueError): restore_checkpoint(checkpoint,components.state,identity=identity)
    assert equal(before,pack_state(components.state)) and equal(outer,rng_state())
    assert components.state['native_step'].global_step==0


def test_common_checkpoint_write_failure_keeps_prior_complete_output(public_checkpoint,tmp_path,monkeypatch):
    from distillschedule.training import checkpoint as checkpoint_module
    config,result=public_checkpoint
    components,_,_,identity=session.construct(config)
    checkpoint_module.restore_checkpoint(result['checkpoint'],components.state,identity=identity)
    original_hash=file_sha256(Path(result['checkpoint'])/'manifest.json')
    def fail(*args,**kwargs): raise OSError('explicit atomic publication failure')
    monkeypatch.setattr(checkpoint_module.os,'rename',fail)
    with pytest.raises(OSError,match='explicit atomic'):
        checkpoint_module.save_checkpoint(tmp_path/'failed',components.state,identity=identity,
                                         cycle=1,cursor=1,versions=dict(native_step=5))
    assert not (tmp_path/'failed').exists() and not list(tmp_path.glob('.failed.tmp-*'))
    assert file_sha256(Path(result['checkpoint'])/'manifest.json')==original_hash
    assert (Path(result['checkpoint'])/'_SUCCESS').is_file()


def test_public_partial_failure_poison_retains_previous_cycle(tmp_path,monkeypatch):
    from distillschedule.training.checkpoint import save_checkpoint
    config=prepared_config(tmp_path,capacity=10,cycles=2)
    original=session.construct;owners=[];outer=[]
    def construct(*args,**kwargs):
        result=original(*args,**kwargs)
        owner=result[0].state['native_step'];owners.append(owner);outer.append(rng_state())
        backward=owner.critic.backward
        def fail(loss):
            if owner.global_step==5: raise RuntimeError('explicit real critic backward failure')
            return backward(loss)
        owner.critic.backward=fail
        return result
    monkeypatch.setattr(session,'construct',construct)
    with pytest.raises(RuntimeError,match='explicit real critic'):
        session.train(config)
    owner=owners[0]
    assert owner._poisoned and owner.global_step==5 and owner.generator.global_steps==2
    assert equal(outer[0],rng_state())
    root=Path(config.trainer['output_dir'])
    failure=json.loads((root/'failure.json').read_text())
    assert failure['completed_cycles_this_process']==1 and failure['cursor']==1
    assert Path(failure['last_complete_checkpoint'],'_SUCCESS').is_file()
    assert not (root/'summary.json').exists()
    with pytest.raises(ValueError,match='failed owner'):
        save_checkpoint(tmp_path/'partial',{'native_step':owner},identity={},cycle=2,cursor=2,versions=dict(native_step=10))
    with pytest.raises(ValueError,match='failed owner'):
        owner.step_input({},absolute_step=5)


@pytest.mark.parametrize('change',['budget','pin','missing','config','nonfinite','label','negative','args','cuda'])
def test_local_asset_contract_refuses_without_native_update(tmp_path,change):
    from safetensors.torch import load_file,save_file
    from distillschedule.training.helios_public import build_model
    assets,sources=fixture_files(tmp_path/'local',capacity=5)
    value=json.loads(assets.read_text())
    params=dict(asset_manifest=str(assets))
    if change=='budget': value['budget']['max_parameters']=1
    elif change=='pin': value['generator']['weights'][0]['sha256']='0'*64
    elif change=='config':
        path=Path(value['generator']['config']['path']);content=json.loads(path.read_text())
        content['num_layers']=2;write_json(path,content);value['generator']['config']=pin(path)
    elif change in ('missing','nonfinite','label'):
        path=Path(value['generator']['weights'][0]['path']);weights=load_file(path)
        if change=='missing': del weights['proj_out.weight']
        elif change=='nonfinite': weights['proj_out.weight'].fill_(float('nan'))
        save_file(weights,path,metadata={'evidence_kind':FIXTURE if change!='label' else 'FOREIGN'})
        value['generator']['weights'][0]=pin(path)
    elif change=='negative':
        path=Path(value['negative']['path'])
        save_file({'negative_prompt_embeds':torch.zeros(1,3,4)},path,metadata={'evidence_kind':FIXTURE})
        value['negative']=pin(path)
    elif change=='args':
        path=Path(value['args']['path']);content=json.loads(path.read_text())
        content['training_config']['gradient_accumulation_steps']=2;write_json(path,content);value['args']=pin(path)
    elif change=='cuda': params['device']='cuda'
    write_json(assets,value)
    outer=rng_state()
    with pytest.raises(ValueError): build_model(params)
    # Admission and meta architecture checks must not seed/advance setup RNG.
    assert equal(outer,rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('gan_start,ema_start',[(0,0),(5,5)])
def test_source_curriculum_boundaries_remain_original(tmp_path,gan_start,ema_start):
    from distillschedule.training.helios_step_owner import expected_clocks
    config=prepared_config(tmp_path,capacity=10,cycles=2,gan_start=gan_start,ema_start=ema_start)
    result=session.train(config)
    payload=read_checkpoint(result['checkpoint'],identity=result['identity'])
    from distillschedule.training.tensor_state import decode
    value=decode(payload['state']['native_step']['value'],{})
    assert value['versions']==expected_clocks(10,ema_start_step=ema_start)


def test_stream_api_validation_and_exact_capacity_preserve_untouched_owner(tmp_path):
    from distillschedule.training.helios_public import build_model
    from distillschedule.data import PreparedDataset
    from distillschedule.data.helios_native_inputs import decode_row
    config=prepared_config(tmp_path,capacity=5,cycles=1)
    components=build_model(config.model['parameters']);owner=components.state['native_step']
    rows=PreparedDataset(config.data['prepared']).sample(0)['native_rows']
    outer=rng_state();native=deepcopy(owner._rng)
    with pytest.raises(ValueError): owner.step_input({},absolute_step=0)
    with pytest.raises(ValueError): owner.step_input(decode_row(rows[0],absolute_step=0),absolute_step=1)
    assert owner.global_step==0 and not owner._poisoned
    assert equal(outer,rng_state()) and equal(native,owner._rng)
    first=decode_row(rows[0],absolute_step=0)
    owner.step_input(first,absolute_step=0)
    before=pack_state(components.state);outer=rng_state()
    with pytest.raises(ValueError,match='already consumed'): owner.step_input(first,absolute_step=1)
    assert equal(before,pack_state(components.state)) and equal(outer,rng_state())
    for index in range(1,5): owner.step_input(decode_row(rows[index],absolute_step=index),absolute_step=index)
    before=pack_state(components.state)
    with pytest.raises(ValueError,match='exhausted'): owner.step_input(first,absolute_step=5)
    assert equal(before,pack_state(components.state)) and not owner._poisoned
    for method in (owner.step,lambda:owner.save(tmp_path/'standalone'),lambda:owner.restore(tmp_path/'standalone')):
        with pytest.raises(ValueError,match='stream-v2'): method()


def test_reference_preparation_and_train_public_cli(tmp_path,capsys):
    from distillschedule.data.cli import main as data_main
    from distillschedule.training.cli import main as training_main
    assets,sources=fixture_files(tmp_path/'local',capacity=5)
    assert data_main(['prepare-helios-gan-reference','--sources',str(sources),'--output',str(tmp_path/'prepared')])==0
    prepared_report=json.loads(capsys.readouterr().out)
    assert prepared_report['samples']==1 and prepared_report['metadata']['evidence_kind']==FIXTURE
    config=dict(schema_version='distillschedule.training.v1',
        model=dict(factory='distillschedule.training.helios_public:build_model',parameters=dict(asset_manifest=str(assets))),
        algorithm=dict(factory='distillschedule.training.helios_public:build_algorithm'),
        data=dict(prepared=str(tmp_path/'prepared/manifest.json')),execution=dict(mode='reference'),
        trainer=dict(cycles=1,seed=773,device='cpu',checkpoint_every=1,output_dir=str(tmp_path/'train')))
    config_path=write_json(tmp_path/'cli.json',config)
    assert training_main(['train',str(config_path)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='COMPLETE' and result['final_versions']==dict(native_step=5)
    assert Path(result['checkpoint'],'_SUCCESS').is_file()



def test_factory_materializes_every_original_nonpersistent_buffer_without_rng_draws(tmp_path):
    from distillschedule.training.helios_public import build_model
    from safetensors.torch import load_file
    assets,sources=fixture_files(tmp_path/'local',capacity=5)
    components=build_model(dict(asset_manifest=str(assets)))
    owner=components.state['native_step']
    manifest=json.loads(assets.read_text())
    for role in ('generator','critic'):
        model=getattr(owner,role).module
        before=rng_state()
        original_rope=type(model.rope)(rope_dim=model.config.rope_dim,theta=model.config.rope_theta)
        assert equal(before,rng_state()), 'original source positional construction consumed RNG'
        nonpersistent={name:value for name,value in model.named_buffers() if name not in model.state_dict()}
        assert nonpersistent.keys()=={'rope.'+name for name,_ in original_rope.named_buffers()}
        assert all(torch.equal(nonpersistent['rope.'+name],value) for name,value in original_rope.named_buffers())
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 and t.isfinite().all()
                   for t in [*model.parameters(),*model.buffers()])
        pinned=load_file(manifest[role]['weights'][0]['path'])
        # LoRA wraps native linear names; unchanged frozen base values remain exact.
        for name,tensor in pinned.items():
            live_name=name.replace('.weight','.base_layer.weight').replace('.bias','.base_layer.bias')
            state=model.state_dict()
            assert torch.equal(state[name] if name in state else state[live_name],tensor)


def test_output_loader_uses_staged_outer_progress_hook_before_restore(public_checkpoint,tmp_path,monkeypatch):
    import shutil
    config,result=public_checkpoint
    target=tmp_path/'wrong-output-checkpoint'
    shutil.copytree(result['checkpoint'],target)
    payload=torch.load(target/'state.pt',weights_only=True)
    metadata=json.loads((target/'manifest.json').read_text())
    payload['cycle']=metadata['cycle']=2
    payload['cursor']=metadata['cursor']=2
    payload['versions']=metadata['versions']=dict(native_step=10)
    torch.save(payload,target/'state.pt')
    metadata['files']['state.pt']=file_sha256(target/'state.pt')
    write_json(target/'manifest.json',metadata)
    (target/'_SUCCESS').write_text(file_sha256(target/'manifest.json')+'\n')
    original=session.construct;owned=[];snapshots=[]
    def construct(*args,**kwargs):
        values=original(*args,**kwargs)
        owned.append(values[0].state);snapshots.append(pack_state(values[0].state))
        return values
    monkeypatch.setattr(session,'construct',construct)
    with pytest.raises(ValueError,match='inner/outer checkpoint progress'):
        session.export(config,target,tmp_path/'export-refused')
    assert equal(snapshots[0],pack_state(owned[0]))
    assert not (tmp_path/'export-refused').exists()


def test_local_base_checks_each_weight_without_rebuilding_state_per_tensor(tmp_path, monkeypatch):
    from distillschedule.training.helios_public_assets import load_base
    from distillschedule.adapters.helios.reference_model import load_reference_model_class

    asset_path, _ = fixture_files(tmp_path / 'local', capacity=5)
    assets = json.loads(asset_path.read_text())
    cls = load_reference_model_class(backend='torch-sdpa')
    original = cls.state_dict
    calls = []
    def record(self, *args, **kwargs):
        calls.append(id(self))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, 'state_dict', record)
    model = load_base(cls, assets['generator'], evidence_kind=FIXTURE,
                      remaining_parameters=assets['budget']['max_parameters'])
    # Layout admission and the final complete value check each need one map;
    # the pinned original constructor has no state-dict hooks or side effects.
    assert len(calls) == 2
    assert all(not m._state_dict_hooks and not m._state_dict_pre_hooks for m in model.modules())
    assert all(t.device.type == 'cpu' and t.isfinite().all() for t in model.parameters())
