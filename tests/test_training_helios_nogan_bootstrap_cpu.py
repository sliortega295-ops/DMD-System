"""D1 state-reader evidence, not original W2 warm-update/lifecycle proof."""

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil

import pytest
import torch

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from helios_reference_fixtures import original_logical as original_logical


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return path


def _pin(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha256(path))


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _commit(engine, value):
    # Deliberately NOT an original W2 forward/backward. This reader fixture
    # exercises real Adam/clipping/LambdaLR with declared admitted gradients.
    for group in engine.optimizer.param_groups:
        for parameter in group['params']:
            parameter.grad = torch.full_like(parameter, value)
    engine.step()


@pytest.fixture(scope='module')
def warm_fixture(tmp_path_factory):
    from peft import LoraConfig
    from safetensors.torch import save_file
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.training.helios_public_assets import FIXTURE, runtime_identity
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.training.helios_nogan_bootstrap import SCHEMA, SCOPE, native_policy, source_identity
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS, _all_linear_targets
    from runtime.precast_checkpoint import convert_pretrained_safetensors
    from runtime.gst_pool_contract import write_negative_prompt_asset
    from runtime.sparse_ema import SparseShardEMA

    assert importlib.metadata.version('torch') == '2.10.0+cpu'
    assert importlib.metadata.version('diffusers') == '0.37.1'
    assert importlib.metadata.version('peft') == '0.17.1'
    root = tmp_path_factory.mktemp('complete-original-state-reader-only')
    logical = root/'logical'; logical.mkdir()
    cls, roles, engines, rows = load_reference_model_class(backend='torch-sdpa'), {}, {}, {}
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        for index, (role, native, subfolder) in enumerate((('generator','generator','transformer_ode'),
                                                         ('critic','score','transformer'))):
            torch.manual_seed(401 + index)
            model = cls(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=16,
                out_channels=16,text_dim=4096,freq_dim=16,ffn_dim=64,num_layers=1,rope_dim=(4,6,6),
                is_use_gan=False,is_use_gan_final=False,is_use_gan_hooks=False,**TRANSFORMER_ADDITIONAL_KWARGS)
            for name, parameter in model.named_parameters():
                parameter.data = parameter.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules)
                                                   else torch.bfloat16)
            native_root = root/native
            source = native_root/subfolder
            model.save_config(source)
            weights = source/'model.safetensors'
            save_file({n:t.detach().contiguous() for n,t in model.state_dict().items()}, weights,
                      metadata={'evidence_kind':FIXTURE})
            _write(source/'diffusion_pytorch_model.safetensors.index.json',dict(metadata={},
                weight_map={n:weights.name for n in model.state_dict()}))
            roles[role] = dict(config=_pin(source/'config.json'),weights=[_pin(weights)])
            precast = root/(native+'-precast')
            convert_pretrained_safetensors(source,precast,target_dtype=torch.bfloat16)
            for name, directory, identity in ((native+'_model',native_root,subfolder+'/config.json'),
                    (native+'_precast',precast,'distillschedule_precast_manifest.json')):
                rows[name] = dict(path=str(directory),identity_file=identity,sha256=file_sha256(directory/identity))
            model.requires_grad_(False)
            targets, excluded = _all_linear_targets(model), ['down','up']
            if role == 'generator':
                targets += ['patch_short','patch_mid','patch_long']
            else:
                patches = ['patch_short','patch_mid','patch_long','patch_embedding']
                targets = [n for n in targets if n not in patches]
                excluded += patches+['gan_heads','gan_final_head']
            model.add_adapter(LoraConfig(r=256,lora_alpha=256.,lora_dropout=0.,init_lora_weights='gaussian',
                target_modules=sorted(set(targets)),exclude_modules=excluded))
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            model.enable_gradient_checkpointing(); model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=2e-6 if role=='generator' else 4e-7,betas=(0.,.999),eps=1e-8,weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            engines[role] = FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=0,
                                             global_batch=1,dp_world_size=1)
    g = engines['generator']
    ema = SparseShardEMA(g.module.named_parameters(),decay=.99,update_after_step=750,base_model=str(root/'generator'))
    before_shadows = deepcopy(ema.shadows)
    updates = []
    for index in range(10):
        if index % 5 == 0:
            _commit(g,(index+1)/100.); updates.append(dict(role='generator',gradient=(index+1)/100.))
        _commit(engines['critic'],(index+1)/200.)
        updates.append(dict(role='critic',gradient=(index+1)/200.))
        ema.step(g.module.parameters())
    assert ema.optimization_step == 10 and any(not torch.equal(a,b) for a,b in zip(before_shadows,ema.shadows))
    manifest = dict(schema_version='distillschedule.logical_checkpoint.v3',global_step=index+1,
        source='STATE_READER_FIXTURE_ONLY',components={},ema=dict(optimization_step=ema.optimization_step))
    expected = {}
    for role, engine in engines.items():
        selected = {n:p for n,p in engine.module.named_parameters() if p.requires_grad}
        records, offset = {}, 0
        for name, p in selected.items():
            records[name] = dict(shape=list(p.shape),numel=p.numel(),offset_elements=offset)
            offset += p.numel()
        payloads = {}
        for key, state_key in (('master',None),('first_moment','exp_avg'),('second_moment','exp_avg_sq')):
            tensors = [p.detach() if state_key is None else engine.optimizer.state[p][state_key] for p in selected.values()]
            payload = logical/(role+'-'+key+'.f32')
            torch.cat([t.flatten() for t in tensors]).numpy().tofile(payload)
            payloads[key] = dict(path=payload.name,bytes=payload.stat().st_size,dtype='torch.float32',sha256=file_sha256(payload))
        manifest['components'][role] = dict(parameter_names=list(selected),records=records,logical_numel=offset,
            optimizer_step=engine.global_steps,parameter_version=engine.global_steps,
            scheduler_state=engine.lr_scheduler.state_dict(),payloads=payloads)
        expected[role] = dict(weights=deepcopy({n:p.detach() for n,p in selected.items()}),
            optimizer=deepcopy(engine.optimizer.state_dict()),scheduler=deepcopy(engine.lr_scheduler.state_dict()))
        assert all(s['step'].item()==engine.global_steps for s in engine.optimizer.state.values())
        assert any(bool(torch.count_nonzero(s['exp_avg'])) for s in engine.optimizer.state.values())
    ema.save_pretrained(logical/'ema',max_shard_bytes=8*1024**2)
    manifest['manifest_hash'] = fingerprint(manifest)
    _write(logical/'manifest.json',manifest)
    rows['logical_checkpoint'] = dict(path=str(logical),identity_file='manifest.json',
        sha256=file_sha256(logical/'manifest.json'),ema_manifest=dict(identity_file='ema/sparse_ema_manifest.json',
        sha256=file_sha256(logical/'ema/sparse_ema_manifest.json')))
    negative = root/'negative'
    write_negative_prompt_asset(negative,embedding=torch.zeros(1,512,4096,dtype=torch.bfloat16),model_root=root)
    rows['negative_prompt'] = dict(path=str(negative),identity_file='manifest.json',sha256=file_sha256(negative/'manifest.json'))
    inputs = _write(root/'reader-gradients.json',dict(evidence_kind='STATE_READER_FIXTURE_ONLY',updates=updates,
        original_forward_backward='NOT_RUN',original_geometry=[1,16,9,48,80]))
    value = dict(schema_version=SCHEMA,scope=SCOPE,precision='WAN_NATIVE_MIXED_BF16',base_evidence_kind=FIXTURE,
        source=source_identity(),runtime=runtime_identity(),request=OriginalHeliosRequest().to_dict(),
        native_assets=dict(schema_version='distillschedule.helios_reference_assets.v1',assets=rows),
        budget=dict(max_parameters=20000000,max_loaded_bytes=400000000),**roles)
    provenance = dict(evidence_kind='STATE_READER_FIXTURE_ONLY',update_source='DECLARED_READER_GRADIENTS_NOT_W2',
        source=value['source'],runtime=value['runtime'],request=value['request'],policy=native_policy(),
        bases={r:value[r] for r in ('generator','critic')},native_assets=value['native_assets'],inputs=[_pin(inputs)],
        clocks=dict(global_step=10,generator=2,critic=10,ema=10))
    value['provenance'] = _pin(_write(root/'provenance.json',provenance))
    asset = _write(root/'assets.json',value)
    expected_ema = deepcopy(ema.shadows)
    for role, engine in engines.items():
        _commit(engine,.03125)
        expected[role]['next'] = deepcopy(engine.optimizer.state_dict())
        expected[role]['next_weights'] = deepcopy({n:p.detach() for n,p in engine.module.named_parameters() if p.requires_grad})
    ema.step(g.module.parameters())
    return dict(root=root,asset=asset,expected=expected,ema=expected_ema,next_ema=deepcopy(ema.shadows))


def test_w2_cpu_warm_materializer_exists():
    from distillschedule.training.helios_nogan_bootstrap import materialize_bootstrap
    assert callable(materialize_bootstrap)


def test_complete_original_layout_real_adam_reader_state_and_next_commit(warm_fixture, monkeypatch):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    from runtime import critic_pool_checkpoint
    copied, ids = [], {}
    original = critic_pool_checkpoint.load_logical_master_into_trainables
    def witnessed(model, package, **kwargs):
        before = {n:id(p) for n,p in model.named_parameters()}
        result = original(model,package,**kwargs)
        assert before == {n:id(p) for n,p in model.named_parameters()}
        copied.append(kwargs['component']); ids[kwargs['component']] = before
        return result
    monkeypatch.setattr(critic_pool_checkpoint,'load_logical_master_into_trainables',witnessed)
    rng = torch.get_rng_state().clone()
    result = bootstrap.materialize_bootstrap(warm_fixture['asset'])
    assert torch.equal(rng,torch.get_rng_state()) and copied == ['generator','critic']
    assert result.identity['evidence_kind'] == 'STATE_READER_FIXTURE_ONLY'
    assert result.identity['original_w2_warm_updates_verified'] is False
    assert result.identity['native_protocol_samples'] == {'generator':32,'critic':160}
    with pytest.raises(TypeError):
        result.identity['native_protocol_samples']['generator'] = 1
    for role, engine in result.engines.items():
        expected = warm_fixture['expected'][role]
        assert type(engine.optimizer) is torch.optim.AdamW
        assert type(engine.lr_scheduler) is torch.optim.lr_scheduler.LambdaLR
        assert engine.dp_world_size == engine.train_batch_size() == 1 and engine._gradient_clip == 1.
        assert engine.global_steps == engine.global_samples == (2 if role=='generator' else 10)
        assert _equal(expected['weights'],{n:p.detach() for n,p in engine.module.named_parameters() if p.requires_grad})
        assert _equal(expected['optimizer'],engine.optimizer.state_dict())
        assert _equal(expected['scheduler'],engine.lr_scheduler.state_dict())
        assert ids[role] == {n:id(p) for n,p in engine.module.named_parameters()}
        assert any(p.dtype==torch.bfloat16 for p in engine.module.parameters() if not p.requires_grad)
        _commit(engine,.03125)
        assert _equal(expected['next'],engine.optimizer.state_dict())
        assert _equal(expected['next_weights'],{n:p.detach() for n,p in engine.module.named_parameters() if p.requires_grad})
    assert _equal(result.ema.shadows,warm_fixture['ema'])
    assert result.ema.optimization_step == 10 and result.ema.update_after_step == 750
    assert result.ema.get_decay(753) == 3/12
    result.ema.step(result.engines['generator'].module.parameters())
    assert _equal(result.ema.shadows,warm_fixture['next_ema'])
    views = result.state_views()
    for view in views.values():
        if hasattr(view,'validate_state_dict'):
            view.validate_state_dict(deepcopy(view.state_dict()))
    wrong = views['generator.clocks'].state_dict()
    wrong['policy']['dp_world_size'] = 16
    with pytest.raises(ValueError,match='clock/policy'):
        views['generator.clocks'].validate_state_dict(wrong)
    assert not torch.cuda.is_initialized()


def _copy_assets(fixture, destination):
    shutil.copytree(fixture['root'],destination)
    # Canonical paths are explicit evidence, not relocatable guesses.
    def relocate(value):
        if isinstance(value,dict):
            return {k:relocate(v) for k,v in value.items()}
        if isinstance(value,list):
            return [relocate(v) for v in value]
        if isinstance(value,str) and value.startswith(str(fixture['root'])+'/'):
            return str(destination)+value[len(str(fixture['root'])):]
        return value
    for name in ('assets.json','provenance.json'):
        _write(destination/name,relocate(json.loads((destination/name).read_text())))
    _refresh(destination)
    return destination/'assets.json'


def _refresh(root):
    asset = json.loads((root/'assets.json').read_text())
    logical = root/'logical'
    manifest = json.loads((logical/'manifest.json').read_text())
    for spec in manifest['components'].values():
        for payload in spec['payloads'].values():
            path = logical/payload['path']
            if path.is_file():
                payload.update(bytes=path.stat().st_size,sha256=file_sha256(path))
    manifest.pop('manifest_hash',None); manifest['manifest_hash'] = fingerprint(manifest)
    _write(logical/'manifest.json',manifest)
    for row in asset['native_assets']['assets'].values():
        row['sha256'] = file_sha256(Path(row['path'])/row['identity_file'])
    row = asset['native_assets']['assets']['logical_checkpoint']
    row['ema_manifest']['sha256'] = file_sha256(logical/row['ema_manifest']['identity_file'])
    for role in ('generator','critic'):
        asset[role]['config'] = _pin(Path(asset[role]['config']['path']))
        asset[role]['weights'] = [_pin(Path(pin['path'])) for pin in asset[role]['weights']]
    proof = json.loads((root/'provenance.json').read_text())
    proof['native_assets'] = asset['native_assets']
    proof['bases'] = {r:asset[r] for r in ('generator','critic')}
    proof['inputs'] = [_pin(Path(pin['path'])) for pin in proof['inputs']]
    asset['provenance'] = _pin(_write(root/'provenance.json',proof))
    _write(root/'assets.json',asset)


@pytest.mark.parametrize('case',[
    'global_clock','generator_clock','critic_clock','optimizer_clock','scheduler_clock','scheduler_policy',
    'scheduler_counter','missing_moment','moment_dtype','moment_nan_generator','moment_nan_critic','master_nan_critic',
    'negative_second_moment','name','order','shape','overlap','gap','duplicate_json','escaped_payload','aliased_payload',
    'ema_clock','ema_decay','ema_update_after','ema_current_decay','ema_indices','ema_count','ema_shadow_nan',
    'ema_shadow_dtype','ema_names','missing_pyramid','base_dtype','gan_config','foreign_request','rank_policy',
    'request_bool','clip_bool','scheduler_flag_type','ema_format_bool',
])
def test_rehashed_malformed_late_owner_rejects_before_any_master_copy(warm_fixture,tmp_path,monkeypatch,case):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    from runtime import critic_pool_checkpoint
    from safetensors.torch import load_file, save_file
    root = tmp_path/'assets'; path = _copy_assets(warm_fixture,root)
    logical = root/'logical'; manifest = json.loads((logical/'manifest.json').read_text())
    g,c = manifest['components']['generator'],manifest['components']['critic']
    ema_path = logical/'ema/sparse_ema_manifest.json'; ema = json.loads(ema_path.read_text())
    if case=='global_clock': manifest['global_step']=0
    elif case=='generator_clock': g['parameter_version']=0
    elif case=='critic_clock': c['parameter_version']=5
    elif case=='optimizer_clock': c['optimizer_step']=9
    elif case=='scheduler_clock': c['scheduler_state']['last_epoch']=9
    elif case=='scheduler_policy': c['scheduler_state']['base_lrs']=[2e-6]
    elif case=='scheduler_counter': c['scheduler_state']['_step_count']=1
    elif case=='scheduler_flag_type': c['scheduler_state']['_get_lr_called_within_step']=0
    elif case=='missing_moment': del c['payloads']['first_moment']
    elif case=='moment_dtype': c['payloads']['first_moment']['dtype']='torch.float64'
    elif case in ('moment_nan_generator','moment_nan_critic','master_nan_critic','negative_second_moment'):
        spec = g if case=='moment_nan_generator' else c
        payload = 'master' if case=='master_nan_critic' else ('second_moment' if case=='negative_second_moment' else 'first_moment')
        file = logical/spec['payloads'][payload]['path']
        raw = bytearray(file.read_bytes()); raw[:4] = torch.tensor([-1. if case=='negative_second_moment' else float('nan')]).numpy().tobytes()
        file.write_bytes(raw)
    elif case=='name':
        old = c['parameter_names'][0]; c['parameter_names'][0]='wrong'; c['records']['wrong']=c['records'].pop(old)
    elif case=='order': c['parameter_names'].reverse()
    elif case=='shape': c['records'][c['parameter_names'][0]]['shape'].reverse()
    elif case=='overlap': c['records'][c['parameter_names'][1]]['offset_elements']=0
    elif case=='gap': c['records'][c['parameter_names'][-1]]['offset_elements']+=1
    elif case=='escaped_payload': c['payloads']['master']['path']='../logical/'+c['payloads']['master']['path']
    elif case=='aliased_payload': c['payloads']['second_moment']=deepcopy(c['payloads']['first_moment'])
    elif case=='ema_clock': ema['optimization_step']=9
    elif case=='ema_format_bool': ema['format_version']=True
    elif case=='ema_decay': ema['decay']=.98
    elif case=='ema_update_after': ema['update_after_step']=0
    elif case=='ema_current_decay': ema['cur_decay_value']=None
    elif case=='ema_indices': ema['tracked_indices'][0]+=1
    elif case=='ema_count': ema['parameter_count']+=1
    elif case=='ema_names': ema['tracked_names'].reverse()
    elif case in ('ema_shadow_nan','ema_shadow_dtype'):
        shard=ema['shards'][-1]; file=ema_path.parent/shard['file']; tensors=torch.load(file,weights_only=True)
        name=next(iter(tensors))
        if case=='ema_shadow_nan': tensors[name].flatten()[0]=float('nan')
        else: tensors[name]=tensors[name].double()
        torch.save(tensors,file); shard['sha256']=file_sha256(file)
    elif case in ('missing_pyramid','base_dtype'):
        asset=json.loads(path.read_text()); file=Path(asset['generator']['weights'][0]['path']); tensors=load_file(file)
        name=next(n for n in tensors if n.startswith('patch_long.'))
        if case=='missing_pyramid': del tensors[name]
        else: tensors[name]=tensors[name].double()
        save_file(tensors,file,metadata={'evidence_kind':asset['base_evidence_kind']})
    elif case=='gan_config':
        asset=json.loads(path.read_text()); file=Path(asset['critic']['config']['path']); config=json.loads(file.read_text())
        config['is_use_gan']=True; _write(file,config)
    elif case=='foreign_request':
        asset=json.loads(path.read_text()); asset['request']['variant']='gan'; _write(path,asset)
    elif case=='request_bool':
        asset=json.loads(path.read_text()); asset['request']['physical_batch']=True; _write(path,asset)
        proof=json.loads((root/'provenance.json').read_text()); proof['request']=asset['request']; _write(root/'provenance.json',proof)
    elif case=='clip_bool':
        proof=json.loads((root/'provenance.json').read_text()); proof['policy']['gradient_clip']=True; _write(root/'provenance.json',proof)
    elif case=='rank_policy':
        proof=json.loads((root/'provenance.json').read_text()); proof['policy']['lora_rank']=1; _write(root/'provenance.json',proof)
    _write(logical/'manifest.json',manifest); _write(ema_path,ema); _refresh(root)
    if case=='duplicate_json':
        text=(logical/'manifest.json').read_text(); (logical/'manifest.json').write_text('{"global_step":10,'+text[1:])
        asset=json.loads(path.read_text()); asset['native_assets']['assets']['logical_checkpoint']['sha256']=file_sha256(logical/'manifest.json')
        _write(path,asset)
    copies=[]
    monkeypatch.setattr(critic_pool_checkpoint,'load_logical_master_into_trainables',lambda *a,**k:copies.append(k))
    with pytest.raises((ValueError,KeyError,TypeError,AssertionError)):
        bootstrap.materialize_bootstrap(path)
    assert copies == [], 'all roles/moments/scheduler/EMA must be admitted before the first master write'
    assert not torch.cuda.is_initialized()


def test_native_distributed_topology_and_unqualified_provenance_reject(warm_fixture,tmp_path,monkeypatch):
    from distillschedule.training.helios_nogan_bootstrap import materialize_bootstrap
    monkeypatch.setenv('WORLD_SIZE','16')
    with pytest.raises(ValueError,match='CPU1'):
        materialize_bootstrap(warm_fixture['asset'])
    monkeypatch.delenv('WORLD_SIZE')
    path=_copy_assets(warm_fixture,tmp_path/'assets')
    proof=json.loads((path.parent/'provenance.json').read_text())
    proof['evidence_kind']='W1_TWO_CYCLES_RELABELLED_AS_W2'
    _write(path.parent/'provenance.json',proof); _refresh(path.parent)
    with pytest.raises(ValueError,match='evidence'):
        materialize_bootstrap(path)


def test_cpu_budget_rejects_before_deserializing_ema(warm_fixture,tmp_path,monkeypatch):
    from distillschedule.training.helios_nogan_bootstrap import materialize_bootstrap
    path = _copy_assets(warm_fixture,tmp_path/'assets')
    value=json.loads(path.read_text()); value['budget']['max_loaded_bytes']=1; _write(path,value)
    def forbidden(*args,**kwargs):
        raise AssertionError('budget must reject before native EMA deserialization')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='budget'):
        materialize_bootstrap(path)


def test_existing_scalar_package_is_only_parser_rejection_evidence(warm_fixture,original_logical,tmp_path,monkeypatch):
    from distillschedule.training.helios_nogan_bootstrap import materialize_bootstrap
    from distillschedule.training.helios_reference_assets import load_bootstrap
    from runtime import critic_pool_checkpoint
    path = _copy_assets(warm_fixture,tmp_path/'assets')
    shutil.rmtree(path.parent/'logical'); shutil.copytree(original_logical,path.parent/'logical')
    _refresh(path.parent)
    # This OLD scalar fixture forges clocks and shadows: native metadata parser
    # acceptance is characterized only to prove it is NOT a D1 positive fixture.
    package,_=load_bootstrap(json.loads(path.read_text())['native_assets'],verify_payloads=True)
    assert package.component('generator')['parameter_names']==['weight']
    copies=[]
    monkeypatch.setattr(critic_pool_checkpoint,'load_logical_master_into_trainables',lambda *a,**k:copies.append(k))
    with pytest.raises(ValueError,match='inventory/order'):
        materialize_bootstrap(path)
    assert copies == []


def test_native_single_module_prefix_ema_names_have_lossless_cpu_state_view(warm_fixture,tmp_path):
    from distillschedule.training.helios_nogan_bootstrap import materialize_bootstrap
    path=_copy_assets(warm_fixture,tmp_path/'assets')
    ema_path=path.parent/'logical/ema/sparse_ema_manifest.json'
    metadata=json.loads(ema_path.read_text())
    metadata['tracked_names']=['module.'+n for n in metadata['tracked_names']]
    for shard in metadata['shards']:
        file=ema_path.parent/shard['file']; values=torch.load(file,weights_only=True)
        torch.save({'module.'+n:t for n,t in values.items()},file)
        shard['names']=['module.'+n for n in shard['names']]; shard['sha256']=file_sha256(file)
    _write(ema_path,metadata); _refresh(path.parent)
    result=materialize_bootstrap(path)
    assert all(n.startswith('module.') for n in result.identity['serialized_ema_names'])
    assert all(not n.startswith('module.') for n in result.ema.names)
    view=result.state_views()['ema']; view.validate_state_dict(view.state_dict())
    assert _equal(result.ema.shadows,warm_fixture['ema'])
    assert not torch.cuda.is_initialized()


def test_non_fp32_default_rejects_before_model_construction(warm_fixture,monkeypatch):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    def forbidden(*args,**kwargs):
        raise AssertionError('default dtype must reject before constructing models')
    monkeypatch.setattr(bootstrap,'load_base',forbidden)
    before=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError,match='FP32'):
            bootstrap.materialize_bootstrap(warm_fixture['asset'])
    finally:
        torch.set_default_dtype(before)


def test_native_policy_is_immutable_and_proof_copies_do_not_change_next_admission(warm_fixture):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    policy=deepcopy(dict(bootstrap.POLICY))
    with pytest.raises(TypeError):
        bootstrap.POLICY['gradient_clip']=10.
    with pytest.raises(TypeError):
        bootstrap.POLICY['betas'][1]=.5
    proof=json.loads((warm_fixture['root']/'provenance.json').read_text())
    proof['policy']['gradient_clip']=10.
    proof['policy']['betas'][1]=.5
    assert bootstrap.POLICY==policy
    result=bootstrap.materialize_bootstrap(warm_fixture['asset'])
    assert result.identity['policy']['gradient_clip']==1.
    assert result.identity['policy']['betas']==(0.,.999)


def test_genuine_cpu_test_warm_provenance_is_distinct_and_requires_actual_receipts(warm_fixture, tmp_path):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    assert bootstrap.CPU_WARM not in (bootstrap.READER_ONLY, bootstrap.ORIGINAL)
    value = json.loads(warm_fixture['asset'].read_text())
    proof = json.loads(Path(value['provenance']['path']).read_text())
    proof['evidence_kind'] = bootstrap.CPU_WARM
    proof['update_source'] = 'ORIGINAL_W2_HELPERS'
    value['provenance'] = _pin(_write(tmp_path/'false-original-proof.json', proof))
    path = _write(tmp_path/'false-original-assets.json', value)
    with pytest.raises(ValueError, match='actual original CPU warm receipt'):
        bootstrap.materialize_bootstrap(path)
