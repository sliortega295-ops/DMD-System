"""Actual Diffusers tiny classes; every randomly initialized asset is SYNTHETIC_TEST_ONLY."""
import importlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='opt in to the isolated modern-flow pinned CPU stack with DMD_MODERN_FLOW_CPU=1')


@pytest.mark.parametrize('family,classname,config', [
    ('flux', 'Flux2Transformer2DModel', dict(patch_size=1, in_channels=16, out_channels=16,
        num_layers=1, num_single_layers=1, num_attention_heads=2, attention_head_dim=16,
        joint_attention_dim=48, timestep_guidance_channels=16, mlp_ratio=2,
        axes_dims_rope=[4,4,4,4], guidance_embeds=False)),
    ('qwen_image', 'QwenImageTransformer2DModel', dict(patch_size=2, in_channels=16, out_channels=4,
        num_layers=1, num_attention_heads=2, attention_head_dim=16, joint_attention_dim=16,
        axes_dims_rope=[4,6,6], guidance_embeds=False, zero_cond_t=False,
        use_additional_t_cond=False, use_layer3d_rope=False)),
])
def test_actual_transformer_saved_format_has_a_common_flow_adapter(tmp_path, family, classname, config):
    import diffusers
    import torch
    native = getattr(diffusers, classname)(**config).float().eval()
    native.save_pretrained(tmp_path / 'transformer', safe_serialization=True)
    assert (tmp_path / 'transformer' / 'diffusion_pytorch_model.safetensors').is_file()
    adapter = importlib.import_module('distillschedule.training.' + family)
    assert callable(adapter.build_model)
    assert callable(adapter.build_inference_model)
    assert torch.cuda.is_initialized() is False


@pytest.fixture(scope='module', params=['flux2-klein-base', 'qwen-image'])
def assets(request, tmp_path_factory):
    from pathlib import Path
    from distillschedule.training.plugins import load_callable
    repo = Path(__file__).resolve().parents[1]
    build, _ = load_callable(str(repo/'recipes/flow_image/create_cpu_fixture.py')+':build_fixture')
    return build(tmp_path_factory.mktemp(request.param)/'fixture', request.param)


def owned(assets):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    config = load_training_config(assets['config'])
    return (config, *construct(config))


def test_actual_forward_text_masks_pack_and_native_vae_roundtrip(assets):
    import torch
    from distillschedule.training.prediction import PredictionSpec
    config, components, dataset, _, _ = owned(assets)
    role = components.state['student']; api = role.api; media = components.state['media']
    assert role.native.training and components.state['fake'].native.training
    assert not components.state['teacher'].training and not components.state['teacher'].native.training
    assert not media.training and not media.vae.training and not media.text_encoder.training
    row = dataset.sample(0)
    empty = media.encode_text([''])
    assert torch.equal(row['null_context'], empty[0]) and torch.equal(row['null_context_mask'], empty[1])
    assert torch.count_nonzero(empty[0]) > 0
    assert row['context'].shape[-1] == role.architecture['joint_attention_dim']
    if assets['family'] == 'flux2-klein-base':
        assert media.text_encoder.config.num_hidden_layers == 28
        assert role.architecture['guidance_embeds'] is False
        assert row['img_ids'][0].tolist() == [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,1,1,0]]
    else:
        assert media.pipeline().prompt_template_encode_start_idx == 34
        assert row['img_shapes'].tolist() == [[1,2,2]]
        assert row['null_context_mask'].sum() != row['context_mask'].sum()
    sigma = torch.tensor([.625], dtype=torch.float32)
    sample = row['latents']
    actual = role.denoise(sample, sigma, row, conditional=True)['image']
    if assets['family'] == 'flux2-klein-base':
        native = role.native(hidden_states=sample['image'], encoder_hidden_states=row['context'], timestep=sigma,
            img_ids=row['img_ids'],txt_ids=row['txt_ids'],guidance=None,return_dict=False)[0]
    else:
        native = role.native(hidden_states=sample['image'],encoder_hidden_states=row['context'],
            encoder_hidden_states_mask=row['context_mask'],timestep=sigma,img_shapes=[[(1,2,2)]],guidance=None,return_dict=False)[0]
    assert torch.equal(actual, native), 'native normalized sigma and v sign must match exactly'
    if assets['family'] == 'qwen-image':
        original_context, original_mask = media.pipeline().encode_prompt(['a bird'],device=torch.device('cpu'),max_sequence_length=1024)
        length = original_context.shape[1]
        assert torch.equal(row['context'][:, :length], original_context)
        assert row['context_mask'][:, :length].all() and not row['context_mask'][:, length:].any()
        original = role.native(hidden_states=sample['image'],encoder_hidden_states=original_context,
            encoder_hidden_states_mask=original_mask,timestep=sigma,img_shapes=[[(1,2,2)]],guidance=None,return_dict=False)[0]
        torch.testing.assert_close(actual,original,rtol=1e-5,atol=1e-6)

    x0 = PredictionSpec('flow').to_x0(actual, sample['image'], sigma=sigma[:,None,None])
    assert torch.equal(x0, sample['image']-sigma[:,None,None]*native)
    from PIL import Image
    import numpy as np
    pixels = torch.from_numpy(np.array(Image.open(assets['root']/'input.png')).copy()).permute(2,0,1).unsqueeze(0).float()/127.5-1
    tokens, shape = media.encode_image(pixels)
    assert torch.equal(tokens, row['latents']['image']) and shape == row['latent_shape'].tolist()
    with torch.no_grad():
        raw = media.vae.encode(pixels if assets['family']=='flux2-klein-base' else pixels.unsqueeze(2)).latent_dist.mode()
        if assets['family']=='flux2-klein-base':
            normalized = (api.classes()[3]._patchify_latents(raw)-media.vae.bn.running_mean.view(1,-1,1,1))/torch.sqrt(
                media.vae.bn.running_var.view(1,-1,1,1)+media.vae.config.batch_norm_eps)
            packed = api.classes()[3]._pack_latents(normalized)
            decoded = media.vae.decode(raw, return_dict=False)[0]
        else:
            normalized = (raw-raw.new_tensor(media.vae.config.latents_mean).view(1,-1,1,1,1))/raw.new_tensor(
                media.vae.config.latents_std).view(1,-1,1,1,1)
            packed = api.classes()[3]._pack_latents(normalized,*shape)
            decoded = media.vae.decode(raw,return_dict=False)[0][:,:,0]
        assert torch.equal(tokens,packed)
        decoded_owned = media.decode_tokens({'image_latents':api.unpack(tokens,shape)})
        torch.testing.assert_close(decoded_owned,decoded,rtol=1e-5,atol=1e-6)
    assert not any(p.requires_grad for p in media.parameters())
    assert not torch.cuda.is_initialized()


def test_real_common_g1_f5_and_existing_immutable_sharing_parity(assets, tmp_path):
    from copy import deepcopy
    import json
    import torch
    from distillschedule.training.session import build_cycle
    from distillschedule.open_workload.torch_backend import Session,sharing_inventory,resolve_module
    from test_training_end_to_end import equal
    results = []; gradients = []
    for share in (False,True):
        config, model, dataset, algorithm, _ = owned(assets)
        teacher = deepcopy(model.state['teacher'].state_dict())
        before = {role:deepcopy(model.state[role].state_dict()) for role in ('student','fake')}
        captured = {role:[] for role in ('student','fake')}
        for role in captured:
            for parameter in model.state[role].parameters():
                if parameter.requires_grad:
                    parameter.register_hook(lambda grad, role=role: captured[role].append(grad.detach().clone()))
        workload = build_cycle(config,model,dataset,algorithm,cycle=0,cursor=0)
        pairs = sharing_inventory(workload)
        assert pairs, 'exact identical frozen native bases must produce real existing share candidates'
        first,second,_ = pairs[0]
        source = resolve_module(model.state,first[0]).get_parameter(first[1])
        target = resolve_module(model.state,second[0]).get_parameter(second[1])
        assert source is not target and source.data_ptr()!=target.data_ptr()
        session = Session(workload,dict(backend='torch-callable-resident-v1',share_immutable=share,checkpoint_bindings=[]),copy_state=False)
        target = resolve_module(session.state,second[0]).get_parameter(second[1])
        assert (source is target) is share
        output = session.run()
        assert all(torch.isfinite(x) for x in output.values())
        assert model.state['dmd_counters'].tolist()==[1,1,5]
        assert session.receipt['dispatcher']['final_versions']['fake']==5
        assert equal(teacher,model.state['teacher'].state_dict())
        assert all(p.grad is None and not p.requires_grad for p in model.state['teacher'].parameters())
        for role in ('student','fake'):
            assert captured[role] and any(torch.count_nonzero(g)>0 for g in captured[role])
            assert any(not torch.equal(value, before[role][name]) for name,value in model.state[role].state_dict().items())
            optimizer = model.state[role+'_optimizer']
            assert optimizer.state and all(torch.isfinite(v['exp_avg']).all() for v in optimizer.state.values())
            assert all(float(v['step'])==(1 if role=='student' else 5) for v in optimizer.state.values())
            assert model.state[role+'_scheduler'].last_epoch==(1 if role=='student' else 5)
            frozen = {name:p for name,p in model.state[role].named_parameters() if not p.requires_grad}
            assert all(torch.equal(p,before[role][name]) for name,p in frozen.items())
        gradients.append(captured)
        state = {key: value.state_dict() if hasattr(value,'state_dict') else value for key,value in model.state.items()}
        from distillschedule.training.checkpoint import pack_state
        results.append(dict(state=deepcopy(state),packed=pack_state(model.state),output={k:v.detach().clone() for k,v in output.items()},storage=session.initial_storage,effects=session.effects.copy()))
    assert equal(results[0]['output'],results[1]['output']) and equal(results[0]['state'],results[1]['state'])
    assert equal(gradients[0],gradients[1]), 'real shared/unshared gradient tensors must match exactly'
    assert equal(results[0]['packed'],results[1]['packed']), 'checkpoint immutable slot flags must be alias-independent'
    assert results[1]['storage']<results[0]['storage']
    assert results[0]['effects']['shared_parameter_aliases']==0 and results[1]['effects']['shared_parameter_aliases']>0
    (tmp_path/'sharing-proof.json').write_text(json.dumps(dict(family=assets['family'],evidence_kind='SYNTHETIC_TEST_ONLY',
        exact_output_gradient_update_optimizer_scheduler_ema_teacher_parity=True,
        shared_parameter_aliases=results[1]['effects']['shared_parameter_aliases'],
        unshared_bytes=results[0]['storage'],shared_bytes=results[1]['storage'],gpu_execution='NOT_RUN',quality_validation='NOT_RUN'),indent=2))


def test_fresh_process_cli_resume_current_student_export_and_new_prompt(assets,tmp_path):
    from copy import deepcopy
    import json
    from pathlib import Path
    import torch
    from distillschedule.data import PreparedDataset
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import _load_for_output
    from test_training_end_to_end import run_cli,equal
    config = assets['config']
    partial=run_cli('train',config,'--cycles',1,'--output',tmp_path/'partial')
    resumed=run_cli('train',config,'--cycles',3,'--resume',partial['checkpoint'],'--output',tmp_path/'resumed')
    whole=run_cli('train',config,'--cycles',3,'--output',tmp_path/'whole')
    a=torch.load(Path(resumed['checkpoint'])/'state.pt',weights_only=True)
    b=torch.load(Path(whole['checkpoint'])/'state.pt',weights_only=True)
    assert equal(a,b),'complete cycle state/RNG/data/versions/Adam/schedulers/EMA must match in new processes'
    assert a['state']['dmd_counters']['value']['counters'].tolist()==[3,3,15]
    assert a['state']['student_scheduler']['value']['last_epoch']==3
    assert a['state']['fake_scheduler']['value']['last_epoch']==15
    manual=json.loads(Path(config).read_text())
    manual['execution']=dict(mode='manual',implementation=dict(backend='torch-callable-resident-v1',share_immutable=True,checkpoint_bindings=[]))
    (tmp_path/'sharing.json').write_text(json.dumps(manual))
    shared_first=run_cli('train',tmp_path/'sharing.json','--cycles',1,'--output',tmp_path/'shared-first')
    shared=run_cli('train',tmp_path/'sharing.json','--cycles',3,'--output',tmp_path/'shared')
    shared_resumed=run_cli('train',tmp_path/'sharing.json','--cycles',3,'--resume',shared_first['checkpoint'],'--output',tmp_path/'shared-resumed')
    shared_payload=torch.load(Path(shared['checkpoint'])/'state.pt',weights_only=True)
    assert equal(shared_payload,torch.load(Path(shared_resumed['checkpoint'])/'state.pt',weights_only=True))
    restored,restored_dataset,restored_payload,_,restored_session=_load_for_output(load_training_config(tmp_path/'sharing.json'),shared_resumed['checkpoint'])
    from distillschedule.open_workload.torch_backend import sharing_inventory,resolve_module
    pairs=sharing_inventory(restored_session.workload,include_existing=True)
    assert restored_session.implementation['share_immutable'] and pairs
    assert any(resolve_module(restored_session.state,first[0]).get_parameter(first[1]) is
        resolve_module(restored_session.state,second[0]).get_parameter(second[1]) for first,second,_ in pairs)
    # Re-established aliases do not weaken frozen Teacher mutation checks.
    from distillschedule.training.session import build_cycle
    from distillschedule.training.dmd import build_algorithm
    mutation_workload=build_cycle(load_training_config(tmp_path/'sharing.json'),restored,restored_dataset,build_algorithm,
        cycle=3,cursor=0,versions=restored_payload['versions'])
    restored_session.bind_cycle(mutation_workload)
    with torch.no_grad(): next(restored.state['teacher'].parameters()).add_(.1)
    with pytest.raises(ValueError,match='immutable'): restored_session.run()
    for key in ('state','rng','cycle','cursor','versions'):
        assert equal(a[key],shared_payload[key]),'public immutable-sharing lifecycle must preserve '+key

    exported=run_cli('export',config,'--checkpoint',resumed['checkpoint'],'--output',tmp_path/'export')
    assert set(exported['model']['modules'])=={'student','media'}
    assert 'assets' not in exported['model']['parameters'] and 'teacher' not in exported['model']['modules']
    from distillschedule.data.flow_image_inputs import SCHEMA
    source=dict(schema_version=SCHEMA,family=assets['family'],assets=str(assets['assets']),mode='prompts',records=[
        dict(id='new-unseen-native-prompt',prompt='a blue boat on a lake',latent_shape=[1,4,4,4])])
    (tmp_path/'new.json').write_text(json.dumps(source))
    run_cli('data','prepare',tmp_path/'new.json','--output',tmp_path/'new-prepared',
        '--preparer','distillschedule.data.flow_image_inputs:prepare_inputs')
    prepared=tmp_path/'new-prepared/manifest.json'
    current=_load_for_output(load_training_config(config),resumed['checkpoint'])[0]
    feeds=PreparedDataset(prepared).sample(0)
    for module in current.state.values():
        if isinstance(module,torch.nn.Module): module.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        generated=current.generate(current.state,feeds,{})
        expected=dict(generated=generated,decoded=current.decode(generated,{}))
    # New prompt features must not be copied from the training record.
    assert not equal(feeds['context'],PreparedDataset(assets['prepared']).sample(0)['context'])
    inference=dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'],parameters=exported['model']['parameters']),algorithm=dict(name='dmd'),
        data=dict(prepared=str(prepared)),execution=dict(mode='reference'),
        trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(tmp_path/'unused')))
    (tmp_path/'inference.json').write_text(json.dumps(inference))
    hidden=assets['root'].with_name(assets['root'].name+'-hidden')
    assets['root'].rename(hidden)
    training_dirs=[]
    for name in ('partial','resumed','whole'):
        (tmp_path/name).rename(tmp_path/(name+'-hidden'));training_dirs.append(name)
    try:
        actual=run_cli('generate',tmp_path/'inference.json','--weights',tmp_path/'export','--prepared',prepared,
            '--output',tmp_path/'generation','--output-kind','image')
    finally:
        hidden.rename(assets['root'])
        for name in training_dirs: (tmp_path/(name+'-hidden')).rename(tmp_path/name)
    artifact=torch.load(actual['artifact'],weights_only=True)
    assert equal(expected,artifact),'portable output must be exact CURRENT restored Student, not base/Teacher/EMA'
    assert artifact['decoded']['image'].shape==(8,8,3)
    assert Path(actual['image']['path']).is_file()
    # The export also owns the real text and tokenizer codecs, not only a VAE.
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import read_export,restore_modules
    factory,_=load_callable(exported['model']['factory']);portable=factory(exported['model']['parameters'])
    export_root,manifest=read_export(tmp_path/'export');restore_modules(portable,export_root,manifest)
    assert equal(portable.state['media'].encode_text(['another real prompt']),current.state['media'].encode_text(['another real prompt']))
    (tmp_path/'lifecycle-proof.json').write_text(json.dumps(dict(family=assets['family'],evidence_kind='SYNTHETIC_TEST_ONLY',
        exact_complete_payload_equality=True,payload_fields=sorted(a),versions=whole['final_versions'],clocks=[3,3,15],
        current_student_portable_image_exact=True,new_prompt=True,training_assets_and_checkpoints_hidden=True,
        actual_image=actual['image'],gpu_execution='NOT_RUN',pretrained_quality='NOT_RUN'),indent=2))


def test_text_cohort_geometry_is_fixed_without_truncating_native_features(assets):
    import torch
    _,components,_,_,_=owned(assets)
    media=components.state['media']
    values=[media.encode_text([prompt]) for prompt in ('a bird','a red bird in the sky','a boat on a quiet blue lake')]
    assert len({tuple(context.shape) for context,mask in values})==1
    if assets['family']=='qwen-image':
        assert values[0][0].shape[1]==media.architecture['text_max_length']
        assert len({int(mask.sum()) for context,mask in values})==3
    with pytest.raises(ValueError,match='length|truncat'):
        media.encode_text(['unbounded '*2048])


@pytest.mark.parametrize('module', ['flux','qwen_image'])
def test_explicit_safe_storage_dtype_policy_is_not_compute_precision(tmp_path,module):
    import torch
    from distillschedule.training.flow_image import load_native
    api=importlib.import_module('distillschedule.training.'+module)
    config=dict(patch_size=1,in_channels=16,out_channels=16,num_layers=1,num_single_layers=1,
        num_attention_heads=2,attention_head_dim=16,joint_attention_dim=48,timestep_guidance_channels=16,
        mlp_ratio=2,axes_dims_rope=[4,4,4,4],guidance_embeds=False) if module=='flux' else dict(
        patch_size=2,in_channels=16,out_channels=4,num_layers=1,num_attention_heads=2,attention_head_dim=16,
        joint_attention_dim=16,axes_dims_rope=[4,6,6],guidance_embeds=False,zero_cond_t=False,
        use_additional_t_cond=False,use_layer3d_rope=False)
    native=api.classes()[0](**config).bfloat16().eval()
    native.save_pretrained(tmp_path/'bf16',safe_serialization=True)
    with pytest.raises(ValueError,match='dtype|FP32'):
        load_native(api.classes()[0],tmp_path/'bf16',10_000_000)
    loaded,_=load_native(api.classes()[0],tmp_path/'bf16',10_000_000,dtype_policy='bf16_or_fp32_to_fp32')
    assert all(p.dtype==torch.float32 for p in loaded.parameters())
    assert all(torch.equal(v.float(),loaded.state_dict()[k]) for k,v in native.state_dict().items())
    native.half().save_pretrained(tmp_path/'fp16',safe_serialization=True)
    with pytest.raises(ValueError,match='dtype|FP32'):
        load_native(api.classes()[0],tmp_path/'fp16',10_000_000,dtype_policy='bf16_or_fp32_to_fp32')


def test_unknown_native_text_config_fields_are_rejected_before_weights(assets):
    from distillschedule.training.flow_image import architecture,json_object
    api=importlib.import_module('distillschedule.training.'+('flux' if assets['family']=='flux2-klein-base' else 'qwen_image'))
    config=json_object(assets['root']/'text_encoder/config.json')
    config['unknown_ignored_text_option']=True
    with pytest.raises(ValueError,match='unknown|field'):
        architecture(api.classes()[2],config)


@pytest.mark.parametrize('spelling', ['public', 'omitted'])
def test_native_raw_default_configs_load_forward_gradient_and_portable_restore(assets, tmp_path, spelling):
    """Tiny native weights with source-derived null/omission, never public weights."""
    from copy import deepcopy
    import json
    import shutil
    import torch
    from distillschedule.data.flow_image_inputs import prepare_inputs
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training import flow_image as safe
    from distillschedule.training.inference import publish_export, read_export, restore_modules
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.session import construct
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal

    original, explicit, explicit_data, _, _ = owned(assets)
    api = explicit.state['student'].api
    root = tmp_path/'raw-default-assets'; shutil.copytree(assets['root'], root)
    manifest = json.loads((root/'assets.json').read_text())
    documents = {}
    for folder in ('base', 'fake'):
        path = root/folder/'config.json'; raw = json.loads(path.read_text())
        if assets['family'] == 'flux2-klein-base':
            if spelling == 'public': raw['out_channels'] = None
            else: raw.pop('out_channels')
        else:
            for key in ('zero_cond_t', 'use_additional_t_cond', 'use_layer3d_rope'):
                raw.pop(key)
            if spelling == 'public': raw['pooled_projection_dim'] = 768  # audited historical unused field
            else: raw.pop('guidance_embeds')
        documents[path] = (json.dumps(raw, indent=3)+'\n').encode()
        path.write_bytes(documents[path])
    for role, folder in (('student','base'), ('teacher','base'), ('fake','fake')):
        manifest['roles'][role]['files']['config.json'] = file_sha256(root/folder/'config.json')
    (root/'assets.json').write_text(json.dumps(manifest))
    source = json.loads((root/'raw.json').read_text()); source['assets'] = str(root/'assets.json')
    (root/'raw.json').write_text(json.dumps(source))
    prepared = prepare_inputs(root/'raw.json', root/'raw-default-prepared')
    config = original.to_dict()
    config['model']['parameters']['assets'] = str(root/'assets.json'); config['data']['prepared'] = str(prepared)
    config = TrainingConfig.from_dict(config, source=root/'train.json')
    components, dataset, _, _ = construct(config)
    row, explicit_row = dataset.sample(0), explicit_data.sample(0)
    role = components.state['student']; reference = explicit.state['student']
    assert all(path.read_bytes() == raw for path, raw in documents.items())
    assert role.architecture == json.loads(documents[root/'base/config.json'])
    assert equal(role.state_dict(), reference.state_dict())
    assert equal({k:v for k,v in row.items() if k != 'encoder_identity'},
                 {k:v for k,v in explicit_row.items() if k != 'encoder_identity'})
    sigma = torch.tensor([.625])
    actual = role.denoise(row['latents'], sigma, row, conditional=True)['image']
    expected = reference.denoise(explicit_row['latents'], sigma, explicit_row, conditional=True)['image']
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual.square().mean(), tuple(p for p in role.parameters() if p.requires_grad))
    b = torch.autograd.grad(expected.square().mean(), tuple(p for p in reference.parameters() if p.requires_grad))
    assert equal(a,b) and any(g.any() for g in a)
    # Unmodified native Diffusers owns the raw saved ABI; the safe loader must
    # produce identical full parameters, forward and gradients (not just LoRA).
    native_cls = api.classes()[0]
    native = native_cls.from_pretrained(root/'base', local_files_only=True).float().eval()
    loaded, retained = safe.load_native(native_cls,root/'base',10_000_000,
        config_transform=getattr(api,'transformer_config',None))
    assert retained == role.architecture and equal(native.state_dict(),loaded.state_dict())
    with torch.device('meta'):
        native_meta = native_cls.from_config(deepcopy(retained))
    effective = retained if not hasattr(api,'transformer_config') else api.transformer_config(retained)
    safe_meta = safe.shell(native_cls,effective)
    assert {k:v.shape for k,v in native_meta.state_dict().items()} == {k:v.shape for k,v in safe_meta.state_dict().items()}
    def forward(model):
        kwargs = dict(hidden_states=row['latents']['image'],encoder_hidden_states=row['context'],
            timestep=sigma,guidance=None,return_dict=False)
        kwargs.update(dict(img_ids=row['img_ids'],txt_ids=row['txt_ids']) if assets['family'] == 'flux2-klein-base' else
            dict(encoder_hidden_states_mask=row['context_mask'],img_shapes=[[(1,2,2)]]))
        return model(**kwargs)[0]
    native_value,safe_value = forward(native),forward(loaded)
    assert torch.equal(native_value,safe_value)
    a = torch.autograd.grad(native_value.square().sum(),tuple(native.parameters()),allow_unused=True)
    b = torch.autograd.grad(safe_value.square().sum(),tuple(loaded.parameters()),allow_unused=True)
    assert equal(a,b) and any(g is not None and g.any() for g in a)
    projection = components.inference_projection(components, config.model['parameters']).validate()
    assert projection.parameters['architecture'] == role.architecture
    _, identity = load_callable(projection.factory)
    model = dict(factory=projection.factory, source=identity, parameters=projection.parameters,
                 metadata=deepcopy(projection.components.metadata))
    publish_export(projection.components, tmp_path/'export', model=model,
                   identity=dict(model=identity, model_metadata=model['metadata']), cycle=0)
    factory, _ = load_callable(projection.factory)
    restored = factory(projection.parameters)
    export_root, exported = read_export(tmp_path/'export'); restore_modules(restored, export_root, exported)
    restored.state['student'].eval()
    assert restored.metadata['architecture'] == role.architecture
    assert torch.equal(actual, restored.state['student'].denoise(row['latents'],sigma,row,conditional=True)['image'])
    assert equal(components.state['media'].encode_text(['a violet kite']), restored.state['media'].encode_text(['a violet kite']))
    assert all(path.read_bytes() == raw for path, raw in documents.items())
    assert not torch.cuda.is_initialized()
    (tmp_path/'raw-config-proof.json').write_text(json.dumps(dict(
        evidence_kind='SYNTHETIC_TINY_NATIVE_CPU_ONLY', family=assets['family'], spelling=spelling,
        raw_config_sha256={str(path.relative_to(root)):file_sha256(path) for path in documents},
        raw_bytes_unchanged=True, native_safe_parameter_forward_gradient_exact=True,
        native_meta_layout_exact=True, explicit_config_lora_forward_gradient_exact=True,
        portable_raw_descriptor_and_new_text_exact=True, cuda_initialized=False,
        pretrained_execution='NOT_RUN', quality_validation='NOT_RUN'),indent=2)+'\n')


def test_native_default_admission_preserves_variant_type_and_geometry_refusals(assets):
    from copy import deepcopy
    from distillschedule.training import flow_image as safe
    module = 'flux' if assets['family'] == 'flux2-klein-base' else 'qwen_image'
    api = importlib.import_module('distillschedule.training.'+module)
    config = safe.json_object(assets['root']/'base/config.json')
    media = dict(vae=safe.json_object(assets['root']/'vae/config.json'),
                 text_encoder=safe.json_object(assets['root']/'text_encoder/config.json'), text_max_length=48)
    changes = [('patch_size', True), ('in_channels', True), ('out_channels', True),
               ('out_channels', 3), ('guidance_embeds', True), ('guidance_embeds', None),
               ('guidance_embeds', 0), ('unknown_default_field', False)]
    if module == 'flux': changes += [('out_channels', 0), ('out_channels', False), ('out_channels', '16')]
    else:
        changes += [(key, value) for key in ('zero_cond_t','use_additional_t_cond','use_layer3d_rope') for value in (True,None,0)]
        changes += [('out_channels', None), ('pooled_projection_dim', True), ('pooled_projection_dim', 769)]
    for key, value in changes:
        current = deepcopy(config); current[key] = value
        with pytest.raises(ValueError): api.validate_architecture(current,media)
    if module == 'qwen_image':
        historical = deepcopy(config); historical['pooled_projection_dim'] = 768
        with pytest.raises(ValueError,match='unknown'): safe.architecture(api.classes()[0],historical)
        assert 'pooled_projection_dim' not in api.transformer_config(historical)
        assert historical['pooled_projection_dim'] == 768


def test_invalid_native_features_and_null_encoding_are_rejected(assets):
    from copy import deepcopy
    import torch
    _,model,dataset,_,_=owned(assets)
    role=model.state['student'];good=dataset.sample(0)
    for kind in ('wrong-family-same-latent-shape','missing-text','invalid-mask','empty-mask','nonfinite','invented-empty','heterogeneous-geometry','causal-cache'):
        row=deepcopy(good)
        if kind=='wrong-family-same-latent-shape': row['encoder_identity'][0]^=1
        elif kind=='missing-text': del row['context']
        elif kind=='invalid-mask': row['context_mask']=row['context_mask'].float()
        elif kind=='empty-mask': row['context_mask'].zero_()
        elif kind=='nonfinite': row['context'][0,0,0]=float('nan')
        elif kind=='invented-empty': row['null_context'].zero_()
        elif kind=='heterogeneous-geometry': row['img_ids' if 'img_ids' in row else 'img_shapes'][...,1]+=1
        else: row['kv_cache']=torch.zeros(1)
        with pytest.raises(ValueError): role.validate_feeds(row)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='AMP|FP32'): role.denoise(good['latents'],torch.tensor([.5]),good,conditional=True)


def test_preencoded_batch_two_rejected_before_training(assets):
    from copy import deepcopy
    _, model, dataset, _, _ = owned(assets)
    role = model.state['student']
    row = deepcopy(dataset.sample(0))
    row['latent_shape'][0] = 2
    row['latents']['image'] = row['latents']['image'].repeat(2, 1, 1)
    for key in ('context', 'null_context'):
        row[key] = row[key].repeat(2, 1, 1)
    for key in ('context_mask', 'null_context_mask'):
        row[key] = row[key].repeat(2, 1)
    row.update(role.api.geometry(row, shape=row['latent_shape'].tolist(),
                                context=row['context'], null_context=row['null_context']))
    with pytest.raises(ValueError, match='batch-one'):
        role.validate_feeds(row)


def test_native_sharded_safe_weights_complete_missing_and_malformed(assets,tmp_path):
    import json
    import torch
    from safetensors.torch import load_file,save_file
    from distillschedule.training.flow_image import load_native
    api=importlib.import_module('distillschedule.training.'+('flux' if assets['family']=='flux2-klein-base' else 'qwen_image'))
    native,_=load_native(api.classes()[0],assets['root']/'base',10_000_000)
    native.save_pretrained(tmp_path/'shards',safe_serialization=True,max_shard_size='10KB')
    loaded,_=load_native(api.classes()[0],tmp_path/'shards',10_000_000)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in native.state_dict().items())
    weights=sorted((tmp_path/'shards').glob('*.safetensors'));assert len(weights)>1
    weights[-1].unlink()
    with pytest.raises(ValueError,match='shard|index'): load_native(api.classes()[0],tmp_path/'shards',10_000_000)
    native.save_pretrained(tmp_path/'malformed',safe_serialization=True)
    path=next((tmp_path/'malformed').glob('*.safetensors'))
    payload=load_file(path);key=next(iter(payload));payload[key]=torch.zeros(1)
    save_file(payload,path)
    with pytest.raises(ValueError,match='shape|dtype'): load_native(api.classes()[0],tmp_path/'malformed',10_000_000)
    with pytest.raises(ValueError,match='budget'): load_native(api.classes()[0],assets['root']/'base',1)


def test_asset_capabilities_and_source_rejected_before_native_load(assets,tmp_path,monkeypatch):
    import json
    import shutil
    from distillschedule.training import flow_image as common
    from distillschedule.training.config import load_training_config
    from distillschedule.workloads.best_known.registry import file_sha256
    from copy import deepcopy
    root=tmp_path/'bundle';shutil.copytree(assets['root'],root)
    manifest=json.loads((root/'assets.json').read_text())
    parameters=dict(load_training_config(assets['config']).model['parameters'],assets=str(root/'assets.json'))
    api=importlib.import_module('distillschedule.training.'+('flux' if assets['family']=='flux2-klein-base' else 'qwen_image'))
    def forbidden(*args,**kwargs): raise AssertionError('native weights were loaded before config/capability admission')
    monkeypatch.setattr(common,'load_native',forbidden)
    for kind in ('source','family','guidance','unknown'):
        current=deepcopy(manifest)
        path=root/'base/config.json';config=json.loads((assets['root']/'base/config.json').read_text())
        if kind=='source':current['source']['revision']='0'*40
        elif kind=='family':current['family']='some-other-image-family'
        else:
            config['guidance_embeds' if kind=='guidance' else 'unknown_ignored_option']=True
        path.write_text(json.dumps(config))
        for role in ('student','teacher'):current['roles'][role]['files']['config.json']=file_sha256(path)
        if kind in ('guidance','unknown'):
            (root/'fake/config.json').write_text(json.dumps(config));current['roles']['fake']['files']['config.json']=file_sha256(root/'fake/config.json')
        (root/'assets.json').write_text(json.dumps(current))
        with pytest.raises(ValueError):api.build_model(parameters)


def test_preencoded_happy_path_and_transactional_invalid_binding(assets,tmp_path):
    from copy import deepcopy
    import json
    import torch
    from distillschedule.data import PreparedDataset
    from distillschedule.data.flow_image_inputs import SCHEMA,prepare_inputs
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal
    good=PreparedDataset(assets['prepared']).sample(0)
    torch.save(good,tmp_path/'good.pt')
    manifest=dict(schema_version=SCHEMA,family=assets['family'],assets=str(assets['assets']),mode='preencoded',
        records=[dict(id='actual-native-preencoded',features=dict(path='good.pt',sha256=file_sha256(tmp_path/'good.pt')))])
    (tmp_path/'source.json').write_text(json.dumps(manifest))
    prepared=prepare_inputs(tmp_path/'source.json',tmp_path/'good-prepared')
    dataset=PreparedDataset(prepared)
    assert equal(good,dataset.sample(0)) and dataset.validate(sources=True)['source_validation']=='PASS'
    invalid=deepcopy(good);invalid['encoder_identity'][0]^=1
    torch.save(invalid,tmp_path/'invalid.pt')
    manifest['records'][0]['features']=dict(path='invalid.pt',sha256=file_sha256(tmp_path/'invalid.pt'))
    (tmp_path/'source.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='binding|stale'):
        prepare_inputs(tmp_path/'source.json',tmp_path/'invalid-prepared')
    assert not (tmp_path/'invalid-prepared').exists()
    assert not list(tmp_path.glob('.invalid-prepared*'))


def test_stale_actual_codec_and_cfg_recipe_rejected(assets,tmp_path):
    import json
    import shutil
    from copy import deepcopy
    import torch
    from safetensors.torch import load_file,save_file
    from distillschedule.training.config import load_training_config,TrainingConfig
    from distillschedule.training.session import construct
    from distillschedule.workloads.best_known.registry import file_sha256
    original=load_training_config(assets['config'])
    wrong=original.to_dict();wrong['algorithm']['parameters']['teacher_cfg']=2.
    with pytest.raises(ValueError,match='CFG|cfg'):
        construct(TrainingConfig.from_dict(wrong,source=original.source))
    root=tmp_path/'bundle';shutil.copytree(assets['root'],root)
    manifest=json.loads((root/'assets.json').read_text())
    weight=next((root/'vae').glob('*.safetensors'))
    payload=load_file(weight);key=next(k for k,v in payload.items() if v.is_floating_point())
    payload[key]=payload[key]+.01;save_file(payload,weight)
    manifest['media']['vae']['files'][weight.name]=file_sha256(weight)
    (root/'assets.json').write_text(json.dumps(manifest))
    wrong=original.to_dict();wrong['model']['parameters']['assets']=str(root/'assets.json')
    with pytest.raises(ValueError,match='codec|binding|family'):
        construct(TrainingConfig.from_dict(wrong,source=original.source))
