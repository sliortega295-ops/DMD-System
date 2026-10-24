"""SYNTHETIC_TEST_ONLY: actual pinned Wan2.2 native CPU lifecycle, not quality."""
import importlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='requires opt-in isolated modern-flow native CPU stack')


def test_actual_native_wan22_has_common_flow_adapter(tmp_path):
    import torch
    from diffusers import WanTransformer3DModel
    native = WanTransformer3DModel(patch_size=(1,2,2), in_channels=4, out_channels=4,
        num_layers=1, num_attention_heads=2, attention_head_dim=16, text_dim=16,
        freq_dim=16, ffn_dim=64, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=16).float().eval()
    native.save_pretrained(tmp_path/'transformer', safe_serialization=True)
    api = importlib.import_module('distillschedule.training.wan22')
    assert callable(api.build_model) and callable(api.build_inference_model)
    assert not torch.cuda.is_initialized()


def test_safe_loader_legacy_config_normalization_keeps_raw_source(tmp_path):
    import json
    import torch
    from diffusers import AutoencoderKLWan
    from distillschedule.training.flow_image import load_native
    native = AutoencoderKLWan(base_dim=4, decoder_base_dim=4, z_dim=4, dim_mult=[1,2,2,2],
        num_res_blocks=1, temperal_downsample=[False,True,True], latents_mean=[.1,-.1,.2,-.2],
        latents_std=[.8,.9,1.1,1.2], is_residual=True, in_channels=12, out_channels=12,
        patch_size=2, scale_factor_temporal=4, scale_factor_spatial=16).float().eval()
    native.save_pretrained(tmp_path/'vae',safe_serialization=True)
    path=tmp_path/'vae/config.json'; config=json.loads(path.read_text()); config['clip_output']=False
    path.write_text(json.dumps(config)); raw=path.read_bytes()
    with pytest.raises(ValueError,match='unknown|constructor'):
        load_native(AutoencoderKLWan,tmp_path/'vae',10_000_000)
    def normalize(value):
        assert value['clip_output'] is False
        return {k:v for k,v in value.items() if k!='clip_output'}
    loaded, original=load_native(AutoencoderKLWan,tmp_path/'vae',10_000_000,config_transform=normalize)
    assert original==config and path.read_bytes()==raw
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in native.state_dict().items())


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    from pathlib import Path
    from distillschedule.training.plugins import load_callable
    builder,_=load_callable(str(Path(__file__).resolve().parents[1]/'recipes/wan22_dmd/create_cpu_fixture.py')+':build_fixture')
    return builder(tmp_path_factory.mktemp('native-wan22')/'fixture')


def owned(assets):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    config=load_training_config(assets['config'])
    return config,*construct(config)


def test_actual_native_forward_text_codec_and_manual_euler(assets, record_property):
    import torch
    from diffusers import WanTransformer3DModel,AutoencoderKLWan,WanPipeline
    from transformers import UMT5EncoderModel,T5TokenizerFast
    from distillschedule.training.prediction import PredictionSpec
    from distillschedule.data.decoding import _video
    config,components,dataset,_,_=owned(assets)
    role=components.state['student'];media=components.state['media'];row=dataset.sample(0)
    assert type(role.native) is WanTransformer3DModel and type(media.vae) is AutoencoderKLWan
    assert type(media.text_encoder) is UMT5EncoderModel and type(media.tokenizer) is T5TokenizerFast
    assert components.prediction==PredictionSpec('flow')
    assert row['latents']['video'].shape==(1,4,2,2,2) and row['context'].shape==(1,226,16)
    assert not components.state['teacher'].training and role.native.training and not media.training
    assert not any(p.requires_grad for p in media.parameters())
    assert media.vae.config.is_residual and media.vae.config.patch_size==2
    empty=media.encode_text([''])
    assert torch.equal(row['null_context'],empty[0]) and torch.equal(row['null_context_mask'],empty[1])
    assert torch.count_nonzero(empty[0])>0
    assert not row['context'][~row['context_mask']].any()
    assert not row['null_context'][~row['null_context_mask']].any()
    native_context=media.pipeline()._get_t5_prompt_embeds(['a bird'],max_sequence_length=226,device=torch.device('cpu'))
    assert torch.equal(native_context,row['context'])
    # Actual native cleaner, including unicode repair and HTML whitespace.
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    prompt='  a &amp;amp; bird\n  flies '
    assert torch.equal(media.encode_text([prompt])[0],media.pipeline()._get_t5_prompt_embeds([prompt],max_sequence_length=226,device=torch.device('cpu')))
    assert prompt_clean(prompt)=='a & bird flies'
    sigma=torch.tensor([.625]);sample=row['latents']
    seen=[]
    hook=role.native.register_forward_pre_hook(lambda m,a,k:seen.append(k),with_kwargs=True)
    actual=role.denoise(sample,sigma,row,conditional=True)['video'];hook.remove()
    assert seen[0]['timestep'].shape==(1,2) and torch.equal(seen[0]['timestep'],torch.full((1,2),625.))
    native=role.native(hidden_states=sample['video'],timestep=(1000*sigma[:,None]).expand(1,2),encoder_hidden_states=row['context'],return_dict=False)[0]
    assert torch.equal(actual,native)
    assert torch.equal(PredictionSpec('flow').to_x0(actual,sample['video'],sigma=sigma[:,None,None,None,None]),sample['video']-sigma[:,None,None,None,None]*native)
    torch.manual_seed(909)
    rolled=role.rollout(row,[1.,.75,.5,.25,0.],'all_steps')['video']
    torch.manual_seed(909);manual=torch.randn_like(sample['video'])
    for s,n in zip([1.,.75,.5,.25],[.75,.5,.25,0.]):
        v=role.native(hidden_states=manual,timestep=torch.full((1,2),1000*s),encoder_hidden_states=row['context'],return_dict=False)[0]
        manual=manual+(n-s)*v
    assert torch.equal(rolled,manual),'native velocity sign and four-step Euler must be exact'
    frames=_video(assets['root']/'frames')
    encoded=media.encode_video(frames)
    pixels=frames.permute(3,0,1,2).unsqueeze(0).float()/127.5-1
    with torch.no_grad():
        raw=media.vae.encode(pixels).latent_dist.mode()
        normalized=(raw-raw.new_tensor(media.vae.config.latents_mean).view(1,-1,1,1,1))/raw.new_tensor(media.vae.config.latents_std).view(1,-1,1,1,1)
        decoded=media.vae.decode(raw,return_dict=False)[0]
        owned_decode=media.decode_latents(encoded)
        # Compare the identical native input, not raw versus a FP32
        # normalization/inverse-normalization roundtrip. These differ by ULPs.
        native_decode=media.vae.decode(encoded*encoded.new_tensor(media.vae.config.latents_std).view(1,-1,1,1,1)
            +encoded.new_tensor(media.vae.config.latents_mean).view(1,-1,1,1,1),return_dict=False)[0]
    assert torch.equal(normalized,encoded) and torch.equal(encoded,row['latents']['video'])
    assert torch.equal(owned_decode,native_decode)
    record_property('raw_roundtrip_decode_max_abs',float((owned_decode-decoded).abs().max()))
    assert media.decode({'video_latents':encoded},{})['video'].shape==(5,32,32,3)
    assert not torch.cuda.is_initialized()


def test_common_real_g1_f5_gradients_ownership_and_immutable_sharing(assets,tmp_path):
    from copy import deepcopy
    import json
    import torch
    from distillschedule.training.session import build_cycle
    from distillschedule.open_workload.torch_backend import Session,sharing_inventory,resolve_module
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    results=[];gradients=[]
    for share in (False,True):
        config,model,dataset,algorithm,_=owned(assets)
        teacher=deepcopy(model.state['teacher'].state_dict())
        before={r:deepcopy(model.state[r].state_dict()) for r in ('student','fake')}
        captured={r:[] for r in ('student','fake')}
        trainables={r:{id(p) for p in model.state[r].parameters() if p.requires_grad} for r in captured}
        assert not trainables['student']&trainables['fake']
        for role in captured:
            optimizer=model.state[role+'_optimizer']
            assert {id(p) for g in optimizer.param_groups for p in g['params']}==trainables[role]
            for parameter in model.state[role].parameters():
                if parameter.requires_grad:parameter.register_hook(lambda g,role=role:captured[role].append(g.detach().clone()))
        workload=build_cycle(config,model,dataset,algorithm,cycle=0,cursor=0)
        pairs=sharing_inventory(workload);assert pairs
        first,second,_=pairs[0]
        source=resolve_module(model.state,first[0]).get_parameter(first[1])
        target=resolve_module(model.state,second[0]).get_parameter(second[1])
        assert source is not target
        session=Session(workload,dict(backend='torch-callable-resident-v1',share_immutable=share,checkpoint_bindings=[]),copy_state=False)
        assert (source is resolve_module(session.state,second[0]).get_parameter(second[1])) is share
        output=session.run()
        assert all(torch.isfinite(v) for v in output.values())
        assert model.state['dmd_counters'].tolist()==[1,1,5]
        assert session.receipt['dispatcher']['final_versions']['fake']==5
        assert equal(teacher,model.state['teacher'].state_dict())
        assert all(p.grad is None and not p.requires_grad for p in model.state['teacher'].parameters())
        for role in captured:
            assert captured[role] and any(torch.count_nonzero(g)>0 for g in captured[role])
            assert any(not torch.equal(v,before[role][k]) for k,v in model.state[role].state_dict().items())
            assert all(torch.equal(p,before[role][k]) for k,p in model.state[role].named_parameters() if not p.requires_grad)
            opt=model.state[role+'_optimizer']
            count=1 if role=='student' else 5
            assert opt.state and all(float(v['step'])==count and torch.isfinite(v['exp_avg']).all() for v in opt.state.values())
            assert model.state[role+'_scheduler'].last_epoch==count
        gradients.append(captured)
        results.append(dict(packed=pack_state(model.state),output={k:v.detach().clone() for k,v in output.items()},
            storage=session.initial_storage,effects=session.effects.copy()))
    assert equal(gradients[0],gradients[1]) and equal(results[0]['packed'],results[1]['packed']) and equal(results[0]['output'],results[1]['output'])
    assert results[1]['storage']<results[0]['storage']
    assert results[0]['effects']['shared_parameter_aliases']==0 and results[1]['effects']['shared_parameter_aliases']>0
    (tmp_path/'sharing-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',exact_gradients_optimizer_scheduler_ema_teacher_parity=True,
        shared_parameter_aliases=results[1]['effects']['shared_parameter_aliases'],unshared_bytes=results[0]['storage'],shared_bytes=results[1]['storage'],gpu_execution='NOT_RUN'),indent=2))


def test_fresh_process_restart_export_new_input_without_training_assets(assets,tmp_path):
    from pathlib import Path
    import json
    import shutil
    import torch
    from distillschedule.data import PreparedDataset
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import _load_for_output,build_cycle
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import read_export,restore_modules
    from distillschedule.data.wan22_inputs import SCHEMA
    from test_training_end_to_end import run_cli,equal
    partial=run_cli('train',assets['config'],'--cycles',1,'--output',tmp_path/'partial')
    resumed=run_cli('train',assets['config'],'--cycles',3,'--resume',partial['checkpoint'],'--output',tmp_path/'resumed')
    whole=run_cli('train',assets['config'],'--cycles',3,'--output',tmp_path/'whole')
    a=torch.load(Path(resumed['checkpoint'])/'state.pt',weights_only=True);b=torch.load(Path(whole['checkpoint'])/'state.pt',weights_only=True)
    assert equal(a,b),'all state/Adam/schedulers/EMA/RNG/data/cursor/clocks/versions must restart exactly'
    assert a['state']['dmd_counters']['value']['counters'].tolist()==[3,3,15]
    assert a['state']['student_scheduler']['value']['last_epoch']==3 and a['state']['fake_scheduler']['value']['last_epoch']==15
    for name in ('student','fake','teacher','student_optimizer','fake_optimizer','student_scheduler','fake_scheduler','student_ema'):
        assert name in a['state']
    rows=[json.loads(line) for line in (tmp_path/'whole/metrics.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in rows]==['native-video-0','native-video-1','native-video-2']
    assert [json.loads(line)['sample_id'] for line in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]==['native-video-1','native-video-2']
    manual=json.loads(Path(assets['config']).read_text())
    manual['execution']=dict(mode='manual',implementation=dict(backend='torch-callable-resident-v1',share_immutable=True,checkpoint_bindings=[]))
    (tmp_path/'sharing.json').write_text(json.dumps(manual))
    shared_first=run_cli('train',tmp_path/'sharing.json','--cycles',1,'--output',tmp_path/'shared-first')
    shared=run_cli('train',tmp_path/'sharing.json','--cycles',3,'--output',tmp_path/'shared')
    shared_resumed=run_cli('train',tmp_path/'sharing.json','--cycles',3,'--resume',shared_first['checkpoint'],'--output',tmp_path/'shared-resumed')
    shared_payload=torch.load(Path(shared['checkpoint'])/'state.pt',weights_only=True)
    assert equal(shared_payload,torch.load(Path(shared_resumed['checkpoint'])/'state.pt',weights_only=True))
    for key in ('state','rng','cycle','cursor','versions'):assert equal(a[key],shared_payload[key]),key
    restored,restored_data,payload,_,session=_load_for_output(load_training_config(tmp_path/'sharing.json'),shared_resumed['checkpoint'])
    from distillschedule.open_workload.torch_backend import sharing_inventory,resolve_module
    pairs=sharing_inventory(session.workload,include_existing=True);assert pairs and session.implementation['share_immutable']
    assert any(resolve_module(session.state,x[0]).get_parameter(x[1]) is resolve_module(session.state,y[0]).get_parameter(y[1]) for x,y,_ in pairs)
    from distillschedule.training.dmd import build_algorithm
    workload=build_cycle(load_training_config(tmp_path/'sharing.json'),restored,restored_data,build_algorithm,cycle=3,cursor=0,versions=payload['versions'])
    session.bind_cycle(workload)
    with torch.no_grad():next(restored.state['teacher'].parameters()).add_(.1)
    with pytest.raises(ValueError,match='immutable'):session.run()
    export=run_cli('export',assets['config'],'--checkpoint',resumed['checkpoint'],'--output',tmp_path/'export')
    assert set(export['model']['modules'])=={'student','media'} and 'assets' not in export['model']['parameters']
    new_source=dict(schema_version=SCHEMA,family='wan22-ti2v5b-t2v',assets=str(assets['assets']),mode='prompts',
        records=[dict(id='new-unseen-video',prompt='a little dog rests by a river',latent_shape=[1,4,2,2,2])])
    (tmp_path/'new.json').write_text(json.dumps(new_source))
    run_cli('data','prepare',tmp_path/'new.json','--output',tmp_path/'new-prepared','--preparer','distillschedule.data.wan22_inputs:prepare_inputs')
    prepared=tmp_path/'new-prepared/manifest.json';feeds=PreparedDataset(prepared).sample(0)
    current=_load_for_output(load_training_config(assets['config']),resumed['checkpoint'])[0]
    assert not equal(feeds['context'],PreparedDataset(assets['prepared']).sample(0)['context'])
    for module in current.state.values():
        if isinstance(module,torch.nn.Module):module.eval()
    torch.manual_seed(909)
    with torch.no_grad():
        generated=current.generate(current.state,feeds,{})
        expected=dict(generated=generated,decoded=current.decode(generated,{}))
    factory,_=load_callable(export['model']['factory']);portable=factory(export['model']['parameters'])
    export_root,manifest=read_export(tmp_path/'export');restore_modules(portable,export_root,manifest)
    assert equal(portable.state['media'].encode_text(['a fresh prompt']),current.state['media'].encode_text(['a fresh prompt']))
    assert type(portable.state['media'].tokenizer) is type(current.state['media'].tokenizer)
    assert portable.state['media'].tokenizer._extra_ids==current.state['media'].tokenizer._extra_ids
    inference=dict(schema_version='distillschedule.training.v1',model=dict(factory=export['model']['factory'],parameters=export['model']['parameters']),
        algorithm=dict(name='dmd'),data=dict(prepared=str(prepared)),execution=dict(mode='reference'),trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(tmp_path/'unused')))
    (tmp_path/'inference.json').write_text(json.dumps(inference))
    # Keep receipt evidence, but remove ALL original training assets/checkpoints
    # during portable generation, not merely Teacher/Fake names in a descriptor.
    shutil.copytree(resumed['checkpoint'],tmp_path/'verified-checkpoint')
    for name in ('partial','resumed','whole','shared-first','shared','shared-resumed'):
        shutil.copyfile(tmp_path/name/'metrics.jsonl',tmp_path/(name+'-metrics.jsonl'))
        shutil.rmtree(tmp_path/name)
    hidden=assets['root'].with_name(assets['root'].name+'-hidden');assets['root'].rename(hidden)
    try:
        actual=run_cli('generate',tmp_path/'inference.json','--weights',tmp_path/'export','--prepared',prepared,'--output',tmp_path/'generation','--output-kind','video')
    finally:hidden.rename(assets['root'])
    artifact=torch.load(actual['artifact'],weights_only=True)
    assert equal(expected,artifact),'portable decoded video must match exact CURRENT Student, not base/Teacher/EMA'
    assert artifact['decoded']['video'].shape==(5,32,32,3) and artifact['decoded']['video'].dtype==torch.uint8
    assert len(list(Path(actual['video']['frames']).glob('*.png')))==5
    (tmp_path/'lifecycle-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',exact_complete_state_equality=True,
        clocks=[3,3,15],sample_ids=[r['sample_id'] for r in rows],versions=whole['final_versions'],export=str(tmp_path/'export'),
        verified_checkpoint=str(tmp_path/'verified-checkpoint'),new_prepared=str(prepared),current_student_portable_video_exact=True,
        assets_hidden_and_training_checkpoints_deleted=True,actual_video=actual['video'],gpu_execution='NOT_RUN',quality_validation='NOT_RUN'),indent=2))


def test_tokenizer_native_extra_ids_and_backend_options_restore_exactly():
    import torch
    from transformers import T5TokenizerFast
    from distillschedule.training.wan22 import tokenizer_state,restore_tokenizer
    tokenizer=T5TokenizerFast(vocab=[('<pad>',0.),('</s>',0.),('<unk>',0.),('▁',-2.),('a',-2.),('<extra_id_1>',0.),('<extra_id_0>',0.)],
        extra_ids=2,model_max_length=512,clean_up_tokenization_spaces=True)
    tokenizer(['a',''],padding='max_length',max_length=226,return_tensors='pt')
    state=tokenizer_state(tokenizer);restored=restore_tokenizer(state)
    assert type(restored) is type(tokenizer) and restored._extra_ids==2 and tokenizer_state(restored)==state
    assert torch.equal(restored(['a <extra_id_0>'],return_tensors='pt').input_ids,tokenizer(['a <extra_id_0>'],return_tensors='pt').input_ids)
    wrong=dict(state,**{'class':'transformers.Qwen2TokenizerFast'})
    with pytest.raises(ValueError,match='T5 tokenizer'):restore_tokenizer(wrong)


def test_prompt_length_native_zero_padding_and_refusals(assets):
    import torch
    from copy import deepcopy
    _,model,dataset,_,_=owned(assets);role=model.state['student'];media=model.state['media'];good=dataset.sample(0)
    values=[media.encode_text([p]) for p in ('a bird','a red bird in the sky','a boat on a quiet blue lake')]
    assert all(v.shape==(1,226,16) for v,m in values)
    assert len({int(m.sum()) for v,m in values})==3
    with pytest.raises(ValueError,match='overflow|truncat'):media.encode_text(['unbounded '*256])
    for kind in ('shape','batch','missing','nonfinite','mask','empty-mask','padded','invented-empty','binding','i2v','causal'):
        row=deepcopy(good)
        if kind=='shape':row['latent_shape'][3]=3
        elif kind=='batch':row['latent_shape'][0]=2
        elif kind=='missing':del row['context']
        elif kind=='nonfinite':row['latents']['video'][0,0,0,0,0]=float('nan')
        elif kind=='mask':row['context_mask']=row['context_mask'].float()
        elif kind=='empty-mask':row['context_mask'].zero_()
        elif kind=='padded':row['context'][0,-1,0]=1.
        elif kind=='invented-empty':row['null_context'].zero_()
        elif kind=='binding':row['encoder_identity'][0]^=1
        elif kind=='i2v':row['image']=torch.zeros(1)
        else:row['kv_cache']=torch.zeros(1)
        with pytest.raises(ValueError):role.validate_feeds(row)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='FP32|AMP'):role.denoise(good['latents'],torch.tensor([.5]),good,conditional=True)
    with pytest.raises(ValueError,match='geometry|resample'):
        media.encode_video(torch.zeros(4,32,32,3,dtype=torch.uint8))
    with pytest.raises(ValueError,match='geometry|resize'):
        media.encode_video(torch.zeros(5,16,32,3,dtype=torch.uint8))
    with pytest.raises(ValueError,match='native RoPE'):
        from distillschedule.data.wan22_inputs import validate_shape
        validate_shape([1,4,17,2,2],role.architecture)


def test_preencoded_exact_and_transactional_invalid_inputs(assets,tmp_path):
    import json
    import torch
    from copy import deepcopy
    from distillschedule.data import PreparedDataset
    from distillschedule.data.wan22_inputs import SCHEMA,prepare_inputs
    from distillschedule.training.wan22 import FAMILY
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal
    good=PreparedDataset(assets['prepared']).sample(0)
    torch.save(good,tmp_path/'good.pt')
    manifest=dict(schema_version=SCHEMA,family=FAMILY,assets=str(assets['assets']),mode='preencoded',
        records=[dict(id='genuine-native',features=dict(path='good.pt',sha256=file_sha256(tmp_path/'good.pt')))])
    source=tmp_path/'source.json';source.write_text(json.dumps(manifest))
    prepared=prepare_inputs(source,tmp_path/'preencoded')
    assert equal(good,PreparedDataset(prepared).sample(0))
    assert PreparedDataset(prepared).validate(sources=True)['source_validation']=='PASS'
    bad=deepcopy(good);bad['encoder_identity'][0]^=1;torch.save(bad,tmp_path/'bad.pt')
    manifest['records'][0]['features']=dict(path='bad.pt',sha256=file_sha256(tmp_path/'bad.pt'))
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='binding|stale'):prepare_inputs(source,tmp_path/'bad-prepared')
    assert not (tmp_path/'bad-prepared').exists() and not list(tmp_path.glob('.bad-prepared*'))
    for kind in ('shape','image','missing-prompt','duplicate','sha'):
        value=dict(schema_version=SCHEMA,family=FAMILY,assets=str(assets['assets']),mode='prompts',
            records=[dict(id='new',prompt='a bird',latent_shape=[1,4,2,2,2])])
        if kind=='shape':value['records'][0].pop('latent_shape')
        elif kind=='image':value['records'][0]['image']='not-i2v'
        elif kind=='missing-prompt':value['records'][0].pop('prompt')
        elif kind=='duplicate':value['records']*=2
        else:
            value['mode']='preencoded';value['records']=[dict(id='bad-hash',features=dict(path='good.pt',sha256='0'*64))]
        source.write_text(json.dumps(value))
        with pytest.raises(ValueError):prepare_inputs(source,tmp_path/('refused-'+kind))
        assert not (tmp_path/('refused-'+kind)).exists()


@pytest.mark.parametrize('kind',['source','family','schema','i2v','experts','clip','unknown','budget'])
def test_source_variant_schema_and_budget_refused_before_weights(assets,tmp_path,monkeypatch,kind):
    from pathlib import Path
    from copy import deepcopy
    import json
    import shutil
    from distillschedule.training import wan22 as api
    from distillschedule.training import flow_image as safe
    from distillschedule.training.config import load_training_config
    from distillschedule.workloads.best_known.registry import file_sha256
    root=tmp_path/'bundle';shutil.copytree(assets['root'],root)
    manifest=json.loads((root/'assets.json').read_text())
    if kind=='source':manifest['source']['revision']='0'*40
    elif kind=='family':manifest['family']='wan21-t2v'
    elif kind=='schema':manifest['schema_version']='distillschedule.flow_image_assets.v1'
    elif kind=='i2v':manifest['pipeline']['task']='i2v'
    elif kind=='experts':manifest['pipeline']['transformer_2']='high-low-experts'
    elif kind=='budget':manifest['roles']['teacher']['max_parameters']=1
    elif kind=='clip':
        path=root/'vae/config.json';value=json.loads(path.read_text());value['clip_output']=True
        path.write_text(json.dumps(value));manifest['media']['vae']['files']['config.json']=file_sha256(path)
    else:
        for role in ('base','fake'):
            path=root/role/'config.json';value=json.loads(path.read_text());value['ignored_unknown_field']=True;path.write_text(json.dumps(value))
        for role,folder in (('student','base'),('teacher','base'),('fake','fake')):
            manifest['roles'][role]['files']['config.json']=file_sha256(root/folder/'config.json')
    (root/'assets.json').write_text(json.dumps(manifest))
    parameters=deepcopy(load_training_config(assets['config']).model['parameters']);parameters['assets']=str(root/'assets.json')
    def forbidden(*a,**k):raise AssertionError('loaded weights before config/capability/budget admission')
    monkeypatch.setattr(safe,'load_native',forbidden)
    with pytest.raises(ValueError):api.build_model(parameters)


def test_native_safe_shards_shape_dtype_and_invented_weight_refusals(assets,tmp_path):
    import torch
    from safetensors.torch import load_file,save_file
    from distillschedule.training import wan22 as api
    from distillschedule.training.flow_image import load_native
    cls=api.classes()[0]
    native,_=load_native(cls,assets['root']/'base',10_000_000)
    native.save_pretrained(tmp_path/'shards',safe_serialization=True,max_shard_size='10KB')
    loaded,_=load_native(cls,tmp_path/'shards',10_000_000)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in native.state_dict().items())
    shards=sorted((tmp_path/'shards').glob('*.safetensors'));assert len(shards)>1;shards[-1].unlink()
    with pytest.raises(ValueError,match='shard|index'):load_native(cls,tmp_path/'shards',10_000_000)
    native.save_pretrained(tmp_path/'badshape',safe_serialization=True)
    path=next((tmp_path/'badshape').glob('*.safetensors'));state=load_file(path);key=next(iter(state));state[key]=torch.zeros(1);save_file(state,path)
    with pytest.raises(ValueError,match='shape|dtype'):load_native(cls,tmp_path/'badshape',10_000_000)
    native.half().save_pretrained(tmp_path/'fp16',safe_serialization=True)
    with pytest.raises(ValueError,match='dtype|FP32'):load_native(cls,tmp_path/'fp16',10_000_000,dtype_policy='bf16_or_fp32_to_fp32')
    native.bfloat16().save_pretrained(tmp_path/'bf16',safe_serialization=True)
    loaded,_=load_native(cls,tmp_path/'bf16',10_000_000,dtype_policy='bf16_or_fp32_to_fp32')
    assert all(torch.equal(v.float(),loaded.state_dict()[k]) for k,v in native.state_dict().items())
    with pytest.raises(ValueError,match='budget'):load_native(cls,assets['root']/'base',1)


def test_wrong_codec_and_algorithm_recipe_are_not_silently_admitted(assets,tmp_path):
    from copy import deepcopy
    import json
    import shutil
    from safetensors.torch import load_file,save_file
    from distillschedule.training.config import TrainingConfig,load_training_config
    from distillschedule.training.session import construct
    from distillschedule.workloads.best_known.registry import file_sha256
    original=load_training_config(assets['config'])
    for parameter,value in (('teacher_cfg',2.),('fake_updates',4),('rollout_sigmas',[1.,.5,0.])):
        wrong=original.to_dict();wrong['algorithm']['parameters'][parameter]=value
        with pytest.raises(ValueError,match='cfg|G1/F5|Euler'):construct(TrainingConfig.from_dict(wrong,source=original.source))
    for key,value in (('device','cuda:-1'),('experts',True),('precision','bf16'),('task','i2v'),('seed',True)):
        wrong=original.to_dict();wrong['model']['parameters'][key]=value
        with pytest.raises(ValueError):construct(TrainingConfig.from_dict(wrong,source=original.source))
    root=tmp_path/'changed-codec';shutil.copytree(assets['root'],root)
    manifest=json.loads((root/'assets.json').read_text());path=next((root/'vae').glob('*.safetensors'))
    state=load_file(path);key=next(k for k,v in state.items() if v.is_floating_point());state[key]=state[key]+.01;save_file(state,path)
    manifest['media']['vae']['files'][path.name]=file_sha256(path);(root/'assets.json').write_text(json.dumps(manifest))
    wrong=original.to_dict();wrong['model']['parameters']['assets']=str(root/'assets.json')
    with pytest.raises(ValueError,match='codec|binding'):construct(TrainingConfig.from_dict(wrong,source=original.source))


def test_fresh_process_wrong_seed_and_corrupt_checkpoint_are_rejected(assets,tmp_path):
    from pathlib import Path
    import json
    import subprocess
    import sys
    from test_training_end_to_end import run_cli
    first=run_cli('train',assets['config'],'--cycles',1,'--output',tmp_path/'first')
    value=json.loads(Path(assets['config']).read_text());value['trainer']['seed']+=1
    path=tmp_path/'wrong-seed.json';path.write_text(json.dumps(value))
    def refused(config,output):
        result=subprocess.run([sys.executable,'-m','distillschedule','train',str(config),'--cycles','3','--resume',first['checkpoint'],'--output',str(output)],capture_output=True,text=True)
        assert result.returncode!=0
        assert not list(output.glob('checkpoints/*/manifest.json'))
        return result.stdout+result.stderr
    assert 'identity' in refused(path,tmp_path/'wrong-seed').lower()
    payload=Path(first['checkpoint'])/'state.pt';payload.write_bytes(payload.read_bytes()+b'corrupt')
    assert 'hash' in refused(assets['config'],tmp_path/'corrupt').lower()


def test_lazy_catalog_compatibility_without_torch_cuda_or_assets():
    import subprocess
    import sys
    script='''
import sys
from distillschedule.training.catalog import describe,resolve,validate_compatibility
row=resolve('model','wan22-ti2v5b-t2v')
assert row['compatibility']['batch_size']==1 and row['compatibility']['precision']=='FP32'
assert row['validation']['gpu']=='NOT_RUN'
assert row['source']['revision']=='b8fff7315c768468a5333511427288870b2e9635'
for algorithm,trainer in [('distillschedule.training.dmd2:build_dmd2',{'device':'cpu'}),('distillschedule.training.self_forcing:build_self_forcing',{'device':'cpu'}),('distillschedule.training.dmd:build_algorithm',{'device':'cpu','distributed':{}})]:
    try:validate_compatibility({'factory':row['factory'],'parameters':{}},{'factory':algorithm},trainer)
    except ValueError:pass
    else:raise AssertionError('unsupported capability admitted')
validate_compatibility({'factory':row['factory'],'parameters':{}},{'factory':'distillschedule.training.dmd:build_algorithm'},{'device':'cuda:0'},{'mode':'reference'})
assert 'torch' not in sys.modules and 'diffusers' not in sys.modules and 'transformers' not in sys.modules
print('PASS_LAZY_NO_MODEL_TORCH_CUDA_OR_ASSET_LOAD')
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_pinned_full_upstream_configs_admit_only_source_meta_not_weights():
    from pathlib import Path
    import json
    from distillschedule.training import wan22 as api
    from distillschedule.training.flow_image import shell
    source=json.loads((Path(__file__).resolve().parents[1]/'recipes/wan22_dmd/upstream-native-configs.json').read_text())
    assert source['model_id']==api.MODEL_ID and source['revision']==api.REVISION
    config=source['configs']['transformer/config.json']
    media=dict(vae=source['configs']['vae/config.json'],text_encoder=source['text_encoder'],text_max_length=226)
    api.validate_architecture(config,media)
    assert config['in_channels']==media['vae']['z_dim']==48 and config['num_layers']==30
    native=shell(api.classes()[0],config)
    assert all(p.is_meta for p in native.parameters())
    assert sum(p.numel() for p in native.parameters())>1_000_000_000
    vae=shell(api.classes()[1],api.vae_config(media['vae']))
    assert all(p.is_meta for p in vae.parameters())
    assert media['vae']['clip_output'] is False,'raw original constructor intent retained'


def test_portable_inference_budget_admission_precedes_concrete_layout(assets,monkeypatch):
    from copy import deepcopy
    from functools import wraps
    import torch
    from distillschedule.training import wan22 as api
    _,components,_,_,_=owned(assets)
    descriptor=deepcopy(components.inference_projection(components,{}).parameters)
    descriptor['cpu_parameter_limits']['vae']=1
    cls=api.classes()[0];original=cls.__init__
    @wraps(original)
    def counted(self,*args,**kwargs):
        assert torch.get_default_device().type=='meta','concrete constructor ran before portable CPU budget admission'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(cls,'__init__',counted)
    with pytest.raises(ValueError,match='portable.*budget'):api.build_inference_model(descriptor)
