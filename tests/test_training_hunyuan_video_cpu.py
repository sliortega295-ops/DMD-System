"""SYNTHETIC_TEST_ONLY: actual HunyuanVideo1.5 T2V CPU contracts, not quality."""
import importlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='requires opt-in isolated modern-flow native CPU stack')


def test_actual_native_hunyuan15_has_common_flow_adapter(tmp_path):
    import torch
    from diffusers import HunyuanVideo15Transformer3DModel
    native = HunyuanVideo15Transformer3DModel(in_channels=9, out_channels=4,
        num_layers=1, num_refiner_layers=1, num_attention_heads=2, attention_head_dim=16,
        mlp_ratio=2, patch_size=1, patch_size_t=1, text_embed_dim=16,
        text_embed_2_dim=16, image_embed_dim=16, rope_axes_dim=[4,6,6],
        target_size=16, task_type='t2v', use_meanflow=False).float().eval()
    native.save_pretrained(tmp_path/'transformer', safe_serialization=True)
    api = importlib.import_module('distillschedule.training.hunyuan_video')
    assert callable(api.build_model) and callable(api.build_inference_model)
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    from pathlib import Path
    from distillschedule.training.plugins import load_callable
    builder,_ = load_callable(str(Path(__file__).resolve().parents[1]/'recipes/hunyuan_video_dmd/create_cpu_fixture.py')+':build_fixture')
    return builder(tmp_path_factory.mktemp('native-hunyuan15')/'fixture')


def owned(assets):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    config = load_training_config(assets['config'])
    return config,*construct(config)


def test_native_dual_text_temporal4_codec_forward_gradient_and_euler_oracles(assets):
    import torch
    from diffusers import HunyuanVideo15Transformer3DModel, AutoencoderKLHunyuanVideo15, HunyuanVideo15Pipeline
    from transformers import Qwen2_5_VLTextModel, T5EncoderModel, Qwen2TokenizerFast, ByT5Tokenizer
    from distillschedule.data.decoding import _video
    from distillschedule.training.prediction import PredictionSpec
    from distillschedule.workloads.best_known.registry import file_sha256
    raw = assets['root']/'text_encoder/config.json'; before = file_sha256(raw)
    _,components,dataset,_,_ = owned(assets)
    role,media = components.state['student'],components.state['media']; row = dataset.sample(0)
    assert type(role.native) is HunyuanVideo15Transformer3DModel and type(media.vae) is AutoencoderKLHunyuanVideo15
    assert type(media.text_encoder) is Qwen2_5_VLTextModel and type(media.text_encoder_2) is T5EncoderModel
    assert type(media.tokenizer) is Qwen2TokenizerFast and type(media.tokenizer_2) is ByT5Tokenizer
    assert media.tokenizer_2.backend == 'custom' and media.vae.config.temporal_compression_ratio == 4
    assert row['latents']['video'].shape == (1,4,2,4,4) and components.prediction == PredictionSpec('flow')
    assert before == file_sha256(raw) and 'rope_scaling' in raw.read_text() and 'rope_parameters' not in raw.read_text()
    assert not media.training and not any(p.requires_grad for p in media.parameters())
    assert role.native.training and not components.state['teacher'].training
    for prompt,index in [('a bird',0),('a sign says "hello"',1)]:
        values = media.encode_text([prompt]); actual = dataset.sample(index)
        native = HunyuanVideo15Pipeline._get_mllm_prompt_embeds(media.text_encoder,media.tokenizer,[prompt],device=torch.device('cpu'))
        native2 = HunyuanVideo15Pipeline._get_byt5_prompt_embeds(media.tokenizer_2,media.text_encoder_2,[prompt],device=torch.device('cpu'))
        for key,value,expected in zip(('context','context_mask','context_2','context_mask_2'),values,(*native,*native2)):
            assert torch.equal(value,expected) and torch.equal(value,actual[key])
        assert values[0].shape == (1,1000,16) and values[1].any() and values[2].shape == (1,256,16)
        assert bool(values[3].any()) == (index == 1)
        if index == 0: assert not values[2].any(), 'native no-glyph zeros/allfalse is legitimate, not missing conditioning'
    empty = media.encode_text([''])
    assert empty[1].any() and empty[0].any() and not empty[2].any() and not empty[3].any()
    assert all(torch.equal(row['null_'+key],value) for key,value in zip(('context','context_mask','context_2','context_mask_2'),empty))
    sigma = torch.tensor([.625]); sample = row['latents']; seen = []
    hook = role.native.register_forward_pre_hook(lambda m,a,k:seen.append(k),with_kwargs=True)
    actual = role.denoise(sample,sigma,row,conditional=True)['video']; hook.remove()
    call = seen[0]
    assert call['hidden_states'].shape == (1,9,2,4,4) and torch.equal(call['hidden_states'][:,:4],sample['video'])
    assert not call['hidden_states'][:,4:].any() and call['image_embeds'].shape == (1,729,16) and not call['image_embeds'].any()
    assert torch.equal(call['timestep'],torch.tensor([625.]))
    def oracle(value,s,*,conditional=True):
        prefix = '' if conditional else 'null_'
        return role.native(hidden_states=torch.cat([value,torch.zeros_like(value),torch.zeros_like(value[:,:1])],dim=1),
            timestep=value.new_full((1,),1000*s),image_embeds=value.new_zeros(1,729,16),
            encoder_hidden_states=row[prefix+'context'],encoder_attention_mask=row[prefix+'context_mask'],
            encoder_hidden_states_2=row[prefix+'context_2'],encoder_attention_mask_2=row[prefix+'context_mask_2'],return_dict=False)[0]
    native = oracle(sample['video'],.625)
    assert torch.equal(actual,native)
    assert torch.equal(role.denoise(sample,sigma,row,conditional=False)['video'],oracle(sample['video'],.625,conditional=False))
    params = tuple(p for p in role.parameters() if p.requires_grad)
    a = torch.autograd.grad(actual.square().mean(),params); b = torch.autograd.grad(native.square().mean(),params)
    assert all(torch.equal(x,y) for x,y in zip(a,b)) and any(g.any() for g in a)
    assert torch.equal(PredictionSpec('flow').to_x0(actual,sample['video'],sigma=sigma[:,None,None,None,None]),sample['video']-sigma[:,None,None,None,None]*native)
    torch.manual_seed(909); rolled = role.rollout(row,[1.,.75,.5,.25,0.],'all_steps')['video']
    torch.manual_seed(909); manual = torch.randn_like(sample['video'])
    for s,n in zip([1.,.75,.5,.25],[.75,.5,.25,0.]): manual = manual+(n-s)*oracle(manual,s)
    assert torch.equal(rolled,manual), 'native velocity sign and all four Euler steps must be exact'
    frames = _video(assets['root']/'frames'); encoded = media.encode_video(frames)
    with torch.no_grad():
        raw_latent = media.vae.encode(frames.permute(3,0,1,2).unsqueeze(0).float()/127.5-1).latent_dist.mode()
        assert torch.equal(encoded,raw_latent*media.vae.config.scaling_factor)
        assert torch.equal(encoded,row['latents']['video'])
        torch.testing.assert_close(media.decode_latents(encoded),media.vae.decode(raw_latent,return_dict=False)[0],rtol=1e-5,atol=1e-6)
    assert media.decode({'video_latents':encoded},{})['video'].shape == (5,16,16,3)
    assert not torch.cuda.is_initialized()


def test_native_unmerged_lora_ownership_real_g1_f5_and_shared_execution(assets,tmp_path):
    from copy import deepcopy
    import json
    import torch
    from distillschedule.training.session import build_cycle
    from distillschedule.open_workload.torch_backend import Session, sharing_inventory, resolve_module
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    results,gradients = [],[]
    for share in (False,True):
        config,model,dataset,algorithm,_ = owned(assets)
        teacher = deepcopy(model.state['teacher'].state_dict())
        before = {r:deepcopy(model.state[r].state_dict()) for r in ('student','fake')}
        captured = {r:[] for r in ('student','fake')}
        trainables = {r:{id(p) for p in model.state[r].parameters() if p.requires_grad} for r in captured}
        assert not trainables['student']&trainables['fake']
        for role in captured:
            assert {id(p) for group in model.state[role+'_optimizer'].param_groups for p in group['params']} == trainables[role]
            for parameter in model.state[role].parameters():
                if parameter.requires_grad: parameter.register_hook(lambda g,role=role:captured[role].append(g.detach().clone()))
        workload = build_cycle(config,model,dataset,algorithm,cycle=0,cursor=0)
        pairs = sharing_inventory(workload); assert pairs
        first,second,_ = pairs[0]
        source = resolve_module(model.state,first[0]).get_parameter(first[1])
        assert source is not resolve_module(model.state,second[0]).get_parameter(second[1])
        session = Session(workload,dict(backend='torch-callable-resident-v1',share_immutable=share,checkpoint_bindings=[]),copy_state=False)
        assert (source is resolve_module(session.state,second[0]).get_parameter(second[1])) is share
        output = session.run()
        assert all(torch.isfinite(value) for value in output.values()) and model.state['dmd_counters'].tolist() == [1,1,5]
        assert session.receipt['dispatcher']['final_versions']['fake'] == 5
        assert equal(teacher,model.state['teacher'].state_dict())
        assert all(p.grad is None and not p.requires_grad for p in model.state['teacher'].parameters())
        for role in captured:
            assert captured[role] and any(g.any() for g in captured[role])
            assert any(not torch.equal(v,before[role][key]) for key,v in model.state[role].state_dict().items())
            assert all(torch.equal(p,before[role][key]) for key,p in model.state[role].named_parameters() if not p.requires_grad)
            count = 1 if role == 'student' else 5
            assert all(float(v['step']) == count and torch.isfinite(v['exp_avg']).all() for v in model.state[role+'_optimizer'].state.values())
            assert model.state[role+'_scheduler'].last_epoch == count
        gradients.append(captured)
        results.append(dict(packed=pack_state(model.state),output={k:v.detach().clone() for k,v in output.items()},storage=session.initial_storage,effects=session.effects.copy()))
    assert equal(gradients[0],gradients[1]) and equal(results[0]['packed'],results[1]['packed']) and equal(results[0]['output'],results[1]['output'])
    assert results[1]['storage'] < results[0]['storage'] and results[1]['effects']['shared_parameter_aliases'] > 0
    (tmp_path/'sharing-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',exact_gradients_optimizer_scheduler_ema_teacher_parity=True,
        shared_parameter_aliases=results[1]['effects']['shared_parameter_aliases'],unshared_bytes=results[0]['storage'],shared_bytes=results[1]['storage'],gpu_execution='NOT_RUN'),indent=2))


def test_fresh_process_complete_recovery_and_portable_current_g_video(assets,tmp_path):
    from pathlib import Path
    import json
    import shutil
    import torch
    from distillschedule.data import PreparedDataset
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import _load_for_output, build_cycle
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import read_export, restore_modules
    from distillschedule.data.hunyuan_video_inputs import SCHEMA
    from test_training_end_to_end import run_cli, equal
    partial = run_cli('train',assets['config'],'--cycles',1,'--output',tmp_path/'partial')
    resumed = run_cli('train',assets['config'],'--cycles',3,'--resume',partial['checkpoint'],'--output',tmp_path/'resumed')
    whole = run_cli('train',assets['config'],'--cycles',3,'--output',tmp_path/'whole')
    a = torch.load(Path(resumed['checkpoint'])/'state.pt',weights_only=True); b = torch.load(Path(whole['checkpoint'])/'state.pt',weights_only=True)
    assert equal(a,b), 'all Adam/schedulers/EMA/RNG/data/cursor/clocks/versions must restart exactly'
    assert a['state']['dmd_counters']['value']['counters'].tolist() == [3,3,15]
    assert a['state']['student_scheduler']['value']['last_epoch'] == 3 and a['state']['fake_scheduler']['value']['last_epoch'] == 15
    for name in ('student','fake','teacher','student_optimizer','fake_optimizer','student_scheduler','fake_scheduler','student_ema'): assert name in a['state']
    rows = [json.loads(line) for line in (tmp_path/'whole/metrics.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in rows] == ['native-video-0','native-video-1','native-video-2']
    assert [json.loads(line)['sample_id'] for line in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()] == ['native-video-1','native-video-2']
    manual = json.loads(Path(assets['config']).read_text())
    manual['execution'] = dict(mode='manual',implementation=dict(backend='torch-callable-resident-v1',share_immutable=True,checkpoint_bindings=[]))
    (tmp_path/'sharing.json').write_text(json.dumps(manual))
    first = run_cli('train',tmp_path/'sharing.json','--cycles',1,'--output',tmp_path/'shared-first')
    shared = run_cli('train',tmp_path/'sharing.json','--cycles',3,'--output',tmp_path/'shared')
    shared_resumed = run_cli('train',tmp_path/'sharing.json','--cycles',3,'--resume',first['checkpoint'],'--output',tmp_path/'shared-resumed')
    shared_payload = torch.load(Path(shared['checkpoint'])/'state.pt',weights_only=True)
    assert equal(shared_payload,torch.load(Path(shared_resumed['checkpoint'])/'state.pt',weights_only=True))
    for key in ('state','rng','cycle','cursor','versions'): assert equal(a[key],shared_payload[key]),key
    restored,data,payload,_,session = _load_for_output(load_training_config(tmp_path/'sharing.json'),shared_resumed['checkpoint'])
    from distillschedule.open_workload.torch_backend import sharing_inventory, resolve_module
    pairs = sharing_inventory(session.workload,include_existing=True); assert pairs
    assert any(resolve_module(session.state,x[0]).get_parameter(x[1]) is resolve_module(session.state,y[0]).get_parameter(y[1]) for x,y,_ in pairs)
    from distillschedule.training.dmd import build_algorithm
    session.bind_cycle(build_cycle(load_training_config(tmp_path/'sharing.json'),restored,data,build_algorithm,cycle=3,cursor=0,versions=payload['versions']))
    with torch.no_grad(): next(restored.state['teacher'].parameters()).add_(.1)
    with pytest.raises(ValueError,match='immutable'): session.run()
    export = run_cli('export',assets['config'],'--checkpoint',resumed['checkpoint'],'--output',tmp_path/'export')
    assert set(export['model']['modules']) == {'student','media'} and 'assets' not in export['model']['parameters']
    new_source = dict(schema_version=SCHEMA,family='hunyuan15-t2v',assets=str(assets['assets']),mode='prompts',records=[
        dict(id='new-unseen-video',prompt='a sign says "新的文字" beside a river',latent_shape=[1,4,2,4,4])])
    (tmp_path/'new.json').write_text(json.dumps(new_source))
    run_cli('data','prepare',tmp_path/'new.json','--output',tmp_path/'new-prepared','--preparer','distillschedule.data.hunyuan_video_inputs:prepare_inputs')
    prepared = tmp_path/'new-prepared/manifest.json'; feeds = PreparedDataset(prepared).sample(0)
    current = _load_for_output(load_training_config(assets['config']),resumed['checkpoint'])[0]
    assert not equal(feeds['context'],PreparedDataset(assets['prepared']).sample(0)['context']) and feeds['context_mask_2'].any()
    for module in current.state.values():
        if isinstance(module,torch.nn.Module): module.eval()
    torch.manual_seed(909)
    with torch.no_grad():
        generated = current.generate(current.state,feeds,{})
        expected = dict(generated=generated,decoded=current.decode(generated,{}))
    factory,_ = load_callable(export['model']['factory']); portable = factory(export['model']['parameters'])
    export_root,manifest = read_export(tmp_path/'export'); restore_modules(portable,export_root,manifest)
    assert equal(portable.state['media'].encode_text(['a fresh "glyph"']),current.state['media'].encode_text(['a fresh "glyph"']))
    assert type(portable.state['media'].tokenizer_2) is type(current.state['media'].tokenizer_2)
    inference = dict(schema_version='distillschedule.training.v1',model=dict(factory=export['model']['factory'],parameters=export['model']['parameters']),
        algorithm=dict(name='dmd'),data=dict(prepared=str(prepared)),execution=dict(mode='reference'),trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(tmp_path/'unused')))
    (tmp_path/'inference.json').write_text(json.dumps(inference))
    shutil.copytree(resumed['checkpoint'],tmp_path/'verified-checkpoint')
    for name in ('partial','resumed','whole','shared-first','shared','shared-resumed'):
        shutil.copyfile(tmp_path/name/'metrics.jsonl',tmp_path/(name+'-metrics.jsonl')); shutil.rmtree(tmp_path/name)
    hidden = assets['root'].with_name(assets['root'].name+'-hidden'); assets['root'].rename(hidden)
    try:
        actual = run_cli('generate',tmp_path/'inference.json','--weights',tmp_path/'export','--prepared',prepared,'--output',tmp_path/'generation','--output-kind','video')
    finally: hidden.rename(assets['root'])
    artifact = torch.load(actual['artifact'],weights_only=True)
    assert equal(expected,artifact), 'portable decoded new input must be exact CURRENT G, not base/Teacher/EMA'
    assert artifact['decoded']['video'].shape == (5,16,16,3) and artifact['decoded']['video'].dtype == torch.uint8
    assert len(list(Path(actual['video']['frames']).glob('*.png'))) == 5
    (tmp_path/'lifecycle-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',exact_complete_state_equality=True,
        clocks=[3,3,15],sample_ids=[r['sample_id'] for r in rows],versions=whole['final_versions'],export=str(tmp_path/'export'),
        verified_checkpoint=str(tmp_path/'verified-checkpoint'),new_prepared=str(prepared),current_student_portable_video_exact=True,
        assets_hidden_and_training_checkpoints_deleted=True,actual_video=actual['video'],gpu_execution='NOT_RUN',quality_validation='NOT_RUN'),indent=2))


def test_native_slow_byt5_and_primary_added_token_options_roundtrip():
    import torch
    from tokenizers import AddedToken
    from transformers import ByT5Tokenizer
    from distillschedule.training.hunyuan_video import tokenizer_state, restore_tokenizer
    tokenizer = ByT5Tokenizer(extra_ids=3,model_max_length=256,clean_up_tokenization_spaces=True)
    tokenizer.add_tokens([AddedToken('glyph-special',single_word=True,lstrip=True,rstrip=False,normalized=False)])
    state = tokenizer_state(tokenizer,secondary=True); restored = restore_tokenizer(state,secondary=True)
    assert type(restored) is ByT5Tokenizer and restored.backend == 'custom' and tokenizer_state(restored,secondary=True) == state
    assert torch.equal(tokenizer('a "测试" glyph-special',return_tensors='pt').input_ids,restored('a "测试" glyph-special',return_tensors='pt').input_ids)
    with pytest.raises(ValueError,match='tokenizer'): restore_tokenizer(state)
    with pytest.raises(ValueError,match='tokenizer'): restore_tokenizer(dict(state,**{'class':'Qwen2TokenizerFast'}),secondary=True)


def test_primary_native_tokenizer_state_and_crop_refusal(assets):
    from distillschedule.training.hunyuan_video import tokenizer_state, restore_tokenizer
    _,model,_,_,_ = owned(assets); media = model.state['media']
    state = tokenizer_state(media.tokenizer)
    restored = restore_tokenizer(state)
    assert tokenizer_state(restored) == state and restored.chat_template == media.tokenizer.chat_template
    original = media.tokenizer.chat_template
    try:
        media.tokenizer.chat_template = "{% for message in messages %}{{ message['content'] if message['role']=='user' else '' }}{% endfor %}"
        with pytest.raises(ValueError,match='crop108|empty context'): media.encode_text([''])
    finally: media.tokenizer.chat_template = original


def test_primary_nondefault_native_option_survives_export():
    from transformers import Qwen2TokenizerFast
    from distillschedule.training.hunyuan_video import tokenizer_state, restore_tokenizer
    tokenizer = Qwen2TokenizerFast(vocab={'<|endoftext|>':0,'a':1},merges=[],add_prefix_space=True,chat_template='{{ messages }}')
    restored = restore_tokenizer(tokenizer_state(tokenizer))
    assert restored.add_prefix_space is tokenizer.add_prefix_space is True


def test_native_masks_empty_prompt_overflow_geometry_and_refusals(assets):
    from copy import deepcopy
    import torch
    _,model,dataset,_,_ = owned(assets); role,media = model.state['student'],model.state['media']; good = dataset.sample(0)
    values = [media.encode_text([p]) for p in ('','a bird','a boat rests on the blue lake')]
    assert len({int(v[1].sum()) for v in values}) == 3
    assert all(v[1].any() and not v[3].any() and not v[2].any() for v in values)
    with pytest.raises(ValueError,match='overflow|truncat'): media.encode_text(['unbounded '*256])
    with pytest.raises(ValueError,match='glyph256|truncat'): media.encode_text(['a sign "'+'x'*260+'"'])
    for kind in ('channels','batch','missing','nonfinite','mask','empty-primary','mask-order','bad-glyph-sentinel','invented-empty','binding','i2v','causal','layout'):
        row = deepcopy(good)
        if kind == 'channels': row['latent_shape'][1] = 9
        elif kind == 'batch': row['latent_shape'][0] = 2
        elif kind == 'missing': del row['context_2']
        elif kind == 'nonfinite': row['latents']['video'][0,0,0,0,0] = float('nan')
        elif kind == 'mask': row['context_mask_2'] = row['context_mask_2'].float()
        elif kind == 'empty-primary': row['context_mask'].zero_()
        elif kind == 'mask-order': row['context_mask'][0,0] = False
        elif kind == 'bad-glyph-sentinel': row['context_2'][0,0,0] = 1.
        elif kind == 'invented-empty': row['null_context'].zero_()
        elif kind == 'binding': row['encoder_identity'][0] ^= 1
        elif kind == 'i2v': row['image'] = torch.zeros(1)
        elif kind == 'causal': row['kv_cache'] = torch.zeros(1)
        else: row['latents']['video'] = row['latents']['video'].permute(0,2,3,4,1)
        with pytest.raises(ValueError): role.validate_feeds(row)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='FP32|AMP'): role.denoise(good['latents'],torch.tensor([.5]),good,conditional=True)
    with pytest.raises(ValueError,match='geometry|resample'): media.encode_video(torch.zeros(4,16,16,3,dtype=torch.uint8))
    with pytest.raises(ValueError,match='geometry|resize'): media.encode_video(torch.zeros(5,15,16,3,dtype=torch.uint8))


def test_preencoded_exact_and_transactional_bad_inputs(assets,tmp_path):
    import json
    import torch
    from copy import deepcopy
    from distillschedule.data import PreparedDataset
    from distillschedule.data.hunyuan_video_inputs import SCHEMA, prepare_inputs
    from distillschedule.training.hunyuan_video import FAMILY
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal
    good = PreparedDataset(assets['prepared']).sample(1); torch.save(good,tmp_path/'good.pt')
    value = dict(schema_version=SCHEMA,family=FAMILY,assets=str(assets['assets']),mode='preencoded',
        records=[dict(id='native-glyph',features=dict(path='good.pt',sha256=file_sha256(tmp_path/'good.pt')))])
    source = tmp_path/'source.json'; source.write_text(json.dumps(value))
    prepared = prepare_inputs(source,tmp_path/'preencoded')
    assert equal(good,PreparedDataset(prepared).sample(0)) and PreparedDataset(prepared).validate(sources=True)['source_validation'] == 'PASS'
    bad = deepcopy(good); bad['encoder_identity'][0] ^= 1; torch.save(bad,tmp_path/'bad.pt')
    value['records'][0]['features'] = dict(path='bad.pt',sha256=file_sha256(tmp_path/'bad.pt')); source.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='binding|stale'): prepare_inputs(source,tmp_path/'bad-prepared')
    assert not (tmp_path/'bad-prepared').exists() and not list(tmp_path.glob('.bad-prepared*'))
    for kind in ('shape','i2v','missing-prompt','duplicate','sha'):
        value = dict(schema_version=SCHEMA,family=FAMILY,assets=str(assets['assets']),mode='prompts',records=[dict(id='new',prompt='a bird',latent_shape=[1,4,2,4,4])])
        if kind == 'shape': value['records'][0].pop('latent_shape')
        elif kind == 'i2v': value['records'][0]['image'] = 'not-admitted'
        elif kind == 'missing-prompt': value['records'][0].pop('prompt')
        elif kind == 'duplicate': value['records'] *= 2
        else:
            value['mode'] = 'preencoded'; value['records'] = [dict(id='bad-hash',features=dict(path='good.pt',sha256='0'*64))]
        source.write_text(json.dumps(value))
        with pytest.raises(ValueError): prepare_inputs(source,tmp_path/('refused-'+kind))
        assert not (tmp_path/('refused-'+kind)).exists()


def test_actual_lossless_video_file_and_raw_rgba_refusal(assets,tmp_path):
    import json
    import shutil
    import subprocess
    import torch
    from PIL import Image
    from distillschedule.data.decoding import _video
    from distillschedule.data import PreparedDataset
    from distillschedule.data.hunyuan_video_inputs import SCHEMA, prepare_inputs
    from distillschedule.data.manifest import source_identity
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal
    assert shutil.which('ffmpeg') and shutil.which('ffprobe'), 'actual raw-video qualification needs existing ffmpeg/ffprobe'
    clip = tmp_path/'lossless.mkv'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','5','-i',str(assets['root']/'frames/%04d.png'),
        '-threads','1','-c:v','ffv1',str(clip)],check=True,capture_output=True)
    assert torch.equal(_video(clip),_video(assets['root']/'frames'))
    value = dict(schema_version=SCHEMA,family='hunyuan15-t2v',assets=str(assets['assets']),mode='raw',records=[
        dict(id='actual-lossless-video',prompt='a bird',video=dict(path=str(clip),sha256=file_sha256(clip)))])
    source = tmp_path/'video.json'; source.write_text(json.dumps(value))
    prepared = prepare_inputs(source,tmp_path/'prepared-video')
    assert equal(PreparedDataset(prepared).sample(0),PreparedDataset(assets['prepared']).sample(0))
    rgba = tmp_path/'rgba'; rgba.mkdir()
    for index in range(5): Image.new('RGBA',(16,16),(30,60,90,128)).save(rgba/f'{index:04d}.png')
    value['records'][0]['video'] = dict(path=str(rgba),sha256=source_identity(rgba,frames=True)['sha256'])
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='RGB|alpha'): prepare_inputs(source,tmp_path/'refused-rgba')
    assert not (tmp_path/'refused-rgba').exists()


def test_local_locker_uses_exact_inventory_without_overwriting_assets(assets,tmp_path):
    import json
    import shutil
    from pathlib import Path
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.hunyuan_video import ASSET_SCHEMA, TEXT_POLICY
    from distillschedule.training.flow_image import asset_files
    locker,_ = load_callable(str(Path(__file__).resolve().parents[1]/'recipes/hunyuan_video_dmd/lock_local_assets.py')+':lock_bundle')
    root = tmp_path/'bundle'; shutil.copytree(assets['root'],root); (root/'assets.json').unlink()
    with pytest.raises(ValueError,match='budget'): locker(root,max_parameters=True)
    path = locker(root,max_parameters=10_000_000,student='base',teacher='base',fake='fake')
    value = json.loads(path.read_text())
    assert value['schema_version'] == ASSET_SCHEMA and value['text_policy'] == TEXT_POLICY
    assert value['evidence_kind'] == 'LOCAL_PRETRAINED_UNVERIFIED'
    for spec in {**value['roles'],**value['media']}.values(): asset_files(root,spec)
    original = path.read_bytes()
    with pytest.raises(FileExistsError): locker(root,max_parameters=10_000_000,student='base',teacher='base',fake='fake')
    assert path.read_bytes() == original


@pytest.mark.parametrize('kind',['source','family','schema','i2v','distilled','meanflow','unknown','wrong-primary',
    'temporal1','bad-rope','teacher-budget','text2-budget','text-policy'])
def test_source_variant_config_and_all_budgets_refused_before_weight_load(assets,tmp_path,monkeypatch,kind):
    from copy import deepcopy
    import json
    import shutil
    from distillschedule.training import hunyuan_video as api, flow_image as safe
    from distillschedule.training.config import load_training_config
    from distillschedule.workloads.best_known.registry import file_sha256
    root = tmp_path/'bundle'; shutil.copytree(assets['root'],root)
    manifest = json.loads((root/'assets.json').read_text())
    if kind == 'source': manifest['source']['revision'] = '0'*40
    elif kind == 'family': manifest['family'] = 'hunyuan-video-original'
    elif kind == 'schema': manifest['schema_version'] = 'distillschedule.wan22_t2v_assets.v1'
    elif kind == 'i2v': manifest['pipeline']['task'] = 'i2v'
    elif kind == 'distilled': manifest['pipeline']['distilled'] = True
    elif kind == 'teacher-budget': manifest['roles']['teacher']['max_parameters'] = 1
    elif kind == 'text2-budget': manifest['media']['text_encoder_2']['max_parameters'] = 1
    elif kind == 'text-policy': manifest['text_policy']['crop_start'] = 34
    elif kind in ('temporal1','wrong-primary'):
        name = 'vae' if kind == 'temporal1' else 'text_encoder'
        path = root/name/'config.json'; value = json.loads(path.read_text())
        if kind == 'temporal1': value['temporal_compression_ratio'] = 1
        else: value['model_type'] = 'qwen2_5_vl'
        path.write_text(json.dumps(value)); manifest['media'][name]['files']['config.json'] = file_sha256(path)
    else:
        for name in ('base','fake'):
            path = root/name/'config.json'; value = json.loads(path.read_text())
            if kind == 'unknown': value['ignored_unknown_field'] = True
            elif kind == 'meanflow': value['use_meanflow'] = True
            else: value['rope_axes_dim'] = [2,2,2]
            path.write_text(json.dumps(value))
        for name,folder in (('student','base'),('teacher','base'),('fake','fake')):
            manifest['roles'][name]['files']['config.json'] = file_sha256(root/folder/'config.json')
    (root/'assets.json').write_text(json.dumps(manifest))
    parameters = deepcopy(load_training_config(assets['config']).model['parameters']); parameters['assets'] = str(root/'assets.json')
    def forbidden(*args,**kwargs): raise AssertionError('concrete weight load before native config/all-role/meta budget admission')
    monkeypatch.setattr(safe,'load_native',forbidden)
    with pytest.raises(ValueError): api.build_model(parameters)


def test_native_safe_shards_aliases_dtype_and_shape_refusals(assets,tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    from distillschedule.training import hunyuan_video as api
    from distillschedule.training.flow_image import load_native
    cls = api.classes()[0]
    native,_ = load_native(cls,assets['root']/'base',10_000_000)
    native.save_pretrained(tmp_path/'shards',safe_serialization=True,max_shard_size='2MB')
    loaded,_ = load_native(cls,tmp_path/'shards',10_000_000)
    assert all(torch.equal(value,loaded.state_dict()[key]) for key,value in native.state_dict().items())
    shards = sorted((tmp_path/'shards').glob('*.safetensors')); assert len(shards)>1; shards[-1].unlink()
    with pytest.raises(ValueError,match='shard|index'): load_native(cls,tmp_path/'shards',10_000_000)
    native.save_pretrained(tmp_path/'badshape',safe_serialization=True)
    path = next((tmp_path/'badshape').glob('*.safetensors')); state = load_file(path); key = next(iter(state)); state[key] = torch.zeros(1); save_file(state,path)
    with pytest.raises(ValueError,match='shape|dtype'): load_native(cls,tmp_path/'badshape',10_000_000)
    native.half().save_pretrained(tmp_path/'fp16',safe_serialization=True)
    with pytest.raises(ValueError,match='dtype|FP32'): load_native(cls,tmp_path/'fp16',10_000_000,dtype_policy='bf16_or_fp32_to_fp32')
    native.bfloat16().save_pretrained(tmp_path/'bf16',safe_serialization=True)
    loaded,_ = load_native(cls,tmp_path/'bf16',10_000_000,dtype_policy='bf16_or_fp32_to_fp32')
    assert all(torch.equal(value.float(),loaded.state_dict()[key]) for key,value in native.state_dict().items())
    text,_ = load_native(api.classes()[3],assets['root']/'text_encoder_2',10_000_000)
    assert text.shared.weight is text.encoder.embed_tokens.weight, 'actual native T5 alias ownership must survive safe loading'


def test_algorithm_codec_tokenizer_and_runtime_drift_refusals(assets,tmp_path,monkeypatch):
    from copy import deepcopy
    import importlib.metadata
    import json
    import shutil
    from safetensors.torch import load_file, save_file
    from distillschedule.training.config import TrainingConfig, load_training_config
    from distillschedule.training.session import construct
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.training import hunyuan_video as api
    original = load_training_config(assets['config'])
    for parameter,value in (('teacher_cfg',2.),('fake_updates',4),('rollout_sigmas',[1.,.5,0.])):
        wrong = original.to_dict(); wrong['algorithm']['parameters'][parameter] = value
        with pytest.raises(ValueError,match='cfg|G1/F5|Euler'): construct(TrainingConfig.from_dict(wrong,source=original.source))
    for key,value in (('device','cuda'),('precision','bf16'),('task','i2v'),('use_meanflow',True),('seed',True)):
        wrong = original.to_dict(); wrong['model']['parameters'][key] = value
        with pytest.raises(ValueError): construct(TrainingConfig.from_dict(wrong,source=original.source))
    for component in ('vae','tokenizer','tokenizer_2'):
        root = tmp_path/component; shutil.copytree(assets['root'],root)
        manifest = json.loads((root/'assets.json').read_text())
        if component == 'vae':
            path = next((root/component).glob('*.safetensors')); state = load_file(path)
            key = next(key for key,v in state.items() if v.is_floating_point()); state[key] += .01; save_file(state,path)
        else:
            path = root/component/'tokenizer_config.json'; value = json.loads(path.read_text()); value['model_max_length'] = 2048
            path.write_text(json.dumps(value))
        manifest['media'][component]['files'][path.name] = file_sha256(path); (root/'assets.json').write_text(json.dumps(manifest))
        wrong = original.to_dict(); wrong['model']['parameters']['assets'] = str(root/'assets.json')
        with pytest.raises(ValueError,match='codec|binding'): construct(TrainingConfig.from_dict(wrong,source=original.source))
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda key:'0.0.0' if key == 'diffusers' else version(key))
    with pytest.raises(ValueError,match='modern-flow|pin'): api.runtime_identity()


def test_fresh_process_seed_and_corrupt_checkpoint_are_refused(assets,tmp_path):
    from pathlib import Path
    import json
    import subprocess
    import sys
    from test_training_end_to_end import run_cli
    first = run_cli('train',assets['config'],'--cycles',1,'--output',tmp_path/'first')
    value = json.loads(Path(assets['config']).read_text()); value['trainer']['seed'] += 1
    path = tmp_path/'wrong-seed.json'; path.write_text(json.dumps(value))
    def refused(config,output):
        result = subprocess.run([sys.executable,'-m','distillschedule','train',str(config),'--cycles','3','--resume',first['checkpoint'],'--output',str(output)],capture_output=True,text=True)
        assert result.returncode != 0 and not list(output.glob('checkpoints/*/manifest.json'))
        return result.stdout+result.stderr
    assert 'identity' in refused(path,tmp_path/'wrong-seed').lower()
    payload = Path(first['checkpoint'])/'state.pt'; saved = payload.read_bytes(); payload.write_bytes(saved+b'corrupt')
    assert 'hash' in refused(assets['config'],tmp_path/'corrupt').lower()
    payload.write_bytes(saved); payload.unlink()
    message = refused(assets['config'],tmp_path/'missing-shard').lower()
    assert 'checkpoint' in message and ('missing' in message or 'no such file' in message)


def test_lazy_discovery_and_compatibility_without_torch_or_weights():
    import subprocess
    import sys
    script = '''
import sys
from distillschedule.training.catalog import resolve,validate_compatibility
row=resolve('model','hunyuan15-t2v')
assert row['source']['revision']=='286be7ce72277246578a3e3cc2487e95ddae5bcf'
assert 'Tencent' in row['source']['license'] and row['validation']['gpu']=='NOT_RUN'
for algorithm,trainer in [('distillschedule.training.dmd2:build_dmd2',{'device':'cpu'}),('distillschedule.training.self_forcing:build_self_forcing',{'device':'cpu'}),('distillschedule.training.dmd:build_algorithm',{'device':'cuda'}),('distillschedule.training.dmd:build_algorithm',{'device':'cpu','distributed':{}})]:
    try:validate_compatibility({'factory':row['factory'],'parameters':{}},{'factory':algorithm},trainer)
    except ValueError:pass
    else:raise AssertionError('unqualified capability admitted')
assert 'torch' not in sys.modules and 'diffusers' not in sys.modules and 'transformers' not in sys.modules
print('PASS_LAZY_NO_MODEL_TORCH_CUDA_OR_ASSET_LOAD')
'''
    result = subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr


@pytest.mark.parametrize('name',['student','vae','text_encoder','text_encoder_2'])
def test_portable_all_meta_budgets_precede_any_concrete_model(assets,monkeypatch,name):
    from copy import deepcopy
    from functools import wraps
    import torch
    from distillschedule.training import hunyuan_video as api
    _,model,_,_,_ = owned(assets)
    descriptor = deepcopy(model.inference_projection(model,{}).parameters)
    descriptor['cpu_parameter_limits'][name] = 1
    cls = api.classes()[0]; original = cls.__init__
    @wraps(original)
    def counted(self,*args,**kwargs):
        assert torch.get_default_device().type == 'meta', 'concrete constructor preceded portable budget admission'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(cls,'__init__',counted)
    with pytest.raises(ValueError,match='portable.*budget'): api.build_inference_model(descriptor)
