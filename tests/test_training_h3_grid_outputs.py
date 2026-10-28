"""W5's converted Grid model is not W6's native eight-step DiT."""
import subprocess
import sys


def test_grid_output_discovery_does_not_import_models_or_cuda():
    result = subprocess.run([sys.executable, '-B', '-c',
        "from distillschedule.training.h3_grid_inference import build_inference_model,export_components;"
        "import sys;assert callable(build_inference_model) and callable(export_components);"
        "assert not any(n in sys.modules for n in ('torch','diffusers','peft','lightx2v_train'))"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture(scope='module')
def actual_grid_case(tmp_path_factory):
    """Actual pinned converted model/PEFT/sampler; small random weights only.

    The table owner is a CPU fixture made from the actual frozen AdaLN outputs,
    not a CUDA mmap/controller qualification or a trained/quality checkpoint.
    """
    import inspect
    import torch
    from peft import LoraConfig
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.checkpoint import rng_state, restore_rng
    root = os.environ.get('DMD_H3_TEST_UPSTREAM')
    assert root, 'existing exact LightX2V source must be supplied; no skip/fake model'
    source = dict(root=root, files=grid.SOURCE_FILES,
                  terms=dict(origin='https://github.com/ModelTC/LightX2V', license='Apache-2.0', accepted=True))
    api, cls = grid.source_api(source), grid.model_class()
    before = rng_state(device='cpu')
    torch.manual_seed(179)
    arch = grid.plain({k:v.default for k,v in inspect.signature(cls.__init__).parameters.items() if k!='self'})
    arch.update(num_layers=1, num_refiner_layers=1, hidden_size=32, num_attention_heads=1, attention_head_dim=32,
                ffn_dim=64, freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=4)
    model = cls(**arch).to(dtype=torch.bfloat16)
    # Mirror the original converted checkpoint's mixed-precision modules.
    for name in cls._keep_in_fp32_modules:
        model.get_submodule(name).float()
    for role in ('student','fake'):
        model.add_adapter(LoraConfig(**deepcopy(grid.LORA)), adapter_name=role)
    model.set_adapter('student')
    with torch.no_grad():
        for name, p in model.named_parameters():
            if '.lora_B.student.' in name: p.normal_(std=.015)
            elif '.lora_B.fake.' in name: p.fill_(.25)
    dims = dict(height=32,width=32,frames=22,text_tokens=3)
    shape, _ = grid.geometry(api,dims,text_dim=5120)
    feeds = dict(prompt_embeds=torch.randn(1,3,5120,dtype=torch.bfloat16),text_token_tags=torch.ones(3,dtype=torch.int64))
    wrapper = object.__new__(api.model)
    wrapper.device,wrapper.running_dtype=torch.device('cpu'),torch.bfloat16
    wrapper.use_autocast,wrapper.transformer,wrapper.patch_size=False,model,(1,2,2)
    owner = object.__new__(api.trainer)
    owner.model,owner._layout_cache,owner.latent_dtype=wrapper,{},torch.float32
    owner.video_sigmas_cpu,owner.audio_sigmas_cpu=grid.schedule(api)
    # A real gradient update makes the Student distinct from its initialization.
    xt = owner.sample_initial_latents(shape)
    optimizer = torch.optim.AdamW([p for n,p in model.named_parameters() if '.student.' in n],lr=.002)
    velocity = owner._predict_velocity(wrapper,xt,(owner.video_sigmas_cpu[0],owner.audio_sigmas_cpu[0]),feeds,shape)
    loss = sum(t.square().mean() for t in velocity)
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    original = {n:t.detach().clone() for n,t in model.state_dict().items()}
    torch.manual_seed(93)
    with torch.no_grad():oracle = owner.run_back_simulation(feeds,shape,3,False)
    table = torch.full((1004,1,6,6,32),float('nan'),dtype=torch.bfloat16)
    bits = grid.rollout_bits(api)
    with torch.no_grad():
        for step,pair in enumerate(bits):
            times=torch.tensor(pair,dtype=torch.int32).view(torch.float32)
            temb=model.time_embedder(model.time_proj(times).float())
            table[1000+step,0]=torch.stack(model.transformer_blocks[0].adaln_proj(temb))
    # Use the actual source's parameter-free Grid handle, without importing its
    # global training registration. Only the small controller below is a double.
    from distillschedule.training.upstream import PinnedSource
    from distillschedule.workloads.best_known.registry import repository_root,file_sha256
    file='src/distillschedule/backends/training/minimax_h3/grid_adaln.py'
    module=file[:-3].replace('/','.')
    loader=PinnedSource(repository_root(),{file:file_sha256(repository_root()/file)},
        definitions={module:['GridAdaLNProjectionHandle']},
        definition_globals={module:dict(nn=torch.nn,torch=torch,GridAdaLNController=object)})
    controller=NS(payload=dict(schema_version='h3_a100.adaln_grid.v1',grid_size=1000,num_entries=1004,
        num_blocks=1,hidden_size=32,rows_per_entry=6,modulation_chunks=6,video_shift=6.0,audio_shift=3.0),
        _entry_by_timestep_bits={tuple(b):1000+i for i,b in enumerate(bits)},_table=table,
        evidence_kind='SYNTHETIC_TEST_ONLY_TABLE_OWNER_FROM_ACTUAL_ADALN')
    model.transformer_blocks[0].adaln_proj=loader.import_module(module).GridAdaLNProjectionHandle(controller,0)
    snapshot,table_receipt=grid.current_student_view(model,controller,api,max_table_bytes=table[1000:].numel()*2)
    snapshot={n:t.cpu().clone() for n,t in snapshot.items()}
    layout={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in snapshot.items()}
    budget=dict(max_parameters=10_000_000,max_loaded_bytes=256*2**20,max_attention_elements=10_000_000,
                max_input_elements=1_000_000,host_reserve_bytes=2**20)
    params=grid.parameters(source=source,architecture=arch,geometry=dims,weight_layout=layout,
        seed=93,evidence_kind=grid.FIXTURE,budget=budget)
    restore_rng(before)
    yield NS(api=api,model=model,controller=controller,snapshot=snapshot,parameters=params,feeds=feeds,
        oracle=oracle,original=original,root=tmp_path_factory.mktemp('grid-output'),table_receipt=table_receipt)


def test_grid_original_source_schedule_and_projection_identity(actual_grid_case):
    import torch
    from distillschedule.training import h3_grid_inference as grid
    case=actual_grid_case
    assert grid.SCHEDULE['num_inference_steps']==4 and grid.LORA['r']==128
    assert case.table_receipt['entry_indices']==[1000,1001,1002,1003]
    assert case.table_receipt['timestep_bits'][0]==[0,0]
    assert case.table_receipt['consolidation'].startswith('NOT_PERFORMED')
    assert all('.fake.' not in n and '.adaln_proj.' not in n for n in case.snapshot)
    assert torch.equal(case.snapshot['transformer.rope.inv_freq'],case.model.rope.inv_freq)
    assert 'inv_freq' in case.model.rope._non_persistent_buffers_set
    assert type(case.model).__name__=='MiniMaxH3Transformer3DModel'
    assert torch.cuda.is_initialized() is False


def test_grid_snapshot_does_not_mutate_roles_modes_parameters_rng_or_table_cache(actual_grid_case):
    import torch
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.checkpoint import rng_state
    case=actual_grid_case
    before={n:t.detach().clone() for n,t in case.model.state_dict().items()}
    modes=[(n,m.training) for n,m in case.model.named_modules()]
    flags=[(n,t.requires_grad) for n,t in case.model.named_parameters()]
    random=rng_state(device='cpu')
    keys=dict(case.controller._entry_by_timestep_bits)
    value,receipt=grid.current_student_view(case.model,case.controller,case.api,max_table_bytes=1_000_000)
    assert receipt==case.table_receipt
    assert keys==case.controller._entry_by_timestep_bits
    assert modes==[(n,m.training) for n,m in case.model.named_modules()]
    assert flags==[(n,t.requires_grad) for n,t in case.model.named_parameters()]
    assert torch.equal(random['torch'],rng_state(device='cpu')['torch'])
    assert all(torch.equal(t,case.model.state_dict()[n]) for n,t in before.items())
    assert all(torch.equal(t,value[n]) for n,t in case.snapshot.items())


@pytest.mark.parametrize('fault',['source','runtime','schedule','lora','full_arch','dtype','missing','fake','grid_projection','budget'])
def test_grid_rejects_substitution_or_incomplete_snapshot_before_output(actual_grid_case,fault):
    from distillschedule.training import h3_grid_inference as grid
    case=actual_grid_case;p=deepcopy(case.parameters);weights=dict(case.snapshot)
    if fault=='source':p['implementation']['model_sha256']='0'*64
    elif fault=='runtime':p['runtime']['torch']['version']='0'
    elif fault=='schedule':p['schedule']['num_inference_steps']=8
    elif fault=='lora':p['lora']['r']=64
    elif fault=='full_arch':p['evidence_kind']=grid.NATIVE
    elif fault=='dtype':p['weight_layout']['rollout_modulations']['dtype']='torch.float32'
    elif fault=='missing':weights.pop('transformer.rope.inv_freq')
    elif fault=='fake':p['weight_layout']['transformer.x.lora_A.fake.weight']=dict(shape=[1],dtype='torch.float32')
    elif fault=='grid_projection':p['weight_layout']['transformer.transformer_blocks.0.adaln_proj.linear.weight']=dict(shape=[1],dtype='torch.bfloat16')
    else:p['budget']['max_loaded_bytes']=100
    with pytest.raises(ValueError):grid.export_components(p,weights=weights)


def test_grid_factory_requires_strict_restore_and_preserves_rng(actual_grid_case):
    import torch
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.checkpoint import rng_state
    case=actual_grid_case;before=rng_state(device='cpu')
    components=grid.build_inference_model(case.parameters)
    assert torch.equal(before['torch'],rng_state(device='cpu')['torch'])
    assert not components.state['student'].restored
    with pytest.raises(ValueError,match='restored'):components.generate(components.state,case.feeds,{})
    components.state['student'].load_state_dict(case.snapshot,strict=True)
    assert components.state['student'].restored
    assert all(not t.requires_grad for t in components.state['student'].parameters())
    with pytest.raises(ValueError,match='resumable'):components.admit_training(None)
    with pytest.raises(ValueError,match='timestep'):
        components.state['student'](timestep=torch.tensor([.123]))
    assert components.state['student'].selected is None


def test_actual_grid_export_restore_four_step_rollout_matches_original_projection(actual_grid_case):
    import torch
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.checkpoint import rng_state
    case=actual_grid_case
    components=grid.build_inference_model(case.parameters)
    components.state['student'].load_state_dict(case.snapshot,strict=True)
    before=rng_state(device='cpu')
    result=components.generate(components.state,case.feeds,{})
    assert torch.equal(before['torch'],rng_state(device='cpu')['torch'])
    assert components.state['student'].training is False
    assert result['latent_schema']['video_tokens']['layout']=='BLD'
    for name,value in zip(('video_tokens','audio_tokens'),case.oracle,strict=True):
        torch.testing.assert_close(result[name],value,rtol=0,atol=0)
    changed=deepcopy(case.feeds);changed['prompt_embeds']=changed['prompt_embeds']+.25
    other=components.generate(components.state,changed,{})
    assert not torch.equal(result['video_tokens'],other['video_tokens'])
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize('fault',['missing','wrong_tag','dtype','shape','extra','options'])
def test_grid_new_condition_rejections(actual_grid_case,fault):
    import torch
    from distillschedule.training import h3_grid_inference as grid
    case=actual_grid_case;parts=grid.build_inference_model(case.parameters)
    parts.state['student'].load_state_dict(case.snapshot,strict=True)
    feeds=deepcopy(case.feeds);options={}
    if fault=='missing':feeds.pop('prompt_embeds')
    elif fault=='wrong_tag':feeds['text_token_tags'][0]=0
    elif fault=='dtype':feeds['prompt_embeds']=feeds['prompt_embeds'].float()
    elif fault=='shape':feeds['prompt_embeds']=feeds['prompt_embeds'][:,:2]
    elif fault=='extra':feeds['previous_chunk']=torch.zeros(1)
    else:options={'steps':8}
    with pytest.raises(ValueError):parts.generate(parts.state,feeds,options)


def test_grid_public_export_and_new_process_generate(actual_grid_case):
    import torch
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.wan_native_inference import standalone_config
    from distillschedule.data.preparation import import_preencoded
    case=actual_grid_case;root=case.root
    _,source=load_callable(grid.FACTORY)
    components=grid.export_components(case.parameters,weights=case.snapshot)
    manifest=publish_export(components,root/'export',model=dict(factory=grid.FACTORY,source=source,
        parameters=case.parameters,metadata=dict(components.metadata)),
        identity=dict(model=source,model_metadata=dict(components.metadata),evidence='RANDOM_ORIGINAL_CLASS_CPU_ONLY'),cycle=1,
        primary_only=True)
    config=standalone_config(manifest,root/'inference.json')
    config['algorithm']['factory']='distillschedule.training.h3_grid_inference:refuse_training'
    (root/'inference.json').write_text(json.dumps(config))
    # Existing generic preencoded-data path, not a fabricated prompt encoder.
    fresh=deepcopy(case.feeds);fresh['prompt_embeds']=fresh['prompt_embeds']+.125
    torch.save(fresh,root/'new-condition.pt')
    from distillschedule.workloads.best_known.registry import file_sha256
    (root/'inputs.json').write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW_NOT_TRAINING',features=dict(path='new-condition.pt',sha256=file_sha256(root/'new-condition.pt')))])))
    import_preencoded(root/'inputs.json',root/'prepared')
    script='''import json,sys,torch
from pathlib import Path
from distillschedule.training.config import load_training_config
from distillschedule.cli import main
torch.set_num_threads(1)
root=Path(sys.argv[1])
assert main(['generate',str(root/'inference.json'),'--weights',str(root/'export'),
 '--prepared',str(root/'prepared/manifest.json'),'--output',str(root/'generated'),'--output-kind','latents'])==0
result=json.loads((root/'generated/generation.json').read_text())
assert result['kind']=='INFERENCE_FROM_EXPORTED_WEIGHTS' and result['sample_id']=='NEW_NOT_TRAINING'
assert not any(n in sys.modules for n in ('distillschedule.training.h3_grid_training',
 'distillschedule.training.h3_ar_training','distillschedule.training.h3_ar_portable'))
assert not torch.cuda.is_initialized()
print('ACTUAL_ORIGINAL_GRID_FOUR_STEP_CPU_NEW_PROCESS')
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(root)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                          capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ACTUAL_ORIGINAL_GRID_FOUR_STEP_CPU_NEW_PROCESS' in result.stdout
    assert (root/'generated/_SUCCESS').is_file()


@pytest.mark.parametrize('fault',['merge','small_budget','missing_entry','table_dtype','source_file'])
def test_grid_snapshot_and_source_refuse_incompatible_lifecycle(actual_grid_case,tmp_path,fault):
    import torch
    from copy import copy
    from distillschedule.training import h3_grid_inference as grid
    case=actual_grid_case
    controller=copy(case.controller);controller._entry_by_timestep_bits=dict(controller._entry_by_timestep_bits)
    layer=next(m for m in case.model.modules() if hasattr(m,'merged_adapters'))
    old=list(layer.merged_adapters)
    if fault=='merge':layer.merged_adapters.append('student')
    elif fault=='missing_entry':controller._entry_by_timestep_bits.pop(tuple(grid.rollout_bits(case.api)[0]))
    elif fault=='table_dtype':controller._table=controller._table.float()
    elif fault=='source_file':
        source=deepcopy(case.parameters['source']);source['root']=str(tmp_path)
        for name,sha in source['files'].items():
            path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# not the pinned source\n')
        with pytest.raises(ValueError,match='SHA256'):grid.source_api(source)
        return
    try:
        with pytest.raises((ValueError,KeyError)):
            grid.current_student_view(case.model,controller,case.api,max_table_bytes=1 if fault=='small_budget' else 1_000_000)
    finally:
        layer.merged_adapters[:]=old
    assert not torch.cuda.is_initialized()
