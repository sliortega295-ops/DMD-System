"""W2 public lifecycle: actual original full class, never a scalar stand-in."""

def test_original_nogan_independent_public_factory_exists():
    from distillschedule.training.helios_nogan_public import build_model, build_algorithm
    from distillschedule.data.helios_nogan_reference_inputs import prepare_inputs
    assert callable(build_model) and callable(build_algorithm) and callable(prepare_inputs)

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_training_helios_nogan_bootstrap_cpu import _write, _pin, _equal
from distillschedule.workloads.best_known.registry import repository_root, file_sha256, fingerprint


def _stack():
    assert importlib.metadata.version('torch') == '2.10.0+cpu'
    assert importlib.metadata.version('diffusers') == '0.37.1'
    assert importlib.metadata.version('peft') == '0.17.1'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)


def _capture(root, config, *, count, seed_base):
    from distillschedule.training.helios_inputs import HeliosCohortTask, HeliosCohortSources, prepare_stream
    from distillschedule.data.helios_nogan_reference_inputs import SCHEMA, SCOPE, prepare_inputs
    from distillschedule.training.helios_public_assets import FIXTURE
    root.mkdir()
    sources = []
    for index in range(count):
        seed, samples = seed_base+index, []
        for offset in range(5):
            names = []
            for rank in range(16):
                local_seed = seed+index*1000000+rank*5+offset
                sections = next(n for n in range(2,64) if all(torch.randint(0,n,(1,),
                    generator=torch.Generator().manual_seed(s)).item() > 0 for s in (local_seed,local_seed+999999)))
                value = .01*(index+1)+.0001*(rank*5+offset)+.001*seed_base
                path = root/f'feature-{index}-{offset}-{rank}.pt'
                text = torch.full((512,4096),value,dtype=torch.bfloat16)
                text[0,0] = rank*5+offset  # exact BF16 source distinction, not just archive names
                torch.save(dict(vae_latent=torch.tensor(value,dtype=torch.bfloat16).expand(sections,16,9,48,80),
                    prompt_embed=text,
                    attention_mask=torch.ones(512,dtype=torch.bool)),path)
                names.append(str(path))
            samples.append(names)
        sources.append(HeliosCohortSources(seed,tuple(tuple(r) for r in samples)))
    stream = prepare_stream(HeliosCohortTask(0).request,sources,config,root/'stream')
    source = _write(root/'source.json',dict(schema_version=SCHEMA,scope=SCOPE,evidence_kind=FIXTURE,
        stream=_pin(root/'stream/stream.json'),input_config=_pin(config)))
    prepared = prepare_inputs(source,root/'prepared')
    return dict(stream=str(root/'stream/stream.json'),source=str(source),prepared=str(prepared),document=stream)


def _cold(root):
    from safetensors.torch import save_file
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.training.helios_public_assets import FIXTURE
    from distillschedule.training.helios_nogan_bootstrap import _model
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS
    from runtime.precast_checkpoint import convert_pretrained_safetensors
    from runtime.sparse_ema import SparseShardEMA
    from runtime.gst_pool_contract import write_negative_prompt_asset
    cls, roles, rows, engines = load_reference_model_class(backend='torch-sdpa'),{},{},{}
    with torch.random.fork_rng(devices=[]):
        for index,(role,native,folder) in enumerate((('generator','generator','transformer_ode'),('critic','score','transformer'))):
            torch.manual_seed(314+index)
            source = root/native/folder
            if not source.exists():
                model = cls(patch_size=(1,2,2),num_attention_heads=1,attention_head_dim=16,in_channels=16,
                    out_channels=16,text_dim=4096,freq_dim=16,ffn_dim=32,num_layers=1,rope_dim=(4,6,6),
                    is_use_gan=False,is_use_gan_final=False,is_use_gan_hooks=False,**TRANSFORMER_ADDITIONAL_KWARGS)
                with torch.no_grad():
                    model.proj_out.weight.normal_(0.,.02)
                for name,p in model.named_parameters():
                    p.data = p.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules) else torch.bfloat16)
                model.save_config(source)
                weights = source/'model.safetensors'
                save_file({n:t.detach().contiguous() for n,t in model.state_dict().items()},weights,metadata={'evidence_kind':FIXTURE})
                _write(source/'diffusion_pytorch_model.safetensors.index.json',dict(metadata={},
                    weight_map={n:weights.name for n in model.state_dict()}))
                del model
            weights = source/'model.safetensors'
            roles[role] = dict(config=_pin(source/'config.json'),weights=[_pin(weights)])
            precast = root/(native+'-precast')
            if not precast.exists():
                convert_pretrained_safetensors(source,precast,target_dtype=torch.bfloat16)
            for key,directory,identity in ((native+'_model',root/native,folder+'/config.json'),
                    (native+'_precast',precast,'distillschedule_precast_manifest.json')):
                rows[key] = dict(path=str(directory),identity_file=identity,sha256=file_sha256(directory/identity))
            value = {role:roles[role], 'base_evidence_kind':FIXTURE}
            model,_ = _model(cls,value,role,100000000)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=2e-6 if role=='generator' else 4e-7,betas=(0.,.999),eps=1e-8,weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            engines[role] = FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=0,global_batch=1,dp_world_size=1,gradient_clip=1.)
    ema = SparseShardEMA(engines['generator'].module.named_parameters(),decay=.99,update_after_step=750,
                         base_model=str(root/'generator'))
    negative_root = root/'negative'
    if not negative_root.exists():
        write_negative_prompt_asset(negative_root,embedding=torch.zeros(1,512,4096,dtype=torch.bfloat16),model_root=root)
    rows['negative_prompt'] = dict(path=str(negative_root),identity_file='manifest.json',sha256=file_sha256(negative_root/'manifest.json'))
    return engines,ema,roles,rows


def _original_cycle(engines,ema,feeds,index):
    from distillschedule.data.helios_nogan_reference_inputs import decode,descriptor,HeliosCohortTask
    from distillschedule.runtime.helios_planner_cycle import _Observer,_prepare_cycle
    from distillschedule.search.helios_implementation import HeliosImplementation
    from distillschedule.training.helios_nogan_public import _p0
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
    trace = decode(feeds,HeliosCohortTask(index))
    class Bound:
        @property
        def optimization_step(self):
            return ema.optimization_step
        def step(self):
            return ema.step(engines['generator'].module.parameters())
    observer = _Observer(torch=torch,device=torch.device('cpu'),rank=0,
        impl=HeliosImplementation('no-gan','world16-fsdp',512,1,False),generator=engines['generator'],critic=engines['critic'],
        ema=Bound(),mode='cpu-diagnostic',version_offsets=dict(generator=0,critic=0,ema=0),cpu_trace=trace)
    negative = torch.zeros(1,512,4096,dtype=torch.bfloat16)
    prepared,graph,order = _prepare_cycle(observer=observer,p0=_p0(),trace=trace,steps=trace.task.source_steps,
        args=SimpleNamespace(timing_mode=False,profile_mode=False,profile_diagnostics=False,ema_device='cpu'),
        policy=StaticFSDP2P0Policy(),negative=negative,cfg=None,diagnostics=StaticProfileDiagnostics(enabled=False,rank=0),
        generator_free_memory=None)
    with observer.installed():
        prepared.run()
    validation = observer.finish()
    assert observer.versions() == graph.metadata['final_versions']
    return dict(task=trace.task.to_dict(),source_steps=list(trace.task.source_steps),descriptor=descriptor(trace.reference_document),
        rank_control_hashes=[trace._control[r].trace_hash for r in range(16)],dispatch=prepared.receipt(),
        actions=observer.actions,semantic_validation=validation)


def original_capacity_canary(root):
    """One ORIGINAL full-geometry cycle; shell imposes wall/address-space bounds."""
    import yaml
    from omegaconf import OmegaConf
    from distillschedule.strategies.wan_capture import native_primitives
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.helios_step_owner import _SparseEMAView
    from distillschedule.training.checkpoint import cpu_tree,rng_state
    from distillschedule.training.helios_inputs import source_fingerprint
    from distillschedule.data import PreparedDataset
    _stack(); root = Path(root); root.mkdir(parents=True,exist_ok=True)
    assert not (root/'canary.json').exists(), 'preserve existing canary; choose explicit reuse instead of overwrite'
    native_primitives(source_only=True)
    from helios.utils.train_config import Args
    template = yaml.safe_load((repository_root()/'configs/distillschedule/templates/helios_stage3_384x640.yaml').read_text())
    template['training_config']['is_random_drop'] = False
    template['training_config']['noise_corrupt_clean_prob_history'] = 1.
    config = root/'capture.yaml'; config.write_text(yaml.safe_dump(template))
    args = OmegaConf.merge(OmegaConf.structured(Args),template)
    args.training_config.max_grad_norm = args.training_config.max_grad_norm_critic = 1.
    args.training_config.max_train_steps = 25
    args.validation_config.use_kv_cache = False
    args_path = _write(root/'args.json',OmegaConf.to_container(args,resolve=True))
    engines,ema,roles,rows = _cold(root)
    warm = _capture(root/'warm-inputs',config,count=2,seed_base=41)
    dataset = PreparedDataset(warm['prepared'])
    first = _original_cycle(engines,ema,dataset.sample(0),0)
    views = engine_state(engines); views['ema'] = _SparseEMAView(ema,engines['generator'])
    for role in ('generator','critic'):
        assert all(s['step'].item() == (1 if role=='generator' else 5) for s in engines[role].optimizer.state.values())
        assert any(torch.count_nonzero(s['exp_avg']) for s in engines[role].optimizer.state.values())
    torch.save(dict(native={k:cpu_tree(v.state_dict()) for k,v in views.items()},rng=cpu_tree(rng_state())),root/'canary-state.pt')
    _write(root/'canary-cycle.json',first)
    _write(root/'canary.json',dict(source=source_fingerprint(),warm=warm,roles=roles,rows=rows,args=_pin(args_path),
        versions=dict(generator=1,critic=5,ema=5),geometry=[1,16,9,48,80],heads=1,head_dim=16,layers=1,ffn=32,
        policy='ORIGINAL_NO_GAN_LORA256_CLIP1',cuda_initialized=torch.cuda.is_initialized()))
    print('W2_ORIGINAL_FULL_GEOMETRY_CPU_ONE_COLD_CYCLE_G1_C5_EMA5_CAPACITY_PASS',flush=True)


def fixture_files(root):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    from distillschedule.training.helios_public_assets import FIXTURE,runtime_identity
    from distillschedule.training.helios_inputs import HeliosCohortTask,source_fingerprint
    from distillschedule.training.helios_step_owner import _SparseEMAView
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import restore_rng
    from distillschedule.data import PreparedDataset
    from distillschedule.training.helios_nogan_public import SCHEMA
    from distillschedule.data.helios_nogan_reference_inputs import SCOPE,PRECISION
    _stack(); root = Path(root)
    if (root/'complete.json').exists():
        result = json.loads((root/'complete.json').read_text())
        assert result['source'] == source_fingerprint(), 'source changed: preserve old fixture, use a new explicit directory'
        return result
    if not (root/'canary.json').exists():
        original_capacity_canary(root)
    canary = json.loads((root/'canary.json').read_text())
    assert canary['source'] == source_fingerprint()
    engines,ema,roles,rows = _cold(root)
    saved = torch.load(root/'canary-state.pt',weights_only=True)
    views = engine_state(engines); views['ema'] = _SparseEMAView(ema,engines['generator'])
    for name,view in views.items():
        view.load_state_dict(saved['native'][name])
    restore_rng(saved['rng'])
    dataset = PreparedDataset(canary['warm']['prepared'])
    second = _original_cycle(engines,ema,dataset.sample(1),1)
    logical = root/'logical'; logical.mkdir()
    manifest = dict(schema_version='distillschedule.logical_checkpoint.v3',global_step=10,
        source=bootstrap.CPU_WARM,components={},ema=dict(optimization_step=10))
    for role,engine in engines.items():
        selected = {n:p for n,p in engine.module.named_parameters() if p.requires_grad}
        assert set(engine.optimizer.state) == set(selected.values())
        records,offset = {},0
        for name,p in selected.items():
            records[name] = dict(shape=list(p.shape),numel=p.numel(),offset_elements=offset); offset += p.numel()
        payloads = {}
        for kind,key in (('master',None),('first_moment','exp_avg'),('second_moment','exp_avg_sq')):
            tensors = [p.detach() if key is None else engine.optimizer.state[p][key] for p in selected.values()]
            payload = logical/(role+'-'+kind+'.f32')
            torch.cat([t.flatten() for t in tensors]).numpy().tofile(payload)
            payloads[kind] = dict(path=payload.name,bytes=payload.stat().st_size,dtype='torch.float32',sha256=file_sha256(payload))
        manifest['components'][role] = dict(parameter_names=list(selected),records=records,logical_numel=offset,
            optimizer_step=engine.global_steps,parameter_version=engine.global_steps,
            scheduler_state=engine.lr_scheduler.state_dict(),payloads=payloads)
        assert engine.global_steps == (2 if role=='generator' else 10)
        assert all(state['step'].item() == engine.global_steps for state in engine.optimizer.state.values())
        assert any(torch.count_nonzero(state['exp_avg']) for state in engine.optimizer.state.values())
    ema.save_pretrained(logical/'ema',max_shard_bytes=8*1024**2)
    manifest['manifest_hash'] = fingerprint(manifest); _write(logical/'manifest.json',manifest)
    rows['logical_checkpoint'] = dict(path=str(logical),identity_file='manifest.json',sha256=file_sha256(logical/'manifest.json'),
        ema_manifest=dict(identity_file='ema/sparse_ema_manifest.json',sha256=file_sha256(logical/'ema/sparse_ema_manifest.json')))
    documents = PreparedDataset(canary['warm']['prepared']).manifest['provenance']['documents']
    input_documents = []
    for pin in documents:
        if pin['kind'] == 'file':
            input_documents.append(_pin(Path(pin['path'])))
    input_documents += [_pin(dataset.path), *[_pin(dataset.path.parent/r['features']['path']) for r in dataset.manifest['records']]]
    # Every actual feature/control/payload is pinned, not a string-only claim.
    assert len(input_documents) >= 65
    receipt = dict(schema_version=bootstrap.CPU_WARM_RECEIPT,evidence_kind=bootstrap.CPU_WARM,
        source=bootstrap.source_identity(),runtime=runtime_identity(),request=HeliosCohortTask(0).request.to_dict(),
        framework_source=source_fingerprint(),execution=dict(device='cpu',world_size=1,source_rank=0,source_world_size=16,
            historical_step0_inputs=False,pretrained_quality='NOT_RUN'),physical_initial=dict(generator=0,critic=0,ema=0),
        physical_final=dict(generator=2,critic=10,ema=10),cycles=[json.loads((root/'canary-cycle.json').read_text()),second],
        input_documents=input_documents,prepared_inputs=_pin(dataset.path),logical_payloads={r:{k:p['sha256'] for k,p in c['payloads'].items()}
            for r,c in manifest['components'].items()},ema_manifest=_pin(logical/'ema/sparse_ema_manifest.json'))
    receipt_path = _write(root/'actual-warm-receipt.json',receipt)
    warm = dict(schema_version=bootstrap.SCHEMA,scope=bootstrap.SCOPE,precision=PRECISION,base_evidence_kind=FIXTURE,
        source=bootstrap.source_identity(),runtime=runtime_identity(),request=HeliosCohortTask(0).request.to_dict(),
        native_assets=dict(schema_version='distillschedule.helios_reference_assets.v1',assets=rows),
        budget=dict(max_parameters=20000000,max_loaded_bytes=400000000),**roles)
    proof = dict(evidence_kind=bootstrap.CPU_WARM,update_source='ORIGINAL_W2_HELPERS',source=warm['source'],runtime=warm['runtime'],
        request=warm['request'],policy=bootstrap.native_policy(),bases=roles,native_assets=warm['native_assets'],
        inputs=[_pin(receipt_path)],clocks=dict(global_step=10,generator=2,critic=10,ema=10))
    warm['provenance'] = _pin(_write(root/'warm-provenance.json',proof))
    warm_path = _write(root/'warm-assets.json',warm)
    reloaded = bootstrap.materialize_bootstrap(warm_path)
    for role,e in engines.items():
        assert _equal(e.optimizer.state_dict(),reloaded.engines[role].optimizer.state_dict())
        assert _equal({n:p.detach() for n,p in e.module.named_parameters()},
                      {n:p.detach() for n,p in reloaded.engines[role].module.named_parameters()})
    assert _equal(ema.shadows,reloaded.ema.shadows)
    del views,engines,ema,reloaded,e; gc.collect()
    training = _capture(root/'training-inputs',root/'capture.yaml',count=3,seed_base=71)
    assets = _write(root/'public-assets.json',dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,
        bootstrap=_pin(warm_path),args=_pin(root/'args.json'),input_config=_pin(root/'capture.yaml')))
    result = dict(assets=str(assets),prepared=training['prepared'],warm=str(warm_path),
        warm_receipt=str(receipt_path),source=source_fingerprint(),root=str(root))
    _write(root/'complete.json',result)
    if os.environ.get('HELIOS_NOGAN_PUBLIC_EVIDENCE'):
        evidence = Path(os.environ['HELIOS_NOGAN_PUBLIC_EVIDENCE'])/'artifacts'
        _write(evidence/'actual-warm-receipt.json',receipt)
        _write(evidence/'fixture-identity.json',result)
    return result


@pytest.fixture(scope='module')
def fixture_assets(tmp_path_factory):
    root = os.environ.get('HELIOS_NOGAN_PUBLIC_FIXTURE')
    return fixture_files(Path(root) if root else tmp_path_factory.mktemp('helios-nogan-original')/'local')


def config_for(files,output,cycles=3,resume=None):
    from distillschedule.training.config import TrainingConfig
    trainer = dict(cycles=cycles,seed=773,device='cpu',output_dir=str(output),checkpoint_every=1)
    if resume:
        trainer['resume_from'] = str(resume)
    return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model=dict(factory='distillschedule.training.helios_nogan_public:build_model',parameters=dict(asset_manifest=files['assets'])),
        algorithm=dict(factory='distillschedule.training.helios_nogan_public:build_algorithm'),
        data=dict(prepared=files['prepared'],sampling=dict(kind='finite',shuffle=False)),
        execution=dict(adapter='program',mode='reference'),trainer=trainer),source=Path(output).parent/'training.json')


def run_capturing(config,monkeypatch):
    from distillschedule.training import session
    owners,receipts,hooks = [],[],[]
    original = session.construct
    def construct(*args,**kwargs):
        result = original(*args,**kwargs); owner = result[0].state['native_cycle']; owners.append(owner)
        assert owner.versions == dict(generator=2,critic=10,ema=10)
        for e in owner._engines.values():
            for p in e.module.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad:hooks.append(float(grad.float().norm())))
        return result
    monkeypatch.setattr(session,'construct',construct)
    def profile(frame,event,arg):
        if event == 'return' and frame.f_code.co_name == 'native_cycle' and isinstance(arg,dict):
            receipts.append(deepcopy(arg))
    try:
        sys.setprofile(profile)
        result = session.train(config)
    finally:
        sys.setprofile(None); monkeypatch.setattr(session,'construct',original)
    return result,owners[0],receipts,hooks


def test_actual_original_warm_public_two_fresh_cycles(fixture_assets,tmp_path,monkeypatch):
    from distillschedule.training.checkpoint import read_checkpoint
    result,owner,receipts,hooks = run_capturing(config_for(fixture_assets,tmp_path/'train',2),monkeypatch)
    assert result['status'] == 'COMPLETE' and result['cursor'] == 2 and owner.cycle == 2
    assert owner.versions == dict(generator=4,critic=20,ema=20) and result['final_versions'] == dict(native_cycle=2)
    assert len(receipts) == 2 and any(hooks)
    for index,row in enumerate(receipts):
        assert row['source_steps'] == list(range(10+5*index,15+5*index))
        assert row['dispatch']['status'] == 'PASS' and row['semantic_validation']['census'] == 'PASS'
        assert row['dispatch']['initial_versions'] == owner.expected(index)
        assert row['dispatch']['final_versions'] == owner.expected(index+1)
        assert len(row['actions']) == 16
        assert sum(a['counts']['generator_commit'] for a in row['actions']) == 1
        assert sum(a['counts']['critic_commit'] for a in row['actions']) == 5
        assert len([a for a in row['actions'] if a['action_id'].startswith('e')]) == 5
        rolls = [a for a in row['actions'] if a['action_id'].startswith('r')]
        assert len(rolls) == len({a['trace_entries'][0]['trace_identity_sha256'] for a in rolls}) == 5
        assert row['telemetry']['max_allocated_bytes'] is None and row['claimable_speedup'] is False
        assert all(a['calibration_seconds'] is None for a in row['actions'])
    assert all(any(n.startswith(p+'.lora_') for n,t in owner.generator.module.named_parameters() if t.requires_grad)
               for p in ('patch_short','patch_mid','patch_long'))
    assert not any(n.startswith(('patch_short.','patch_mid.','patch_long.','patch_embedding.','gan_heads.','gan_final_head.'))
                   for n,p in owner.critic.module.named_parameters() if p.requires_grad)
    for role,e in owner._engines.items():
        assert all(s['step'].item() == owner.versions[role] for s in e.optimizer.state.values())
        assert any(torch.count_nonzero(s['exp_avg']) for s in e.optimizer.state.values())
    snapshot = read_checkpoint(result['checkpoint'],identity=result['identity'])
    assert snapshot['cursor'] == snapshot['cycle'] == 2
    assert not torch.cuda.is_initialized()


def test_fresh_process_exact_uninterrupted3_vs1_plus2(fixture_assets,tmp_path,monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.checkpoint import read_checkpoint
    full,owner,receipts,_ = run_capturing(config_for(fixture_assets,tmp_path/'full'),monkeypatch)
    expected = read_checkpoint(full['checkpoint'],identity=full['identity'])
    partial = session.train(config_for(fixture_assets,tmp_path/'split',1))
    prefix = read_checkpoint(Path(full['checkpoint']).parent/'cycle-000001',identity=full['identity'])
    split = read_checkpoint(partial['checkpoint'],identity=partial['identity'])
    assert _equal(prefix['state'],split['state']) and _equal(prefix['rng'],split['rng'])
    torch.save(dict(state=expected['state'],rng=expected['rng'],receipts=receipts[1:]),tmp_path/'expected.pt')
    _write(tmp_path/'resume.json',config_for(fixture_assets,tmp_path/'resumed',resume=partial['checkpoint']).to_dict())
    script = r"""
from pathlib import Path
import torch,sys,pytest
from test_training_helios_nogan_public_cpu import run_capturing,_equal
from distillschedule.training.config import load_training_config
from distillschedule.training.checkpoint import read_checkpoint,rng_state
root=Path(sys.argv[1])
with pytest.MonkeyPatch.context() as patch:
 result,owner,receipts,_=run_capturing(load_training_config(root/'resume.json'),patch)
expected=torch.load(root/'expected.pt',weights_only=True)
actual=read_checkpoint(result['checkpoint'],identity=result['identity'])
assert _equal(expected['state'],actual['state']), 'named weights/Adam/LR/EMA/clocks/native RNG/source cursor differ'
assert _equal(expected['rng'],actual['rng']) and _equal(actual['rng'],rng_state()), 'outer RNG differs'
assert _equal(expected['receipts'],receipts), 'actual native action/forward/version receipts differ'
assert owner.versions==dict(generator=5,critic=25,ema=25) and owner.cycle==3
assert result['resume_cycle']==1 and result['cursor']==3 and result['final_versions']==dict(native_cycle=3)
assert not torch.cuda.is_initialized()
print('W2_FULL_ORIGINAL_CPU_FRESH_PROCESS_EXACT_3_VS_1_PLUS_2_G5_C25_EMA25_PASS')
"""
    process = subprocess.run([sys.executable,'-c',script,str(tmp_path)],cwd=repository_root(),env=dict(os.environ),
                             text=True,capture_output=True,timeout=600)
    assert process.returncode == 0, process.stdout+process.stderr
    if os.environ.get('HELIOS_NOGAN_PUBLIC_EVIDENCE'):
        (Path(os.environ['HELIOS_NOGAN_PUBLIC_EVIDENCE'])/'artifacts/restart-process.stdout').write_text(process.stdout)
    assert 'FRESH_PROCESS_EXACT' in process.stdout


def test_partial_g_teacher_c_and_restore_failures_poison_all_owners(fixture_assets,tmp_path,monkeypatch):
    from distillschedule.training import session
    from distillschedule.data import PreparedDataset
    from distillschedule.data.helios_nogan_reference_inputs import decode,descriptor,HeliosCohortTask
    from distillschedule.runtime.helios_planner_cycle import _initial_adapter_state
    from distillschedule.training.checkpoint import rng_state,cpu_tree
    components,_,_,_ = session.construct(config_for(fixture_assets,tmp_path/'unused'))
    owner = components.state['native_cycle']; dataset = PreparedDataset(fixture_assets['prepared'])
    first = decode(dataset.sample(0),HeliosCohortTask(0)); owner.run(first,descriptor(first.reference_document))
    safe = owner.state_dict()
    trace = decode(dataset.sample(1),HeliosCohortTask(1)); row = descriptor(trace.reference_document)
    parameter_ids = {r:{n:id(p) for n,p in e.module.named_parameters()} for r,e in owner._engines.items()}
    for failure in ('g','teacher','c'):
        calls = []
        def fail_step(optimizer,args,kwargs):
            calls.append(True)
            if failure == 'g' or len(calls) == 2:
                raise RuntimeError('injected original optimizer post-commit failure')
        def fail_teacher(model,args,kwargs):
            if not _initial_adapter_state(model):
                raise RuntimeError('injected original Teacher admission failure')
        handle = (owner.critic.module.register_forward_pre_hook(fail_teacher,with_kwargs=True) if failure == 'teacher'
            else (owner.generator if failure == 'g' else owner.critic).optimizer.register_step_post_hook(fail_step))
        outer = cpu_tree(rng_state())
        try:
            with pytest.raises(RuntimeError,match='injected original'):
                owner.run(trace,row)
        finally:
            handle.remove()
        assert owner.poisoned and _equal(outer,rng_state())
        with pytest.raises(ValueError,match='poisoned'):
            owner.state_dict()
        with pytest.raises(ValueError,match='poisoned'):
            owner.run(trace,row)
        before = {r:{n:p.detach().clone() for n,p in e.module.named_parameters()} for r,e in owner._engines.items()}
        bad = deepcopy(safe)
        first_g = next(iter(bad['native']['generator.weights']['tensors']))
        bad['native']['generator.weights']['tensors'][first_g].add_(.25)
        cstate = bad['native']['critic.optimizer']['optimizer']['state']
        next(iter(cstate.values()))['exp_avg'].flatten()[0] = float('nan')
        with pytest.raises(ValueError):
            owner.load_state_dict(bad)
        assert all(_equal(before[r],{n:p.detach() for n,p in e.module.named_parameters()}) for r,e in owner._engines.items())
        owner.load_state_dict(safe)
        assert not owner.poisoned and _equal(safe,owner.state_dict())
        assert parameter_ids == {r:{n:id(p) for n,p in e.module.named_parameters()} for r,e in owner._engines.items()}
    restore = owner._views['critic.optimizer'].load_state_dict
    def fail_restore(value):
        raise RuntimeError('injected staged restore copy failure')
    monkeypatch.setattr(owner._views['critic.optimizer'],'load_state_dict',fail_restore)
    with pytest.raises(RuntimeError,match='staged restore'):
        owner.load_state_dict(safe)
    assert owner.poisoned
    monkeypatch.setattr(owner._views['critic.optimizer'],'load_state_dict',restore)
    owner.load_state_dict(safe)
    assert _equal(safe,owner.state_dict())
    owner.run(trace,row)
    assert owner.versions == dict(generator=4,critic=20,ema=20)
    assert safe['native']['generator.clocks']['counters']['global_steps'] == 3
    assert all(s['step'].item() == 3 for s in safe['native']['generator.optimizer']['optimizer']['state'].values())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('key,value', [('max_grad_norm',10.),('learning_rate',1e-5),('ema_start_step',751),('is_use_gan',True)])
def test_public_native_policy_changes_reject_before_model_construction(fixture_assets,tmp_path,monkeypatch,key,value):
    from distillschedule.training import helios_nogan_public as public
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    assets = json.loads(Path(fixture_assets['assets']).read_text())
    args = json.loads(Path(assets['args']['path']).read_text()); args['training_config'][key] = value
    assets['args'] = _pin(_write(tmp_path/'args.json',args))
    path = _write(tmp_path/'assets.json',assets)
    def forbidden(*a,**kw):
        raise AssertionError('invalid public policy reached model materialization')
    monkeypatch.setattr(bootstrap,'materialize_bootstrap',forbidden)
    with pytest.raises(ValueError,match='W2 original'):
        public.build_model(dict(asset_manifest=str(path)))


def test_reader_only_w1_scope_empty_adam_and_cursor_reject(fixture_assets,tmp_path,monkeypatch):
    from distillschedule.training import helios_nogan_public as public
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    from distillschedule.training import session
    assets = json.loads(Path(fixture_assets['assets']).read_text())
    warm = json.loads(Path(assets['bootstrap']['path']).read_text())
    proof = json.loads(Path(warm['provenance']['path']).read_text())
    proof['evidence_kind'],proof['update_source'] = bootstrap.READER_ONLY,'DECLARED_READER_GRADIENTS_NOT_W2'
    warm['provenance'] = _pin(_write(tmp_path/'reader-proof.json',proof))
    assets['bootstrap'] = _pin(_write(tmp_path/'reader-warm.json',warm))
    with pytest.raises(ValueError,match='state-reader fixture'):
        public.build_model(dict(asset_manifest=str(_write(tmp_path/'reader-assets.json',assets))))
    assets = json.loads(Path(fixture_assets['assets']).read_text()); assets['scope'] = 'CPU1_RANK0_PER_SAMPLE_ORIGINAL_W1_NOT_WORLD16'
    with pytest.raises(ValueError,match='independent W2 public'):
        public.admit_assets(dict(asset_manifest=str(_write(tmp_path/'w1-assets.json',assets))))
    components,_,_,_ = session.construct(config_for(fixture_assets,tmp_path/'unused'))
    owner = components.state['native_cycle']
    from distillschedule.data import PreparedDataset
    from distillschedule.data.helios_nogan_reference_inputs import decode,descriptor,HeliosCohortTask
    feeds = PreparedDataset(fixture_assets['prepared']).sample(0)
    trace = decode(feeds,HeliosCohortTask(0)); owner.run(trace,descriptor(trace.reference_document))
    safe = owner.state_dict()
    for change in ('adam','clocks','ema','cursor','flags','precision'):
        bad = deepcopy(safe)
        if change == 'adam':
            bad['native']['critic.optimizer']['optimizer']['state'] = {}
        elif change == 'clocks':
            bad['native']['critic.clocks']['counters']['global_steps'] = 10
        elif change == 'ema':
            bad['native']['ema']['optimization_step'] = 10
        elif change == 'cursor':
            bad['consumed'] = []
        elif change == 'flags':
            first = next(iter(bad['native']['critic.weights']['requires_grad']))
            bad['native']['critic.weights']['requires_grad'][first] = 1
        else:
            tensors = bad['native']['generator.weights']['tensors']; first = next(iter(tensors))
            tensors[first] = tensors[first].to(torch.bfloat16)
        with pytest.raises(ValueError):
            owner.load_state_dict(bad)
        assert _equal(safe,owner.state_dict())
    with pytest.raises(ValueError,match='inner/outer cursor'):
        owner.validate_checkpoint_progress(safe,dict(cycle=1,cursor=2,versions=dict(native_cycle=1)))
    with pytest.raises(ValueError,match='replay|reordered'):
        owner.run(trace,descriptor(trace.reference_document))
    assert not torch.cuda.is_initialized()


def test_native_generator_global_step_is_not_relabelled_as_cold_step0():
    from distillschedule.strategies.wan_capture import native_primitives
    sample = native_primitives(source_only=True).post.sample_dynamic_timestep
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777)
        cold = sample(B=4096,global_step=0,dynamic_step=500,dynamic_alpha=4.,dynamic_beta=1.5,
                      dynamic_sample_type='beta',device='cpu')
        torch.manual_seed(777)
        actual_source = sample(B=4096,global_step=10,dynamic_step=500,dynamic_alpha=4.,dynamic_beta=1.5,
                               dynamic_sample_type='beta',device='cpu')
    assert not torch.equal(cold,actual_source), 'source step10 data must not be described as source step0 warm data'


def test_d1_nested_identity_is_canonical_plain_json_not_mapping_proxy():
    from types import MappingProxyType
    from distillschedule.training.helios_nogan_public import plain
    frozen = MappingProxyType({'outer':MappingProxyType({'value':(MappingProxyType({'a':1}),)})})
    assert plain(frozen) == {'outer':{'value':[{'a':1}]}}
    assert json.loads(json.dumps(plain(frozen))) == plain(frozen)


def test_outer_zero_actual_warm_boundary_recovers_first_partial_cycle(fixture_assets,tmp_path):
    from distillschedule.training import session
    from distillschedule.data import PreparedDataset
    from distillschedule.data.helios_nogan_reference_inputs import decode,descriptor,HeliosCohortTask
    from distillschedule.training.checkpoint import cpu_tree,rng_state
    owner = session.construct(config_for(fixture_assets,tmp_path/'zero'))[0].state['native_cycle']
    safe = owner.state_dict()
    assert safe['cycle'] == 0 and safe['consumed'] == [] and safe['versions'] == dict(generator=2,critic=10,ema=10)
    assert safe['native']['generator.clocks']['counters'] == dict(global_steps=2,global_samples=2,_step_calls=0)
    assert safe['native']['critic.clocks']['counters'] == dict(global_steps=10,global_samples=10,_step_calls=0)
    # RED first: outer0 is a complete WARM boundary, not a cold inner Step0.
    owner.validate_state_dict(safe)
    with pytest.raises(ValueError,match='complete original W2 cycle'):
        owner.validate_checkpoint_boundary()  # Generic trainer publication policy stays unchanged.
    trace = decode(PreparedDataset(fixture_assets['prepared']).sample(0),HeliosCohortTask(0))
    row = descriptor(trace.reference_document)
    before_rng = cpu_tree(rng_state())
    def fail_after_real_g(optimizer,args,kwargs):
        raise RuntimeError('first cycle real G Adam already mutated')
    handle = owner.generator.optimizer.register_step_post_hook(fail_after_real_g)
    try:
        with pytest.raises(RuntimeError,match='Adam already mutated'):
            owner.run(trace,row)
    finally:
        handle.remove()
    assert owner.poisoned and _equal(before_rng,rng_state())
    before = {r:{n:p.detach().clone() for n,p in e.module.named_parameters()} for r,e in owner._engines.items()}
    for corruption in ('second_role','ema'):
        bad = deepcopy(safe)
        next(iter(bad['native']['generator.weights']['tensors'].values())).add_(1.)
        if corruption == 'second_role':
            state = next(iter(bad['native']['critic.optimizer']['optimizer']['state'].values()))
            state['exp_avg'].flatten()[0] = float('nan')
        else:
            bad['native']['ema']['shadows'][-1].flatten()[0] = float('nan')
        with pytest.raises(ValueError):
            owner.load_state_dict(bad)
        assert owner.poisoned
        assert all(_equal(before[r],{n:p.detach() for n,p in e.module.named_parameters()}) for r,e in owner._engines.items())
    owner.load_state_dict(safe)
    assert not owner.poisoned and _equal(safe,owner.state_dict())
    owner.run(trace,row)
    actual = owner.state_dict()
    oracle = session.construct(config_for(fixture_assets,tmp_path/'oracle'))[0].state['native_cycle']
    oracle.run(trace,row)
    assert _equal(actual,oracle.state_dict()), 'restored genuine warm boundary changed the next original update'
    assert safe['versions'] == dict(generator=2,critic=10,ema=10) and safe['cycle'] == 0
    assert all(s['step'].item() == 2 for s in safe['native']['generator.optimizer']['optimizer']['state'].values())


def test_complete_warm_outer_zero_state_is_staged_restorable(fixture_assets):
    from distillschedule.training.helios_nogan_public import build_model
    owner = build_model(dict(asset_manifest=fixture_assets['assets'])).state['native_cycle']
    snapshot = owner.state_dict()
    assert snapshot['cycle'] == 0 and snapshot['versions'] == dict(generator=2,critic=10,ema=10)
    owner.validate_state_dict(snapshot)
    owner.load_state_dict(snapshot)
    assert _equal(snapshot,owner.state_dict())


@pytest.mark.parametrize('corruption',['descriptor','rank_controls','prepared_inputs','coverage','commits','logical_payload'])
def test_genuine_cpu_warm_receipt_is_bound_to_inputs_actions_and_serialized_state(fixture_assets,tmp_path,monkeypatch,corruption):
    from distillschedule.training import helios_nogan_bootstrap as bootstrap
    value = json.loads(Path(fixture_assets['warm']).read_text())
    proof = json.loads(Path(value['provenance']['path']).read_text())
    receipt = json.loads(Path(proof['inputs'][0]['path']).read_text())
    if corruption == 'descriptor':
        receipt['cycles'][0]['descriptor']['content'][next(iter(receipt['cycles'][0]['descriptor']['content']))] = '0'*64
    elif corruption == 'rank_controls':
        receipt['cycles'][0]['rank_control_hashes'].reverse()
    elif corruption == 'prepared_inputs':
        receipt['prepared_inputs'] = _pin(Path(fixture_assets['prepared']))
    elif corruption == 'coverage':
        receipt['input_documents'] = receipt['input_documents'][:-1]
    elif corruption == 'commits':
        receipt['cycles'][0]['actions'][0]['counts']['generator_optimizer'] = 0
    else:
        receipt['logical_payloads']['critic']['first_moment'] = '0'*64
    proof['inputs'] = [_pin(_write(tmp_path/'receipt.json',receipt))]
    value['provenance'] = _pin(_write(tmp_path/'proof.json',proof))
    path = _write(tmp_path/'warm.json',value)
    def forbidden(*a,**kw):
        raise AssertionError('malformed CPU warm provenance reached model construction')
    monkeypatch.setattr(bootstrap,'_model',forbidden)
    with pytest.raises(ValueError):
        bootstrap.materialize_bootstrap(path)


@pytest.mark.parametrize('tamper',['decay','names','indices','shadow_layout'])
def test_poisoned_live_ema_ownership_tamper_refuses_before_any_copy(fixture_assets,tmp_path,tamper):
    from distillschedule.training import session
    from distillschedule.data import PreparedDataset
    from distillschedule.data.helios_nogan_reference_inputs import decode,descriptor,HeliosCohortTask
    owner = session.construct(config_for(fixture_assets,tmp_path/'live-ema'))[0].state['native_cycle']
    safe = owner.state_dict()
    trace = decode(PreparedDataset(fixture_assets['prepared']).sample(0),HeliosCohortTask(0))
    def fail_after_native_g(optimizer,args,kwargs):
        raise RuntimeError('actual original G Adam committed before live EMA tamper')
    handle = owner.generator.optimizer.register_step_post_hook(fail_after_native_g)
    try:
        with pytest.raises(RuntimeError,match='G Adam committed'):
            owner.run(trace,descriptor(trace.reference_document))
    finally:
        handle.remove()
    assert owner.poisoned
    before = {r:{n:p.detach().clone() for n,p in e.module.named_parameters()} for r,e in owner._engines.items()}
    if tamper == 'decay':
        owner.ema.decay = .98
    elif tamper == 'names':
        owner.ema.names[0],owner.ema.names[1] = owner.ema.names[1],owner.ema.names[0]
    elif tamper == 'indices':
        owner.ema.indices[0],owner.ema.indices[1] = owner.ema.indices[1],owner.ema.indices[0]
    else:
        owner.ema.shadows[-1] = owner.ema.shadows[-1].flatten()[:1]
    with pytest.raises(ValueError):
        owner.load_state_dict(safe)
    assert owner.poisoned
    assert all(_equal(before[r],{n:p.detach() for n,p in e.module.named_parameters()}) for r,e in owner._engines.items()), (
        'malformed live EMA ownership was rejected only AFTER already copying G/C state')
    # Do not reset the policy to make the test pass: this tampered owner stays poisoned.
