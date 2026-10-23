"""Actual W1 CPU1 BF16/FP32 original P0 reference, never world16 proof."""
import os
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_helios_step_owner_cpu import equal
from test_training_helios_public_cpu import STACK_AVAILABLE, FIXTURE, pin, write_json
from distillschedule.training.config import TrainingConfig
from distillschedule.training import session
from distillschedule.training.checkpoint import read_checkpoint, rng_state
from distillschedule.workloads.best_known.registry import repository_root

if os.environ.get('WAN_PUBLIC_REQUIRE_STACK') == '1' and not STACK_AVAILABLE:
    raise RuntimeError('required full W1 CPU stack missing: Diffusers0.37.1/PEFT0.17.1')
pytestmark = pytest.mark.skipif(not STACK_AVAILABLE, reason='separate full original Diffusers0.37.1/PEFT0.17.1 CPU proof')


def test_original_wan_public_program_factory_exists():
    from distillschedule.training.wan_public import build_model, build_algorithm
    from distillschedule.data.wan_reference_inputs import prepare_inputs
    assert callable(build_model) and callable(build_algorithm) and callable(prepare_inputs)


def fixture_files(root, *, cohorts=4):
    """Seeded complete ORIGINAL class mixed bases, ORIGINAL capture, no trainer fake."""
    from safetensors.torch import save_file
    from omegaconf import OmegaConf
    import yaml
    from distillschedule.adapters.helios.reference_model import load_reference_model_class, reference_model_identity
    from distillschedule.training.helios_public_assets import runtime_identity
    from distillschedule.strategies.wan_capture import native_primitives, capture
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_inputs import WanInputSources
    from distillschedule.strategies.wan_planning import source_fingerprint
    from distillschedule.training.wan_public import SCHEMA
    from distillschedule.data.wan_reference_inputs import SCHEMA as INPUTS, SCOPE, PRECISION, prepare_inputs
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    native_primitives(source_only=True)
    from helios.utils.train_config import Args
    template = yaml.safe_load((repository_root()/'configs/distillschedule/templates/helios_stage3_384x640.yaml').read_text())
    template['training_config']['is_random_drop'] = False  # declared native input choice, not hidden resampling
    capture_config = root/'capture.yaml'; capture_config.write_text(yaml.safe_dump(template))
    args = OmegaConf.merge(OmegaConf.structured(Args),template)
    args.training_config.max_train_steps = 5*cohorts
    args.training_config.lr_scheduler = 'constant_with_warmup'
    args.training_config.lr_warmup_steps = 1
    args.training_config.ema_start_step = 0  # declared fixture policy makes current G distinguishable from EMA
    args.model_config.lora_rank = args.model_config.critic_lora_rank = 1
    args.model_config.lora_alpha = args.model_config.critic_lora_alpha = 1.
    args.validation_config.stage2_simulated_inference_steps = [1,1,1]
    args.validation_config.use_kv_cache = False
    args_path = write_json(root/'args.json',OmegaConf.to_container(args,resolve=True))
    task = OriginalWanTask(video_height=128,video_width=128,video_frames=33,first_step=0,lora_rank=1,stage2_inference_steps=(1,1,1))
    cls = load_reference_model_class(backend='torch-sdpa')
    roles = {}
    with torch.random.fork_rng(devices=[]):
        for index,role in enumerate(('generator','critic')):
            torch.manual_seed(131+index)
            model = cls(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=16,out_channels=16,
                text_dim=4096,freq_dim=16,ffn_dim=64,num_layers=1,rope_dim=(4,6,6),
                is_use_gan=False,is_use_gan_final=False,is_use_gan_hooks=False,**TRANSFORMER_ADDITIONAL_KWARGS)
            with torch.no_grad():
                model.proj_out.weight.normal_(0.,.02)
            for name,tensor in model.named_parameters():
                tensor.data = tensor.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules) else torch.bfloat16)
            model.save_config(root/role)
            weights = root/role/'model.safetensors'
            save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},weights,metadata={'evidence_kind':FIXTURE})
            roles[role] = dict(config=pin(root/role/'config.json'),weights=[pin(weights)])
    negative = root/'negative.safetensors'
    save_file({'negative_prompt_embeds':torch.zeros(1,2,4096,dtype=torch.bfloat16)},negative,metadata={'evidence_kind':FIXTURE})
    assets = dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,evidence_kind=FIXTURE,
        source=reference_model_identity(backend='torch-sdpa'),runtime=runtime_identity(),args=pin(args_path),task=task.to_dict(),input_config=pin(capture_config),
        negative=pin(negative),geometry=dict(noise_shape=list(task.latent_shape),prompt_shape=[1,2,4096]),
        budget=dict(max_parameters=2000000,max_loaded_bytes=20000000),**roles)
    assets_path = write_json(root/'assets.json',assets)
    captured = []
    for cycle in range(cohorts):
        files = []
        for rank in range(16):
            path = root/f'feature-{cycle:02d}-{rank:02d}.pt'
            generator = torch.Generator().manual_seed(811+cycle*16+rank)
            # Fixed 42-section fixture + seed811 exercises the NORMAL native
            # branch for all declared ranks/cohorts, without production retries.
            torch.save(dict(vae_latent=torch.randn((42,*task.latent_shape[1:]),generator=generator),
                prompt_embed=torch.randn(2,4096,generator=generator),attention_mask=torch.ones(2,dtype=torch.int64)),path)
            files.append(str(path))
        current = replace(task,first_step=5*cycle)
        report = capture(current,WanInputSources(tuple(files),seed=811,cohort=cycle),capture_config,root/f'trace-{cycle:02d}')
        captured.append(dict(root=report['trace_bundle'],bundle_hash=report['expected_trace_hash'],task=current.to_dict()))
    source = write_json(root/'inputs.json',dict(schema_version=INPUTS,scope=SCOPE,evidence_kind=FIXTURE,
        source=source_fingerprint(),task=task.to_dict(),cohorts=captured,input_config=pin(capture_config)))
    prepared = prepare_inputs(source,root/'prepared')
    return dict(assets=assets_path,prepared=prepared,sources=source,task=task)


@pytest.fixture(scope='module')
def fixture_assets(tmp_path_factory):
    return fixture_files(tmp_path_factory.mktemp('wan-public')/'local')


def config_for(files, output, cycles=4, resume=None):
    trainer = dict(cycles=cycles,seed=773,device='cpu',output_dir=str(output),checkpoint_every=1)
    if resume:
        trainer['resume_from'] = str(resume)
    return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model=dict(factory='distillschedule.training.wan_public:build_model',parameters=dict(asset_manifest=str(files['assets']))),
        algorithm=dict(factory='distillschedule.training.wan_public:build_algorithm'),
        data=dict(prepared=str(files['prepared']),sampling=dict(kind='finite',shuffle=False)),
        execution=dict(adapter='program',mode='reference'),trainer=trainer),source=Path(output).parent/'training.json')


def run_capturing(config, monkeypatch):
    owners, receipts, events, hooks = [], [], [], []
    original = session.construct
    def construct(*args,**kwargs):
        result = original(*args,**kwargs); owner = result[0].state['native_cycle']; owners.append(owner)
        assert owner.versions == dict(generator=0,critic=0,ema=0)
        for engine in (owner.generator,owner.critic):
            assert not engine.optimizer.state
            for group in engine.optimizer.param_groups:
                for parameter in group['params']:
                    parameter.register_hook(lambda grad: hooks.append(float(grad.float().norm())))
        return result
    monkeypatch.setattr(session,'construct',construct)
    from distillschedule.training.wan_public import _p0
    p0 = _p0()
    functions = {getattr(p0,name).__code__:name for name in ('_run_nogan_generator_update_p0','_precompute_critic_rollouts','_run_nogan_critic_precomputed')}
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code in functions:
            events.append((functions[frame.f_code],frame.f_locals.get('step',frame.f_locals.get('steps'))))
        if event == 'return' and frame.f_code.co_name == 'native_cycle' and isinstance(arg,dict):
            receipts.append(deepcopy(arg))
    try:
        sys.setprofile(profile)
        result = session.train(config)
    finally:
        sys.setprofile(None); monkeypatch.setattr(session,'construct',original)
    return result,owners[0],receipts,dict(events=events,hooks=hooks)


def test_real_original_bf16_program_two_distinct_cohorts(fixture_assets,tmp_path,monkeypatch):
    config = config_for(fixture_assets,tmp_path/'train',2)
    result,owner,receipts,observed = run_capturing(config,monkeypatch)
    assert result['status'] == 'COMPLETE' and result['cursor'] == 2 and result['final_versions'] == dict(native_cycle=2)
    assert owner.versions == dict(generator=2,critic=10,ema=10) and owner.cycle == 2
    assert len(receipts) == 2 and len(observed['events']) == 22 and any(observed['hooks'])
    for i,row in enumerate(receipts):
        assert row['dispatch']['status'] == 'PASS' and row['dispatch']['native_versions'] == owner.expected(i+1)
        native = row['native']
        assert native['generator']['rollout']['output_dtype'] == 'torch.bfloat16'
        assert native['generator']['rollout']['output_requires_grad'] is True
        assert native['generator']['physical_score_forwards'] == 3
        assert len(native['rollouts']) == len(native['critic']) == 5
        assert len({r['member_trace_identities'][0] for r in native['rollouts']}) == 5
        assert all(not r['output_requires_grad'] and r['memory']['status'] == 'NOT_APPLICABLE_CPU_REFERENCE' for r in native['rollouts'])
    for engine in (owner.generator,owner.critic):
        assert any(torch.count_nonzero(s['exp_avg']) for s in engine.optimizer.state.values())
        assert all(p.dtype == torch.float32 for p in engine.module.parameters() if p.requires_grad)
        assert any(p.dtype == torch.bfloat16 for p in engine.module.parameters() if not p.requires_grad)
    assert all(any(name.startswith(patch+'.lora_') for name,p in owner.generator.module.named_parameters() if p.requires_grad)
               for patch in ('patch_short','patch_mid','patch_long'))
    assert not torch.cuda.is_initialized()


def test_fresh_os_process_full4_vs3_plus_exact_restart(fixture_assets,tmp_path,monkeypatch):
    full,owner,receipts,_ = run_capturing(config_for(fixture_assets,tmp_path/'full'),monkeypatch)
    expected = read_checkpoint(full['checkpoint'],identity=full['identity'])
    partial = session.train(config_for(fixture_assets,tmp_path/'split',3))
    prefix = read_checkpoint(Path(full['checkpoint']).parent/'cycle-000003',identity=full['identity'])
    split = read_checkpoint(partial['checkpoint'],identity=partial['identity'])
    assert equal(prefix['state'],split['state']) and equal(prefix['rng'],split['rng'])
    torch.save(dict(state=expected['state'],rng=expected['rng'],receipts=receipts[-1:]),tmp_path/'expected.pt')
    config = config_for(fixture_assets,tmp_path/'resumed',resume=partial['checkpoint'])
    write_json(tmp_path/'resume.json',config.to_dict())
    script = r"""
from pathlib import Path
import torch,sys,pytest
from test_training_wan_public_cpu import run_capturing,equal
from distillschedule.training.config import load_training_config
from distillschedule.training.checkpoint import read_checkpoint,rng_state
root=Path(sys.argv[1])
with pytest.MonkeyPatch.context() as patch:
 result,owner,receipts,_=run_capturing(load_training_config(root/'resume.json'),patch)
expected=torch.load(root/'expected.pt',weights_only=True)
actual=read_checkpoint(result['checkpoint'],identity=result['identity'])
assert equal(expected['state'],actual['state']), 'mixed weights/Adam/LR/EMA/clocks/native RNG/cursor drift'
assert equal(expected['rng'],actual['rng']) and equal(actual['rng'],rng_state())
assert equal(expected['receipts'],receipts), 'next ORIGINAL P0 call/version/control trace differs'
assert owner.versions==dict(generator=4,critic=20,ema=20) and owner.cycle==4
assert result['resume_cycle']==3 and result['cursor']==4 and result['final_versions']==dict(native_cycle=4)
assert not torch.cuda.is_initialized()
print('W1_FULL_ORIGINAL_BF16_FP32_CPU_FRESH_PROCESS_EXACT_4_VS_3_PLUS_RESTART_PASS')
"""
    process = subprocess.run([sys.executable,'-c',script,str(tmp_path)],cwd=repository_root(),env=dict(os.environ),
        text=True,capture_output=True,timeout=300)
    assert process.returncode == 0, process.stdout+process.stderr
    assert 'FRESH_PROCESS_EXACT' in process.stdout
    if os.environ.get('WAN_PUBLIC_EVIDENCE'):
        destination = Path(os.environ['WAN_PUBLIC_EVIDENCE'])/'artifacts'
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'restart-process.stdout').write_text(process.stdout)


def test_partial_cycle_poison_and_safe_all_owner_restore(fixture_assets,tmp_path):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.wan_reference_inputs import decode,descriptor
    components = session.construct(config_for(fixture_assets,tmp_path/'not-started',2))[0]
    owner = components.state['native_cycle']
    dataset = PreparedDataset(fixture_assets['prepared'])
    trace = decode(dataset.sample(0),owner.task)
    owner.run(trace,descriptor(trace.reference_document))
    saved = deepcopy(owner.state_dict())
    next_trace = decode(dataset.sample(1),replace(owner.task,first_step=5))
    from distillschedule.training.wan_public import _p0
    native_critic = _p0()._run_nogan_critic_precomputed
    calls = 0
    def failure(frame,event,arg):
        nonlocal calls
        if frame.f_code is native_critic.__code__ and event == 'return':
            calls += 1
            if calls == 2:
                raise RuntimeError('injected AFTER two real C commits')
    try:
        sys.setprofile(failure)
        with pytest.raises(RuntimeError,match='two real C'):
            owner.run(next_trace,descriptor(next_trace.reference_document))
    finally:
        sys.setprofile(None)
    assert owner.poisoned and owner.versions == dict(generator=2,critic=7,ema=6)
    with pytest.raises(ValueError,match='poison'):
        owner.state_dict()
    with pytest.raises(ValueError,match='poison'):
        owner.run(next_trace,descriptor(next_trace.reference_document))
    before = {k:deepcopy(v.state_dict()) for k,v in owner._views.items()}
    before_rng = deepcopy(rng_state())
    for key in ('ema','generator.optimizer','critic.clocks','consumed','versions','adapters','buffers','modes','integer_flags'):
        invalid = deepcopy(saved)
        if key == 'ema':
            invalid['native'][key]['optimization_step'] += 1
        elif key == 'critic.clocks':
            invalid['native'][key]['counters']['global_steps'] += 1
        elif key == 'consumed':
            invalid[key][0]['control_hash'] = '0'*64
        elif key == 'versions':
            invalid[key]['generator'] = True
        elif key == 'adapters':
            invalid[key]['generator']['config']['default']['r'] += 1
        elif key == 'buffers':
            next(iter(invalid['native']['generator.weights']['buffers'].values())).add_(1)
        elif key == 'modes':
            invalid['native']['generator.weights']['training'][''] = False
        elif key == 'integer_flags':
            weights = invalid['native']['generator.weights']
            weights['requires_grad'] = {name:int(flag) for name,flag in weights['requires_grad'].items()}
            # Integer flags compare equal to bools but must not reach a copy.
            next(iter(weights['tensors'].values())).add_(.5)
        else:
            next(iter(invalid['native'][key]['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
        with pytest.raises(ValueError):
            owner.load_state_dict(invalid)
        assert owner.poisoned and equal(before,{k:v.state_dict() for k,v in owner._views.items()})
        assert equal(before_rng,rng_state())
    owner.load_state_dict(saved)
    assert not owner.poisoned and equal(saved,owner.state_dict())
    with pytest.raises(ValueError,match='replay|reordered'):
        owner.run(trace,descriptor(trace.reference_document))
    # Exact source Teacher calls mask C's adapters/grad flags. Recovery must
    # stage safely even when failure happens BEFORE C can reenable them.
    original_g = _p0()._run_nogan_generator_update_p0
    def after_g(frame,event,arg):
        if frame.f_code is original_g.__code__ and event == 'return':
            raise RuntimeError('injected AFTER real G commit, Teacher mask still active')
    try:
        sys.setprofile(after_g)
        with pytest.raises(RuntimeError,match='Teacher mask'):
            owner.run(next_trace,descriptor(next_trace.reference_document))
    finally:
        sys.setprofile(None)
    assert owner.poisoned and owner.versions == dict(generator=2,critic=5,ema=5)
    assert not any(p.requires_grad for p in owner.critic.module.parameters())
    owner.load_state_dict(saved)
    assert equal(saved,owner.state_dict())
    # Genuine failed backward may leave partial native gradients: restore
    # discards them, never fabricates/zeros any optimizer/scheduler/EMA clocks.
    parameter = owner.critic.optimizer.param_groups[0]['params'][0]
    def failed_gradient(grad):
        raise RuntimeError('injected DURING actual C backward')
    hook = parameter.register_hook(failed_gradient)
    try:
        with pytest.raises(RuntimeError,match='actual C backward'):
            owner.run(next_trace,descriptor(next_trace.reference_document))
    finally:
        hook.remove()
    assert owner.poisoned and owner.versions == dict(generator=2,critic=5,ema=5)
    owner.load_state_dict(saved)
    assert equal(saved,owner.state_dict())
    owner.run(next_trace,descriptor(next_trace.reference_document))
    assert owner.versions == dict(generator=2,critic=10,ema=10)


def test_asset_task_precision_pyramid_and_topology_fail_closed(fixture_assets,tmp_path,monkeypatch):
    from distillschedule.training.wan_public import build_model
    from safetensors.torch import load_file,save_file
    assets = json.loads(fixture_assets['assets'].read_text())
    invalids = []
    for key,value in [('precision','FP32'),('scope','CPU1_W2')]:
        invalid = deepcopy(assets); invalid[key] = value; invalids.append(invalid)
    invalid = deepcopy(assets); invalid['task']['model_family'] = 'helios14b'; invalids.append(invalid)
    invalid = deepcopy(assets)
    weights = load_file(invalid['generator']['weights'][0]['path']); del weights['patch_mid.weight']
    missing = tmp_path/'missing.safetensors'; save_file(weights,missing,metadata={'evidence_kind':FIXTURE})
    invalid['generator']['weights'] = [pin(missing)]; invalids.append(invalid)
    for index,invalid in enumerate(invalids):
        path = write_json(tmp_path/f'assets-{index}.json',invalid)
        with pytest.raises(ValueError):
            build_model(dict(asset_manifest=str(path)))
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='topology'):
        build_model(dict(asset_manifest=str(fixture_assets['assets'])))
    assert not torch.cuda.is_initialized()


def test_operation_control_hash_input_shape_and_unqualified_cpu_reject(fixture_assets):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.wan_reference_inputs import decode,cpu_telemetry
    feeds = PreparedDataset(fixture_assets['prepared']).sample(0)
    for change in ('branch','task','tensor'):
        value = deepcopy(feeds)
        if change == 'tensor':
            document = json.loads(bytes(value['control'][:value['control_length'].item()].tolist()))
            index = sorted(document['manifests'][0]['tensors']).index('step/0/generator/rollout_noise')
            value['tensors'][index] = value['tensors'][index][...,:8]
        else:
            control = json.loads(bytes(value['control'][:value['control_length'].item()].tolist()))
            if change == 'branch':
                control['manifests'][0]['operations'][0]['control']['gan'] = True
            else:
                control['task']['first_step'] = 5
            raw = json.dumps(control).encode()
            value['control'].zero_(); value['control'][:len(raw)] = torch.tensor(list(raw),dtype=torch.uint8)
            value['control_length'][0] = len(raw)
        with pytest.raises(ValueError):
            decode(value,fixture_assets['task'])
    with pytest.raises(ValueError,match='validated'):
        cpu_telemetry(torch.device('cpu'),object())


def test_current_projection_nonmutating_next_actual_update_and_immutable_rope(fixture_assets,monkeypatch):
    from distillschedule.training.wan_public import build_model,_p0
    from distillschedule.data import PreparedDataset
    from distillschedule.data.wan_reference_inputs import decode,descriptor
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    components = build_model(dict(asset_manifest=str(fixture_assets['assets'])))
    a = components.state['native_cycle']
    dataset = PreparedDataset(fixture_assets['prepared'])
    first = decode(dataset.sample(0),a.task)
    a.run(first,descriptor(first.reference_document))
    before = deepcopy(a.state_dict()); outer = rng_state()
    projection = components.inference_projection(components,dict(seed=773))
    assert projection.components.state['generator'] is a.generator.module
    assert equal(before,a.state_dict()) and equal(outer,rng_state())
    cls = load_reference_model_class(backend='torch-sdpa')
    with torch.random.fork_rng(devices=[]):
        original = cls(**projection.parameters['architecture'])
    # Full source constructor owns nonpersistent RoPE contents/dtypes.
    assert equal(dict(original.named_buffers()),dict(a.generator.module.named_buffers()))
    b = build_model(dict(asset_manifest=str(fixture_assets['assets']))).state['native_cycle']
    b.load_state_dict(before)
    next_trace = decode(dataset.sample(1),replace(a.task,first_step=5))
    ar = a.run(next_trace,descriptor(next_trace.reference_document))
    br = b.run(next_trace,descriptor(next_trace.reference_document))
    assert equal(ar,br) and equal(a.state_dict(),b.state_dict())
    snapshot = deepcopy(a.state_dict())
    p0 = _p0()
    with monkeypatch.context() as patch:
        patch.setattr(p0,'_run_nogan_generator_update_p0',lambda **kw: None)
        with pytest.raises(ValueError,match='binding|source'):
            a.state_dict()
    assert equal(snapshot,a.state_dict())
    layer = next(m for m in a.generator.module.modules() if hasattr(m,'scaling'))
    with monkeypatch.context() as patch:
        patch.setitem(layer.scaling,'default',2*layer.scaling['default'])
        with pytest.raises(ValueError,match='scaling'):
            a.state_dict()
    assert equal(snapshot,a.state_dict())
    processor = a.generator.module.blocks[0].attn1.processor
    with monkeypatch.context() as patch:
        patch.setattr(processor,'cache_enabled',True)
        with pytest.raises(ValueError,match='cache'):
            a.state_dict()
    assert equal(snapshot,a.state_dict())
