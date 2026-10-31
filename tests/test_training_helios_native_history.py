"""Opt-in W2 native-history protocol tests; CPU fixtures are not GPU evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from distillschedule.training.helios_inputs import HeliosCohortTask, TRACE_KEY, _keys
from distillschedule.search.helios_request import OriginalHeliosRequest
from distillschedule.training.config import TrainingConfig

POLICY = 'source_native_amplified_v1'
KIND = 'original_helios_native_history_cohort_stream.v1'


def task(index=0):
    return HeliosCohortTask(index, native_policy=POLICY)


def controls(value):
    """SYNTHETIC_TEST_ONLY: exact metadata shape/branches; no numerical claim."""
    result = {}
    for rank in range(16):
        ops, tensors = [], {}
        for step in value.source_steps:
            ops.append(dict(key=f'step/{step}/branch', control=dict(global_step=step,
                dfake_gen_update_ratio=5, train_generator=step == value.first_step,
                train_critic=True, gan=False, reward=False, vae_decode=False)))
            for phase in (('generator', 'critic') if step == value.first_step else ('critic',)):
                for role, key in _keys(step, phase).items():
                    shape, dtype = list(value.latent_shape), 'bfloat16'
                    if role == 'text_embedding': shape = [2,4096]
                    elif role == 'attention_mask': shape, dtype = [2], 'bool'
                    elif role.startswith('indices_'):
                        shape, dtype = [1, {'indices_hidden_states':9, 'indices_history_short':2,
                            'indices_history_mid':2, 'indices_history_long':16}[role]], 'int64'
                    elif role.endswith('_timestep'): shape, dtype = [1], 'int64'
                    elif role == 'block_noise_stage1': shape[-2:] = [24,40]
                    elif role in ('history_short', 'history_mid', 'history_long', 'continuous_history'):
                        shape[2] = dict(history_short=2, history_mid=2, history_long=16, continuous_history=19)[role]
                    tensors[key] = dict(shape=shape, dtype=dtype)
                zero = rank % 2 == 0
                sequence = []
                for stage, hw in enumerate(value.stage_spatial_shapes):
                    calls = (rank + stage) % 4 + 1 if zero else 1
                    sequence.append(dict(stage=stage, call_count=calls, denoising_exit=calls-1,
                        denoising_step_budget=4 if zero else 2, timestep=[1000.-i for i in range(calls)],
                        latent_shape=[*value.latent_shape[:3],*hw]))
                control = dict(pyramid_stage_call_sequence=sequence,
                    call_count=sum(x['call_count'] for x in sequence), all_history_zero=zero, rank=rank,
                    stage_schedule_policy=POLICY, initial_synchronized_pyramid_stage_exit=[0,0,0],
                    stage_schedule_scope='native_rank0_initial_with_rank_local_amplified')
                ops.append(dict(key=f'step/{step}/{phase}/rollout', control=control))
                ops.append(dict(key=f'step/{step}/{phase}/' + ('dmd' if phase == 'generator' else 'train'), control={}))
        contract = {TRACE_KEY:value.to_dict(), 'resolution':[384,640], 'full_latent_shape':list(value.latent_shape),
            'stage_spatial_shapes':[list(x) for x in value.stage_spatial_shapes], 'dtype':'bfloat16',
            'dfake_gen_update_ratio':5}
        manifest = dict(contract=contract, operations=ops, tensors=tensors, evidence_kind='SYNTHETIC_TEST_ONLY')
        indexed = {op['key']:op for op in ops}
        result[rank] = NS(manifest=manifest, operation=lambda key, rows=indexed:rows[key])
    return result


def test_native_task_is_versioned_and_original_default_is_byte_identical():
    old = HeliosCohortTask(0)
    assert old.to_dict() == dict(schema_version='distillschedule.helios_training_cohort.v1', index=0,
        request=OriginalHeliosRequest().to_dict(), source_steps=[10,11,12,13,14])
    new = task(1)
    assert new.to_dict()['schema_version'] == 'distillschedule.helios_native_history_cohort.v1'
    assert new.stage_schedule_policy == POLICY and new.request == OriginalHeliosRequest()
    assert new.source_steps == (15,16,17,18,19)
    assert HeliosCohortTask.from_dict(new.to_dict()) == new and new.digest != HeliosCohortTask(1).digest
    bad = new.to_dict(); bad['schema_version'] = old.to_dict()['schema_version']
    with pytest.raises(ValueError): HeliosCohortTask.from_dict(bad)
    with pytest.raises(ValueError): HeliosCohortTask(0, native_policy='benchmark_v4_strict_global')
    with pytest.raises(ValueError): OriginalHeliosRequest(stage_schedule_policy=POLICY)


def test_all_rank_native_controls_allow_real_different_prefix_lengths_without_resident_claim():
    from distillschedule.training.helios_inputs import validate_controls
    value = task(); rows = controls(value)
    actual = validate_controls(value, rows)
    assert actual['status'] == 'PASS' and actual['cross_rank_call_count_parity'] is False
    assert actual['rank_count'] == 16 and actual['local_exception_entry_count'] == 48
    assert actual['execution_requirement'] == 'FSDP2_ALIGNED_MATERIALIZATION_WITH_ORIGINAL_REDUCE_AND_STEP'
    assert 'RESIDENT' not in actual['execution_requirement']
    assert [r['logical_call_count'] for r in actual['operations'][0]['ranks']] == [6,3,8,3]*4
    with pytest.raises(ValueError): validate_controls(HeliosCohortTask(0), rows)


@pytest.mark.parametrize('mutation', ['budget','rank','initial','normal_exit','total','branch','geometry','stale_kind','old_hash'])
def test_bad_native_metadata_is_rejected(mutation):
    from distillschedule.training.helios_inputs import validate_controls
    value = task(); rows = controls(value); c = rows[1].operation('step/10/generator/rollout')['control']
    if mutation == 'budget': c['pyramid_stage_call_sequence'][0]['denoising_step_budget'] = 4
    if mutation == 'rank': c['rank'] = 0
    if mutation == 'initial': c['initial_synchronized_pyramid_stage_exit'] = [1,0,0]
    if mutation == 'normal_exit': c['pyramid_stage_call_sequence'][0]['denoising_exit'] = 1
    if mutation == 'total': c['call_count'] = 4
    if mutation == 'branch': rows[1].operation('step/10/branch')['control']['gan'] = True
    if mutation == 'geometry': rows[1].manifest['contract']['resolution'] = [64,64]
    if mutation == 'stale_kind': rows[1].manifest['contract'][TRACE_KEY] = HeliosCohortTask(0).to_dict()
    if mutation == 'old_hash': rows[1].manifest['contract']['distillschedule_original_helios_request_sha256'] = 'a'*64
    with pytest.raises(ValueError): validate_controls(value, rows)


def test_public_config_is_explicit_reference_not_saved_plan(tmp_path):
    from test_training_helios_stream import config
    from distillschedule.training import native
    raw = config(tmp_path).to_dict(); raw['data']['input_kind'] = KIND
    conf = TrainingConfig.from_dict(raw, source=tmp_path/'training.json')
    assert native._options(conf,'W2') == 'reference'
    described = native.describe_native(conf)
    assert described['capabilities']['cycles']['fresh_inputs'] == KIND
    assert described['capabilities']['native_history']['execution_requirement'].startswith('FSDP2_ALIGNED')
    for patch in (dict(mode='plan',plan='old.json'), dict(mode='manual',preset='helios_nogan.safe_stream_ed8.v1')):
        bad = deepcopy(raw); bad['execution'].update(patch)
        with pytest.raises(ValueError): native._options(TrainingConfig.from_dict(bad,source=tmp_path/'bad.json'),'W2')


def test_data_cli_reaches_native_policy_without_new_gpu_owner(tmp_path, monkeypatch, capsys):
    from distillschedule.cli import main
    from distillschedule.training import helios_inputs
    values = [dict(seed=17+i, samples=[[f'/features/{i}/{s}/{r}.pt' for r in range(16)] for s in range(5)])
              for i in range(2)]
    source = tmp_path/'sources.json'; source.write_text(json.dumps(values)); seen=[]
    def producer(request, sources, training, output, *, input_policy, native_policy=None):
        seen.append((request,sources,native_policy)); return dict(status='SYNTHETIC_TEST_ONLY',gpu_execution='NOT_RUN')
    monkeypatch.setattr(helios_inputs,'prepare_stream',producer)
    assert main(['data','prepare-helios-stream','--sources',str(source),'--training-config',str(tmp_path/'training.yaml'),
                 '--output',str(tmp_path/'prepared'),'--history-policy',POLICY]) == 0
    assert seen[0][2] == POLICY and 'NOT_RUN' in capsys.readouterr().out


@pytest.fixture
def phase_protocol(monkeypatch):
    """SYNTHETIC_TEST_ONLY control plane, not actual FSDP/model math evidence.

    The component's actual Torch2.10/PEFT HSDP proof remains separate. Here the
    real new model validator, phase owner and graph dispatch run together.
    """
    import sys
    import types
    import torch.distributed.fsdp as fsdp
    from test_runtime_fsdp2_variable_forward import Unit, BlockUnit, Engine
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    monkeypatch.setattr(fsdp,'FSDPModule',Unit)
    upstream = types.ModuleType('helios.modules.transformer_helios')
    upstream.HeliosTransformerBlock = BlockUnit
    monkeypatch.setitem(sys.modules,'helios.modules.transformer_helios',upstream)
    engine = Engine()
    engine.module.units = [engine.module] + [BlockUnit(engine.events,f'block-{i}') for i in range(40)]
    engine.module.gradient_checkpointing = True
    engine.module._gradient_checkpointing_func = fsdp2_metadata_tolerant_checkpoint
    return engine, Unit, BlockUnit


def test_whole_native_checkpoint_class_and_original_checkpoint_function_are_required(phase_protocol):
    from functools import partial
    from distillschedule.runtime.helios_native_history import admit_engine
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    engine, Unit, BlockUnit = phase_protocol
    assert admit_engine(engine)['fsdp_units'] == 41
    engine.module._gradient_checkpointing_func = partial(fsdp2_metadata_tolerant_checkpoint, operation_source=True)
    assert admit_engine(engine)['checkpoint_policy'] == 'original_metadata_tolerant'
    engine.module._gradient_checkpointing_func = lambda *a, **k: None
    with pytest.raises(ValueError, match='checkpoint function'): admit_engine(engine)
    engine.module._gradient_checkpointing_func = fsdp2_metadata_tolerant_checkpoint
    # Same number of units, wrong ownership granularity: an interior unit is
    # NOT the whole Helios checkpoint block even when every other unit agrees.
    engine.module.units[-1] = Unit(engine.events,'interior-lora-linear')
    with pytest.raises(ValueError, match='whole checkpoint blocks'): admit_engine(engine)
    engine.module.units[-1] = BlockUnit(engine.events,'restored-block')
    engine.module.gradient_checkpointing = False
    with pytest.raises(ValueError, match='active original checkpoint'): admit_engine(engine)


@pytest.mark.parametrize('fail', [False, True])
def test_actual_graph_callback_uses_entire_grad_phase_and_releases_before_original_step(phase_protocol, fail):
    from contextlib import contextmanager
    from distillschedule.runtime.helios_planner_cycle import _prepare_cycle
    from distillschedule.search.helios_implementation import HeliosImplementation
    from distillschedule.runtime.role_runner import OperationContext
    from distillschedule.runtime.version_sync import VersionLedger
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    engine, _, _ = phase_protocol
    class Observer:
        mode, rank, device = 'native', 0, 'cpu'
        impl = HeliosImplementation('no-gan','world16-fsdp',512,1,False)
        generator, critic = engine, object()
        rows = []
        def versions(self): return dict(generator=engine.global_steps, critic=0, ema=0)
        @contextmanager
        def action(self, op, *, source_steps):
            row = dict(action_id=op.operation_id, source_steps=source_steps, memory_after={})
            self.rows.append(row)
            yield row
        def produced(self, step, output): engine.events.append(('produced',step,output))
    observer = Observer(); marker = object(); calls=[]
    def original_g_helper(**kwargs):
        calls.append('original_g_helper')
        assert kwargs['trace']._trace.task == task()
        view = kwargs['generator_engine']
        assert view is not engine and all(unit.group.is_unsharded for unit in engine.module.units)
        for index in range(7): view(index)
        engine.events.append(('original_score_calls',))
        if fail: raise RuntimeError('original_score_failure')
        view.backward('original_loss'); view.step()
        return dict(original_helper_return=marker)
    def original_detached_helper(**kwargs):
        calls.append('original_detached_helper')
        view = kwargs['generator_engine']
        assert view is not engine and all(unit.group.is_unsharded for unit in engine.module.units)
        # The actual original helper forwards through .module; phase census
        # records acquire/release separately from observed application calls.
        view(8)
        return ({10:marker},[dict(original_rollout=True)],dict(enabled=False,restored=True))
    p0 = NS(_run_nogan_generator_update_p0=original_g_helper,
            _precompute_critic_rollouts=original_detached_helper)
    prepared, graph, order = _prepare_cycle(observer=observer,p0=p0,
        args=NS(timing_mode=True,profile_mode=False,profile_diagnostics=False),
        policy=StaticFSDP2P0Policy(rollout_batch_size=1, pack_real_score_pair=False),
        steps=task().source_steps,trace=NS(task=task()),negative=marker,cfg=None,
        diagnostics=object(),generator_free_memory={})
    operation, callback = prepared._bound[0]
    assert operation.operation_id == 'g'
    ctx = OperationContext(operation,VersionLedger(dict(generator=0,critic=0,ema=0)))
    if fail:
        with pytest.raises(RuntimeError,match='original_score_failure'): callback(ctx)
        assert engine.global_steps == 0
        assert observer.rows[0]['native_history_phase']['restored'] is True
        assert observer.rows[0]['native_history_phase']['commit_count'] == 0
        assert all(not unit.group.is_unsharded for unit in engine.module.units)
        return
    assert callback(ctx)['original_helper_return'] is marker
    phase = observer.rows[0]['native_history_phase']
    assert phase['unshard_count'] == phase['reshard_count'] == 41 and phase['restored']
    assert phase['engine_forward_count'] == 7 and phase['backward_count'] == phase['commit_count'] == 1
    assert engine.events[-1] == ('original_clip_adam_scheduler_zero_grad',)
    operation, callback = next((op,fn) for op,fn in prepared._bound if op.operation_id == 'r0')
    assert callback(OperationContext(operation,VersionLedger(dict(generator=1,critic=0,ema=0)))) == {10:marker}
    phase = observer.rows[-1]['native_history_phase']
    assert not phase['with_grad'] and phase['unshard_count'] == phase['reshard_count'] == 41
    assert phase['backward_count'] == phase['commit_count'] == 0 and phase['restored']
    assert calls == ['original_g_helper','original_detached_helper']
    assert all(not unit.group.is_unsharded for unit in engine.module.units)


def test_native_kind_is_not_silently_accepted_by_old_loader(tmp_path):
    from distillschedule.training.helios_inputs import load_stream
    from distillschedule.workloads.best_known.registry import fingerprint
    doc = dict(schema_version='distillschedule.'+KIND, status='READY_FRESH_COHORT_STREAM',
               identity=dict(native_history_policy=POLICY))
    doc['stream_sha256'] = fingerprint(doc)
    path = tmp_path/'stream.json'; path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='identity changed'): load_stream(path,cycles=2)


def test_runtime_census_requires_real_phase_receipt_not_only_native_flag():
    from distillschedule.runtime.helios_native_history import validate_action, SEMANTICS, SCHEMA, REQUIREMENT
    value=task(); control=controls(value)[0].operation('step/10/generator/rollout')['control']
    entry = dict(native_history_policy=POLICY, native_history_semantics=SEMANTICS, source_rank=0,
                 all_history_zero=True, stage_schedule=control['pyramid_stage_call_sequence'], native_task_sha256=value.digest)
    row = dict(action_id='g',source_rank=0,trace_entries=[entry])
    with pytest.raises(ValueError,match='actual phase'): validate_action(row)
    row['native_history_phase'] = dict(schema_version=SCHEMA,action_id='g',execution_requirement=REQUIREMENT,
        task_sha256=value.digest,with_grad=True,unshard_count=41,reshard_count=41,backward_count=1,commit_count=1,restored=True)
    validate_action(row)
    row['native_history_phase']['commit_count']=0
    with pytest.raises(ValueError,match='actual phase'): validate_action(row)


@pytest.fixture
def metadata_stream(tmp_path, monkeypatch):
    """Real prepare/bundle/load plumbing; producer-boundary synthetic metadata.

    Deliberately no tensors.pt: this fixture CANNOT pass worker payload gates
    and must never be used as native qualification or model math evidence.
    """
    from pathlib import Path
    from distillschedule.training import helios_inputs as inputs
    from distillschedule.workloads.best_known.registry import file_sha256
    from runtime.operation_trace import OperationTrace, _manifest_hash
    source_rows = []
    for index in range(2):
        rows=[]
        for offset in range(5):
            names=[]
            for rank in range(16):
                p=tmp_path/f'synthetic-not-features-{index}-{offset}-{rank}.pt'
                p.write_bytes(f'SYNTHETIC_TEST_ONLY {index} {offset} {rank}'.encode()); names.append(str(p))
            rows.append(names)
        source_rows.append(inputs.HeliosCohortSources(17+index, tuple(tuple(r) for r in rows)))
    cfg=tmp_path/'training.yaml';cfg.write_text('SYNTHETIC_TEST_ONLY; no native values loaded')
    def inventory(value, sources):
        return [dict(rank=r,step=value.source_steps[s],path=name,sha256=file_sha256(Path(name)),
                     latent_shape=[1,16,9,48,80],text_shape=[2,4096])
                for s,names in enumerate(sources.samples) for r,name in enumerate(names)]
    monkeypatch.setattr(inputs,'_inventory',inventory)
    import distillschedule.strategies.wan_capture as capture
    monkeypatch.setattr(capture,'read_training_config',lambda _:dict(evidence_kind='SYNTHETIC_TEST_ONLY'))
    def capture_rank(value,sources,inventory,cfg,*,rank,output,provenance):
        manifest=deepcopy(controls(value)[rank].manifest)
        manifest['source']={**provenance,'rank':rank,'samples':[r for r in inventory if r['rank']==rank],
            'seed':sources.seed,'cohort':value.index,'rng_scope':'new_native_CPU_streams_not_prior_training_rng_continuation'}
        manifest['trace_hash']=_manifest_hash(manifest)
        output.mkdir(parents=True); (output/'trace.json').write_text(json.dumps(manifest))
        return OperationTrace(root=output,manifest=manifest,tensors={})
    monkeypatch.setattr(inputs,'capture_rank',capture_rank)
    output=tmp_path/'native-prepared'
    doc=inputs.prepare_stream(OriginalHeliosRequest(),source_rows,cfg,output,native_policy=POLICY)
    return output,doc


def test_prepare_to_load_to_public_native_admission_and_original_p0_argv(
        metadata_stream, tmp_path, monkeypatch):
    from distillschedule.training.helios_inputs import load_stream
    from distillschedule.training import helios_worker as worker
    from distillschedule.training.helios_reference_assets import SCHEMA as ASSETS
    from test_training_helios_stream import config
    root,doc=metadata_stream
    assert load_stream(root/'stream.json',cycles=2,expected_kind=KIND) == doc
    with pytest.raises(ValueError): load_stream(root/'stream.json',cycles=2)
    # Only expensive frozen asset payloads are an explicit test boundary. Real
    # stream reader, public options, selected descriptor and argv builder run.
    assets=dict(schema_version=ASSETS, assets={name:dict(path=f'/assets/{name}') for name in (
        'generator_model','score_model','generator_precast','score_precast','logical_checkpoint','negative_prompt')})
    path=tmp_path/'assets.json';path.write_text(json.dumps(assets))
    monkeypatch.setattr(worker,'reference_assets',lambda value:value)
    monkeypatch.setattr(worker,'verify_reference_pins',lambda value:dict(evidence_kind='SYNTHETIC_TEST_ONLY'))
    monkeypatch.setattr(worker,'load_reference_bootstrap',lambda *a,**k:(NS(manifest_hash='a'*64),{}))
    raw=config(tmp_path,cycles=2).to_dict(); raw['data'].update(input_kind=KIND,prepared=str(root/'stream.json'))
    conf=TrainingConfig.from_dict(raw,source=tmp_path/'native-training.json')
    value=worker.admit_reference(conf,OriginalHeliosRequest().to_dict(),assets,
                                dict(assets=path,output=tmp_path/'run'))
    assert value['selected']['native_history']['execution_requirement'].startswith('FSDP2_ALIGNED')
    assert value['stream']['identity']['native_history_policy'] == POLICY
    args=value['argv'];assert args[args.index('--trace-bundle')+1] == doc['cohorts'][0]['trace_bundle']
    assert args[args.index('--mesh-policy')+1] == 'world16-fsdp'
    assert '--no-generator-nograd-persistent-unshard' in args and '--no-pack-real-score-pair' in args
    from distillschedule.training.helios_lifecycle import current_training,_checkpoint_policy
    from distillschedule.runtime.helios_native_history import descriptor
    import sys
    saved_argv=sys.argv
    def original_entry_boundary():
        binding=current_training()
        assert binding['stream'] is value['stream']
        assert binding['identity']['native_history'] == descriptor()
        assert binding['identity']['schema_version']=='distillschedule.helios_native_history_training_identity.v1'
        assert _checkpoint_policy(binding['identity'],optimized=False)['schema_version'] == (
            'distillschedule.helios_native_history_checkpoint_policy.v1')
        assert '--trace-bundle' in sys.argv and binding['owned_rank_output'].endswith('/ranks/rank_00.json')
        return dict(status='SYNTHETIC_TEST_ONLY_ORIGINAL_MAIN_BOUNDARY; NOT_EXECUTED_GPU')
    monkeypatch.setenv('RANK','0')
    entered=worker._run_admitted(conf,value,dict(identity=dict(head='SYNTHETIC_TEST_ONLY')),
        ownership={},backend=NS(main=original_entry_boundary))
    assert entered['status'].startswith('SYNTHETIC_TEST_ONLY') and current_training() is None and sys.argv is saved_argv
    with pytest.raises(FileNotFoundError):
        load_stream(root/'stream.json',cycles=2,expected_kind=KIND,verify_rank=0)


def test_real_native_zero_history_capture_and_payload_grid_replay(tmp_path):
    """Native upstream CPU history/scheduler producer; full-size BF16 tensors.

    Required target-environment test. The local minimal lane may lack upstream
    dependencies; absence is an explicit SKIP, never GPU admission.
    """
    pytest.importorskip('ftfy');pytest.importorskip('omegaconf')
    import random
    import torch
    import yaml
    from pathlib import Path
    from runtime.operation_trace import OperationTrace
    from distillschedule.strategies.wan_capture import native_primitives, read_training_config
    from distillschedule.training.helios_inputs import capture_rank, HeliosCohortSources, validate_rank_payload
    from distillschedule.workloads.best_known.registry import repository_root,file_sha256
    native=native_primitives(source_only=True);value=task();rank=0
    cfg=yaml.safe_load((repository_root()/'configs/distillschedule/templates/helios_stage3_384x640.yaml').read_text())
    cfg['training_config']['noise_corrupt_clean_prob_history']=1.
    path=tmp_path/'native-config.yaml';path.write_text(yaml.safe_dump(cfg));cfg=read_training_config(path)
    samples=[];inventory=[]
    for offset,step in enumerate(value.source_steps):
        names=[]
        for r in range(16):
            source=tmp_path/f'feature-{offset}-{r}.pt';names.append(str(source))
            if r==rank:
                torch.save(dict(vae_latent=torch.zeros(1,dtype=torch.bfloat16).expand(2,16,9,48,80),
                    prompt_embed=torch.full((2,4096),.01*(offset+1),dtype=torch.bfloat16),
                    attention_mask=torch.ones(2,dtype=torch.bool)),source)
                inventory.append(dict(rank=r,step=step,path=str(source),sha256=file_sha256(source)))
        samples.append(tuple(names))
    sources=HeliosCohortSources(17,tuple(samples));before=torch.get_rng_state().clone();py_before=random.getstate()
    captured=capture_rank(value,sources,inventory,cfg,rank=rank,output=tmp_path/'rank',
                          provenance=dict(evidence_kind='CPU_SYNTHETIC_TEST_ONLY'))
    assert torch.equal(before,torch.get_rng_state()) and py_before==random.getstate()
    loaded=OperationTrace.load(captured.root,verify=True)
    trace=NS(task=value,logical_steps=value.source_steps,_load=lambda _:loaded)
    validate_rank_payload(trace,rank,sources)
    for op in loaded.manifest['operations']:
        if op['key'].endswith('/rollout'):
            control=op['control']
            assert control['all_history_zero'] is True and control['rank']==rank
            assert all(s['denoising_step_budget']==4 for s in control['pyramid_stage_call_sequence'])
    # Real CPU history values, not a metadata flag, decide the native branch.
    # tensor() returns a clone by default; mutate the actual admitted payload,
    # not an unused temporary, so validation really observes the changed value.
    loaded.tensor('step/10/generator/history_short',clone=False).flatten()[0]=1.
    with pytest.raises(ValueError,match='stage grid/RNG exits'):
        validate_rank_payload(trace,rank,sources)
    assert not torch.cuda.is_initialized()




def test_native_history_recipe_discovery_needs_no_torch_cuda_or_assets():
    import os
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code=r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'))
from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import describe_native,_options
path=root/'recipes/helios_nogan/training.native-history.json'
conf=TrainingConfig.from_dict(json.loads(path.read_text()),source=path)
assert _options(conf,'W2')=='reference'
value=describe_native(conf)
assert value['capabilities']['native_history']['fsdp_units']==41
assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
print(json.dumps(dict(status='PASS_NO_MODEL_IMPORT',input_kind=conf.data['input_kind'])))
'''
    proc=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(root)],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=15)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)['status']=='PASS_NO_MODEL_IMPORT'


def test_old_default_graph_helper_receives_same_engine_without_new_phase(phase_protocol,monkeypatch):
    from contextlib import contextmanager
    from distillschedule.runtime import helios_native_history as native
    from distillschedule.runtime.helios_planner_cycle import _prepare_cycle
    from distillschedule.search.helios_implementation import HeliosImplementation
    from distillschedule.runtime.role_runner import OperationContext
    from distillschedule.runtime.version_sync import VersionLedger
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    engine,_,_=phase_protocol;seen=[]
    monkeypatch.setattr(native,'admit_engine',lambda *_:pytest.fail('old default entered native phase'))
    class Observer:
        mode,rank,device='native',0,'cpu'
        impl=HeliosImplementation('no-gan','world16-fsdp',512,1,False)
        generator,critic=engine,object()
        def versions(self):return dict(generator=0,critic=0,ema=0)
        @contextmanager
        def action(self,op,*,source_steps):
            row=dict(memory_after={});seen.append(row);yield row
    def original_helper(**kwargs):
        assert kwargs['generator_engine'] is engine
        return dict(original_engine_identity=True)
    p0=NS(_run_nogan_generator_update_p0=original_helper)
    prepared,_,_=_prepare_cycle(observer=Observer(),p0=p0,
        args=NS(timing_mode=True,profile_mode=False,profile_diagnostics=False),
        policy=StaticFSDP2P0Policy(rollout_batch_size=1,pack_real_score_pair=False),
        steps=HeliosCohortTask(0).source_steps,trace=NS(task=HeliosCohortTask(0)),
        negative=None,cfg=None,diagnostics=None,generator_free_memory={})
    op,fn=prepared._bound[0]
    assert fn(OperationContext(op,VersionLedger(dict(generator=0,critic=0,ema=0))))['original_engine_identity']
    assert 'native_history_phase' not in seen[0]
    assert not engine.events


def _cpu_import_capability_probe(probe, *, cuda_version):
    """Test-only exception normalization for the original optional import.

    Probe the real runtime first. Only the CPU-wheel missing-CUDA assertion is
    equivalent to the unsupported-GPU RuntimeError already handled upstream.
    """
    def checked(*args, **kwargs):
        try:
            return probe(*args, **kwargs)
        except AssertionError as error:
            if cuda_version is None and str(error) == 'Torch not compiled with CUDA enabled':
                raise RuntimeError('CPU-only topology fixture has no CUDA attention capability') from error
            raise
    return checked


def test_cpu_topology_import_probe_calls_original_and_preserves_success():
    seen = []
    value = object()
    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return value
    probe = _cpu_import_capability_probe(original, cuda_version=None)
    assert probe('device', explicit=True) is value
    assert seen == [(('device',), {'explicit': True})]


@pytest.mark.parametrize('cuda_version,error,translated', [
    (None, AssertionError('Torch not compiled with CUDA enabled'), True),
    ('12.8', AssertionError('Torch not compiled with CUDA enabled'), False),
    (None, AssertionError('unrelated assertion'), False),
    (None, RuntimeError('real unsupported GPU'), False),
    (None, ValueError('unrelated failure'), False),
])
def test_cpu_topology_import_probe_only_normalizes_missing_cpu_cuda(cuda_version, error, translated):
    seen = []
    def original():
        seen.append(True)
        raise error
    probe = _cpu_import_capability_probe(original, cuda_version=cuda_version)
    with pytest.raises(RuntimeError if translated else type(error)) as caught:
        probe()
    assert seen == [True]
    if translated:
        assert caught.value.__cause__ is error
        assert str(caught.value) == 'CPU-only topology fixture has no CUDA attention capability'
    else:
        assert caught.value is error
