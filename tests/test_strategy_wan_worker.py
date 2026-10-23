"""CPU orchestration tests; never pretend injected native handlers ran a GPU."""
import json
from types import SimpleNamespace as NS
import pytest
from distillschedule.strategies.wan_worker import execute_cycles,topology,OriginalWanAssets
from distillschedule.strategies.wan_strategy import WanStrategy,strategy_environment
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_bridge import OriginalWanSession


@pytest.mark.parametrize('n',[1,3,8])
def test_selected_native_order_drives_original_session_over_two_cycles(n):
    t=OriginalWanTask(fake_updates=n,first_step=2*n,lora_rank=64)
    strategy=WanStrategy('resident',2048,True,n)
    g,c,ema=NS(global_steps=2),NS(global_steps=10),NS(optimization_step=10)
    emitted=[]
    def gupdate(**kw):g.global_steps+=1;emitted.append('g');return {}
    def rollout(**kw):
        s=kw['steps'][0];emitted.append('r'+str(t.source_steps.index(s)))
        kw['entry_outputs'][s]=NS(tensors={});return {s:(s,g.global_steps)},[{}],{}
    def update(**kw):
        assert kw['x0']==(kw['step'],g.global_steps)
        emitted.append('c'+str(t.source_steps.index(kw['step'])));c.global_steps+=1;return {}
    def estep():ema.optimization_step+=1;emitted.append('e')
    ema.step=estep
    p0=NS(_run_nogan_generator_update_p0=gupdate,_precompute_critic_rollouts=rollout,
          _run_nogan_critic_precomputed=update)
    fair=NS(base=NS(_build_engines=lambda **kw:(g,c,ema,{}),p0=p0))
    observed=[]
    cuda=NS(synchronize=lambda device:None)
    dist=NS(barrier=lambda:None)
    # Deliberately feed a legal explicit order, not a legacy Full5 recipe.
    order=('g',*(f'r{i}' for i in range(n)),*(key for i in range(n) for key in (f'c{i}',f'e{i}')))
    with strategy_environment(strategy):
        s=OriginalWanSession(task=t,strategy=strategy,fair=fair,bootstrap=NS(manifest={'lora_rank':64}),
            model_root='fixture',device='CPU_HANDLER_FIXTURE',rank=0,
            trace=NS(variant='no-gan',logical_steps=t.source_steps),negative=None,diagnostics=None)
        r=execute_cycles(s,cycles=2,mode='run',torch=NS(cuda=cuda),distributed=dist,
                         operation_order=order,progress=lambda *x:observed.append(x))
    assert r['final_versions']==dict(generator=4,critic=10+2*n,ema=10+2*n)
    assert r['warm_initial_versions']==dict(generator=3,critic=10+n,ema=10+n)
    assert len(observed)==2
    assert all(row['prepared']['executed_operations']==list(order) for row in r['cycles'])


@pytest.mark.parametrize('environment',[{}, {'WORLD_SIZE':'8','RANK':'0','LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'},
    {'WORLD_SIZE':'16','RANK':'8','LOCAL_RANK':'1','LOCAL_WORLD_SIZE':'8'}])
def test_topology_not_inferred_or_silently_changed(environment):
    with pytest.raises(ValueError):topology(environment)


def test_world16_topology():
    assert topology({'WORLD_SIZE':'16','RANK':'15','LOCAL_RANK':'7','LOCAL_WORLD_SIZE':'8'})==(15,7)


def test_cli_uses_original_contract_not_new_euler(tmp_path,capsys):
    from distillschedule.cli import main
    p=tmp_path/'task.json';p.write_text(json.dumps(OriginalWanTask(fake_updates=3,first_step=6).to_dict()))
    code=main(['workloads','native','validate','--task',str(p)])
    output=json.loads(capsys.readouterr().out)
    assert code==0 and output['protocol']=='ORIGINAL_WAN_HELIOS_PYRAMID_NOGAN'
    assert output['task']['fake_updates']==3 and output['native_execution'].startswith('NOT_RUN')
    assert any(v['placement']=='fsdp' for v in output['implementations'])
    assert any(v['placement']=='resident' and v['chunk_tokens']==2048 for v in output['implementations'])


def test_execute_required_before_original_worker_imports_or_allocates(tmp_path,monkeypatch,capsys):
    from distillschedule.cli import main
    import distillschedule.strategies.wan_worker as worker
    def reject(*a,**kw):raise AssertionError('not authorized to allocate')
    monkeypatch.setattr(worker,'run_worker',reject)
    p=tmp_path/'task.json';p.write_text(json.dumps(OriginalWanTask().to_dict()))
    assert main(['workloads','native','calibrate','--task',str(p),'--assets','absent',
                 '--implementation','absent','--output',str(tmp_path/'not-created')])==0
    assert not (tmp_path/'not-created').exists()


def test_original_plan_cli_uses_observed_optimized_arm(tmp_path,capsys):
    from test_strategy_wan_planning import collection
    from distillschedule.cli import main
    t=OriginalWanTask(fake_updates=3,first_step=6)
    p=tmp_path/'task.json';p.write_text(json.dumps(t.to_dict()))
    source=collection(tmp_path,t,[(WanStrategy('fsdp',512,False),100,50,10,200),
                                  (WanStrategy('resident',2048,True),30,70,10,200)])
    target=tmp_path/'plan.json'
    assert main(['workloads','native','plan','--task',str(p),'--calibration',str(source),'--output',str(target)])==0
    data=json.loads(target.read_text())
    assert data['selected']['strategy']['chunk_tokens']==2048
    assert data['selected']['strategy']['lora_epilogue'] is True
    assert data['evidence_scope']=='CPU_FIXTURE_ONLY'


@pytest.mark.parametrize('n',[1,3,8])
def test_original_cuda_observer_protocol_parameterizes_stage_inventory(n):
    from test_refactor_wan_stage_calibration import FakeBackend,SOURCE,DEVICE
    from test_refactor_wan_action_calibration import Tensor
    from distillschedule.strategies.wan_calibration import OriginalWanCalibrator,two_cycles
    t=OriginalWanTask(fake_updates=n,first_step=2*n)
    initial=dict(generator=2,critic=10,ema=10)
    s=WanStrategy('resident',2048,True)
    graph=two_cycles(t,initial,1)
    backend=FakeBackend();backend.is_tensor=lambda x:isinstance(x,Tensor)
    c=OriginalWanCalibrator(task=t,initial_versions=initial,work={op.metadata['action']:{} for op in graph.operations},
        torch=backend,distributed=backend,device=DEVICE,rank=0,implementation=s,source_head=SOURCE,
        clock=backend.clock,evidence_scope='CPU_FIXTURE_ONLY')
    for i,op in enumerate(graph.operations):
        with c.stage(op.metadata['cycle'],op.metadata['action']) as sample:
            backend.body(allocated=120,peak_allocated=180)
        if op.metadata['action'].startswith('r'):
            sample.record_output(Tensor(100+i,20))
    result=c.finish()
    assert result['status']=='PASS' and result['evidence_scope']=='CPU_FIXTURE_ONLY'
    assert len(result['stages'])==2*(1+3*n)
    assert len(result['output_lifetimes'])==2*n and result['retained_input_bytes_at_finish']==0
