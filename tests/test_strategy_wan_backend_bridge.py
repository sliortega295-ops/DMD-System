"""Verify that choices reach the OLD native handlers, rather than a toy backend."""
from types import SimpleNamespace as NS
import os
import pytest
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import WanStrategy,implementations,strategy_environment
from distillschedule.strategies.wan_bridge import OriginalWanSession
from distillschedule.search.wan_implementation import current_implementation


@pytest.mark.parametrize('n',[1,3,8])
@pytest.mark.parametrize('impl', [('fsdp',512,False),('resident',512,False),('resident',1024,True),('resident',2048,True)])
def test_original_routes_and_variable_updates(n,impl):
    task=OriginalWanTask(fake_updates=n,first_step=2*n,lora_rank=64)
    strategy=WanStrategy(*impl,rollout_window=n)
    g,c,ema=NS(global_steps=2),NS(global_steps=10),NS(optimization_step=10)
    calls=[];construction=[]
    def build(**kw):
        construction.append(kw['args']);assert current_implementation()==strategy.backend
        return g,c,ema,{'fixture':'handler-only'}
    def gupdate(**kw):calls.append('g');g.global_steps+=1;return {}
    def rollout(**kw):
        step=kw['steps'][0];calls.append(('r',step))
        kw['entry_outputs'][step]=NS(tensors={})
        return {step:(step,g.global_steps)},[{}],{}
    def critic(**kw):
        assert kw['x0']==(kw['step'],3);calls.append(('c',kw['step']));c.global_steps+=1;return {}
    def estep():ema.optimization_step+=1;calls.append('e')
    ema.step=estep
    p0=NS(_run_nogan_generator_update_p0=gupdate,_precompute_critic_rollouts=rollout,
          _run_nogan_critic_precomputed=critic)
    fair=NS(base=NS(_build_engines=build,p0=p0))
    with strategy_environment(strategy):
        session=OriginalWanSession(task=task,strategy=strategy,fair=fair,
            bootstrap=NS(manifest={'lora_rank':64}),model_root='local-model',device='CPU_HANDLER_FIXTURE',rank=0,
            trace=NS(logical_steps=task.source_steps,variant='no-gan'),negative=None,diagnostics=None)
        assert construction[0].original_wan_task is task
        assert construction[0].placement==strategy.backend.native_placement
        assert os.environ.get('NCCL_ALGO')==('Tree' if strategy.placement=='resident' else None)
        assert os.environ['DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE']==str(int(strategy.lora_epilogue))
        prepared=session.prepare();assert not calls
        result=prepared.run()
        assert len(result['critic'])==n and prepared.receipt()['peak_live_inputs']==n
        assert session.versions()==dict(generator=3,critic=10+n,ema=10+n)
    assert 'DISTILLSCHEDULE_SEARCH_CANDIDATE_ID' not in os.environ


def test_candidate_set_contains_original_winner_and_control_without_hardcoding_winner():
    task=OriginalWanTask()
    candidates=implementations(task,triton_available=True)
    assert len(candidates)==7
    assert WanStrategy('resident',2048,True) in candidates
    assert WanStrategy('fsdp',512,False) in candidates
    assert len(implementations(task,triton_available=False))==4
    with pytest.raises(ValueError): WanStrategy('fsdp',2048,True)


def test_environment_restored_on_failure(monkeypatch):
    monkeypatch.setenv('NCCL_ALGO','Ring')
    with pytest.raises(RuntimeError):
        with strategy_environment(WanStrategy('resident',2048,True)):
            assert os.environ['NCCL_ALGO']=='Tree';raise RuntimeError('fixture')
    assert os.environ['NCCL_ALGO']=='Ring'


def test_existing_modulation_implementation_is_actually_applied():
    from distillschedule.runtime.wan13b_fair import configure_modulation_chunk_tokens
    for chunk in (512,1024,2048):
        strategy=WanStrategy('resident',chunk,False)
        module=NS(blocks=[NS() for _ in range(30)])
        with strategy_environment(strategy):
            receipt=configure_modulation_chunk_tokens(module,placement=strategy.backend.native_placement)
        assert receipt['effective_tokens']==[chunk]
        assert all(b._distillschedule_modulation_chunk_tokens==chunk for b in module.blocks)
