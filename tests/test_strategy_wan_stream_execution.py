"""Persistent original dispatcher with real CPU Adam on explicit scalar fixtures.

Helper call interfaces are original; scalar losses/model boundaries are fixtures,
not original trained-Wan numerical or GPU performance evidence.
"""
from types import SimpleNamespace as NS
import pytest

torch = pytest.importorskip('torch')
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import WanStrategy, strategy_environment
from distillschedule.strategies.wan_bridge import OriginalWanSession
from distillschedule.strategies.wan_stream import task_at, bind_cohort
from distillschedule.strategies.wan_worker import execute_cycles


class Engine:
    def __init__(self):
        self.p = torch.nn.Parameter(torch.tensor([.25]))
        self.optimizer = torch.optim.Adam([self.p], lr=.01)
        self.global_steps = 0

    def update(self, target):
        self.optimizer.zero_grad(set_to_none=True)
        ((self.p-target)**2).mean().backward()
        self.optimizer.step(); self.global_steps += 1


def session_and_helpers(task, strategy):
    g, c = Engine(), Engine(); ema = NS(optimization_step=0, shadow=g.p.detach().clone())
    seen = []; builds = []
    def build(**kw):
        builds.append(kw); return g, c, ema, {'fixture': True}
    def gupdate(**kw):
        first = kw['trace'].logical_steps[0]
        g.update(torch.tensor([first/10]))
        seen.append(('g', first)); return {}
    def rollout(**kw):
        step = kw['steps'][0]
        assert step in kw['trace'].logical_steps
        seen.append(('r', step))
        value = g.p.detach().clone() + step/100
        kw['entry_outputs'][step] = NS(tensors={})
        return {step: value}, [{'batch_size': 1}], {}
    def critic(**kw):
        seen.append(('c', kw['step']))
        c.update(kw['x0']); return {}
    def estep():
        ema.shadow.mul_(.9).add_(g.p.detach(), alpha=.1); ema.optimization_step += 1
    ema.step = estep
    helpers = NS(_run_nogan_generator_update_p0=gupdate, _precompute_critic_rollouts=rollout,
                 _run_nogan_critic_precomputed=critic)
    session = OriginalWanSession(task=task, strategy=strategy,
        fair=NS(base=NS(_build_engines=build, p0=helpers)), bootstrap=NS(manifest={'lora_rank':64}),
        model_root='SCALAR_FIXTURE_NOT_WAN_WEIGHTS', device='cpu', rank=0,
        trace=NS(variant='no-gan', logical_steps=task.source_steps), negative=None, diagnostics=None)
    return session, builds, seen


@pytest.mark.parametrize('n,ema_every', [(1,1), (3,2), (5,3)])
def test_new_cohorts_keep_adam_and_obey_absolute_ema(n, ema_every, monkeypatch):
    import distillschedule.strategies.wan_trace as module
    task = OriginalWanTask(fake_updates=n, first_step=2*n, lora_rank=64, ema_every=ema_every)
    doc = {'stream_sha256':'fixture-stream', 'cohorts':[{
        'task':task_at(task, i).to_dict(), 'trace_bundle':f'fixture-cohort-{i}', 'trace_hash':str(i)} for i in range(3)]}
    monkeypatch.setattr(module, 'load_original_trace', lambda t,p,**kw: NS(variant='no-gan',logical_steps=t.source_steps))
    stub_torch = NS(cuda=NS(synchronize=lambda d: None)); dist=NS(barrier=lambda: None)
    signatures=[]
    for window in (1,n):
        strategy = WanStrategy('resident',2048,False,window)
        chosen = ('g', *(f'r{i}' for i in range(n)), *(f'c{i}' for i in range(n))) if window == n else (
            'g', *(key for i in range(n) for key in (f'r{i}',f'c{i}')))
        with strategy_environment(strategy):
            session, builds, seen = session_and_helpers(task,strategy)
            optimizers = (session.g.optimizer,session.c.optimizer)
            out = execute_cycles(session,cycles=3,mode='run',torch=stub_torch,distributed=dist,
                before_cycle=lambda cycle,current:bind_cohort(current,task,doc,cycle-1,chosen))
        assert len(builds)==1 and session.g.optimizer is optimizers[0] and session.c.optimizer is optimizers[1]
        assert out['final_versions']==dict(generator=3,critic=3*n,ema=sum(len(task_at(task,i).ema_after) for i in range(3)))
        assert [s for role,s in seen if role=='c']==list(range(task.first_step,task.first_step+3*n))
        assert [r['cohort']['cohort_index'] for r in out['cycles']]==[0,1,2]
        assert all(r['input_and_execution_wall_ns']>=r['duration_ns'] for r in out['cycles'])
        params=[session.g.p.detach().clone(),session.c.p.detach().clone(),session.ema.shadow.clone()]
        for engine in (session.g,session.c):
            params.extend(v.clone() for v in engine.optimizer.state[engine.p].values())
        signatures.append(params)
    assert all(torch.equal(a,b) for a,b in zip(*signatures))


def test_transition_cannot_replace_engines_before_training():
    task=OriginalWanTask(fake_updates=1,first_step=2,lora_rank=64)
    strategy=WanStrategy('resident',512,False,1)
    with strategy_environment(strategy):
        session,builds,seen=session_and_helpers(task,strategy)
        def reset(cycle,current):
            current.g=Engine()
            return {'operation_order':('g','r0','c0','e0'),'receipt':{}}
        with pytest.raises(ValueError,match='reset model'):
            execute_cycles(session,cycles=2,mode='run',torch=NS(cuda=None),distributed=None,before_cycle=reset)
    assert not seen
