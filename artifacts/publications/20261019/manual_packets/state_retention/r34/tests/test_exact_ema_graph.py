"""SYNTHETIC_TEST_ONLY real CPU tensors/recurrence, fake graph submission."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,sys
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from exact_ema_graph import ExactEMAStep,schedule,identities
from distillschedule.runtime.wan13b_upper_bound import FreshTrainableEMA
from distillschedule.runtime.fsdp2_static import ema_update_reference

def ema():
    x=FreshTrainableEMA.__new__(FreshTrainableEMA)
    x.implementation='reference';x.optimization_step=10;x.decay=.999;x.update_after_step=0;x.min_decay=0.;x.cur_decay_value=None;x.last_transfer_bytes=0
    x._parameters=[torch.nn.Parameter(torch.arange(35,dtype=torch.float32).reshape(5,7)/37),torch.nn.Parameter(torch.linspace(-.1,.1,9))]
    x._shadows=[torch.ones_like(p)*.013 for p in x._parameters]
    return x

def driver(x):
    graphs={}
    for step,decay in schedule(x).items():
        def replay(d=decay):
            for s,p in zip(x._shadows,x._parameters,strict=True):ema_update_reference(s,p.detach().float(),d)
        graphs[step]=NS(replay=replay)
    receipt=dict(graph_replays=0,observed_steps=[])
    return ExactEMAStep(x,graphs,receipt),receipt

def test_ten_real_cpu_updates_bitwise_output_metadata_and_rng():
    a=ema();b=copy.deepcopy(a);candidate,receipt=driver(b);rng=torch.get_rng_state().clone()
    for step in range(11,21):
        if step==16:
            for x,y in zip(a._parameters,b._parameters,strict=True):
                with torch.no_grad():x.add_(.0000123);y.add_(.0000123)
        value=FreshTrainableEMA.step(a);other=candidate.step()
        assert value==other and a.optimization_step==b.optimization_step and a.cur_decay_value==b.cur_decay_value
        assert all(torch.equal(x,y)for x,y in zip(a._shadows,b._shadows,strict=True))
    assert receipt==dict(graph_replays=10,observed_steps=list(range(11,21)))
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError,match='extra update'):candidate.step()

@pytest.mark.parametrize('change',['skip','pointer','dtype','shape','decay'])
def test_identity_and_version_fail_closed(change):
    x=ema();d,r=driver(x)
    if change=='skip':x.optimization_step+=1
    elif change=='pointer':x._shadows[0]=x._shadows[0].clone()
    elif change=='dtype':x._parameters[0]=torch.nn.Parameter(x._parameters[0].double())
    elif change=='shape':x._shadows[0]=x._shadows[0].flatten()
    elif change=='decay':x.get_decay=lambda step:.123
    with pytest.raises(ValueError):d.step()
    assert r['graph_replays']==0

@pytest.mark.parametrize('change',['foreach','start','steps','missing'])
def test_unknown_implementation_or_graph_set_rejected(change):
    x=ema()
    if change=='foreach':x.implementation='foreach'
    elif change=='start':x.optimization_step=11
    if change=='steps':
        with pytest.raises(ValueError):schedule(x,11)
    elif change=='missing':
        with pytest.raises(ValueError):ExactEMAStep(x,{},dict(graph_replays=0,observed_steps=[]))
    else:
        with pytest.raises(ValueError):schedule(x)

def test_capture_function_contains_only_original_math_not_foreach_or_merge():
    import inspect,exact_ema_graph
    body=inspect.getsource(exact_ema_graph.capture)
    assert 'ema_update_reference(shadow,parameter.detach().float(),decay)'in body
    assert '_foreach_'not in body and 'get_delta_weight'not in body
