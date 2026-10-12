"""Original-helper dispatch tests; no mocked timing or GPU speed claims."""
from types import SimpleNamespace as NS
import pytest
from distillschedule.strategies.wan_cycle import prepare_parameterized_wan_cycle


def case(n=3, window=1, ema_after=(), **overrides):
    calls = []
    g, c = NS(global_steps=7), NS(global_steps=23)
    ema = NS(optimization_step=9)
    def e_step(): ema.optimization_step += 1; calls.append("ema")
    ema.step = e_step
    steps = tuple(range(100, 100+n))
    trace = NS(logical_steps=steps, variant="no-gan")
    def gen(**kw):
        assert kw["trace"] is trace and kw["generator_engine"] is g
        assert kw["critic_engine"] is c and kw["score_pack_parity_check"] is False
        g.global_steps += 1; calls.append("g")
        return {"logical_step": 10}
    def rollout(**kw):
        assert kw["trace"] is trace and kw["generator_engine"] is g
        assert g.global_steps == 8
        step, = kw["steps"]
        payload, entry = object(), NS(tensors={})
        kw["entry_outputs"][step] = entry
        calls.append(f"r{step}")
        return {step:payload}, [{"payload":payload}], {}
    def critic(**kw):
        assert kw["critic_engine"] is c and kw["trace"] is trace
        assert kw["x0"] is not None and kw["entry"] is not None
        c.global_steps += 1; calls.append(f"c{kw['step']}")
        return {"step": kw["step"]}
    args = dict(p0=NS(_run_nogan_generator_update_p0=gen,
                _precompute_critic_rollouts=rollout,_run_nogan_critic_precomputed=critic),
                source_steps=steps, ema_after=ema_after, max_live_rollouts=window,
                rank=0,device="cpu",generator_engine=g,critic_engine=c,ema=ema,
                trace=trace,negative=None,diagnostics=None,
                policy=NS(rollout_batch_size=1,pack_real_score_pair=False,generator_nograd_persistent_unshard=False),
                clean_runtime=True,runtime_cache={},deferred_finite_checks=[])
    args.update(overrides)
    return args, calls


@pytest.mark.parametrize("n", [1,2,3,5,8])
@pytest.mark.parametrize("all_ema", [False,True])
def test_selected_frequency_executes_original_helpers(n, all_ema):
    positions = tuple(range(n)) if all_ema else ()
    args, calls = case(n,n,positions)
    cycle = prepare_parameterized_wan_cycle(**args)
    out = cycle.run(); receipt=cycle.receipt()
    assert len(out["rollouts"]) == len(out["critic"]) == n
    assert out["generator"]["logical_step"] == 100
    assert receipt["native_versions"] == dict(generator=8,critic=23+n,ema=9+len(positions))
    assert receipt["status"] == "PASS" and receipt["peak_live_inputs"] == n
    assert calls[0] == "g" and calls[1:n+1] == [f"r{100+i}" for i in range(n)]
    with pytest.raises(ValueError): cycle.run()


def test_no_trace_relabel_or_wrong_protocol():
    args,_=case(); args["source_steps"]=(100,)
    with pytest.raises(ValueError,match="trace steps"): prepare_parameterized_wan_cycle(**args)
    args,_=case(); args["trace"].variant="gan"
    with pytest.raises(ValueError,match="GAN"): prepare_parameterized_wan_cycle(**args)


def test_native_commit_mismatch_does_not_pass():
    args,_=case(1); args["p0"]._run_nogan_critic_precomputed=lambda **kw: {}
    cycle=prepare_parameterized_wan_cycle(**args)
    with pytest.raises(ValueError,match="native engine commits"):cycle.run()
    assert cycle.receipt()["status"]=="FAIL"


def test_selected_order_is_not_ignored():
    args,calls=case(3,1,order=("g","r0","c0","r1","c1","r2","c2"))
    cycle=prepare_parameterized_wan_cycle(**args);cycle.run()
    assert calls==["g","r100","c100","r101","c101","r102","c102"]
    args,_=case(3,1,order=("g","r0","r1","r2","c0","c1","c2"))
    with pytest.raises(ValueError,match="retained window"):prepare_parameterized_wan_cycle(**args)
    args,_=case(1,1,order=("g","c0","r0"))
    with pytest.raises(ValueError):prepare_parameterized_wan_cycle(**args)
