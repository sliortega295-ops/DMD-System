"""CPU-only capacity-candidate contracts; never evidence of GPU performance."""
from copy import deepcopy
import inspect
from types import SimpleNamespace as NS
import pytest

from distillschedule.runtime import livewan_system_factorial_r2 as r2
from distillschedule.runtime import livewan_crossscale_worker as w
from distillschedule.runtime.livewan_cpu_optimizer import CPUAdamW
from distillschedule.runtime.livewan_baseline_placement import BaselinePlacement
from test_livewan_system_factorial_r2 import ROOT, HEAD, env
from test_livewan_system_factorial_chain import controller, controller_args


def test_explicit_capacity_arm_changes_no_other_A_or_B_configuration(env):
    control = r2.make_plan(ROOT, arm="only_A", source_head=HEAD)
    candidate = r2.make_plan(ROOT, arm=r2.CAPACITY_ARM, source_head=HEAD)
    assert r2.only_a(candidate, mode="pipeline")
    assert r2.generator_optimizer_cpu(candidate, mode="pipeline")
    assert not r2.generator_optimizer_cpu(control, mode="pipeline")
    for arm in ("baseline", "full"):
        p = r2.make_plan(ROOT, arm=arm, source_head=HEAD)
        assert not r2.generator_optimizer_cpu(p, mode="baseline" if arm == "baseline" else "pipeline")
    candidate.pop("id");control.pop("id")
    candidate["metadata"].pop(r2.CAPACITY_KEY)
    candidate["metadata"][r2.KEY]["arm"] = "only_A"
    assert candidate == control


@pytest.mark.parametrize("change", ("critic_cpu", "activation", "wrong_role", "B_cache", "batch", "offload_on_parent"))
def test_capacity_override_cannot_relax_frozen_configuration(env, change):
    p = r2.make_plan(ROOT, arm=r2.CAPACITY_ARM, source_head=HEAD)
    if change == "critic_cpu":p["metadata"][r2.CAPACITY_KEY]["critic_optimizer_device"] = "cpu"
    elif change == "activation":p["metadata"][r2.CAPACITY_KEY]["activation_offload"] = True
    elif change == "wrong_role":p["metadata"][r2.CAPACITY_KEY]["generator_optimizer_device"] = "cuda"
    elif change == "B_cache":p["metadata"]["generator_rollout_cast_cache"] = {"enabled": True}
    elif change == "batch":p["workload"]["shape"]["batch_size"] = 8
    else:p["metadata"][r2.KEY]["arm"] = "only_A"
    with pytest.raises(ValueError):r2.validate(p, mode="pipeline", repo_root=ROOT)


def test_original_global_CPU_placement_rejections_remain(env):
    for name in ("legacy-hybrid", "world16-fsdp"):
        with pytest.raises(ValueError):BaselinePlacement(name, optimizer_device="cpu")
    with pytest.raises(ValueError, match="BLOCKED"):
        r2.make_plan(ROOT, arm="only_B", source_head=HEAD)


def test_real_generator_factory_cpu_initialization_grad_mean_clip_and_Adam(env):
    torch = pytest.importorskip("torch")
    torch.manual_seed(101)
    p = r2.make_plan(ROOT, arm=r2.CAPACITY_ARM, source_head=HEAD)
    factory = w._generator_optimizer_factory(torch, plan=p, mode="pipeline",
                                            baseline_spec=BaselinePlacement())
    values = [torch.nn.Parameter(torch.randn(5, 7)), torch.nn.Parameter(torch.randn(9))]
    refs = [torch.nn.Parameter(v.detach().clone()) for v in values]
    rng = torch.get_rng_state().clone()
    kw = dict(lr=2e-6, betas=(0., .99), eps=1e-8, weight_decay=0., foreach=True)
    off = factory(values, **kw);assert isinstance(off, CPUAdamW)
    off.materialize_state(step=0);ref = torch.optim.AdamW(refs, **kw)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(values, refs))
    for step in range(4):
        for a,b in zip(values, refs):
            summed = torch.randn_like(a)
            a.grad = summed.clone().div_(16);b.grad = summed.clone().div_(16)
        n1=torch.nn.utils.clip_grad_norm_(values, 1.);n2=torch.nn.utils.clip_grad_norm_(refs, 1.)
        assert torch.equal(n1,n2)
        off.step();ref.step()
        for a,b in zip(values,refs):
            assert torch.equal(a,b)
            for key in ("step","exp_avg","exp_avg_sq"):
                assert torch.equal(off.state[a][key],ref.state[b][key])
        off.zero_grad();ref.zero_grad()
    receipt=off.receipt()
    assert receipt["steps"]==4 and receipt["transfers_and_cpu_update_inside_timer"]
    assert receipt["gradient_d2h_bytes"]==receipt["updated_parameter_h2d_bytes"]>0
    assert all(a.grad is None for a in values)
    assert not torch.cuda.is_initialized()


def test_original_factories_and_Critic_path_unchanged(env):
    torch=pytest.importorskip("torch")
    for arm in ("baseline","only_A","full"):
        p=r2.make_plan(ROOT,arm=arm,source_head=HEAD)
        assert w._generator_optimizer_factory(torch,plan=p,mode="baseline" if arm=="baseline" else "pipeline",baseline_spec=BaselinePlacement()) is torch.optim.AdamW
    source=inspect.getsource(w.build_models)
    assert source.count("_generator_optimizer_factory(")==1
    assert 'if baseline_spec.optimizer_device == "cpu":\n            from distillschedule.runtime.livewan_cpu_optimizer import CPUAdamW\n            critic_optimizer_factory' in source
    assert inspect.getsource(w.run_worker).index("timing_start_ns =") < inspect.getsource(w.run_worker).index("system_r2.attach_capacity_optimizer_receipt(")


def test_native_controller_and_shell_bind_explicit_child_not_global_cpu(env):
    import subprocess,os
    c=controller();a=c.parser().parse_args(controller_args(r2.CAPACITY_ARM))
    for node in (0,1):
        text=" ".join(c.command(a,node,"192.0.2.1"))
        assert f"LIVEWAN_SYSTEM_FACTORIAL_ARM={r2.CAPACITY_ARM}" in text
        assert "LIVEWAN_OPTIMIZER_DEVICE=cuda" in text
    script=(ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    start=script.index('if [[ "${RUN_KIND}" == system-factorial-timing ]]')
    branch=script[start:script.index('\nfi',start)+3]
    e=dict(os.environ,RUN_KIND=r2.RUN_KIND,LIVEWAN_SYSTEM_FACTORIAL_SEED=str(r2.SEED),PROFILE_TWO_CYCLES='0',DISTILLSCHEDULE_NSYS_BIN='',DISTILLSCHEDULE_NCU_BIN='',LIVEWAN_SYSTEM_FACTORIAL_ARM=r2.CAPACITY_ARM,EXECUTION_MODE='pipeline',PIPELINE_VARIANT='c51-continuous-b16-generator-fused-rms')
    assert subprocess.run(['bash','-c',branch],env=e,capture_output=True).returncode==0


def test_real_optimizer_receipt_producer_only_touches_generator_rank(env):
    torch=pytest.importorskip("torch")
    p=r2.make_plan(ROOT,arm=r2.CAPACITY_ARM,source_head=HEAD)
    parameter=torch.nn.Parameter(torch.ones(2))
    opt=CPUAdamW(torch,[parameter],lr=2e-6);opt.materialize_state()
    parameter.grad=torch.ones_like(parameter);opt.step()
    b=NS(generator_optimizer=opt,metadata={"student":{},"critic":None})
    r2.attach_capacity_optimizer_receipt(b,plan=p,mode="pipeline",rank=0)
    proof=b.metadata['student']['cpu_optimizer']
    assert {k:proof[k] for k in opt.receipt()}==opt.receipt()
    assert proof['cpu_gradient_buffer_bytes']==8
    assert proof['pinned_parameter_mirror_bytes']==proof['pinned_gradient_buffer_bytes']==0
    assert b.metadata['capacity_offload']['scope']=='GENERATOR_OPTIMIZER_ONLY'
    untouched=NS(metadata={"student":None,"critic":{"optimizer":{"state_device":"cuda"}}})
    r2.attach_capacity_optimizer_receipt(untouched,plan=p,mode="pipeline",rank=8)
    assert untouched.metadata['critic']=={"optimizer":{"state_device":"cuda"}}
    assert untouched.metadata['capacity_offload']['execution_status']=='NOT_APPLICABLE_SCORE_RANK'


def test_child_reuses_real_four_cycle_loop_writer_collector_and_keeps_mock_unclaimable(monkeypatch,env):
    # The existing synthetic fixture stubs only model/transport leaves. Its
    # capacity byte counts below are SYNTHETIC_TEST_ONLY, not a model allocation.
    from test_livewan_system_factorial_chain import produce_run
    from distillschedule.runtime import livewan_system_factorial_collect as collector
    import json
    trace=produce_run(monkeypatch,env,r2.CAPACITY_ARM)
    with pytest.raises(ValueError,match='CPU mock'):
        collector.collect(env,arm=r2.CAPACITY_ARM,expected_head=HEAD,trace=trace)
    s=collector.collect(env,arm=r2.CAPACITY_ARM,expected_head=HEAD,trace=trace,allow_synthetic=True)
    assert s['status']=='PASS_CPU_MOCK_ONLY' and s['cycle_2_seconds'] is None
    assert not s['native_descriptive_timing_eligible']
    p=env/'receipts/rank_000.json';row=json.loads(p.read_text())
    row['build']['student']['cpu_optimizer']['steps']=3
    w.atomic_json(p,row)
    s=collector.collect(env,arm=r2.CAPACITY_ARM,expected_head=HEAD,trace=trace,allow_synthetic=True)
    assert s['status']=='FAIL' and any('steps' in x for x in s['errors'])
