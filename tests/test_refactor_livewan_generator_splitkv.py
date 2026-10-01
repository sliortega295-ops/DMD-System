"""C49 default-off policy, real dispatch and one-to-one runtime receipt gates."""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime import livewan_generator_splitkv as m

ROOT = Path(__file__).resolve().parents[1]


def test_lazy_default_off():
    env=dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONDONTWRITEBYTECODE="1")
    p=subprocess.run([sys.executable,"-c",
        "import sys;from distillschedule.runtime.livewan_generator_splitkv import enabled;"
        "assert not enabled({},mode='pipeline');assert 'torch' not in sys.modules"],
        env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert not any(x in inspect.getsource(m.ReadOnlySplitKV) for x in
                   (".cpu(", ".tolist(", ".item(", "synchronize(", "copy_("))


@pytest.fixture
def plans(monkeypatch):
    from distillschedule.core.config import load_plan
    for key in ("LIVEWAN_STUDENT_CHECKPOINT","LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT","LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP","DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key,"/unread/"+key)
    d=ROOT/"configs/distillschedule/plans"
    return tuple(load_plan(d/file).plan.to_dict() for file in (
        "livewan_wan13b_14b_sf_dmd_c38_continuous_b16_score_fa3.json",
        "livewan_wan13b_14b_sf_dmd_c49_generator_readonly_splitkv.json"))


def test_only_one_factor_and_worker_plan_gate(plans):
    from distillschedule.runtime.livewan_crossscale_worker import (
        _validate_plan_identity,_apply_trajectory_contract)
    a,b=copy.deepcopy(plans)
    assert not m.enabled(a,mode="pipeline")
    assert m.enabled(b,mode="pipeline")
    assert b["metadata"].pop("generator_readonly_splitkv")==m.CONTRACT
    b["id"]=a["id"];assert a==b
    b=copy.deepcopy(plans[1])
    _apply_trajectory_contract(b,cycles=2,mode="pipeline")
    _validate_plan_identity(b,mode="pipeline")
    with pytest.raises(RuntimeError):m.enabled(b,mode="baseline")


@pytest.mark.parametrize("field,value", [
    ("num_splits",4),("reuse_previous_kv",True),("append_kv",True),
    ("grad_backend_unchanged",False),("query_tokens",5520),
    ("application_calls_unchanged",False),("kernel_dtype","float32")])
def test_plan_fail_closed(plans,field,value):
    p=copy.deepcopy(plans[1]);p["metadata"]["generator_readonly_splitkv"][field]=value
    with pytest.raises(RuntimeError):m.enabled(p,mode="pipeline")


def dummy_kvcache(q,k,v,**kwargs):
    return q


def tensor(q_tokens=2760,k=False):
    return SimpleNamespace(shape=(1,q_tokens,12,128),dtype="bf16",requires_grad=False,
                           is_cuda=True,is_contiguous=lambda:True,_version=0)


@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.setattr(m,"EXPECTED_KVCACHE_SOURCE_SHA256",
        hashlib.sha256(inspect.getsource(dummy_kvcache).encode()).hexdigest())
    torch=SimpleNamespace(bfloat16="bf16",is_grad_enabled=lambda:False)
    return m.ReadOnlySplitKV(torch,SimpleNamespace(flash_attn_with_kvcache=dummy_kvcache))


def options():
    return dict(dropout_p=0.,softmax_scale=128**(-.5),causal=False,
                window_size=(-1,-1),deterministic=False)


def test_actual_readonly_call_and_input_inventory(proxy):
    q,k,v=tensor(),tensor(19320),tensor(19320)
    ledger=[]
    def function(qq,kk,vv,**kwargs):
        ledger.append(kwargs);return qq
    proxy.function=function
    def forbidden(*args,**kwargs):raise AssertionError("unexpected reference fallback")
    assert proxy.execute(q,k,v,reference=forbidden,reference_options=options()) is q
    assert ledger[0]["k"] is ledger[0]["v"] is ledger[0]["cache_seqlens"] is None
    assert ledger[0]["num_splits"]==2 and ledger[0]["rotary_cos"] is None
    assert proxy.receipt["eligible_nograd_calls"]==1
    proxy.receipt["reference_calls"]=1
    assert not m.runtime_receipt_errors(proxy.receipt,rank=0,total_calls=2)
    assert m.runtime_receipt_errors(proxy.receipt,rank=0,total_calls=3)


@pytest.mark.parametrize("case",["grad_mode","input_grad","cross","short","other_query"])
def test_gradient_and_ineligible_calls_never_use_kvcache(proxy,case):
    q,k,v=tensor(),tensor(19320),tensor(19320)
    if case=="grad_mode":proxy.torch.is_grad_enabled=lambda:True
    if case=="input_grad":q.requires_grad=True
    if case=="cross":k=v=tensor(512)
    if case=="short":k=v=tensor(5520)
    if case=="other_query":q=tensor(19320)
    proxy.function=lambda *a,**kw:pytest.fail("KV function must not run")
    seen=[]
    def reference(*args,**kwargs):seen.append(kwargs);return q
    assert proxy.execute(q,k,v,reference=reference,reference_options=options()) is q
    assert seen==[options()] and proxy.receipt["eligible_nograd_calls"]==0


@pytest.mark.parametrize("field,value",[("causal",True),("dropout_p",.1),
                                      ("deterministic",True),("window_size",(1,1))])
def test_eligible_bad_options_do_not_fallback(proxy,field,value):
    args=options();args[field]=value
    with pytest.raises(RuntimeError):
        proxy.execute(tensor(),tensor(19320),tensor(19320),
                      reference=lambda *a,**k:None,reference_options=args)
    assert proxy.receipt["contract_failure_count"]==1


def test_kernel_source_identity_fail_closed(monkeypatch):
    monkeypatch.setattr(m,"EXPECTED_KVCACHE_SOURCE_SHA256","f"*64)
    with pytest.raises(RuntimeError):
        m.ReadOnlySplitKV(None,SimpleNamespace(flash_attn_with_kvcache=dummy_kvcache))


@pytest.mark.parametrize("rank",range(16))
def test_rank_receipt_must_prove_real_execution(rank,proxy):
    if rank<8:
        assert m.runtime_receipt_errors(proxy.receipt,rank=rank,total_calls=0)
    else:
        receipt=dict(enabled=True,name=m.CONTRACT["name"],contract=m.CONTRACT,
                     production_installed=False,execution_status="NOT_APPLICABLE",eligible_nograd_calls=0)
        assert not m.runtime_receipt_errors(receipt,rank=rank)
        receipt["eligible_nograd_calls"]=1
        assert m.runtime_receipt_errors(receipt,rank=rank)


@pytest.mark.parametrize("kind",["trajectory","two-cycle-timing"])
def test_launcher_routes_exact_default_off_child(tmp_path,kind):
    args=[sys.executable,str(ROOT/"scripts/distillschedule/run_livewan_crossscale_world16.py"),
        "--hosts","node0,node1","--run-id","c49-dryrun","--run-kind",kind,
        "--execution-mode","pipeline","--pipeline-variant",m.VARIANT,
        "--expected-head","a"*40,"--master-addr","127.0.0.1","--dry-run",
        "--local-evidence-root",str(tmp_path)]
    for key in ("remote-root","remote-output-root","student-checkpoint","critic-checkpoint",
                "teacher-checkpoint","trace","bootstrap","source-root","source-receipt","python-overlay"):
        args+=["--"+key,str(tmp_path/key)]
    if kind=="trajectory":
        args+=["--trajectory-cycles","2","--trajectory-arm-id","c49","--trajectory-seed","20260830"]
    else:
        args+=["--two-cycle-arm-id","c49","--two-cycle-seed","20260830",
               "--correctness-receipt",str(tmp_path/"gate.json"),"--correctness-sha256","b"*64]
    r=subprocess.run(args,check=True,capture_output=True,text=True,timeout=15)
    assert all(m.VARIANT in " ".join(v) for v in json.loads(r.stdout)["commands"])
    shell=(ROOT/"scripts/distillschedule/launch_livewan_crossscale_world16_node.sh").read_text()
    assert shell[:shell.index("  PLAN=")].count(m.VARIANT)>=3
    assert "livewan_wan13b_14b_sf_dmd_c49_generator_readonly_splitkv.json" in shell
