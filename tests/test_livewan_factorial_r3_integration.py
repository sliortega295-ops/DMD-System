"""Real lowerer/worker/collector bindings; no GPU or synthetic speed claims."""
from copy import deepcopy
from pathlib import Path
import os

import pytest

from distillschedule.runtime import livewan_system_factorial_r2 as r2
from distillschedule.runtime import livewan_factorial_r3 as r3
from distillschedule.runtime import livewan_crossscale_worker as worker

ROOT = Path(__file__).resolve().parents[1]
HEAD = "a"*40


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT", "LIVEWAN_TEACHER_CHECKPOINT",
                 "LIVEWAN_OPERATION_TRACE", "LIVEWAN_SYSTEMS_BOOTSTRAP"):
        monkeypatch.setenv(name, str(tmp_path/name))
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(tmp_path/"run"))
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "SYNTHETIC_TEST_ONLY")
    monkeypatch.setenv("DMD_SYSTEM_ROOT", str(ROOT))
    monkeypatch.setenv("LIVEWAN_EXPECTED_HEAD", HEAD)


@pytest.mark.parametrize("arm", r3.ARMS)
def test_real_lowerer_worker_original_shape_and_runtime_flags(env, arm):
    plan = r2.make_plan(ROOT, arm=arm, source_head=HEAD)
    mode = r2.mode_for_arm(arm)
    r2.validate(plan, mode=mode, repo_root=ROOT)
    worker._validate_plan_identity(plan, mode=mode)
    assert plan["workload"]["shape"]["resolution"] == dict(height=368,width=640)
    assert plan["workload"]["shape"]["batch_size"] == 16
    assert plan["evidence"]["steps"] == 20
    assert worker._generator_runahead_transport_enabled(plan,mode=mode) == (arm==r3.A_ARM)
    assert worker._continuous_pipeline_contract(plan,mode=mode) is not None if arm==r3.A_ARM else True
    assert worker._full_context_lens_elision(plan,mode=mode) == (arm==r3.B_ARM)
    assert not worker._critic_attention_checkpoint_enabled(plan,mode=mode)
    assert not worker._generator_rollout_cast_cache_enabled(plan,mode=mode)
    if arm == r3.A_ARM:
        assert plan["training_backends"]["generator"]["kind"] == "fsdp"
        assert plan["training_backends"]["generator"]["options"]["reshard_after_forward"] is True
        assert plan["training_backends"]["generator"]["options"]["resident"] is False
        assert plan["metadata"][r3.KEY]["B"] is False
        assert r3.generator_fsdp8(plan,mode=mode)
    else:
        assert r3.complete_b(plan,mode=mode)
        active = set(r2.B_METADATA)-{"checkpoint_composition","capacity_recovery"}
        assert set(plan["metadata"][r3.KEY]["mechanism_contracts"]) == active
        assert plan["metadata"]["baseline_placement"]["name"] == "world16-fsdp"
        assert plan["training_backends"]["generator"]["options"]["optimizer_impl"] == "fused_adamw"


@pytest.mark.parametrize("arm",r3.ARMS)
@pytest.mark.parametrize("key",("shape","batch","seed","source","mechanism","reshard"))
def test_r3_tamper_rejected(env,arm,key):
    plan=r2.make_plan(ROOT,arm=arm,source_head=HEAD)
    if key=="shape":plan["workload"]["shape"]["latent"]["frames"]-=1
    elif key=="batch":plan["workload"]["shape"]["batch_size"]=8
    elif key=="seed":plan["metadata"][r2.KEY]["seed"]+=1
    elif key=="source":plan["metadata"][r2.KEY]["source_head"]="b"*40
    elif key=="mechanism":plan["metadata"][r3.KEY]["B"]=not plan["metadata"][r3.KEY]["B"]
    else:plan["training_backends"]["generator"]["options"]["reshard_after_forward"]=False
    with pytest.raises(ValueError):r2.validate(plan,mode=r2.mode_for_arm(arm),repo_root=ROOT)


def test_legacy_name_stays_blocked(env):
    with pytest.raises(ValueError,match="BLOCKED"):
        r2.make_plan(ROOT,arm="only_B",source_head=HEAD)


def test_no_training_rng_for_b_installer_or_roles():
    import ast
    tree=ast.parse((ROOT/"src/distillschedule/runtime/livewan_factorial_r3_b.py").read_text())
    names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert not names.intersection({"manual_seed","rand","randn","random_","normal_"})


def test_r3_collector_rejects_requested_flags_without_runtime():
    assert r3.mechanism_errors({r3.KEY:{"complete_B":True}},arm=r3.B_ARM,rank=0)
    assert r3.mechanism_errors({r3.KEY:{"complete_B":False}},arm=r3.A_ARM,rank=0)


def test_actual_pipeline_commit_uses_fsdp_mean_without_extra_ar(monkeypatch):
    from types import SimpleNamespace as NS
    calls=[]
    monkeypatch.setattr(worker,"_mean_replicated_gradients",lambda *a,**k:calls.append(k) or {"policy":"AR"})
    param=NS(to_local=lambda:None)
    student=NS(_livewan_r3_fsdp8=True,parameters=lambda:[param])
    bundle=NS(student=student,generator_group_size=8,generator_group="SYNTHETIC_TEST_ONLY")
    assert worker.pipeline_generator_gradient_sync(None,bundle)==dict(policy="fsdp2_reduce_scatter_sum_divide_8")
    assert not calls
    student._livewan_r3_fsdp8=False
    assert worker.pipeline_generator_gradient_sync(None,bundle)["policy"]=="AR"
    assert calls[0]["group_size"]==8


def test_fsdp8_sample_gradient_mean_matches_original_global16():
    import torch
    # 8 G ranks, two real samples each, 3 selected blocks each. Native FSDP
    # SUM/8 acts on per-sample half-weight losses; no second manual AR.
    contributions=torch.arange(48,dtype=torch.float64).reshape(8,2,3)
    resident16=contributions.sum()/48
    fsdp8=(contributions*(.5/3)).sum()/8
    assert torch.equal(resident16,fsdp8)


def test_existing_frozen_plans_unchanged_by_r3(env):
    # Load the actual parent source implementation in an isolated module and
    # compare complete serialized plans, not a hand-selected flag list.
    import subprocess,types
    raw=subprocess.check_output(["git","-C",str(ROOT),"show",
        "42f4f855ac52002470cf0b39eb17187a7bc44caf:src/distillschedule/runtime/livewan_system_factorial_r2.py"],text=True)
    parent=types.ModuleType("_r3_parent_r2")
    exec(compile(raw,"PARENT_42f4_R2","exec"),parent.__dict__)
    for arm in parent.ARMS:
        assert r2.make_plan(ROOT,arm=arm,source_head=HEAD)==parent.make_plan(ROOT,arm=arm,source_head=HEAD)


def test_b_receipt_uses_real_lora_counter_shape():
    from types import SimpleNamespace as NS
    from distillschedule.runtime.livewan_factorial_r3_b import CompleteB
    # Only receipt plumbing is exercised; no fake claim of kernel execution.
    helper=CompleteB.__new__(CompleteB)
    helper.kernel_receipts={"critic_lora_kernel":dict(autocast_input_elision_invocation_count=11,
        autocast_contract_failure_count=0)}
    helper.dispatch=NS(receipt=lambda:{})
    helper.kv=helper.checkpoint=None
    helper.retention=NS(receipt=lambda:{})
    helper.generator_version=helper.commits=0
    assert helper.receipt()["kernels"]["critic_lora_kernel"]["execution_status"]=="PASS"
    helper.kernel_receipts["critic_lora_kernel"]["autocast_input_elision_invocation_count"]=0
    assert helper.receipt()["kernels"]["critic_lora_kernel"]["execution_status"]=="NOT_RUN"


@pytest.mark.parametrize("arm",r3.ARMS)
def test_r3_producer_node_writer_retains_exact_plan_and_localrank_owner(env,arm):
    import json
    plan=r2.make_plan(ROOT,arm=arm,source_head=HEAD)
    output=Path(os.environ["DISTILLSCHEDULE_OUTPUT_DIR"])
    for node in (0,1):
        for local_rank in range(8):
            wrote=worker._write_effective_plan_for_run(plan=plan,output_dir=output,
                run_kind=r2.RUN_KIND,local_rank=local_rank,node_rank=node)
            assert wrote==(local_rank==0)
    nodes=[json.loads((output/f"effective_plan.node{i}.json").read_text()) for i in (0,1)]
    assert nodes[0]==nodes[1]
    assert {k:v for k,v in nodes[0].items() if k!="effective_plan_sha256"}==plan
    assert nodes[0]["effective_plan_sha256"]==r2.digest(plan)
