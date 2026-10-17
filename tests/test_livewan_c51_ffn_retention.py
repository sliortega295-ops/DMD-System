"""CPU/source qualification only; not world16 correctness or performance."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from distillschedule.core.config import load_plan
from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime import livewan_c51_ffn_retention as feature
from distillschedule.runtime import livewan_replay_ablation as replay
from distillschedule.runtime import livewan_ffn_down_checkpoint as down

@pytest.fixture
def plans(monkeypatch):
 for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT','LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
  monkeypatch.setenv(key,'/unread/'+key)
 root=Path(__file__).resolve().parents[1]/'configs/distillschedule/plans'
 return load_plan(root/replay.PARENT_FILE).plan.to_dict(),load_plan(root/feature.PLAN_FILE).plan.to_dict()

def test_complete_parent_and_real_installer_lowering(plans,monkeypatch):
 from distillschedule.runtime import livewan_score_fa3 as fa3
 parent,child=plans
 assert not feature.enabled(parent,mode='pipeline') and feature.enabled(child,mode='pipeline')
 assert feature.parent_plan(child,mode='pipeline')==parent
 assert child['workload']==parent['workload'] and child['rank_groups']==parent['rank_groups']
 for lower in (lambda p:worker._apply_trajectory_contract(p,cycles=2,mode='pipeline'),lambda p:worker._apply_two_cycle_timing_contract(p,mode='pipeline')):
  a,b=copy.deepcopy(parent),copy.deepcopy(child);lower(a);lower(b)
  worker._validate_plan_identity(a,mode='pipeline');worker._validate_plan_identity(b,mode='pipeline')
  assert feature.parent_plan(b,mode='pipeline')==a
  assert worker._effective_plan_sha256(a)!=worker._effective_plan_sha256(b)
 calls=[];sentinel=object()
 monkeypatch.setattr(fa3,'load_flash_attn3',lambda:(sentinel,None))
 monkeypatch.setattr(down,'CriticFFNDownCheckpointProxy',lambda cp,t,flash:calls.append((cp,t,flash)) or sentinel)
 assert worker._build_critic_fa3_checkpoint_proxy(child,'checkpoint','torch') is sentinel
 assert calls==[('checkpoint','torch',sentinel)]

@pytest.mark.parametrize('change',[
 lambda p:p['workload']['shape'].__setitem__('batch_size',8),
 lambda p:p['metadata'].__setitem__('generator_rollout_cast_cache',None),
 lambda p:p['metadata'].__setitem__('critic_attention_checkpoint',replay.RECOMPUTE_CONTRACT),
 lambda p:p['metadata']['critic_ffn_down_checkpoint'].__setitem__('cpu_offload',True),
 lambda p:p['metadata'].pop('c51_ffn_retention'),
 lambda p:p.__setitem__('id',replay.PLAN_ID),
])
def test_rejects_all_other_parent_or_policy_changes(plans,change):
 _,child=plans;change(child)
 with pytest.raises(RuntimeError):worker._validate_plan_identity(child,mode='pipeline')

def test_old_conditional_and_parent_stay_separate(plans):
 parent,child=plans
 with pytest.raises(RuntimeError):feature.enabled(child,mode='baseline')
 assert not replay.enabled(child,mode='pipeline')
 assert not down.enabled(parent,mode='pipeline')
 bad=copy.deepcopy(parent);bad['metadata']['critic_ffn_down_checkpoint']=copy.deepcopy(down.CONTRACT)
 with pytest.raises(RuntimeError):worker._validate_plan_identity(bad,mode='pipeline')
 # Existing generic child-policy consumer authenticates the actual new plan.
 replay.validate_effective_policy(child)


def test_real_node_header_rejects_old_child_gate_and_accepts_exact_new_child(tmp_path):
 import json
 from test_refactor_livewan_replay_ablation import _header_fixture,_node_timing_header
 path=tmp_path/'SYNTHETIC_TEST_ONLY_gate.json';gate=_header_fixture(True)
 path.write_text(json.dumps(gate))
 assert _node_timing_header(path,variant=feature.VARIANT,mode='pipeline').returncode!=0
 gate['arms']['candidate']['pipeline_variant']=feature.VARIANT;path.write_text(json.dumps(gate))
 assert _node_timing_header(path,variant=feature.VARIANT,mode='pipeline').returncode==0
 assert _node_timing_header(path,variant=replay.VARIANT,mode='pipeline').returncode!=0
