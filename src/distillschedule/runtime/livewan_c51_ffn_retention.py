"""Default-off exact C51 + prior audited C44 frozen-down reuse composition."""
from __future__ import annotations
import copy
from distillschedule.runtime import livewan_replay_ablation as replay
from distillschedule.runtime.livewan_ffn_down_checkpoint import CONTRACT as DOWN, enabled as down_enabled
VARIANT = 'c52-c51-critic-ffn-down-retention-v1'
PLAN_ID = 'livewan_wan13b_14b_sf_dmd_c52_c51_critic_ffn_down_retention_v1'
PLAN_FILE = PLAN_ID + '.json'
CONTRACT = {'schema': 'livewan_c51_ffn_retention_v1', 'parent_variant': 'c51-continuous-b16-generator-fused-rms', 'parent_source_head': 'aa30527db5d048e2ab23222eef587cb7cfe909f6', 'parent_plan_sources_sha256': '919db26012f4dee2429e434ffc1c6508957b149cdeb2427e1e9c674035e4152d', 'factor': 'critic_frozen_ffn_down_checkpoint_output_reuse', 'reuse_enabled': True, 'application_calls_unchanged': True, 'optimizer_placement_and_schedule_unchanged': True, 'scope': 'conditional_on_full_c51_not_layout_only'}

def validate_effective_policy(plan):
 metadata=plan.get('metadata',{})
 if plan.get('id')!=PLAN_ID or metadata.get('c51_ffn_retention')!=CONTRACT or metadata.get('critic_ffn_down_checkpoint')!=DOWN:
  raise RuntimeError('C52 full C51 frozen-down retention identity/policy changed')

def parent_plan(plan,*,mode):
 validate_effective_policy(plan)
 if mode!='pipeline':raise RuntimeError('C52 requires unchanged pipeline role layout')
 # Reuse the source-pinned C51-parent authenticator by forming an in-memory
 # validation-only view of the already admitted old conditional child.
 view=copy.deepcopy(plan);view['id']=replay.PLAN_ID
 view['metadata'].pop('c51_ffn_retention');view['metadata'].pop('critic_ffn_down_checkpoint')
 view['metadata']['checkpoint_replay_ablation']=copy.deepcopy(replay.CONTRACT)
 view['metadata']['critic_attention_checkpoint']=copy.deepcopy(replay.RECOMPUTE_CONTRACT)
 return replay.parent_plan(view,mode=mode)

def enabled(plan,*,mode):
 if 'c51_ffn_retention' not in plan.get('metadata',{}):
  if plan.get('id')==PLAN_ID:raise RuntimeError('C52 retention contract missing')
  return False
 parent_plan(plan,mode=mode)
 if not down_enabled(plan,mode=mode):raise RuntimeError('C52 original frozen-down runtime not installed')
 return True
