"""Versioned C51 conditional ablation: recompute, rather than reuse, FA3 outputs.

This is NOT a layout-only candidate. All C51 layout, optimizer, kernel and
other state-reuse mechanisms remain enabled. Canonical C19/C30/C51 contracts
are not loosened: only this new plan may replace its parent replay policy.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_attention_checkpoint import (
    AttentionCheckpointContexts, CriticAttentionCheckpointProxy,
)
from distillschedule.runtime.livewan_score_fa3 import (
    INTERFACE_SHA256, OP_NAME, SAC_CONTRACT, CriticFA3CheckpointProxy,
    attention_metadata, discover_flash_forward,
)

VARIANT = 'c51-critic-fa3-recompute-v1'
PARENT_VARIANT = 'c51-continuous-b16-generator-fused-rms'
PARENT_HEAD = 'aa30527db5d048e2ab23222eef587cb7cfe909f6'
PARENT_PLAN_SOURCES_SHA256 = '919db26012f4dee2429e434ffc1c6508957b149cdeb2427e1e9c674035e4152d'
PARENT_FILE = 'livewan_wan13b_14b_sf_dmd_c51_generator_fused_rms.json'
PLAN_FILE = 'livewan_wan13b_14b_sf_dmd_c51_critic_fa3_recompute_v1.json'
PLAN_ID = 'livewan_wan13b_14b_sf_dmd_c51_critic_fa3_recompute_v1'
CONTRACT = {
    'schema': 'livewan_c51_checkpoint_replay_ablation_v1',
    'parent_variant': PARENT_VARIANT,
    'parent_source_head': PARENT_HEAD,
    'parent_plan_sources_sha256': PARENT_PLAN_SOURCES_SHA256,
    'factor': 'critic_grad_self_attention_output_reuse',
    'reuse_enabled': False,
    'replacement': 'same_pinned_fa3_forward_recompute',
    'preserve_rng_state': True,
    'application_calls_unchanged': True,
    'optimizer_placement_and_schedule_unchanged': True,
    'scope': 'conditional_on_full_c51_not_layout_only',
}
RECOMPUTE_CONTRACT = {
    **SAC_CONTRACT,
    'name': 'critic_fa3_checkpoint_recompute_ablation_v1',
    'cache_location': 'none_recompute',
}


def validate_effective_policy(plan: Mapping[str, Any]) -> None:
    """Check the serialized effective policy without resolving asset env vars."""
    if plan.get('id') == 'livewan_wan13b_14b_sf_dmd_c52_c51_critic_ffn_down_retention_v1' or 'c51_ffn_retention' in plan.get('metadata', {}):
        from distillschedule.runtime.livewan_c51_ffn_retention import validate_effective_policy as validate_retention
        return validate_retention(plan)
    if plan.get('id') == 'livewan_wan13b_14b_sf_dmd_c53_c51_critic_tail4_retention_v1' or 'critic_block_retention' in plan.get('metadata', {}):
        from distillschedule.runtime.livewan_block_retention import validate_effective_policy as validate_retention
        return validate_retention(plan)
    metadata = plan.get('metadata', {})
    if (plan.get('id') != PLAN_ID
        or metadata.get('checkpoint_replay_ablation') != CONTRACT
        or metadata.get('critic_attention_checkpoint') != RECOMPUTE_CONTRACT):
        raise RuntimeError('C51 replay ablation identity/policy changed')


def parent_plan(plan: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Authenticate the whole candidate against the exact resolved C51 parent.

    Only the two existing in-memory lowering transforms are recognized. All
    assets, hardware, workload, precision, kernels, caches and optimizer fields
    must equal that parent; restoring a validation view does not execute it.
    The actual effective plan retains the explicit no-cache policy and its ID.
    """
    validate_effective_policy(plan)
    if mode != 'pipeline':
        raise RuntimeError('C51 replay ablation is pipeline-only')
    metadata = plan['metadata']
    parent_path = Path(__file__).resolve().parents[3] / 'configs/distillschedule/plans' / PARENT_FILE
    loaded = load_plan(parent_path)
    root = Path(__file__).resolve().parents[3]
    sources = sorted((str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in loaded.source_files)
    source_sha = hashlib.sha256(json.dumps(sources, separators=(',', ':')).encode()).hexdigest()
    if source_sha != PARENT_PLAN_SOURCES_SHA256:
        raise RuntimeError('C51 frozen parent plan/include source changed')
    expected = loaded.plan.to_dict()
    from distillschedule.runtime import livewan_crossscale_worker as worker
    continuous = metadata.get('continuous_pipeline', {})
    if continuous.get('name') == worker.TRAJECTORY_CONTINUOUS_NAME:
        if continuous.get('cycles') != 2:
            raise RuntimeError('C51 replay ablation requires exactly two cycles')
        worker._apply_trajectory_contract(expected, cycles=2, mode=mode)
    elif continuous.get('name') == worker.TWO_CYCLE_TIMING_CONTINUOUS_NAME:
        worker._apply_two_cycle_timing_contract(expected, mode=mode)
    actual = copy.deepcopy(dict(plan))
    actual['id'] = expected['id']
    actual['metadata'].pop('checkpoint_replay_ablation')
    actual['metadata']['critic_attention_checkpoint'] = copy.deepcopy(SAC_CONTRACT)
    if actual != expected:
        raise RuntimeError('C51 replay ablation changed an unregistered parent field')
    return expected


def enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    if 'checkpoint_replay_ablation' not in plan.get('metadata', {}):
        if plan.get('id') == PLAN_ID:
            raise RuntimeError('C51 replay ablation contract missing')
        return False
    parent_plan(plan, mode=mode)
    return True


def validation_checkpoint_contract(plan: Mapping[str, Any], *, mode: str):
    """Keep original validators strict; authorize only the new exact parent."""
    if enabled(plan, mode=mode):
        return SAC_CONTRACT
    return plan.get('metadata', {}).get('critic_attention_checkpoint')


class RecomputeAttentionContexts(AttentionCheckpointContexts):
    def _attention_policy(self, checkpoint_policy: Any):
        return checkpoint_policy.PREFER_RECOMPUTE

    def receipt(self):
        result = super().receipt()
        result['replay_fa_recompute_calls'] = result['replay_fa_cache_reads']
        result['replay_fa_cache_reads'] = 0
        return result


class CriticFA3RecomputeProxy(CriticAttentionCheckpointProxy):
    def __init__(self, checkpoint_module: Any, torch: Any, flash_module: Any):
        self.module = checkpoint_module
        op, identity = discover_flash_forward(flash_module)
        self.contexts = RecomputeAttentionContexts(
            torch, op, identity, self_attention_only=True,
            metadata_reader=attention_metadata, receipt_name=RECOMPUTE_CONTRACT['name'])
        self.receipt_errors = runtime_receipt_errors


def build_proxy(plan, checkpoint_module, torch, flash_module):
    if enabled(plan, mode='pipeline'):
        return CriticFA3RecomputeProxy(checkpoint_module, torch, flash_module)
    return CriticFA3CheckpointProxy(checkpoint_module, torch, flash_module)


def runtime_receipt_errors(r: dict[str, Any], *, rank: int, expected_calls: int) -> list[str]:
    if rank < 8:
        return [] if (r.get('execution_status') == 'NOT_APPLICABLE'
                      and r.get('checkpoint_calls') == 0) else ['Generator replay policy changed']
    required = dict(name=RECOMPUTE_CONTRACT['name'], execution_status='PASS',
                    production_installed=True, self_attention_only=True,
                    contract_failures=0, checkpoint_calls=expected_calls,
                    original_fa_calls=expected_calls, replay_fa_cache_reads=0,
                    replay_fa_recompute_calls=expected_calls,
                    expected_checkpoint_calls=expected_calls,
                    allow_cache_entry_mutation=False, cpu_offload=False,
                    rng_preservation='UNCHANGED_CHECKPOINT_DEFAULT_TRUE')
    if (expected_calls <= 0 or r.get('source', {}).get('interface_sha256') != INTERFACE_SHA256
        or r.get('source', {}).get('operator') != OP_NAME.replace('::', '.') + '.default'
        or any(r.get(k) != v for k, v in required.items())):
        return ['C51 conditional FA3 recompute source/replay/census contract failed']
    return []
