"""Opt-in, read-only manual-incumbent match audit over existing native evidence.

This does not launch runs, normalize legacy provenance into current receipts, or
replace workload-owned gates. Missing manual/capacity/source observations remain
NOT_ESTABLISHED. The result is an addendum, not a new execution receipt schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import finite_number, require, semantic_contract
from .execution import validated_profile
from .helios_action_costs import observed_work
from .helios_implementation import HeliosImplementation
from .helios_native import bound, checked, launch_evidence, normalize_native
from .plan_ir import validate_binding
from .runtime_identity import verify_measurement_runtime
from distillschedule.workloads.best_known.registry import fingerprint, repository_root, write_new_json

INPUTS = ('preregistration', 'manual_execution', 'selected_execution',
          'manual_measurement', 'selected_measurement', 'selected_plan')
EMPTY_DIFF = hashlib.sha256(b'').hexdigest()


def _trace(rows):
    result = {}
    for row in rows:
        for cycle in row['cycles']:
            for action in cycle['actions']:
                for entry in action['trace_entries']:
                    key = (row['rank'], cycle['cycle'], entry['source_step'], entry['rollout_operation'])
                    require(key not in result or result[key] == entry, 'conflicting repeated trace identity')
                    result[key] = entry
    return result


def _capacity(policy):
    require(policy.get('gpu_metric') == 'peak_allocated_bytes', 'capacity metric missing/unsupported')
    for field, count in (('memory_limits', 16), ('host_limits', 2)):
        values = policy.get(field, {})
        require(set(values) == {str(i) for i in range(count)}, f'common capacity {field} missing')
        require(all(type(v) is int and v > 0 for v in values.values()), 'invalid common capacity limit')
    for field in ('gpu_reserve_bytes', 'host_reserve_bytes'):
        require(type(policy.get(field)) is int and policy[field] >= 0, f'common {field} missing')


def _unmerged_samples(witness):
    samples = witness.get('model_metadata_samples', [])
    require(len(samples) == 4 and {(s['point'], s['role']) for s in samples} ==
            {(p, r) for p in ('AFTER_SETUP_BEFORE_BOTH_TIMERS', 'AFTER_BOTH_TIMERS')
             for r in ('generator', 'critic')}, 'N70 model-state coverage missing')
    require(all(s.get('N70_model_state_present') is False and s.get('PEFT_merged_layers') == 0
                for s in samples), 'N70 model state/merged layers observed')


def _nomerge(rows, *, candidate_id=None):
    # New source-bound cohorts observe guarded PEFT calls and actual models;
    # do not manufacture the deleted N70 implementation's historical counters.
    for row in rows:
        if 'gan_manual_runtime' in row.get('build', {}):
            require(candidate_id == 'joint-helios1-gan-fsdp-c2048-p1-w5-kgan-fa3-down40-nomerge-v1'
                    and row.get('candidate_id') == candidate_id, 'new no-merge candidate identity differs')
            impl = HeliosImplementation.parse(candidate_id)
            require(row.get('planner_binding', {}).get('candidate_bundle') ==
                    impl.candidate_bundle.cost_identity(), 'new no-merge bundle/source cost identity differs')
            witness = row['build']['gan_manual_runtime']
            require(witness.get('installed') is True and witness.get('restored') is True
                    and not witness.get('failed', False)
                    and witness.get('effective_attention_backend') == 'pinned-sm80-fa3',
                    'new no-merge runtime/backend/restoration missing or failed')
            for key, expected in (('unsupported_request_count', 0), ('merge_attempt_count', 0),
                                  ('guarded_merge_methods', 812), ('lora_layer_count', 812),
                                  ('merged_layer_count', 0)):
                require(type(witness.get(key)) is int and witness[key] == expected,
                        'new no-merge actual observation missing/failed: ' + key)
            _unmerged_samples(witness)
            require(all(type(s.get('LoRA_count')) is int and s['LoRA_count'] == 406
                        for s in witness['model_metadata_samples']),
                    'new no-merge per-role LoRA inventory missing/different')
            if 'no_merge_audit' not in row:
                continue
        # Preserve strict historical validation when that witness is present.
        witness = row.get('no_merge_audit', {})
        for key in ('merge_count', 'PEFT_merge_count', 'N70_window_construction_count',
                    'enabled_install_count', 'enabled_scope_count'):
            require(type(witness.get(key)) is int and witness[key] == 0, f'N70 no-merge witness missing/failed: {key}')
        require(witness.get('guard_errors') == [], 'N70 guard did not complete')
        for key in ('execution_shadow', 'once_refresh', 'rollout_merged_shadow'):
            require(witness.get(key) is False, f'N70 shadow not proved disabled: {key}')
        _unmerged_samples(witness)


def validate_manual_incumbent_match(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Consume six path/SHA256 references; never accept historical scalar times.

    Preregistration must contain identity (the existing six-field planning
    binding), manual_candidate, registered_unix_ns, max_planner_overhead=0.03,
    and capacity_policy. Both existing node-resource receipts must independently
    record that same policy and source_identity={head,dirty,diff_sha256}.
    Legacy receipts lack those observations: NOT_ESTABLISHED, not inferred PASS.
    """
    report = dict(status='NOT_ESTABLISHED', formal_timing_eligible=False,
        statistical_equivalence='NOT_ESTABLISHED', global_optimum='NOT_CLAIMED',
        numerical_status='NOT_RUN', claim_boundary='DESCRIPTIVE_SINGLE_PAIR_ONLY')
    missing = [key for key in INPUTS if not evidence.get(key)]
    if missing:
        return report | {'missing_evidence': missing}
    try:
        data = {key: bound(evidence[key]) for key in INPUTS}
        protocol = data['preregistration']; identity = protocol['identity']
        validate_binding(identity)
        require(protocol.get('max_planner_overhead') == 0.03, 'preregistered overhead limit must be 0.03')
        registered = protocol.get('registered_unix_ns')
        require(type(registered) is int and registered > 0, 'preregistration time missing')
        policy = protocol['capacity_policy']; _capacity(policy)
        manual, selected = data['manual_execution'], data['selected_execution']
        require(manual['search_candidate'] == protocol['manual_candidate'], 'manual incumbent identity changed')
        manual_impl = HeliosImplementation.parse(manual['search_candidate']['id'])
        require(manual_impl.candidate_bundle.enabled, 'manual arm is not a named complete incumbent bundle')
        require(manual['run_id'] != selected['run_id'], 'independent manual/selected native runs required')
        for key in ('profile_id', 'integration_head', 'asset_fingerprint', 'workload_semantics_hash'):
            require(manual[key] == selected[key], f'paired {key} differs')
        require(semantic_contract(manual['fully_resolved_profile']) ==
                semantic_contract(selected['fully_resolved_profile']), 'paired frozen workload differs')
        saved = data['selected_plan']
        require(bool(saved.get('model_plan')), 'selected plan lacks saved search problem/cost/prediction')
        validated_profile(saved, root=repository_root())
        for key in ('profile_id', 'integration_head', 'asset_fingerprint', 'workload_semantics_hash',
                    'search_candidate', 'profile_sha256', 'model_plan'):
            require(selected.get(key) == saved.get(key), f'selected native execution lost saved {key}')
        require(saved['model_plan']['problem']['binding'] == identity, 'selected plan runtime binding differs')
        require(saved['model_plan']['problem']['memory_limits'] == policy['memory_limits'],
                'planner memory admission differs from common capacity policy')
        rows_by_arm = {}; times = {}; numeric = {}
        for arm, receipt in (('manual', manual), ('selected', selected)):
            out, summary, rows, _ = checked(receipt)
            require(receipt['mode'] == 'diagnostic-timing' and summary['mode'] == 'native', 'not native timing')
            require(all(registered < r['cycles'][0]['start_unix_ns'] for r in rows), 'target registered after timing began')
            preparation = dict(assets=receipt['asset_config'],
                comparison=bound(receipt['asset_config']['correctness'])['comparison'])
            normalized = normalize_native(receipt, preparation, identity)
            measured = data[arm + '_measurement']
            require(all(measured.get(k) == v for k, v in normalized.items()), 'measurement differs from actual native receipts')
            verify_measurement_runtime(measured, identity)
            resources, _ = launch_evidence(receipt, out)
            for resource in resources:
                require(resource.get('capacity_policy') == policy, 'common capacity policy observation missing/different')
                require(resource.get('source_identity') == dict(head=identity['source_head'], dirty=False,
                    diff_sha256=EMPTY_DIFF), 'same clean source/diff observation missing/different')
            for rank in measured['ranks']:
                require(rank['peak_allocated_bytes'] <= policy['memory_limits'][str(rank['rank'])], 'GPU capacity exceeded')
                require(rank['min_driver_free_bytes'] >= policy['gpu_reserve_bytes'], 'GPU reserve not met')
                require(rank['host_peak_bytes'] + policy['host_reserve_bytes'] <=
                        policy['host_limits'][str(rank['rank']//8)], 'host capacity exceeded')
            if manual_impl.variant == 'gan':
                _nomerge(rows, candidate_id=receipt['search_candidate']['id'])
            rows_by_arm[arm] = rows
            times[arm] = finite_number(summary['cycle_wall_seconds']['2'], 'cycle2', positive=True)
            numeric[arm] = measured['numerical_status']
        require(_trace(rows_by_arm['manual']) == _trace(rows_by_arm['selected']), 'paired rank/sample/RNG differs')
        require(observed_work(rows_by_arm['manual']) == observed_work(rows_by_arm['selected']),
                'paired actual shape/frames/batch/call inventory differs')
        ratio = times['selected'] / times['manual']
        return report | dict(status='MATCHED_DIAGNOSTIC_TARGET_MET' if ratio <= 1.03 else
            'MATCHED_DIAGNOSTIC_TARGET_MISSED', cycle2_seconds=times, selected_over_manual=ratio,
            planner_overhead_fraction=ratio-1, max_planner_overhead=0.03, numerical_status=numeric,
            evidence=dict(evidence), source_head=identity['source_head'], independent_pairs=1)
    except (KeyError, ValueError, TypeError, OSError) as error:
        return report | {'reason': str(error), 'error_type': type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate_manual_incumbent_match(json.loads(args.evidence.read_text()))
    if args.output:
        write_new_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 2 if result['status'] == 'NOT_ESTABLISHED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
