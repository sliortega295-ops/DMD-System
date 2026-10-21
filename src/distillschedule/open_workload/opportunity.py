"""Explain measured action costs without proposing an unmeasured speedup.

This is a serial full-world resource bound (as admitted by the current open
executor), not a critical-path analysis for unscheduled cross-pool pipelines.
Per-call projected improvement is a WHAT-IF upper bound, not a new optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from distillschedule.core.contracts import require

SCHEMA='distillschedule.call_opportunity_report.v1'


def report(selected, variant, graph, rules, costs, manifest):
    total=selected['predicted_makespan_ns']
    require(type(total) is int and total>0, 'non-positive cost prediction',path='open.opportunity')
    bindings={op.operation_id:op.metadata.get('binding','system_barrier') for op in graph.operations}
    rows=[]
    for key in selected['order']:
        record=costs.get(rules[key].cost_key)
        duration=record.duration_ns
        if key=='__open_finish__' or duration==0: continue
        receipt=json.loads(Path(record.evidence['source_receipt']).read_text())
        observations=receipt.get('rank_observations',[])
        skew=[]
        for sample in observations:
            ns=[x['duration_ns'] for x in sample]
            if ns and statistics.median(ns)>0:
                skew.append(max(ns)/statistics.median(ns))
        portion=duration/total
        # Upper bound if the component became free and nothing else changed.
        # For a component using all world resources, serial makespan is additive.
        upper=None if duration>=total else round(total/(total-duration),4)
        rows.append({'call':key,'binding':bindings.get(key,'unknown'),
                     'observed_median_ns':duration,'fraction_of_serial_prediction':round(portion,6),
                     'optimistic_total_speedup_if_call_zero':upper,
                     'hypothetical_total_speedup_if_call_2x':round(total/(total-duration/2),4),
                     'observed_max_to_median_rank_time':round(statistics.median(skew),3) if skew else None,
                     'rank_count':manifest.get('replicas',1)})
    rows.sort(key=lambda x:(-x['observed_median_ns'],x['call']))
    binding_totals={}
    for row in rows:binding_totals[row['binding']]=binding_totals.get(row['binding'],0)+row['observed_median_ns']
    return {'schema_version':SCHEMA,'scope':'source_bound_serial_all_rank_native_call_costs',
            'gpu_family':(manifest['hardware'].get('accelerator',{}).get('gpu_family','UNKNOWN')
                          if 'replicas' not in manifest['hardware'] else
                          sorted({r['accelerator']['gpu_family'] for r in manifest['hardware']['replicas']})),
            'predicted_makespan_ns':total,'measured_variant':variant['id'],
            'ranked_calls':rows[:16],
            'binding_cost_share':{key:round(value/total,6) for key,value in sorted(binding_totals.items())},
            'note':'Amdahl what-if assumes serial execution; no reduction is promised or observed',
            'mfu':'NOT_CLAIMED_WITHOUT_COMPLETE_FLOPS_AND_PEAK_PRECISION_MODEL',
            'cross_role_overlap':'NOT_MODELED_BY_THIS_SERIAL_BACKEND',
            'gpu_execution':'NOT_RUN_BY_REPORT','claimable_speedup':False}
