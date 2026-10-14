"""Amdahl what-if bound uses only trusted component receipts."""
import json
from pathlib import Path
import pytest
from distillschedule.open_workload.planning import calibrate,plan,run
from distillschedule.workloads.best_known.registry import write_new_json
from test_open_workload_execution import build


def test_new_task_produces_ranked_measured_costs_not_fake_speedups(tmp_path):
    w=build(n=2,width=5,length=8)
    m=calibrate(w,tmp_path/'cal',samples=1,max_candidates=2,max_program_runs=3)
    p=plan(tmp_path/'cal/calibration.json',memory_limit_bytes=10**7)
    report=p['opportunity_report']
    assert report['predicted_makespan_ns']==p['selected']['predicted_makespan_ns']
    assert report['gpu_family']=='CPU_NOT_GPU_EVIDENCE'
    assert report['mfu'].startswith('NOT_CLAIMED') and not report['claimable_speedup']
    assert report['ranked_calls'] and len(report['ranked_calls'])>2
    assert all(row['observed_median_ns']>=0 for row in report['ranked_calls'])
    assert all(row['hypothetical_total_speedup_if_call_2x']>=1. for row in report['ranked_calls'])
    assert all(row['observed_max_to_median_rank_time']==1. for row in report['ranked_calls'])
    assert list(report['ranked_calls'])==sorted(report['ranked_calls'],
        key=lambda x:(-x['observed_median_ns'],x['call']))
    write_new_json(tmp_path/'plan.json',p)
    assert run(w,tmp_path/'plan.json',tmp_path/'run')[0]['status']=='PASS_SELECTED_PROGRAM'


def test_cost_analysis_rejects_missing_evidence(tmp_path):
    w=build(n=1)
    m=calibrate(w,tmp_path/'cal',samples=1,max_candidates=1,max_program_runs=2)
    # The planning code already verifies SHA of cost sources before analysis.
    record=next((tmp_path/'cal').glob('candidate-*/op-*.json'))
    doc=json.loads(record.read_text());doc['rank_observations'][0][0]['duration_ns']=0
    record.write_text(json.dumps(doc))
    with pytest.raises(ValueError):plan(tmp_path/'cal/calibration.json',memory_limit_bytes=10**7)
