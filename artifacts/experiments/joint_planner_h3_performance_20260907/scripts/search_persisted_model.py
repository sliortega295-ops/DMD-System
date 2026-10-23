"""One completed calibration -> shared planner -> one native executable plan."""
import argparse, copy, csv, json, sys
from pathlib import Path
from distillschedule.search.h3_action_model_adapter import assemble_calibration, SCHEMA
from distillschedule.search.h3_implementation import H3Implementation
from distillschedule.search.model_search import search_plans
from distillschedule.search.native_macro_adapter import lower_prediction
from distillschedule.search.native import binding
from distillschedule.workloads.best_known.registry import write_new_json, fingerprint, repository_root
from distillschedule.workloads.best_known.dispatch import validate_local

p=argparse.ArgumentParser()
p.add_argument('--calibration-plan',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--analysis-summary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--confirmation-run-id',required=True)
a=p.parse_args()
assert 'torch' not in sys.modules
plan=json.loads(a.calibration_plan.read_text()); measured=json.loads(a.analysis_summary.read_text())
assert measured['status']=='PASS_ACTION_CALIBRATION'
assert measured['run_id']==plan['run_id'] and measured['source_head']==plan['integration_head']
run=Path(plan['asset_config']['run_root'])/plan['run_id']
a.output.mkdir(parents=True,exist_ok=False)
layout=dict(schema_version=SCHEMA,binding=measured['binding'],implementation_id=plan['search_candidate']['id'],
    rank_receipts=[binding(run/'execution/ranks'/f'rank_{i:03d}.json') for i in range(16)],
    execution_receipt=binding(a.calibration_plan),runtime_identity_evidence=binding(a.runtime))
lp=a.output/'layout.json';write_new_json(lp,layout)
# Read the existing immutable serialized component model. This is the same
# file-based path consumed by the public planner CLI, not retiming calibration.
from distillschedule.search.plan_ir import PlanningProblem
from distillschedule.search.cost_model import CostModel
prior=a.output.parent/'search-r1/calibration'
problem=PlanningProblem.from_dict(json.loads((prior/'problem.json').read_text()))
costs=CostModel.from_dict(json.loads((prior/'cost-model.json').read_text()))
costs.verify_receipts()
write_new_json(a.output/'reused-model-sources.json',dict(problem=binding(prior/'problem.json'),cost_model=binding(prior/'cost-model.json'),new_gpu_calibration=False))
result=search_plans(problem,costs,exhaustive=True,max_proposals=20,seed=20260907)
write_new_json(a.output/'search-result.json',result)
ranked=result['ranked']
def record(x):
    return dict(candidate_id=H3Implementation.from_assignment(x['assignment']).id,status=x['status'],
        predicted_cycle1_s=x.get('predicted_cycle1_ns',0)/1e9,predicted_cycle2_s=x.get('predicted_cycle2_ns',0)/1e9,
        predicted_peak_bytes=max((m['peak_bytes'] for m in x.get('memory',[])),default=0),
        min_predicted_headroom_bytes=min((m['predicted_headroom_bytes'] for m in x.get('memory',[])),default=0),
        rejection=x.get('detail',''))
rr=[record(x) for x in [*ranked,*result['rejections']]]
with (a.output/'search-candidates.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
if result['status']!='PASS_CPU_SEARCH':
    print(json.dumps(dict(status=result['status'],gpu_confirmation='NOT_RUN',output=str(a.output))))
    raise SystemExit(2)
selected=result['predicted_best'];impl=H3Implementation.from_assignment(selected['assignment'])
assets=copy.deepcopy(plan['asset_config'])
gate=json.loads(Path(assets['correctness']['path']).read_text())
gate['integration_binding']['profile_sha256']=fingerprint(impl.lower_profile())
gp=a.output/'prior-action-gate-selected.json';write_new_json(gp,gate)
assets['correctness']=binding(gp)
ap=a.output/'assets-selected.json';write_new_json(ap,assets)
native=lower_prediction(result,problem,costs,run_id=a.confirmation_run_id,assets_path=ap,mode='diagnostic-timing')
validation=validate_local(native,root=repository_root())
write_new_json(a.output/'execution-plan-confirmation.json',native)
write_new_json(a.output/'confirmation-validation.json',validation)
ties=[H3Implementation.from_assignment(x['assignment']).id for x in ranked
      if x['predicted_cycle2_ns']==selected['predicted_cycle2_ns']]
summary=dict(status='PASS_CPU_MODEL_SEARCH_NATIVE_CONFIRMATION_NOT_RUN',source_head=plan['integration_head'],
    calibration_run_id=plan['run_id'],component_count=len(costs.records),observed_operations_per_rank=22,
    search_space=result['space_size'],evaluated=result['unique_evaluated'],feasible=len(ranked),
    uncalibrated=sum(x['status']=='UNCALIBRATED' for x in result['rejections']),selected=impl.id,
    predicted_cycle1_seconds=selected['predicted_cycle1_ns']/1e9,
    predicted_cycle2_seconds=selected['predicted_cycle2_ns']/1e9,
    equal_latency_candidates=ties,tie_break='EXISTING_DETERMINISTIC_FINGERPRINT_TIEBREAK_NOT_MEASURED_SPEEDUP',
    predicted_best_unique_latency=len(ties)==1,
    cost_source='ACTUAL_FENCED_ACTIONS_MAX_RANK; PHYSICAL_RANK_LAYOUTS_AND_WARM_STATE_BOUND',
    missing_costs='UNMEASURED_DEPTH_TILE_REMAIN_UNCALIBRATED',
    overlap='SERIAL_WORLD16_GPU_RESOURCE_ENVELOPES; NO_SYNTHETIC_FREE_OVERLAP',
    validation='PASS_BUT_CPU_ONLY',strict_numerical_status='REVIEW',claimable_speedup=None,
    confirmation_run_id=a.confirmation_run_id)
write_new_json(a.output/'summary.json',summary)
print(json.dumps(summary,indent=2))
