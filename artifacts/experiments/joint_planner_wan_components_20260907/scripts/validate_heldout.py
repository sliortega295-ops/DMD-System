"""One held-out model-selected native order; reuse completed source-bound gates.

No anchor timing matrix, no repeat of window3, no calibration-as-speedup claim.
"""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess

from collect_components import audit
from distillschedule.search.contracts import require, validate_measurement
from distillschedule.search.execution import NativeTrialRunner
from distillschedule.search.registry import get_candidate
from distillschedule.search.wan_implementation import WAN, WanImplementation
from distillschedule.workloads.best_known.registry import (
    file_sha256, fingerprint, git_head, repository_root, write_new_json,
)

ROOT = Path(__file__).resolve().parents[1]
HEAD = "ceb2921985e5a855ff93d6c9d7e21e7528b6ef5a"
RUN = "jointp_wan_components_20260907_r1"
ANALYSIS = "/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/component_costs_r1/analysis_r1"


def lower_remote(assets, prediction):
    node = assets['nodes'][0]
    code = r'''
import json,os,sys
from pathlib import Path
p=json.load(sys.stdin);repo=Path(p['repo']);sys.path[:0]=[str(repo/'src'),str(repo)]
from distillschedule.search.plan_ir import PlanningProblem
from distillschedule.search.cost_model import CostModel
from distillschedule.search.wan_model_adapter import lower_prediction
from distillschedule.workloads.best_known.registry import write_new_json,file_sha256,git_head
assert git_head(repo)==p['head']
root=Path(p['analysis']);result=json.loads((root/'search.json').read_text())
assert result['predicted_best']==p['predicted_best']
problem=PlanningProblem.from_dict(json.loads((root/'assembly/problem.json').read_text()))
costs=CostModel.from_dict(json.loads((root/'assembly/cost-model.json').read_text()))
asset_path=root/'heldout-assets.json';write_new_json(asset_path,p['assets'])
plan=lower_prediction(result,problem,costs,run_id=p['run_id'],assets_path=asset_path)
path=root/'heldout-model-plan.json';write_new_json(path,plan)
print(json.dumps(dict(status='PASS_MODEL_TO_NATIVE_LOWERING',candidate=plan['search_candidate'],
 path=str(path),sha256=file_sha256(path),problem_sha256=problem.digest,cost_model_sha256=costs.digest,
 predicted_cycle2_ns=result['predicted_best']['predicted_cycle2_ns'],
 integration_head=plan['integration_head'],asset_fingerprint=plan['asset_fingerprint'],
 workload_semantics_hash=plan['workload_semantics_hash'])))
'''
    cmd = ['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no',
           '-oControlPath=none','-p',str(node['ssh_port']),node['host'],
           shlex.join(['env','PYTHONDONTWRITEBYTECODE=1',str(Path(node['python_env'])/'bin/python'),'-c',code])]
    payload = dict(repo=node['repo_root'],head=HEAD,analysis=ANALYSIS,assets=assets,
                   run_id=RUN+'-heldout-model',predicted_best=prediction)
    write_new_json(ROOT/'manifests/heldout-lowering-command.json',dict(command=cmd,payload=payload))
    r=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=240)
    for name,data in [('stdout',r.stdout),('stderr',r.stderr)]:
        with (ROOT/f'logs/heldout-lowering.{name}.log').open('x') as f:f.write(data)
    require(r.returncode==0,r.stderr[-4000:])
    return json.loads(r.stdout)


def main():
    require(git_head(repository_root())==HEAD,'source changed')
    require(not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'dirty source')
    complete=json.loads((ROOT/'manifests/collection-complete.json').read_text())
    require(complete['status']=='PASS_NATIVE_ACTION_COLLECTION','incomplete collection')
    search=json.loads((ROOT/'reports/component-search.json').read_text())
    prediction=search['predicted_best']
    impl=WanImplementation.from_assignment(prediction['assignment'])
    require(impl.id=='joint-wan2-resident-c2048-e1-w2','unexpected heldout selection')
    require(impl.id not in {r['implementation_id'] for r in complete['results']},'not held-out')
    assets=json.loads((ROOT/'configs/assets.json').read_text())
    identity=json.loads((ROOT/'manifests/execution-preflight.json').read_text())['binding']
    controller=ROOT/'controller'
    directory=controller/'heldout-w2-native'
    with (controller/'exclusive-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observations=[audit(node) for node in assets['nodes']]
        require(all(r['gpu_processes']['returncode']==0 and not r['gpu_processes']['stdout'].strip()
                    and r['lock']['returncode']==0 and 31527 not in r['listening_ports']
                    for r in observations),'resource ownership/lock/port conflict')
        proof=lower_remote(assets,prediction)
        candidate=get_candidate(WAN,impl.id)
        require(proof['candidate']==candidate.to_dict(),'lowered native candidate differs')
        require(proof['integration_head']==HEAD and proof['asset_fingerprint']==identity['asset_fingerprint']
                and proof['workload_semantics_hash']==identity['semantic_fingerprint'],'lowering identity differs')
        directory.mkdir(exist_ok=False)
        write_new_json(directory/'model-lowering-proof.json',proof)
        write_new_json(ROOT/'manifests/heldout-start.json',dict(status='GO_HELDOUT_NATIVE',source_head=HEAD,
            run_id=RUN+'-heldout-w2-native',candidate_id=impl.id,lowering=proof,resource_audit=observations,
            max_new_correctness_runs=1,max_new_unprofiled_two_cycle_runs=1,baseline_repeats=0,
            incumbent_timing_repeats=0,completed_window3_repeats=0,
            claim='PREDICTION_ERROR_ONLY; NO_NEW_BASELINE_SPEEDUP; STRICT_NUMERIC_REVIEW_PRESERVED',
            lock='native launcher owns both node locks through teardown'))
        runner=NativeTrialRunner(workload=WAN,identity=identity,assets=assets,output=controller,
            run_prefix=RUN,reference_id='joint-wan2-fsdp-c512-e0-w5',
            bootstrap_peer='joint-wan2-resident-c2048-e1-w5')
        try:
            raw=runner(candidate,'heldout',directory)
            write_new_json(directory/'executor-receipt.json',raw)
            result=validate_measurement(raw,identity,candidate)
            plan=json.loads((directory/'execution-plan.json').read_text())
            require(plan['search_candidate']==proof['candidate'],'measured candidate differs from lowering')
            predicted=prediction['predicted_cycle2_ns']/1e9
            summary=dict(status='PASS_HELDOUT_NATIVE_FEEDBACK',source_head=HEAD,candidate_id=impl.id,
                native=result,predicted_cycle2_seconds=predicted,
                relative_prediction_error=predicted/result['cycle2_seconds']-1,
                model_memory_peak_bytes=max(r['peak_bytes'] for r in prediction['memory']),
                native_receipt=dict(path=str(directory/'executor-receipt.json'),sha256=file_sha256(directory/'executor-receipt.json')),
                lowering=proof,extra_baseline_runs=0,extra_incumbent_runs=0,
                historical_window3_repeats=0,new_speedup_claim=False,
                tie_boundary='All five Resident window orders predict equal action-sum latency; w2 is deterministic tie-break, not a proven optimum.',
                strict_equivalence=result['strict_eligible'])
            write_new_json(ROOT/'reports/heldout-feedback.json',summary)
            print(json.dumps(summary,indent=2),flush=True)
        except BaseException as error:
            write_new_json(ROOT/'manifests/heldout-failure.json',dict(status='FAIL',source_head=HEAD,
                type=type(error).__name__,message=str(error),next_action='PRESERVE_NO_AUTOMATIC_RETRY'))
            raise


if __name__=='__main__':main()
