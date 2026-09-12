"""Keep raw calibration remote; run the common planner/session via a CPU pipe.

GPU execution remains the existing source-bound NativeTrialRunner. The pipe
only transfers small candidate/receipt JSON, never tensors, credentials or costs
invented on the controller. Each phase is explicit, bounded and non-retrying.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

E = Path(__file__).resolve().parents[1]
ROOT = Path('/home/lyy/Helios/DMD-System/worktrees/joint-planner-20260907')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from distillschedule.search.execution import NativeTrialRunner
from distillschedule.search.helios_action_costs import anchors
from distillschedule.search.helios_implementation import HeliosImplementation, NO_GAN, GAN
from distillschedule.search.registry import get_candidate
from distillschedule.workloads.best_known.registry import write_new_json, file_sha256

REMOTE = r'''
import csv,json,subprocess,sys
from pathlib import Path
from distillschedule.search.helios_model_adapter import assemble_calibrations,build_problem
from distillschedule.search.helios_implementation import HeliosImplementation
from distillschedule.search.cost_model import CostModel
from distillschedule.search.model_validation import ModelValidationSession
from distillschedule.workloads.best_known.registry import git_head,repository_root,write_new_json,file_sha256
x=json.loads(sys.stdin.readline());root=repository_root()
assert git_head(root)==x['head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
out=Path(x['remote_output']);stage=x['stage']
assert stage=='validate', 'r2 only continues existing CPU search; never repeats calibration'
import hashlib,importlib.util
source=x['cpu_controller_source'].encode()
assert hashlib.sha256(source).hexdigest()==x['cpu_controller_sha256']
controller_path=out/'controller-r2/model_validation.py'
controller_path.parent.mkdir(parents=True,exist_ok=False)
with controller_path.open('xb') as f:f.write(source)
write_new_json(out/'controller-r2/identity.json',dict(
 native_worker_source_head=x['head'],component_cost_source_head=x['head'],
 cpu_controller_module=str(controller_path),cpu_controller_sha256=x['cpu_controller_sha256'],
 workload_code_modified=False,memory_margin_changed=False,
 reason='Keep predicted-capacity-rejected incumbent NOT_RUN; validate safe baseline/selection only'))
name='distillschedule.search.model_validation'
spec=importlib.util.spec_from_file_location(name,controller_path)
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
ModelValidationSession=module.ModelValidationSession
if stage=='plan':
 out.mkdir(parents=True,exist_ok=False)
 results=[]
 for index,row in enumerate(x['calibrations']):
  path=out/'inputs'/f'calibration-{index}.json';write_new_json(path,row);results.append(path)
 problem,costs=assemble_calibrations(results,output=out/'model',expected_head=x['head'])
 command=[sys.executable,str(root/'scripts/distillctl.py'),'workloads','search',
  '--strategy','model-based','--workload',x['workload'],
  '--planning-layout',str(out/'model/layout.json'),'--cost-model',str(out/'model/cost-model.json'),
  '--exhaustive','--max-proposals','100','--search-timeout','120','--max-trials',str(x['budget']),
  '--run-id',x['prefix']+'-cpu-search','--output',str(out/'search.json')]
 write_new_json(out/'command.json',dict(argv=command,gpu='NOT_RUN',source_head=x['head']))
 with (out/'cli.stdout').open('x') as so,(out/'cli.stderr').open('x') as se:
  subprocess.run(command,cwd=root,stdout=so,stderr=se,check=True,timeout=600)
 prediction=json.loads((out/'search.json').read_text());assert prediction['status']=='PASS_CPU_SEARCH'
 candidates=[]
 for row in [*prediction['ranked'],*prediction['rejections']]:
  candidates.append(dict(candidate_id=HeliosImplementation.from_assignment(row['assignment']).id,
   status=row['status'],predicted_cycle1_seconds=row.get('predicted_cycle1_ns',0)/1e9,
   predicted_cycle2_seconds=row.get('predicted_cycle2_ns',0)/1e9,detail=row.get('detail','')))
 with (out/'candidates.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
 summary=dict(status='PASS_MEASURED_CPU_SEARCH_NATIVE_VALIDATION_NOT_RUN',source_head=x['head'],
  workload=x['workload'],component_costs=len(costs.records),space_size=prediction['space_size'],
  feasible=len(prediction['ranked']),uncalibrated=sum(r['status']=='UNCALIBRATED' for r in candidates),
  candidates=candidates,selected=HeliosImplementation.from_assignment(prediction['predicted_best']['assignment']).id,
  binding=problem.binding,seeds=[HeliosImplementation.from_assignment(v).id for v in problem.seeds],
  problem_sha256=problem.digest,cost_model_sha256=costs.digest,remote_output=str(out),gpu='NOT_RUN',
  historical_costs_used=False,native_validation_budget=x['budget'])
 write_new_json(out/'summary.json',summary)
 print(json.dumps(dict(type='RESULT',result=summary)),flush=True)
else:
 problem=build_problem(out/'model/layout.json')
 costs=CostModel.from_dict(json.loads((out/'model/cost-model.json').read_text()))
 prediction=json.loads((out/'search.json').read_text())
 session=ModelValidationSession(problem=problem,costs=costs,prediction=prediction,
  output=out/'validation-r2',max_trials=x['budget'])
 def execute(candidate,phase,directory):
  print(json.dumps(dict(type='EXECUTE_NATIVE',candidate_id=candidate.id,phase=phase,
   trial_id=directory.name,remote_lowering=str(directory/'model-lowering.json'),
   lowering_sha256=file_sha256(directory/'model-lowering.json'),
   identity=session.identity)),flush=True)
  response=json.loads(sys.stdin.readline())
  assert response['candidate_id']==candidate.id and response['trial_id']==directory.name
  if 'error' in response:raise RuntimeError('Local native executor stopped: '+response['error'])
  return response['measurement']
 result=session.run(execute)
 print(json.dumps(dict(type='RESULT',result=result)),flush=True)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--variant', choices=('no-gan', 'gan'), required=True)
    p.add_argument('--stage', choices=('plan', 'validate'), required=True)
    p.add_argument('--calibration-unit', action='append', default=[])
    p.add_argument('--budget', type=int, choices=(3, 4), required=True)
    a = p.parse_args()
    deploy = json.loads(a.deployment.read_text()); head = deploy['source_head']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == head
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    workload = NO_GAN if a.variant == 'no-gan' else GAN
    assets = deploy['assets'][a.variant]
    controller = E / 'controller' / head[:12] / a.variant
    output = controller / 'model-search'
    output.mkdir(exist_ok=True)
    stage = output / (a.stage+'-r2'); stage.mkdir(exist_ok=False)
    with (E / 'model-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prefix = 'jointp_hv_20260908_' + head[:7] + '_' + a.variant.replace('-', '')
        payload = dict(stage=a.stage, head=head, workload=workload, budget=a.budget,
            cpu_controller_source=(E/'pending-anchor-rejection/model_validation.py').read_text(),
            cpu_controller_sha256=file_sha256(E/'pending-anchor-rejection/model_validation.py'),
            prefix=prefix, remote_output=deploy['remote_root']+'/model-search-'+head[:12]+'/'+a.variant,
            calibrations=[json.loads((controller / 'units' / unit / 'result.json').read_text())
                          for unit in a.calibration_unit])
        assert (len(payload['calibrations']) >= 2) if a.stage == 'plan' else not payload['calibrations']
        node = assets['nodes'][0]
        command = ['ssh', '-4', '-T', '-oBatchMode=yes', '-oConnectTimeout=15',
            '-oControlMaster=no', '-oControlPath=none', '-p', str(node['ssh_port']), node['host'],
            shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1',
                'PYTHONPATH='+deploy['target']+'/src:'+deploy['target'],
                'LD_LIBRARY_PATH='+node['python_env']+'/lib',
                node['python_env']+'/bin/python', '-u', '-c', REMOTE])]
        write_new_json(stage/'request.json', dict(command=command, payload=payload,
            script_sha256=file_sha256(Path(__file__)), source_head=head))
        runner = None
        with (stage/'stderr.log').open('x') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, text=True, bufsize=1)
            try:
                process.stdin.write(json.dumps(payload)+'\n'); process.stdin.flush()
                complete = False
                for line in process.stdout:
                    message = json.loads(line)
                    with (stage/'messages.jsonl').open('a') as log:
                        log.write(json.dumps(message, sort_keys=True)+'\n')
                    if message['type'] == 'RESULT':
                        write_new_json(stage/'result.json', message['result'])
                        print(json.dumps(message), flush=True); complete = True
                        break
                    assert a.stage == 'validate' and message['type'] == 'EXECUTE_NATIVE'
                    identity = message['identity']
                    assert identity['source_head'] == head and identity['workload'] == workload
                    if runner is None:
                        baseline, incumbent = anchors(workload)
                        runner = NativeTrialRunner(workload=workload, identity=identity, assets=assets,
                            output=controller, run_prefix=prefix, reference_id=baseline.id,
                            bootstrap_peer=incumbent.id)
                    else:
                        assert runner.identity == identity
                    candidate = get_candidate(workload, message['candidate_id'])
                    directory = stage/'trials'/message['trial_id']
                    directory.mkdir(parents=True, exist_ok=False)
                    write_new_json(directory/'request.json', message)
                    print(json.dumps(dict(stage='NATIVE_START', candidate_id=candidate.id,
                        phase=message['phase'], trial_id=message['trial_id'])), flush=True)
                    response = dict(candidate_id=candidate.id, trial_id=message['trial_id'])
                    try:
                        response['measurement'] = runner(candidate, message['phase'], directory)
                        write_new_json(directory/'measurement.json', response['measurement'])
                    except Exception as error:
                        response['error'] = type(error).__name__+': '+str(error)
                        write_new_json(directory/'failure.json', response)
                    process.stdin.write(json.dumps(response)+'\n'); process.stdin.flush()
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    print(json.dumps(dict(stage='NATIVE_DONE', candidate_id=candidate.id,
                        cycle_wall_seconds=response['measurement']['cycle_wall_seconds'])), flush=True)
                process.stdin.close()
                returncode = process.wait(timeout=60)
                assert returncode == 0 and complete, (returncode, str(stage/'stderr.log'))
            finally:
                if process.poll() is None:
                    process.stdin.close()
                    try: process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.terminate(); process.wait(timeout=15)


if __name__ == '__main__':
    main()
