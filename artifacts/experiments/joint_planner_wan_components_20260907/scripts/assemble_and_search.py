"""Assemble newly completed native action costs beside their remote receipts.

Metadata only: no GPU launch and no old calibration or correctness rerun.
"""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HEAD = "ceb2921985e5a855ff93d6c9d7e21e7528b6ef5a"
REMOTE = "/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/component_costs_r1/analysis_r1"
ARMS = ("joint-wan2-fsdp-c512-e0-w5", "joint-wan2-resident-c2048-e1-w5")


def main():
    complete = json.loads((ROOT / "manifests/collection-complete.json").read_text())
    assert complete["status"] == "PASS_NATIVE_ACTION_COLLECTION"
    assert complete["source_head"] == HEAD
    assets = json.loads((ROOT / "configs/assets.json").read_text())
    node = assets["nodes"][0]
    results = {}
    for arm in ARMS:
        path = ROOT / "controller" / ("calibration-" + arm) / "result.json"
        value = json.loads(path.read_text())
        assert value["status"] == "PASS_NATIVE_STAGE_CALIBRATION"
        assert value["source_head"] == HEAD and value["rank_count"] == 16
        assert value["stages_per_rank"] == 32
        results[arm] = value
    script = r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin)
repo=Path(p['repo'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==p['head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
root=Path(p['output']);root.mkdir(parents=True,exist_ok=False)
(root/'inputs').mkdir();(root/'logs').mkdir()
paths=[]
for arm,value in p['results'].items():
 f=root/'inputs'/(arm+'.json');f.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');paths.append(f)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(repo/'src'),
     'LD_LIBRARY_PATH':str(Path(sys.executable).parent.parent/'lib')}
commands=[
 [sys.executable,str(repo/'scripts/distillschedule/assemble_wan_action_calibrations.py'),
  *[x for f in paths for x in ('--calibration-result',str(f))],
  '--expected-head',p['head'],'--output',str(root/'assembly')],
 [sys.executable,str(repo/'scripts/distillctl.py'),'workloads','search',
  '--strategy','model-based','--workload','wan13b_nogan_resident_best',
  '--problem',str(root/'assembly/problem.json'),'--cost-model',str(root/'assembly/cost-model.json'),
  '--exhaustive','--max-proposals','60','--output',str(root/'search.json')]]
(root/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for index,command in enumerate(commands):
 q=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,timeout=650)
 (root/'logs'/f'{index}.stdout.log').write_text(q.stdout)
 (root/'logs'/f'{index}.stderr.log').write_text(q.stderr)
 (root/'logs'/f'{index}.exit.json').write_text(json.dumps({'returncode':q.returncode})+'\n')
 if q.returncode:raise RuntimeError(q.stderr[-4000:])
search=json.loads((root/'search.json').read_text())
assembly=json.loads((root/'assembly/summary.json').read_text())
assert search['status']=='PASS_CPU_SEARCH'
from collections import Counter
summary={'status':'PASS_MEASURED_COMPONENT_SEARCH','source_head':p['head'],
 'gpu_execution':'NOT_RUN','claimable_speedup':False,'assembly':assembly,
 'space_size':search['space_size'],'unique_evaluated':search['unique_evaluated'],
 'feasible':len(search['ranked']),'rejections':dict(Counter(r['status'] for r in search['rejections'])),
 'predicted_best':search['predicted_best'],
 'ranked':[dict(assignment=r['assignment'],predicted_cycle2_ns=r['predicted_cycle2_ns'],
                memory=r.get('memory',r.get('peak_bytes_per_rank'))) for r in search['ranked']],
 'paths':{'root':str(root),'search':str(root/'search.json')}}
(root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
sha={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
     for f in root.rglob('*') if f.is_file()}
(root/'artifact_sha256.json').write_text(json.dumps(sha,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary))
'''
    command = ["ssh", "-4", "-T", "-oBatchMode=yes", "-oConnectTimeout=15",
               "-oControlMaster=no", "-oControlPath=none", "-p", str(node["ssh_port"]),
               node["host"], shlex.join([str(Path(node["python_env"]) / "bin/python"), "-c", script])]
    payload = dict(repo=node["repo_root"], head=HEAD, output=REMOTE, results=results)
    with (ROOT / "manifests/assembly-start.json").open("x") as f:
        json.dump(dict(command=command, payload=payload, gpu_execution="NOT_RUN"), f, indent=2)
    result = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True, timeout=1500)
    for name, data in (("stdout", result.stdout), ("stderr", result.stderr)):
        with (ROOT / f"logs/assembly.{name}.log").open("x") as f:
            f.write(data)
    with (ROOT / "manifests/assembly-exit.json").open("x") as f:
        json.dump(dict(returncode=result.returncode), f)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    value = json.loads(result.stdout)
    with (ROOT / "reports/component-search.json").open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
    print(json.dumps({k: value[k] for k in ("status", "space_size", "unique_evaluated", "feasible", "rejections", "paths")}, indent=2))


if __name__ == "__main__":
    main()
