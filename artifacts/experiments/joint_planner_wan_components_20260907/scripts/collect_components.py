"""Collect the missing action envelopes, not the completed window3 canary.

One controller shares current-source preparation across two calibrated arms.
Bound: two prerequisite correctness runs + two new component calibrations.
No old run is retried; no formal timing, profiler or historical matrix is run.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess

from distillschedule.search.execution import NativeTrialRunner
from distillschedule.search.registry import get_candidate
from distillschedule.search.runtime_identity import observed_binding
from distillschedule.search.wan_implementation import WAN, PRECISION
from distillschedule.workloads.best_known.registry import (
    git_head, repository_root, fingerprint, write_new_json,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'ceb2921985e5a855ff93d6c9d7e21e7528b6ef5a'
REFERENCE = 'joint-wan2-fsdp-c512-e0-w5'
CANDIDATE = 'joint-wan2-resident-c2048-e1-w5'
RUN = 'jointp_wan_components_20260907_r1'

def audit(node):
    code = '''import json,socket,subprocess
from pathlib import Path
def run(*args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=45)
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(host=socket.gethostname(),
 gpu_processes=run('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),
 lock=run('flock','-n','/tmp/distillschedule_gpu_node.lock','true'),
 listening_ports=[int(line.split()[1].rsplit(':',1)[1],16) for name in ('tcp','tcp6') for line in (Path('/proc/net')/name).read_text().splitlines()[1:] if line.split()[3]=='0A'],
 cgroup_events=Path('/sys/fs/cgroup/memory.events').read_text())))'''
    ssh=['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none',
         '-p',str(node['ssh_port']),node['host']]
    command=[*ssh,shlex.join([str(Path(node['python_env'])/'bin/python'),'-c',code])]
    p=subprocess.run(command,capture_output=True,text=True,timeout=90)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)

def work_signatures(node, trace):
    code = '''import json,torch,sys
from pathlib import Path
from runtime.static_world16_trace import StaticWorld16Trace
from distillschedule.runtime.fsdp2_p0 import expected_trace_rollout_call_inventory
from distillschedule.search.wan_action_costs import trace_work_signatures
from distillschedule.workloads.best_known.registry import fingerprint
t=StaticWorld16Trace(Path(sys.argv[1]),variant='no-gan')
ops=['step/10/generator/rollout']+[f'step/{s}/critic/rollout' for s in range(10,15)]
inventory=[dict(operation=op,inventory=expected_trace_rollout_call_inventory(t.rank_control_validation,operation=op,expected_rank_count=16)) for op in ops]
work=trace_work_signatures(t,inventory)
assert len(work)==16 and not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_REAL_TRACE_METADATA',work_signatures=work,work_sha256=fingerprint(work),trace_hash=t.bundle_hash,cuda_initialized=False)))'''
    environment={**node['environment'],'PYTHONDONTWRITEBYTECODE':'1'}
    command=['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none',
             '-p',str(node['ssh_port']),node['host'],
             shlex.join(['env',*(f'{k}={v}' for k,v in environment.items()),node['python'],'-c',code,str(trace)])]
    p=subprocess.run(command,capture_output=True,text=True,timeout=180)
    write_new_json(ROOT/'manifests/real-trace-command.json',{'command':command,'returncode':p.returncode,'stderr':p.stderr})
    if p.returncode:raise RuntimeError(p.stderr[-5000:])
    value=json.loads(p.stdout)
    write_new_json(ROOT/'manifests/real-trace-metadata.json',value)
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    assert git_head(repository_root())==SOURCE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repository_root(),text=True).strip()
    assets=json.loads((ROOT/'configs/assets.json').read_text())
    controller=ROOT/'controller';controller.mkdir(exist_ok=True)
    identity=dict(source_head=SOURCE,asset_fingerprint=fingerprint(assets['assets']))
    runner=NativeTrialRunner(workload=WAN,identity=identity,assets=assets,output=controller,
        run_prefix=RUN,reference_id=REFERENCE,bootstrap_peer=CANDIDATE)
    plan=runner._rpc('resolve',workload=WAN,candidate_id=CANDIDATE,assets=assets,
        run_id=RUN+'-preflight',mode='correctness')
    nodes=[runner._rpc('runtime_identity',_node_rank=i,node_rank=i,receipt=plan) for i in (0,1)]
    identity.update(observed_binding(nodes,source_head=SOURCE),precision_fingerprint=PRECISION.digest,
                    semantic_fingerprint=plan['workload_semantics_hash'])
    observations=[audit(node) for node in assets['nodes']]
    assert all(r['gpu_processes']['returncode']==0 and not r['gpu_processes']['stdout'].strip()
               and r['lock']['returncode']==0 and 31527 not in r['listening_ports'] for r in observations)
    tag='execution-preflight' if args.execute else 'validate-only'
    write_new_json(ROOT/f'manifests/{tag}.json',dict(status='GO',source_head=SOURCE,run_id=RUN,
        binding=identity,nodes=nodes,resource_audit=observations,
        gpu_execution='NOT_RUN',max_correctness_runs=2,max_component_calibrations=2,
        formal_timing_runs=0,old_window3_canary_repeats=0,
        rationale='New 32-action costs and retained lifetimes have never been measured; legacy macro costs are not equivalent. Shared current-source preparation is reused across both collectors.',
        native_node_lock='Existing launcher holds flock over audit, construction, run and teardown'))
    if not args.execute:
        work_signatures(plan['nodes'][0],assets['assets']['trace']['path'])
        print(json.dumps(dict(status='PASS_PREFLIGHT',gpu_execution='NOT_RUN',binding=identity)),flush=True)
        return
    assert json.loads((ROOT/'manifests/real-trace-metadata.json').read_text())['status']=='PASS_REAL_TRACE_METADATA'
    with (controller/'exclusive-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_new_json(ROOT/'manifests/collection-start.json',dict(status='START',run_id=RUN,source_head=SOURCE))
        results=[]
        try:
            for candidate in (CANDIDATE,REFERENCE):
                print(json.dumps(dict(status='COLLECTING',candidate=candidate)),flush=True)
                result=runner.calibrate(get_candidate(WAN,candidate),directory=controller/('calibration-'+candidate))
                results.append(result)
                print(json.dumps(dict(status=result['status'],run_id=result['run_id'],candidate=candidate,
                                      ranks=result['rank_count'],actions=result['stages_per_rank'])),flush=True)
            write_new_json(ROOT/'manifests/collection-complete.json',dict(status='PASS_NATIVE_ACTION_COLLECTION',
                source_head=SOURCE,results=results,formal_timing_runs=0,old_window3_canary_repeats=0))
        except BaseException as error:
            write_new_json(ROOT/'manifests/collection-failure.json',dict(status='FAIL',source_head=SOURCE,
                error_type=type(error).__name__,error=str(error),completed_results=results,
                next_action='PRESERVE_FAILURE; NO_AUTOMATIC_RETRY_OR_MATRIX'))
            raise

if __name__=='__main__':main()
