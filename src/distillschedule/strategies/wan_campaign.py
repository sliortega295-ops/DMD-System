"""Budgeted two-node calibration -> planning -> native confirmation.

Prepare is CPU-only. Run this same campaign on each of the two existing nodes;
no SSH, credentials, scheduler submission, background agent, or implicit GPU
allocation is used. Both nodes require the same shared campaign directory.
"""
from __future__ import annotations
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256,fingerprint,repository_root,write_new_json
from .wan_task import OriginalWanTask
from .wan_strategy import implementations
from .wan_worker import OriginalWanAssets,_json
from .wan_planning import assemble,plan,source_fingerprint

SCHEMA='distillschedule.original_wan_campaign.v1'


def prepare(task,assets,output,*,include_fusion=True,max_launches=8):
    require(isinstance(task,OriginalWanTask) and isinstance(assets,OriginalWanAssets),
            'validated task/assets required',path='wan.campaign')
    require(type(max_launches) is int and 2<=max_launches<=8,'total native launch budget must be 2..8',path='wan.campaign')
    candidates=implementations(task,triton_available=include_fusion)
    require(len(candidates)+1<=max_launches,
            'budget cannot cover requested component arms plus confirmation; no silent candidate truncation',path='wan.campaign')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    taskpath=root/'task.json';assetpath=root/'assets.json'
    write_new_json(taskpath,task.to_dict());write_new_json(assetpath,assets.__dict__)
    jobs=[]
    for i,candidate in enumerate(candidates):
        implementation=root/f'implementation-{i}.json';write_new_json(implementation,candidate.to_dict())
        jobs.append({'id':f'calibrate-{i}','implementation':str(implementation),
                     'implementation_sha256':file_sha256(implementation)})
    result={'schema_version':SCHEMA,'task_sha256':task.digest,'source_fingerprint':source_fingerprint(),
            'task':str(taskpath),'task_file_sha256':file_sha256(taskpath),
            'assets':str(assetpath),'asset_file_sha256':file_sha256(assetpath),'jobs':jobs,
            'max_native_launches':max_launches,'planned_native_launches':len(jobs)+1,
            'reference':'original world16 FSDP control is a calibration arm, not a formal speed baseline',
            'confirmation_cycles':2,'gpu_execution':'NOT_RUN','claimable_speedup':False,
            'requirements':'two nodes / eight GPUs each / shared absolute campaign and asset paths'}
    result['campaign_sha256']=fingerprint(result)
    write_new_json(root/'campaign.json',result)
    return result


def _load(path):return json.loads(Path(path).read_text())


def verify(path):
    path=Path(path).resolve();doc=_load(path);unsigned=dict(doc);digest=unsigned.pop('campaign_sha256',None)
    require(doc.get('schema_version')==SCHEMA and fingerprint(unsigned)==digest,'campaign changed',path='wan.campaign')
    require(source_fingerprint()==doc['source_fingerprint'],'campaign code changed; retain old budget record',path='wan.campaign')
    for k in ('task','assets'):
        require(file_sha256(Path(doc[k]))==doc['task_file_sha256' if k=='task' else 'asset_file_sha256'],
                f'campaign {k} changed',path='wan.campaign')
    for job in doc['jobs']:
        require(file_sha256(Path(job['implementation']))==job['implementation_sha256'],
                'campaign strategy changed',path='wan.campaign')
    require(doc['planned_native_launches']==len(doc['jobs'])+1<=doc['max_native_launches']<=8,
            'campaign budget changed',path='wan.campaign')
    return doc


def _await(path,timeout):
    deadline=time.monotonic()+timeout
    while not Path(path).is_file():
        if time.monotonic()>=deadline:raise TimeoutError(f'peer/coordinator did not publish {path}')
        time.sleep(0.25)
    return _load(path)


def run(path,*,node_rank,master_addr,master_port,execute=False,resume=False,timeout_seconds=7200):
    doc=verify(path);root=Path(path).resolve().parent
    require(type(node_rank) is int and node_rank in (0,1),'two-node rank must be 0 or 1',path='wan.campaign')
    require(isinstance(master_addr,str) and master_addr and type(master_port) is int and 1024<=master_port<=65535,
            'explicit rendezvous host/port required',path='wan.campaign')
    require(type(timeout_seconds) in (int,float) and 60<=timeout_seconds<=86400,'invalid launch timeout',path='wan.campaign')
    if not execute:
        return {'status':'VALIDATED_CAMPAIGN_NOT_EXECUTED','planned_native_launches':doc['planned_native_launches'],
                'gpu_execution':'NOT_RUN_REQUIRES_EXPLICIT_EXECUTE','campaign_sha256':doc['campaign_sha256']}
    node=root/f'node-{node_rank}';node.mkdir(exist_ok=True)
    lock=node/'controller.lock'
    with lock.open('x') as f:f.write(f'{os.getpid()}\n')
    journal=node/'journal.json'
    binding={'campaign_sha256':doc['campaign_sha256'],'master_addr':master_addr,'master_port':master_port}
    try:
        if journal.exists():
            require(resume,'campaign already attempted: use explicit resume without resetting its budget',path='wan.campaign')
            ledger=_load(journal);require(ledger['binding']==binding,'campaign identity changed',path='wan.campaign')
            require(all(j['state']=='FINISHED' for j in ledger['launches']),
                    'interrupted launch has unknown outcome; do not relaunch or reset budget',path='wan.campaign')
        else:
            ledger={'binding':binding,'launches':[]};_json(journal,ledger)
        def launch(jobid,arguments):
            previous=next((j for j in ledger['launches'] if j['id']==jobid),None)
            if previous is not None:return previous['returncode']
            require(len(ledger['launches'])<doc['max_native_launches'],'native launch budget exhausted',path='wan.campaign')
            command=[sys.executable,'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=8',
                     f'--node-rank={node_rank}',f'--master-addr={master_addr}',f'--master-port={master_port}',
                     '--max-restarts=0',str(repository_root()/'scripts/distillctl.py'),
                     'workloads','native',*arguments,'--task',doc['task'],'--assets',doc['assets'],'--execute']
            entry={'id':jobid,'state':'STARTED','command':command};ledger['launches'].append(entry);_json(journal,ledger)
            try:
                with (node/f'{jobid}.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=repository_root(),stdout=log,stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    try:
                        code=process.wait(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        # Only this controller's child process group is terminated.
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait()
                        code=124
            except OSError as error:
                entry['launch_error']=str(error);code=127
            entry.update(state='FINISHED',returncode=code);_json(journal,ledger)
            _json(root/f'{jobid}-node{node_rank}.json',{'returncode':code,'binding':binding})
            return code
        for job in doc['jobs']:
            jobid=job['id'];out=root/jobid
            launch(jobid,['calibrate','--implementation',job['implementation'],'--output',str(out)])
            peer=_await(root/f'{jobid}-node{1-node_rank}.json',timeout_seconds)
            require(peer['binding']==binding,'peer is running a different campaign',path='wan.campaign')
            if node_rank==0:
                status=root/f'{jobid}-assembled.json'
                if not status.exists():
                    local=_load(root/f'{jobid}-node0.json')
                    if local['returncode']==peer['returncode']==0:
                        try:
                            collection_root=root/f'{jobid}-costs'
                            assemble(OriginalWanTask.from_dict(_load(doc['task'])),
                                     [out/f'rank_{r:02d}.json' for r in range(16)],collection_root)
                            row={'status':'CALIBRATED','collection':str(collection_root/'collection.json')}
                        except (ValueError,KeyError,OSError) as error:
                            row={'status':'REJECTED','error':str(error)}
                    else:row={'status':'FAILED_NATIVE_ARM','node0':local['returncode'],'node1':peer['returncode']}
                    _json(status,row)
            _await(root/f'{jobid}-assembled.json',timeout_seconds)
        ready=root/'plan-ready.json';planpath=root/'selected-plan.json'
        if node_rank==0 and not ready.exists():
            try:
                valid=[_load(root/f"{j['id']}-assembled.json") for j in doc['jobs']]
                paths=[x['collection'] for x in valid if x['status']=='CALIBRATED']
                selected=plan(OriginalWanTask.from_dict(_load(doc['task'])),paths)
                write_new_json(planpath,selected)
                _json(ready,{'status':'PLANNED','plan':str(planpath),'plan_file_sha256':file_sha256(planpath)})
            except (ValueError,KeyError,OSError) as error:_json(ready,{'status':'NO_PLAN','error':str(error)})
        selected=_await(ready,timeout_seconds)
        require(selected['status']=='PLANNED','no plan available; inspect preserved per-arm outcomes',path='wan.campaign')
        require(file_sha256(planpath)==selected['plan_file_sha256'],'selected plan changed',path='wan.campaign')
        result=launch('confirmation',['run','--plan',str(planpath),'--output',str(root/'confirmation')])
        peer=_await(root/f'confirmation-node{1-node_rank}.json',timeout_seconds)
        require(result==peer['returncode']==0,'selected plan native execution failed',path='wan.campaign')
        comparison_path=root/'matched-comparison.json'
        if node_rank==0 and not comparison_path.exists():
            try:
                from .wan_evidence import compare_runs
                comparison=compare_runs(root/doc['jobs'][0]['id'],root/'confirmation')
            except (ValueError,KeyError,OSError,RuntimeError) as error:
                comparison={'status':'UNVERIFIED','error':str(error),'claimable_speedup':False}
            _json(comparison_path,comparison)
        comparison=_await(comparison_path,timeout_seconds)
        return {'status':'COMPLETED_WITH_REFERENCE_PARITY' if comparison['status']=='PASS'
                else 'COMPLETED_WITH_NUMERICAL_REVIEW','plan':str(planpath),
                'native_launches':len(ledger['launches']),'gpu_execution':'NATIVE_RUN',
                'reference_comparison':comparison,'claimable_speedup':False,'quality_validation':'NOT_RUN'}
    finally:lock.unlink()
