"""Bounded H100 W6 three-arm launcher; native tensor/model code is elsewhere."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback

from run_h3_h100_three_arm_r1 import proc_record, discover_owned_children, cleanup_owned

ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
CAMPAIGN=ROOT/'research_artifacts/five_workload_h100_20260930'
OWNER=CAMPAIGN/'h100_resident_baselines_20261014_r1'
REPO=ROOT/'repos/DMD-System-h100-w6-native-logical-b16-20261014-r1'
HEAD='3003a9903a96e0ccaafa10faec956e1846e2448e'
PYTHON=ROOT/'envs/five-workload-h100-20260930/bin/python'
ACTIVATE=OWNER/'scripts/h3_w6_native_b16_environment_r1.sh'
WORKER=REPO/'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
INPUT=OWNER/'artifacts/h3_w6_preparation/native_inputs_cpu_gate_r1.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2);f.write('\n')

def run(command):
    return subprocess.check_output(list(map(str,command)),text=True,timeout=60).strip()

def source_gate():
    assert run(['git','-C',REPO,'rev-parse','HEAD'])==HEAD
    assert not run(['git','-C',REPO,'status','--porcelain'])

def host():
    events={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}
    stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return {'current':int(Path('/sys/fs/cgroup/memory.current').read_text()),'events':events,
            'memory_stat':stat,'peak':'NOT_AVAILABLE_ON_THIS_KERNEL' if not Path('/sys/fs/cgroup/memory.peak').exists() else Path('/sys/fs/cgroup/memory.peak').read_text().strip()}

def aggregate(rows,tiny):
    assert len(rows)==8 and [r['rank'] for r in rows]==list(range(8))
    initial={json.dumps(r['initial_LoRA_sha256'],sort_keys=True) for r in rows};assert len(initial)==1
    wanted='PASS_TINY_NATIVE_INTEGRATION_ONLY' if tiny else 'PASS_REAL_NATIVE_AR8_G1F1_LOGICAL_B16'
    samples=[]
    for r in rows:
        assert r['status']==wanted and r['logical_slots']==[r['rank'],r['rank']+8]
        assert r['actual_optimizer_counts']=={'student':2,'critic':2} and r['final_versions']=={'student':2,'fake':2,'ema':0}
        assert len(r['cycles'])==2 and r['actual_runtime_dtype_receipts']
        samples.extend(s['identity'] for s in r['samples'])
    assert len(samples)==len(set(samples))==16
    second=[r['cycles'][1] for r in rows]
    seconds=max(r['timer']['stop'] for r in second)-min(r['timer']['start'] for r in second)
    return {'status':'PASS_REAL_8_RANK_16_LOGICAL_SLOT_NATIVE_GATE' if not tiny else 'PASS_TINY_8_RANK_INTEGRATION_ONLY',
            'initial_LoRA_sha256':rows[0]['initial_LoRA_sha256'],'physical_ranks':8,'independent_logical_samples':16,
            'timed_C2_seconds':None if tiny else seconds,'warmup_C1_seconds':None if tiny else max(r['cycles'][0]['timer']['seconds'] for r in rows),
            'timer_basis':'same-node max original C2 stop minus min original C2 start; one warmup cycle excluded',
            'peak_allocated_bytes':max(r['memory_after_timer']['peak_allocated'] for r in second),
            'peak_reserved_bytes':max(r['memory_after_timer']['peak_reserved'] for r in second),
            'strict_cross_arm_numerical_parity':'NOT_RUN','profiled':False}

def arm(parent,placement,mode,inputs,tiny):
    out=parent/(('tiny_' if tiny else '')+placement+'_'+mode);out.mkdir(exist_ok=False)
    for d in ('logs','manifests','rank_receipts'):(out/d).mkdir()
    source_gate();assert not run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
    env={k:v for k,v in os.environ.items() if not k.startswith(('H3_','CUDA_','NCCL_','HF_','TRANSFORMERS_'))}
    env.update(H3_W6_RUN_GPU_LOCK_HELD='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               OMP_NUM_THREADS='4',MALLOC_ARENA_MAX='2',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
               NCCL_DEBUG='WARN',PYTORCH_ALLOC_CONF='expandable_segments:True',
               TORCH_EXTENSIONS_DIR=str(ROOT/'cache/h3-w6-h100-b16/extensions'),
               TRITON_CACHE_DIR=str(ROOT/'cache/h3-w6-h100-b16/triton'))
    command=['bash',str(ACTIVATE),'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8',
             '--master_addr=127.0.0.1','--master_port=29640',str(WORKER),'--placement',placement,'--mode',mode,
             '--config',inputs['config_path'],'--cache-manifest',inputs['cache_manifest'],
             '--checkpoint',inputs['model_root'],'--receipt-dir',str(out/'rank_receipts')]
    if tiny:command.append('--tiny')
    with socket.socket() as s:s.bind(('127.0.0.1',29640))
    write(out/'manifests/command.json',{'command':command,'head':HEAD,'worker_sha256':sha(WORKER),'activation_sha256':sha(ACTIVATE),'environment':{k:v for k,v in env.items() if k.startswith(('H3_W6_','CUDA_','NCCL_','TORCH_','TRITON_','PYTORCH_')) or k in ('OMP_NUM_THREADS','MALLOC_ARENA_MAX')}})
    before=host();owned={};child=None;started=time.monotonic();peak=before['current'];report={'arm':placement,'mode':mode,'status':'FAIL','GPU':'NOT_RUN'}
    try:
        with (out/'logs/launcher.log').open('x') as log,(out/'logs/host_samples.jsonl').open('x') as samples:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record=proc_record(child.pid);assert record is not None;owned[child.pid]=record;write(out/'manifests/launch.json',record)
            report['GPU']='LAUNCHED_REAL_WORKER';deadline=time.monotonic()+(300 if tiny else 7200)
            while child.poll() is None:
                discover_owned_children(owned);now=host();peak=max(peak,now['current']);samples.write(json.dumps(now)+'\n');samples.flush()
                if any(now['events'][k]!=before['events'][k] for k in ('oom','oom_kill')):raise RuntimeError('HOST_CGROUP_OOM_COUNTER_CHANGED')
                if time.monotonic()>deadline:raise TimeoutError('bounded native W6 worker deadline')
                time.sleep(2 if tiny else 10)
            report['exit_code']=child.returncode
        if child.returncode:raise RuntimeError('native worker failed; preserved rank-specific traceback')
        rows=[json.loads((out/f'rank_receipts/rank_{i:02d}.json').read_text()) for i in range(8)]
        report.update(aggregate(rows,tiny))
    except BaseException as exc:
        report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        remaining=cleanup_owned(owned) if owned else []
        after=host();apps=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
        delta={k:after['events'][k]-before['events'][k] for k in ('oom','oom_kill')}
        report.update(before=before,after=after,cgroup_oom_delta=delta,observed_host_cgroup_peak=peak,
                      owned_alive_after_cleanup=remaining,GPU_contexts_after=apps,process_wall_excluded_from_formal=time.monotonic()-started)
        if remaining or apps or any(delta.values()):report['status']='FAIL_TEARDOWN_OR_CGROUP'
        source_gate();write(out/'summary.json',report)
    return report,out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--tiny-only',action='store_true');args=ap.parse_args()
    out=CAMPAIGN/args.run_id;out.mkdir(exist_ok=False)
    source_gate();inputs=json.loads(INPUT.read_text());assert inputs['status']=='PASS_CPU_NATIVE_16_REAL_ROWS_MODEL_LAYOUT'
    assert sha(inputs['cache_manifest'])==inputs['cache_manifest_sha256'] and sha(inputs['config_path'])==inputs['config_sha256']
    report={'run_id':args.run_id,'head':HEAD,'GPU':'NOT_RUN','status':'PASS_CPU_ADMISSION',
            'native_AR8_G1F1':True,'Grid1000':'NOT_APPLICABLE_CONTINUOUS_SIGMA','replications':1,
            'strict_numerical_parity':'NOT_RUN','CV':'NOT_MEASURED_SINGLE_TRIAL','results':[]}
    try:
        if args.execute:
            assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS')=='1'
            subprocess.run([str(PYTHON),str(ROOT/'repos/DMD-System-h100-helios-b16-20260930/scripts/distillschedule/verify_inherited_gpu_locks.py'),'--project-root',str(ROOT),'--asset-fd','7'],check=True,pass_fds=(7,8,9))
            for placement in ('resident','fsdp','candidate'):
                modes=('correctness',) if args.tiny_only else ('correctness','timing')
                for mode in modes:
                    value,path=arm(out,placement,mode,inputs,args.tiny_only);report['results'].append({'placement':placement,'mode':mode,'summary':str(path/'summary.json'),**value})
                    if not value['status'].startswith('PASS_'):
                        text=(path/'logs/launcher.log').read_text(errors='replace')
                        capacity=('CUDA out of memory' in text or 'torch.OutOfMemoryError' in text) and not any(value['cgroup_oom_delta'].values()) and not value['GPU_contexts_after'] and not value['owned_alive_after_cleanup']
                        if not capacity or args.tiny_only:raise RuntimeError('NON_CAPACITY_OR_TINY_FAILURE_REQUIRES_REVIEW')
                        break
            timing={v['placement']:v['timed_C2_seconds'] for v in report['results'] if v['mode']=='timing' and v['status'].startswith('PASS_')}
            report.update(GPU='COMPLETED',status='PASS_MATRIX_EXECUTION_REVIEW_INDIVIDUAL_ARMS',timing_seconds=timing,
                          single_run_ratios={p:timing[p]/timing['candidate'] for p in ('resident','fsdp') if p in timing and 'candidate' in timing},
                          optimization_scope='native attention result reuse only; other A100 kernel mechanisms NOT_YET_PORTED')
    except BaseException as exc:
        report.update(status='FAIL_PRESERVED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    write(out/'result.json',report);print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    return 0 if report['status'].startswith('PASS_') else 1

if __name__=='__main__':raise SystemExit(main())
