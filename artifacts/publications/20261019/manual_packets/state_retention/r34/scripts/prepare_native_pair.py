"""Admit only the GPU-qualified exact EMA graph; all source additions opt-in."""
from pathlib import Path
import base64,fcntl,hashlib,importlib.util,json
R=Path(__file__).resolve().parents[1];B=R.parent;P=B.parent
RR='/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r34'

def main():
    s=importlib.util.spec_from_file_location('io',P/'livewan_paper_three_arm_20261016/r1/scripts/remote_io.py');io=importlib.util.module_from_spec(s);s.loader.exec_module(io);io.R=R;io.RR=RR
    with(B/'r1/artifacts/gpu-scheduler.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=json.loads((B/'r33/reports/component-result.json').read_text());result=proof['component']
        assert proof['lifecycle']['cleanup_pass']and proof['GPU_empty']and len(result['checks'])==10 and all(x['bitwise']for x in result['checks'])and result['RNG_unchanged']and result['capture_actual_state_unchanged']
        assert hashlib.sha256((R/'scripts/exact_ema_graph.py').read_bytes()).hexdigest()==result['component_scripts']['exact']['sha256']
        old=json.loads((B/'r30/configs/overlap.json').read_text());files={};refs={}
        def add(path,rel):
            data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();files[rel]=base64.b64encode(data).decode();ref=dict(path=RR+'/'+rel,sha256=digest);refs[rel]=ref;return ref
        for n in ('exact_ema_graph.py','manual_entry.py','native_node.py'):add(R/'scripts'/n,'scripts/'+n)
        for n in ('test_exact_ema_graph.py','test_manual_ema_integration.py'):add(R/'tests'/n,'tests/'+n)
        arms={}
        for arm,port in [('control',29881),('graph',29891)]:
            runid='wan-exact-ema-'+arm+'-20261018-r34'
            backend=json.loads((B/'r30/configs/overlap-backend.json').read_text());backend['run_id']=runid
            p=R/'configs'/(arm+'-backend.json');io.json_new(p,backend);bref=add(p,'configs/'+p.name)
            stream=json.loads((B/'r30/configs/overlap-stream.json').read_text());stream.update(run_id=runid,backend_packet=bref['path'],backend_packet_sha256=bref['sha256'])
            p=R/'configs'/(arm+'-stream.json');io.json_new(p,stream);sref=add(p,'configs/'+p.name)
            packet=dict(schema='dmd.wan_exact_ema_manual.v1',source_head=old['HEAD'],arm=arm,run_id=runid,entry_files={n:refs['scripts/'+n]['sha256']for n in ('manual_entry.py','exact_ema_graph.py')},parent_entry=old['manual_entry'],parent_entry_sha256=old['input_metadata_sha256'][old['manual_entry']],stream_packet=sref['path'],stream_packet_sha256=sref['sha256'])
            p=R/'configs'/(arm+'-ema.json');io.json_new(p,packet);eref=add(p,'configs/'+p.name)
            keep={k:v for k,v in old['input_metadata_sha256'].items()if k not in(old['experiment_packet'],old['stream_packet'])}
            keep.update({refs['scripts/'+n]['path']:refs['scripts/'+n]['sha256']for n in ('exact_ema_graph.py','manual_entry.py')});keep.update({x['path']:x['sha256']for x in(bref,sref,eref)})
            cfg={**old,'schema_version':'dmd.wan_exact_ema_graph_manual_pair.v1','run_id':runid,'artifact_root':RR+'/runs/'+arm,'master_port':port,'ema_arm':arm,'ema_source_sha256':refs['scripts/exact_ema_graph.py']['sha256'],'ema_packet':eref['path'],'ema_packet_sha256':eref['sha256'],'manual_entry':refs['scripts/manual_entry.py']['path'],'adapter_sha256':refs['scripts/native_node.py']['sha256'],'experiment_packet':bref['path'],'experiment_packet_sha256':bref['sha256'],'stream_packet':sref['path'],'stream_packet_sha256':sref['sha256'],'input_metadata_sha256':keep,'admission':'GPU_COMPONENT_ALL618_SHAPES_TEN_EMA_UPDATES_BITWISE_RNG_PASS; SAME_R30_STREAM_ORDER_BOTH_ARMS; ORIGINAL_FULL_STATE_REVIEW_RETAINED','graph_scope':'KNOWN_DECAY11_TO20_TWO_CYCLE_EXPERIMENT_ONLY_NOT_UNBOUNDED_TRAINING_READY'}
            p=R/'configs'/(arm+'.json');io.json_new(p,cfg);ref=add(p,'configs/'+p.name)
            arms[arm]=dict(config=ref,node_adapter=refs['scripts/native_node.py'],local_config=str(p))
        code=r'''from pathlib import Path
import base64,json,os
r=json.loads(RECORD);root=Path(r['root'])
for rel,blob in r['files'].items():
 p=root/rel;data=base64.b64decode(blob);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:
  t=p.with_name(p.name+'.tmp-'+str(os.getpid()))
  with t.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.link(t,p);t.unlink()
print('PUBLISHED='+str(len(r['files'])))
'''.replace('RECORD',repr(json.dumps(dict(root=RR,files=files))))
        io.call(0,code,'publish-qualified-native-source-packet-batch',100)
        targets=[]
        for node in (0,1):
            record=dict(root=RR,source=old['source'],head=old['HEAD'],refs=refs)
            code=r'''from pathlib import Path
import fcntl,hashlib,json,os,subprocess
r=json.loads(RECORD);root=Path(r['root']);source=Path(r['source'])
with open('/tmp/distillschedule_gpu_node.lock','a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==r['head']and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for ref in r['refs'].values():assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
 program="import sys,json,torch;assert torch.__version__=='2.10.0+cu128' and not torch.cuda.is_initialized();sys.path.append('/share/project/shuchenweng/yongyan_liu/DMD-System_assets/lightx2v_minimax_dmd_20260817/env/lib/python3.10/site-packages');import pytest;rc=pytest.main(['-q','-p','no:cacheprovider',*json.loads(sys.argv[1])]);assert not torch.cuda.is_initialized();print('CPU='+json.dumps(dict(returncode=rc,torch=torch.__version__,CUDA_initialized=False)));raise SystemExit(rc)"
 tests=[str(root/'tests'/n)for n in('test_exact_ema_graph.py','test_manual_ema_integration.py')]+[str(root.parent/'r30/tests/test_stream_execution.py')]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'scripts')+':'+str(source/'src')+':'+str(source),LD_LIBRARY_PATH='/tmp/distillschedule-helios-official-torch210/lib',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 p=subprocess.run(['/tmp/distillschedule-helios-official-torch210/bin/python','-s','-B','-c',program,json.dumps(tests)],cwd=source,env=env,text=True,capture_output=True,timeout=180)
 print(p.stdout);print(p.stderr);assert p.returncode==0 and '25 passed'in p.stdout
'''.replace('RECORD',repr(json.dumps(record)))
            out=io.call(node,code,'qualified-native-target-CPU',220);cpu=json.loads(next(x[4:]for x in out.splitlines()if x.startswith('CPU=')));assert cpu['returncode']==0 and not cpu['CUDA_initialized']
            io.json_new(R/f'manifests/target-CPU-node{node}.json',cpu);targets.append(cpu)
        io.json_new(R/'manifests/manual-pair.json',dict(source=dict(source=old['source'],head=old['HEAD'],source_addition={n:r for n,r in refs.items()if n.startswith('scripts/')}),arms=arms,max_native_starts=2,prior_charged_owner_phases=84,profiler=False,planner_integration=False))
        io.json_new(R/'configs/GO-native.json',dict(status='GO_EXACT_EMA_COMPONENT_QUALIFIED_TWO_NATIVE_ARMS',prior_charged=84,max_native_starts=2,target_CPU=targets,source=old['HEAD'],GPU_component_evidence=str(B/'r33/reports/component-result.json'),original_reserves=True,original_comparator=True,profiler=False,planner=False,no_retry=True))
        print('BOTH_TARGET25_PASS_TWO_QUALIFIED_NATIVE_ARMS_READY')

if __name__=='__main__':main()
