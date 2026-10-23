from pathlib import Path
import ast,base64,concurrent.futures,hashlib,importlib.util,json
R=Path(__file__).resolve().parents[1];B=R.parent;P=B.parent
RR='/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r46'
OLD='/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/planner_manual_best_match_20261017/all_six/h3_ar_cpu/preparation_r6'
def main():
 spec=importlib.util.spec_from_file_location('io',P/'livewan_paper_three_arm_20261016/r1/scripts/remote_io.py');io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io);io.R=R;io.RR=RR
 source=json.loads((B/'r43/manifests/source.json').read_text());target=json.loads((R/'manifests/target-checkpoint-both.json').read_text());assert all(x['rc']==0 and '18 passed'in x['tail']for x in target['nodes'])
 refs={name:dict(path=RR+'/scripts/native_r2/'+name,sha256=hashlib.sha256((R/'scripts/native_r2'/name).read_bytes()).hexdigest())for name in ('h3_ar_match_controller.py','h3_ar_match_node.py','metadata_barrier_worker.py','native_metadata_barrier.py')}
 packet=dict(schema='dmd.h3_ar_manual_metadata_barrier.v1',source_head=source['head'],scripts=dict(worker=refs['metadata_barrier_worker.py'],counter=refs['native_metadata_barrier.py']))
 io.json_new(R/'configs/manual-packet.json',packet);packet_bytes=(R/'configs/manual-packet.json').read_bytes();packet_ref=dict(path=RR+'/configs/manual-packet.json',sha256=hashlib.sha256(packet_bytes).hexdigest())
 record=dict(source=source,refs=refs,root=RR,old=OLD,local_root=str(R/'native_controller'),packet_ref=packet_ref,packet_bytes=base64.b64encode(packet_bytes).decode())
 code=r'''from pathlib import Path
import base64,json,sys
r=json.loads(RECORD);root=Path(r['root']);source=Path(r['source']['source']);sys.path[:0]=[str(root/'scripts/native_r2'),str(source/'src'),str(source)]
import h3_ar_match_controller as c
for name,ref in r['refs'].items():assert c.sha(ref['path'])==ref['sha256']
p=Path(r['packet_ref']['path']);p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(r['packet_bytes']);assert __import__('hashlib').sha256(data).hexdigest()==r['packet_ref']['sha256']
if p.exists():assert p.read_bytes()==data
else:
 with p.open('xb')as f:f.write(data)
old=json.loads((Path(r['old'])/'configs/controller.json').read_text())
cfg=dict(old,source=str(source),expected_head=r['source']['head'],run_root=str(root/'native'),local_root=r['local_root'],cache_root='/tmp/lyy-experiments/DMD-System/h3-ar-native-barrier-r46',master_ports={'manual-control':33741,'manual-child':33742},max_native_starts=2,manual_packet=r['packet_ref'])
cfg['task']=c.ref(source/'configs/distillschedule/original_h3_ar/w6_reference.json');assert cfg['task']['sha256']==old['task']['sha256']
cfg['scripts']={k:{v:r['refs']['h3_ar_match_'+k+'.py'][v]for v in('path','sha256')}for k in('controller','node')}
cfg['schema_version']=c.DEPENDENCY_SCHEMA
cfg['native_dependency']=dict(old['native_dependency'],overlay_root=str(root/'assets/native-accelerate-1.14'))
for phase in c.PHASES:
 command=c.gpu_command(cfg,0,phase);assert command[0]==cfg['nodes'][0]['python']['path']and command[command.index('--restore-native-barrier')+1]==('1'if phase=='manual-child'else'0')
 assert command[command.index('--manual-policy')+1]=='uncached-rollout-fa3-varlen-v1'and 'nsys'not in command
c.source_state(cfg);c.native_manifest(cfg);c.native_dependency_identity(cfg,create=True)
assert json.loads(c.bound(cfg['capacity_policy']).read_text())==dict(schema_version='distillschedule.original_h3_ar_capacity.v2',gpu_limit_bytes=41875931136,gpu_reserve_bytes=0,host_limits={'0':962072674304,'1':816043786240},host_reserve_bytes=0)
path=root/'configs/controller.json';c.write(path,cfg);c.load_config(path)
from distillschedule.strategies.h3_ar_assets import validate_checkpoint_audit
original=Path(r['old'])/'native_match/artifacts/checkpoint-audit.json';audit=json.loads(original.read_text())
validate_checkpoint_audit(Path(audit['checkpoint_root']),c.ref(original),{k:v['sha256']for k,v in audit['files'].items()})
dest=c.paths(cfg)['checkpoint_audit'];dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('xb')as f:f.write(original.read_bytes())
print('CONFIG='+json.dumps(dict(config=c.ref(path),source=r['source'],checkpoint_provenance=c.ref(original),GPU_started=0)))
'''.replace('RECORD',repr(json.dumps(record)))
 out=io.call(0,code,'manual-native-configure',240);config=json.loads(next(x[7:]for x in out.splitlines()if x.startswith('CONFIG=')));io.json_new(R/'manifests/native-configured.json',config)
 def prepare(node):
  code=r'''from pathlib import Path
import json,os,subprocess
r=json.loads(RECORD);source=Path(r['source']['source']);scripts=Path(r['root'])/'scripts/native_r2'
program="from pathlib import Path;import json,sys;sys.path.insert(0,SCRIPTS);import h3_ar_match_controller as c;import h3_ar_match_node as n;cfg=c.load_config(Path(CONFIG));cfg['_config_path']=CONFIG;result=n.run(cfg,NODE,'prepare',c.sha(CONFIG),Path(CONFIG));assert 'torch'not in sys.modules;print('PREPARE='+json.dumps(dict(node=NODE,proof=result,GPU_started=0)))".replace('SCRIPTS',repr(str(scripts))).replace('CONFIG',repr(r['config']['path'])).replace('NODE',str(r['node']))
compile(program,'original-W6-CPU-owner','exec')
env={k:os.environ[k]for k in('HOME','USER','LOGNAME','LANG','SSH_AUTH_SOCK')if k in os.environ};env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(source/'src')+':'+str(source),PATH='/tmp/distillschedule-helios-official-torch210/bin:/usr/local/bin:/usr/bin:/bin',LD_LIBRARY_PATH='/tmp/distillschedule-helios-official-torch210/lib')
p=subprocess.run(['/tmp/distillschedule-helios-official-torch210/bin/python','-s','-B','-c',program],cwd=source,env=env,capture_output=True,text=True,timeout=300)
print(p.stdout);print(p.stderr);assert p.returncode==0
'''.replace('RECORD',repr(json.dumps(dict(record,config=config['config'],node=node))))
  ast.parse(code);out=io.call(node,code,'original-owned-cpu-preparation',360);row=json.loads(next(x[8:]for x in out.splitlines()if x.startswith('PREPARE=')));io.json_new(R/f'manifests/native-prepare-node{node}.json',row);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:nodes=list(pool.map(prepare,(0,1)))
 code=r'''from pathlib import Path
import json,os,subprocess
r=json.loads(RECORD);source=Path(r['source']['source']);root=Path(r['root'])
program="from pathlib import Path;import json,sys;sys.path.insert(0,SCRIPTS);import h3_ar_match_controller as c;cfg=c.load_config(Path(CONFIG));cfg['_config_path']=CONFIG;v=c.freeze_assets(cfg,c.sha(CONFIG));assert 'torch'not in sys.modules;print('READY='+json.dumps(dict(admission=c.ref(c.paths(cfg)['admission']),admission_text=c.paths(cfg)['admission'].read_text(),config=c.ref(Path(CONFIG)),config_text=Path(CONFIG).read_text(),GPU_started=0)))".replace('SCRIPTS',repr(str(root/'scripts/native_r2'))).replace('CONFIG',repr(r['config']['path']))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(source/'src')+':'+str(source),LD_LIBRARY_PATH='/tmp/distillschedule-helios-official-torch210/lib')
p=subprocess.run(['/tmp/distillschedule-helios-official-torch210/bin/python','-s','-B','-c',program],cwd=source,env=env,capture_output=True,text=True,timeout=300);print(p.stdout);print(p.stderr);assert p.returncode==0
'''.replace('RECORD',repr(json.dumps(dict(record,config=config['config']))))
 out=io.call(0,code,'freeze-new-source-assets',360);ready=json.loads(next(x[6:]for x in out.splitlines()if x.startswith('READY=')))
 for name in('config','admission'):
  raw=ready.pop(name+'_text').encode();assert hashlib.sha256(raw).hexdigest()==ready[name]['sha256']
  with (R/'configs'/(name+'.json')).open('xb')as f:f.write(raw)
 io.json_new(R/'manifests/native-ready.json',dict(ready,source=source,nodes=nodes,status='REAL_NATIVE_CPU_OWNER_IMPORT_ABI_ASSETS_READY_GPU_NOT_RUN',native_starts=0,profiler=False))
 print('R46_SAME_SOURCE_SAME_BACKEND_METADATA_BARRIER_PAIR_READY_NO_GPU_YET')
if __name__=='__main__':main()
