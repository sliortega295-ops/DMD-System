"""Scoped source deployment + source/proof/environment gate. No model loading."""
import concurrent.futures,copy,json,shlex,subprocess,datetime
from pathlib import Path
E=Path(__file__).resolve().parents[1]
R=Path('/home/lyy/Helios/DMD-System/worktrees/joint-planner-20260907')
H='bf59e39b8a6fe158ad15ebae07be7df71369f5fd'; B='878aff490ce28c661ac1d927f08f8846992baf45'
old=json.loads((E.parent/'h3_actions_r1/manifests/deployment.json').read_text())
P='/share/project/shuchenweng/yongyan_liu'
T=P+'/DMD-System_deploy/joint-h3-performance-'+H[:12]+'-20260907'
OUT=P+'/DistillSchedule_runs/joint_planner_20260907/h3_performance_r1'
a=copy.deepcopy(old['assets'])
for n in a['nodes']: n['repo_root']=T
# A long but bounded calibration, not an unbounded retry or C50.
a.update(master_port=31491,run_root=OUT+'/native',timeout_seconds=3300,
         node_local_cache_root='/tmp/lyy-experiments/DMD-System/joint_h3_performance_20260907_r1')
a.pop('correctness',None)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==H
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
def ssh(n,command,**kw):
 return subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',a['nodes'][n]['host'],command],**kw)
pack=subprocess.check_output(['git','pack-objects','--stdout','--revs'],cwd=R,input=(H+'\n^'+B+'\n').encode())
cmd=f'''set -eu
test ! -e {T}; test -d {OUT}; test ! -e {OUT}/preflight-r2-r2
git clone --shared --no-checkout {old['target']} {T}
git -C {T} unpack-objects -q
git -C {T} checkout --detach {H}
test -z "$(git -C {old['target']}/third_party/lightx2v status --porcelain)"
cp -a {old['target']}/third_party/lightx2v/. {T}/third_party/lightx2v/
test "$(git -C {T}/third_party/lightx2v rev-parse HEAD)" = d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be
test "$(git -C {T}/third_party/lightx2v rev-parse --show-toplevel)" = {T}/third_party/lightx2v
test -z "$(git -C {T} status --porcelain)"
mkdir -p {OUT}/scripts {OUT}/preflight-r2
'''
s=ssh(0,cmd,input=pack,capture_output=True,timeout=180)
(E/'logs/deploy-r2.stdout').write_bytes(s.stdout);(E/'logs/deploy-r2.stderr').write_bytes(s.stderr);s.check_returncode()
code=r'''
import fcntl,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
from distillschedule.search.execution import resolve_candidate
from distillschedule.search.h3_implementation import H3Implementation,H3
from distillschedule.search.h3_action_admission import core_paths,validate_action_admission
from distillschedule.search.runtime_identity import collect_planned_node_identity
from distillschedule.workloads.best_known.registry import fingerprint,file_sha256,write_new_json
from distillschedule.workloads.best_known.dispatch import validate_local,clean_environment
x=json.load(sys.stdin);root=Path(x['repo']);out=Path(x['out']);n=x['node'];assets=x['assets']
lock=open('/tmp/distillschedule_gpu_node.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def cmd(v):return subprocess.check_output(v,text=True).strip()
assert cmd(['git','-C',str(root),'rev-parse','HEAD'])==x['head']
assert not cmd(['git','-C',str(root),'status','--porcelain','--untracked-files=all'])
assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
s=socket.socket();s.bind(('',assets['master_port']));s.close()
events_before=Path('/sys/fs/cgroup/memory.events').read_text()
impl=H3Implementation(2,128,3)
p=impl.lower_profile(root=root)
refs=[dict(path=str(Path(x['prior_run'])/'execution/ranks'/f'rank_{r:03d}.json')) for r in range(16)]
for ref in refs:ref['sha256']=file_sha256(Path(ref['path']))
proof=dict(schema_version='distillschedule.h3_prior_action_proof.v1',source_head=x['prior_head'],
           candidate_id=impl.id,asset_fingerprint=fingerprint(assets['assets']),ranks=refs,
           native_core_sha256={p:file_sha256(root/p) for p in core_paths(root)})
gate=dict(status='PASS_NATIVE_ACTION_CONTRACTS_REUSED',numerical_status='REVIEW_NOT_STRICT_PASS',
          integration_binding=dict(integration_head=x['head'],profile_sha256=fingerprint(p),
                                   asset_fingerprint=fingerprint(assets['assets'])),h3_action_proof=proof)
gatepath=out/f'prior-action-gate-node{n}.json';write_new_json(gatepath,gate)
assets['correctness']=dict(path=str(gatepath),sha256=file_sha256(gatepath))
ap=out/f'assets-node{n}.json';write_new_json(ap,assets)
plan=resolve_candidate(H3,impl.id,root=root,assets_path=ap,run_id='jointp_h3_actions_perf_20260907_r1-calibration',mode='calibration')
validated=validate_local(plan,root=root)
identity=collect_planned_node_identity(plan,node_rank=n)
node=plan['nodes'][n];env=clean_environment(node)
probe=r"""
import json,torch,diffusers,transformers,peft
from distillschedule.backends.training.minimax_h3.cycle_actions import tensor_evidence
from distillschedule.backends.training.minimax_h3 import grid_entrypoint
from kernels.utils import get_local_kernel
from pathlib import Path
import os
v=torch.ones((2,3),dtype=torch.bfloat16);before=torch.get_rng_state().clone()
assert set(tensor_evidence(torch,v,value_mode='metadata')[0])=={'shape','dtype','device','bytes'}
assert tensor_evidence(torch,v)[0]['finite']
k=get_local_kernel(Path(os.environ['H3_FLASH_ATTN3_LOCAL_PATH']),'flash_attn3');assert callable(k.flash_attn_func)
assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_CPU_METADATA_AND_PINNED_MODULE_IMPORTS',torch=torch.__version__,cuda=torch.version.cuda,
  diffusers=diffusers.__version__,transformers=transformers.__version__,peft=peft.__version__,
  model_loaded=False,cuda_context_created=False)))
"""
s=subprocess.run([node['python'],'-c',probe],cwd=root,env=env,text=True,capture_output=True,timeout=180)
(out/f'imports-node{n}.stdout').write_text(s.stdout);(out/f'imports-node{n}.stderr').write_text(s.stderr)
assert s.returncode==0,s.stderr
assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
events_after=Path('/sys/fs/cgroup/memory.events').read_text();assert events_before==events_after
write_new_json(out/f'execution-plan-node{n}.json',plan)
write_new_json(out/f'identity-node{n}.json',identity)
print(json.dumps(dict(status='PASS',node=n,hostname=socket.gethostname(),validation=validated,imports=json.loads(s.stdout),
  identity=identity,plan=plan,assets=assets,cgroup_before=events_before,cgroup_after=events_after,
  gpu_ownership='IDLE_OWNED_NODES_UNDER_LOCK',cleanup='PASS_GPU_EMPTY')))
'''
def prepare(n):
 node=a['nodes'][n]
 command=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+T+'/src:'+T,
   'LD_LIBRARY_PATH='+node['python_env']+'/lib',node['python_env']+'/bin/python','-c',code])
 s=ssh(n,command,input=json.dumps(dict(repo=T,out=OUT+'/preflight-r2',node=n,assets=a,head=H,prior_head=B,
       prior_run=old['remote_root']+'/native/jointp_h3_actions_20260907_r1-w3-correctness')),text=True,capture_output=True,timeout=360)
 (E/'logs'/f'preflight-r2-node{n}.stdout').write_text(s.stdout);(E/'logs'/f'preflight-r2-node{n}.stderr').write_text(s.stderr)
 assert s.returncode==0,(n,s.stderr[-7000:]);return json.loads(s.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(prepare,[0,1]))
assert rows[0]['identity']['environment']==rows[1]['identity']['environment']
# Shared evidence: node0 bound gate applies to both nodes after verification.
plan=rows[0]['plan']
(E/'manifests/execution-plan-r2.json').write_text(json.dumps(plan,indent=2)+'\n')
(E/'manifests/deployment-r2.json').write_text(json.dumps(dict(source_head=H,target=T,remote_root=OUT,
  assets=rows[0]['assets'],results=rows,source_transfer_bytes=len(pack),previous_runs='PRESERVED_NOT_REPEATED',
  gpu_run='NOT_RUN',calibration='NEXT',native_timing='AFTER_SEARCH'),indent=2)+'\n')
print(json.dumps(dict(status='GO_SOURCE_PROOF_AND_ENV_BOTH_NODES',head=H,target=T,source_bytes=len(pack),
  imports=[r['imports'] for r in rows],cuda_oom=0,cgroup_oom_delta=0,next='ONE_WORLD16_CALIBRATION')))
