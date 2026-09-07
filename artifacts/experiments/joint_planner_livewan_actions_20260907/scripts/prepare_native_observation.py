"""Fresh exact source and metadata-only assets for one missing observation run."""
import json, shlex, subprocess
from pathlib import Path

E=Path(__file__).resolve().parents[1]
R=Path('/home/lyy/Helios/DMD-System/worktrees/joint-planner-20260907')
P='/share/project/shuchenweng/yongyan_liu'
old=json.loads((E/'manifests/deployment-plan.json').read_text())
assets=json.loads((E.parent/'native_macro_closed_loop_r2/configs/livewan-assets.json').read_text())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
assert head=='78e501bc1727013d94e356f697051b4d11cba4be'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
target=P+'/DMD-System_deploy/joint-livewan-observation-'+head[:12]+'-20260907'
remote=old['remote_root'];out=remote+'/native-inputs'
node=assets['nodes'][0]
def ssh(command,**kw):
 return subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',node['host'],command],**kw)
pack=subprocess.check_output(['git','pack-objects','--stdout','--revs'],cwd=R,input=(head+'\n^'+old['source_head']+'\n').encode())
cmd=f'''set -eu
test ! -e {target}
git clone --shared --no-checkout {old['deployment']} {target}
git -C {target} unpack-objects -q
git -C {target} checkout --detach {head}
oldsource={P}/DMD-System_deploy/joint-native-macros-06d5e67aab7e-20260907
for mod in livewan wan21; do
 test -z "$(git -C $oldsource/third_party/$mod status --porcelain)"
 cp -a $oldsource/third_party/$mod/. {target}/third_party/$mod/
 test "$(git -C {target}/third_party/$mod rev-parse --show-toplevel)" = {target}/third_party/$mod
 pin=$(git -C {target} ls-tree HEAD third_party/$mod | awk '{{print $3}}')
 test "$(git -C {target}/third_party/$mod rev-parse HEAD)" = "$pin"
done
test -z "$(git -C {target} status --porcelain)"
'''
p=ssh(cmd,input=pack,capture_output=True,timeout=180)
(E/'logs/observation-deploy.stdout').write_bytes(p.stdout);(E/'logs/observation-deploy.stderr').write_bytes(p.stderr)
p.check_returncode()
for n in assets['nodes']: n['repo_root']=target
assets.update(run_root=remote+'/native',master_port=29938,timeout_seconds=2400,
              node_local_cache_root='/tmp/lyy-experiments/DMD-System/joint_livewan_actions_20260907_r1')
payload=dict(head=head,target=target,out=out,assets=assets)
code=r'''
import copy,json,subprocess,sys
from pathlib import Path
from distillschedule.workloads.best_known.registry import fingerprint,file_sha256,write_new_json,git_head
from distillschedule.workloads.best_known.dispatch import validate_local
from distillschedule.search.execution import resolve_candidate
x=json.load(sys.stdin);root=Path(x['target']);out=Path(x['out']);a=x['assets']
assert git_head(root)==x['head']
out.mkdir(parents=True,exist_ok=False)
old=copy.deepcopy(a['assets'])
for k in ('source_receipt','bootstrap','trace'):
 assert file_sha256(Path(old[k]['path']))==old[k]['sha256']
s=json.loads(Path(old['source_receipt']['path']).read_text());b=json.loads(Path(old['bootstrap']['path']).read_text())
assert b['source_receipt']['sha256']==s['source_receipt_sha256']
assert fingerprint({k:v for k,v in s.items() if k!='source_receipt_sha256'})==s['source_receipt_sha256']
assert fingerprint({k:v for k,v in b.items() if k!='bootstrap_sha256'})==b['bootstrap_sha256']
ns=out/'source_receipt.json'
command=['bash',str(root/'scripts/distillschedule/prepare_livewan_crossscale_source.sh'),'--output',s['source_root'],'--receipt',str(ns)]
with (out/'source-prepare.log').open('x') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90)
new=json.loads(ns.read_text())
for k in ('schema_version','status','livewan_commit','wan21_commit','source_root'):assert new[k]==s[k]
assert new['files'].keys()==s['files'].keys()
for k in s['files']:
 for f in ('bytes','sha256'):assert s['files'][k][f]==new['files'][k][f],(k,f)
derived=copy.deepcopy(b);derived['source_receipt']={'path':str(ns),'sha256':new['source_receipt_sha256']}
derived.pop('bootstrap_sha256');derived['bootstrap_sha256']=fingerprint(derived)
assert {k for k in derived if derived[k]!=b[k]}=={'source_receipt','bootstrap_sha256'}
nb=out/'bootstrap.json';write_new_json(nb,derived)
for k,p in [('source_receipt',ns),('bootstrap',nb)]:a['assets'][k]={'path':str(p),'sha256':file_sha256(p)}
ap=out/'assets.json';write_new_json(ap,a)
plan=resolve_candidate('livewan_crossscale_pipeline_best','joint-livewan1-pipeline-m1-g1-r1',root=root,assets_path=ap,
    run_id='jointp_livewan_actions_20260907_r1-native-observation',mode='action-observation')
checked=validate_local(plan,root=root)
assert all(n['environment']['LIVEWAN_TRAJECTORY_CYCLES']=='2' and n['environment']['DISTILLSCHEDULE_LIVEWAN_OPERATION_CALIBRATION']=='1' for n in plan['nodes'])
write_new_json(out/'execution-plan.json',plan)
proof=dict(status='PASS_SOURCE_CONFIG_ASSETS_ONLY',head=x['head'],target=str(root),assets=a,plan=plan,
 source_relocation_only=['source_receipt','bootstrap_sha256'],payload_identity_preserved=True,validation=checked,
 training='NOT_RUN',formal_timing='NOT_RUN',native_model_script_unchanged=True)
write_new_json(out/'prepare-receipt.json',proof)
print(json.dumps(proof))
'''
command=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+target+'/src:'+target,
                   'LD_LIBRARY_PATH='+node['python_env']+'/lib',node['python_env']+'/bin/python','-c',code])
p=ssh(command,input=json.dumps(payload),capture_output=True,text=True,timeout=180)
(E/'logs/observation-prepare.stdout').write_text(p.stdout);(E/'logs/observation-prepare.stderr').write_text(p.stderr)
p.check_returncode();result=json.loads(p.stdout)
(E/'configs/native-observation-plan.json').write_text(json.dumps(result['plan'],indent=2)+'\n')
(E/'manifests/native-observation-prepare.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'head':head,'target':target,'plan':'configs/native-observation-plan.json'}))
