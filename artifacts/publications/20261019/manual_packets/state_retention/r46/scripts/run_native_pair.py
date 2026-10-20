"""Single source manual control/bank. Original two owner streams per phase."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys
R=Path(__file__).resolve().parents[1];B=R.parent
HEAD='69a9464997de16deafd5786338f538ba7930d9a5'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
 ready=read(R/'manifests/native-ready.json');cfgpath=R/'configs/config.json';cfg=read(cfgpath)
 assert ready['source']['head']==cfg['expected_head']==HEAD
 assert sha(cfgpath)==ready['config']['sha256']and sha(R/'configs/admission.json')==ready['admission']['sha256']
 if not a.execute:print('PLAN_ONLY_NO_GPU');return
 assert '18 passed'in(R/'logs/local-counter-consumer.log').read_text()
 target=read(R/'manifests/target-checkpoint-both.json')
 assert all(x['rc']==0 and '18 passed'in x['tail']and x['CUDA_initialized']is False for x in target['nodes'])
 diagnosis=read(B/'r44/reports/conclusion.json')
 assert diagnosis['rank_count']==16 and diagnosis['actual_QKV_strides_match_component']is True
 assert (B/'r44/reports/nsys-ai-both.json').exists()
 assert read(B/'r43/reports/review-and-closure.json')['saved_s']<0
 sys.path.insert(0,str(R/'scripts/native_r2'))
 import h3_ar_match_controller as c
 c.load_config(cfgpath);assert sha(c.__file__)==cfg['scripts']['controller']['sha256']
 with (B/'r1/artifacts/gpu-scheduler.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert read(B/'reports/continuous_progress.json')['this_campaign_charged']==102
  assert not (B/'STOP_AFTER_CURRENT').exists()
  ledger=R/'native-launches.jsonl';assert not ledger.exists(),'failures count, no implicit retry'
  go=dict(status='GO_SAME_SOURCE_SAME_FA3_ONE_FACTOR_RESTORE_NATIVE_METADATA_BARRIER',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=HEAD,config_sha256=sha(cfgpath),prior_owner_phases=102,stage_cap=2,total_cap=104,profiler=False,cycles=2,scheduler_integration=False,capacity_policy_unchanged=cfg['capacity_policy'],workload='original W6 768x1344 previous56+current56frames,8steps,G1F1,continuous,LoRA64',same_FA3_backend_both_arms=True,external_source_packet=cfg['manual_packet'])
  write(R/'configs/GO-native.json',go)
  os.environ.update(CUDA_VISIBLE_DEVICES='',SSH_AUTH_SOCK='/run/user/1000/gcr/ssh')
  for i,phase in enumerate(c.PHASES,1):
   assert not (B/'STOP_AFTER_CURRENT').exists(),'root/user stop pending; do not start next phase'
   row=dict(number=102+i,stage_start=i,phase=phase,workload='W6',source=HEAD,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),failure_counts=True,profiler=False,cycles=2)
   with ledger.open('a')as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   subprocess.run([sys.executable,str(B/'r5/scripts/update_continuous_progress.py')],check=True)
   print('ROOT_START='+json.dumps(row),flush=True)
   result=c.launch(cfg,cfgpath,phase,sha(cfgpath),ready['config']['path'],admission_proof=R/'configs/admission.json')
   write(R/'artifacts'/f'{phase}-eof.json',result)
   print('ROOT_EOF='+json.dumps(dict(phase=phase,status=result['status'])),flush=True)
  print('NATIVE_PAIR_OWNERS_EOF_CPU_COLLECTION_REQUIRED',flush=True)
if __name__=='__main__':main()
