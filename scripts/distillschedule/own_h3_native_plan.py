"""Native H3 continuous/Grid comparison bounded owner, identical sealed lifecycle policy. Reuses sealed R9 birth/permit/reap guards.

No Torch import here, no borrowed GPU PID authority, no global cleanup. New
workload contracts replace (not bypass) old hard-coded Student1/Fake5 gates.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, importlib.util, json, os, pathlib, signal, socket, subprocess, sys, time
from types import SimpleNamespace
P=pathlib.Path
B=P('/share/project/shuchenweng/yongyan_liu')
OP=B/'research_artifacts/DMD-System/five_workload_cycle2_speedup_20260930/rounds/helios_optimizer_portfolio_20261002_r2'
R9=OP/'h3_lora_swiglu_consumer_fusion_20261005_r1/ffn_operation_markers_r1/cuda_qualifier_r2/scripts/bounded_owner_verified_ssh_ancestor_r9.py'
R9_SHA='e40ac3518c481e163040e77c338b00b8ab57c4ead9c3591692dca4634acace86'
GUARD=OP/'reports/h3_native_qkv_owner_20261005_r1/scripts/native_qkv_prefix_owner_r2.py'
GUARD_SHA='91333522d660c8ebfaa1328a933757536d82d86a9ffa48481453770676dc6bd0'
ROOT_DONOR=OP/'reports/priority_followup_20261007_r2/root_H3_split_release_native_two_cycle_20261008_r3/scripts/root_owner_draft_r1.py'

def require(ok,why):
 if not ok: raise RuntimeError(why)
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def write(p,row):
 # Publish a complete immutable JSON object, preserving exclusive no-overwrite.
 # Peer readers must never observe a zero-length/incomplete ready receipt.
 p=P(p);tmp=p.with_name(p.name+'.publish.'+str(os.getpid()))
 try:
  with tmp.open('x') as f:
   json.dump(row,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,p)  # atomic visibility, FileExistsError still fails closed
 finally:
  if tmp.exists():tmp.unlink()
def load(p,digest,name):
 require(sha(p)==digest,'SEALED_HELPER:'+str(p))
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 require('torch' not in sys.modules,'OWNER_MUST_NOT_IMPORT_TORCH');return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--node',type=int,choices=(0,1),required=True);ap.add_argument('--execute',action='store_true')
 a=ap.parse_args();require(sha(a.manifest)==a.manifest_sha256,'MANIFEST_SHA');cfg=json.loads(P(a.manifest).read_text())
 require(sha(__file__)==cfg['owner_sha256'],'OWNER_SHA')
 r9=load(R9,R9_SHA,'ar_r9');helper=r9.load_module(r9.HELPER,r9.HELPER_SHA,'ar_pid_helper');old=r9.load_module(r9.TRACKER,r9.TRACKER_SHA,'ar_tracker')
 guard=load(GUARD,GUARD_SHA,'ar_assignment_guard');donor=load(ROOT_DONOR,cfg['lifecycle_donor_sha256'],'ar_fd_reconcile')
 require(cfg['kind']=='native_h3_serialized_plan_v1','EXPLICIT_USER_H3_COMPARISON')
 require(cfg['cycles']==2 and cfg['numerical_status']=='GRID1000_REVIEW_POLICY_ONLY','TWO_CYCLE_DIAGNOSTIC_NUMERIC_BOUNDARY')
 require(socket.gethostname()==cfg['hosts'][a.node],'PINNED_JOB_HOST')
 require(sha(cfg['python'])==cfg['python_sha256'],'PYTHON_IDENTITY')
 source=P(cfg['source_root']);head=cfg['source_head']
 require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==head,'EXACT_REPO_HEAD')
 require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),'CLEAN_REPO')
 from historical_ssh_read_only_r2 import install as bind_historical_transport
 historical_transport=bind_historical_transport(r9,cfg,a.node)
 # Diagnostics only: keep the original FD rejection; identify its exact witness.
 original_unreadable = r9.unreadable_verified_ssh_ancestor
 def diagnosed_unreadable(helper, pid, chain, telemetry, path, exc):
  try:return original_unreadable(helper,pid,chain,telemetry,path,exc)
  except BaseException:
   witness={'pid':pid,'birth':helper.pid_identity(pid),'path':str(path),'errno':exc.errno,
            'original_policy':'REJECTION_UNCHANGED_NO_NEW_SIGNAL_AUTHORITY'}
   try:witness['comm']=P('/proc',str(pid),'comm').read_text().strip()
   except OSError:pass
   print(json.dumps({'stage':'FD_UNREADABLE_REJECTION_WITNESS','witness':witness}),flush=True)
   raise
 r9.unreadable_verified_ssh_ancestor=diagnosed_unreadable
 # Monitor-only deadline compatibility: no sealed helper or workload changed.
 original_command=r9.command
 r9.command=lambda argv,timeout=30:original_command(argv,timeout=max(timeout,90))
 for name,h in cfg['file_sha256'].items():require(sha(name)==h,'PINNED_SOURCE_ASSET:'+name)
 for node in (0,1):
  evidence=cfg['qualification']['source_cpu_'+str(node)]
  require(sha(evidence['path'])==evidence['sha256'],'H3_CPU_QUALIFICATION_SHA')
  row=json.loads(P(evidence['path']).read_text())
  require(row['status']=='PASS_H3_SELECTED_PLAN_CPU_NO_GPU' and row['source_head']==head,'CURRENT_CPU_IMPORT_PASS')
  require(row['candidate_id']==cfg['candidate_id'] and row['selected_plan_sha256']==cfg['selected_plan']['sha256'],'CPU_SELECTED_PLAN_BINDING_FRESH_PROCESS')
 from selected_plan_contract import validate_owner_config
 validate_owner_config(cfg,a.node)
 if cfg.get('owned_worker_oom_fail_fast', False):
  import native_worker_oom_guard_r1 as owned_oom_guard
  require(sha(owned_oom_guard.__file__)==cfg['file_sha256'][owned_oom_guard.__file__],'PREEXEC_OWNED_OOM_HELPER_PIN')
 if cfg.get('verified_zombie_fd_rescan', False):
  import verified_zombie_fd_rescan_r1 as zombie_rescan
  require(sha(zombie_rescan.__file__)==cfg['file_sha256'][zombie_rescan.__file__],'PREEXEC_ZOMBIE_HELPER_PIN')
 if not a.execute:
  print(json.dumps({'status':'PASS_PLAN_ONLY_GPU_NOT_RUN','run_id':cfg['run_id'],'source_head':head}));return 0
 root=P(cfg['run_root']);local=root/('node_'+str(a.node));local.mkdir(parents=True,exist_ok=False)
 coordination=root/'coordination';coordination.mkdir(exist_ok=True)
 rec={'run_id':cfg['run_id'],'node':a.node,'source_head':head,'manifest_sha256':a.manifest_sha256,'status':'NOT_RUN','cleanup_pass':False,'phases':[]}
 rec['historical_transport_read_only_qualification']=historical_transport
 lock=open('/tmp/distillschedule_gpu_node.lock','a');trackers=[];start=None;cancelled=[];log=None
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda s,f:cancelled.append(s))
 r9.UUID=cfg['ordered_UUIDs_by_node'][a.node][0]
 def remaining():require(start is not None and not cancelled and time.monotonic()-start<cfg['work_seconds'],'BOUNDED_RUN_OR_CANCELLED')
 def peer_check():
  if cfg['kind']=='native_h3_serialized_plan_v1':require(not(coordination/('abort_'+str(1-a.node)+'.json')).exists(),'PEER_ABORT')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();rec['lock_held']=True
  rec['subreaper']=r9.enable_subreaper();rec['controller_birth']=helper.pid_identity(os.getpid())
  require(rec['controller_birth'] and not r9.direct_children(helper,rec['controller_birth']),'FRESH_OWNER_NO_CHILDREN')
  ancestors=r9.bind_ssh_ancestors(helper,rec['controller_birth']);rec['bound_ancestor_chain']=ancestors
  rec['before']=r9.audit();rec['cgroup_before']=helper.events()
  inventory=guard.assigned_inventory_compat(r9,SimpleNamespace(HOSTS=cfg['hosts']),cfg,a.node,rec['before']);rec['assigned_device_inventory']=inventory
  require(r9.idle(rec['before']),'ALL8_IDLE_UNDER_HELD_LOCK')
  telemetry=[];handles=r9.device_handles(helper,ancestors,telemetry);require(not handles,'FOREIGN_GPU_FD_NO_SIGNAL_NO_RUN')
  rec['initial_FD_audit']={'handles':handles,'telemetry':telemetry}
  current=int(P('/sys/fs/cgroup/memory.current').read_text());maximum=int(P('/sys/fs/cgroup/memory.max').read_text())
  require(maximum-current>=64<<30,'HOST_CGROUP_RESERVE_64GIB')
  if cfg['kind']=='native_h3_serialized_plan_v1':
   sock=socket.socket();sock.bind(('0.0.0.0',cfg['port']));sock.close()
   write(coordination/('ready_'+str(a.node)+'.json'),{'manifest_sha256':a.manifest_sha256,'node':a.node,'controller':rec['controller_birth'],'lock_held':True})
   deadline=time.monotonic()+90
   peer=coordination/('ready_'+str(1-a.node)+'.json')
   while not peer.exists():remaining();peer_check();require(time.monotonic()<deadline,'PEER_READY_DEADLINE');time.sleep(.25)
   require(json.loads(peer.read_text())['manifest_sha256']==a.manifest_sha256,'PEER_MANIFEST_MISMATCH')
  env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG','NVIDIA_VISIBLE_DEVICES') if k in os.environ}
  env.update(cfg['environment_by_node'][a.node]);P(env['TMPDIR']).mkdir(parents=True,exist_ok=True);env.update(DISTILLSCHEDULE_RUN_ID=cfg['run_id'],DISTILLSCHEDULE_EXPECTED_SOURCE_RECEIPT=head,DISTILLSCHEDULE_RUN_ROOT=str(root))
  env['DMD_SELECTED_PLAN_MANIFEST_SHA256']=a.manifest_sha256
  env['CUDA_VISIBLE_DEVICES']=','.join(cfg['ordered_UUIDs_by_node'][a.node])
  argv=[cfg['python'],'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=8','--max-restarts=0','--node-rank='+str(a.node),'--master-addr='+cfg['master_addr'],'--master-port='+str(cfg['port']),cfg['worker_script']]
  argv += cfg.get('worker_flags', [])
  if cfg.get('profile', False):
   profiler = cfg['nsys']
   require(sha(profiler) == cfg['nsys_sha256'], 'NSYS_BINARY_IDENTITY')
   argv = [profiler, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
           '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--wait=all',
           '--trace-fork-before-exec=true', '--force-overwrite=false',
           '--output=' + str(local / 'profile')] + argv
  phase={'phase':cfg['kind'],'argv':argv};rec['phases'].append(phase);tracker=r9.build_tracker(old,helper,phase,rec['controller_birth']);trackers.append(tracker)
  rec['environment']=env;log=(local/'worker.log').open('x')
  permit=local/'exec_permit.json';command=[cfg['python'],'-I','-S','-c',old.BOOTSTRAP_CODE,str(permit),json.dumps(argv),a.manifest_sha256]
  # Last moment audit before the sealed bootstrap may execute Torch.
  state=r9.audit();require(r9.idle(state),'GPU_BECAME_BUSY_BEFORE_POPEN')
  require(not r9.device_handles(helper,ancestors,[]),'GPU_FD_APPEARED_BEFORE_POPEN');remaining();peer_check()
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=source);tracker.attach(child);tracker.issue_permit(permit,argv,a.manifest_sha256)
  print(json.dumps({'run_id':cfg['run_id'],'node':a.node,'stage':'WORKER_STARTED','PID':child.pid}),flush=True)
  next_audit=0;next_status=0;min_free=[None]*8;host_peak=current
  while tracker.child_running():
   tracker.live();remaining();peer_check();donor.sticky_discovery_check(trackers,rec)
   if time.monotonic()>=next_audit:
    state=r9.audit();state['unreadable_verified_ssh_ancestors']=[]
    state['device_handle_owners']=r9.device_handles(helper,ancestors,state['unreadable_verified_ssh_ancestors'])
    live=tracker.live();live=donor.reconcile_observed_fd_births(r9,helper,tracker,state,live,rec['controller_birth'])
    state,live=donor.refresh_ended_fd_observations(r9,helper,tracker,state,live,rec['controller_birth'],ancestors)
    if cfg.get('verified_zombie_fd_rescan', False):
     from verified_zombie_fd_rescan_r1 import refresh as refresh_verified_zombie
     state,live=refresh_verified_zombie(r9,helper,tracker,state,live,rec['controller_birth'],ancestors,enabled=True)
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    state,live=refresh_finished_probe(r9,helper,tracker,state,live,rec['controller_birth'],ancestors)
    try:
     donor.exclusive_all8(r9,helper,inventory,state,live)
    except BaseException:
     # Diagnostic snapshot only. Original ownership rejection propagates.
     rec['failed_exclusive_snapshot'] = state
     rec['failed_exclusive_live_births'] = dict(live)
     rec['failed_exclusive_known_births'] = dict(tracker.known)
     rec['failed_exclusive_monotonic'] = time.monotonic()
     print(json.dumps({'stage':'OWNERSHIP_GUARD_REJECTED', 'node':a.node,
                       'FD_birth_reconciliation':state.get('FD_birth_reconciliation'),
                       'FD_snapshot_exit_reconciliation':state.get('FD_snapshot_exit_reconciliation'),
                       'unmatched_handle_births':[h['birth'] for h in state['device_handle_owners']
                        if not r9.same_birth(live.get(h['birth']['pid']),h['birth'])]}),flush=True)
     raise
    free=r9.command(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'])
    require(free['rc']==0,'NVML_FREE_QUERY')
    free_rows=list(csv.reader(free['stdout'].splitlines()));require(len(free_rows)==8,'NVML_FREE_ALL8')
    for device in free_rows:
     index=int(device[0]);value=int(device[1])<<20;min_free[index]=value if min_free[index] is None else min(min_free[index],value)
    mem=int(P('/sys/fs/cgroup/memory.current').read_text());host_peak=max(host_peak,mem)
    require(helper.events()['oom']==rec['cgroup_before']['oom'] and helper.events()['oom_kill']==rec['cgroup_before']['oom_kill'],'CGROUP_OOM_DETECTED')
    rec.setdefault('audits',[]).append({'time':time.monotonic(),'state':state,'cgroup_current':mem})
    if cfg.get('owned_worker_oom_fail_fast', False):
     from native_worker_oom_guard_r1 import read as read_owned_worker_oom
     failure=read_owned_worker_oom(local/'worker.log')
     if failure:
      rec['worker_cuda_oom_evidence']=failure
      raise RuntimeError('ACTUAL_OWNED_WORKER_CUDA_OOM_NO_TIMING')
    with (local/'worker.log').open('rb') as fatal_log:
     fatal_log.seek(max(0,(local/'worker.log').stat().st_size-262144));fatal_tail=fatal_log.read().decode(errors='replace')
    if 'OSError: AF_UNIX path too long' in fatal_tail:
     rec['worker_runtime_failure']='DATALOADER_AF_UNIX_PATH_TOO_LONG'
     raise RuntimeError('ACTUAL_OWNED_WORKER_AF_UNIX_STARTUP_FAILURE_NO_TIMING')
    next_audit=time.monotonic()+5
   if time.monotonic()>=next_status:
    paths=sorted((root/'receipts').glob('*_final.json')) if (root/'receipts').exists() else []
    try:
     with (local/'worker.log').open('rb') as progress_log:
      progress_log.seek(max(0,(local/'worker.log').stat().st_size-32768));tail=progress_log.read().decode(errors='replace')
     progress=[line[-2000:] for line in tail.splitlines() if '"phase"' in line or 'OutOfMemory' in line or 'Error' in line][-3:]
    except OSError:progress=[]
    print(json.dumps({'run_id':cfg['run_id'],'node':a.node,'stage':'RUNNING','elapsed':time.monotonic()-start,'completed_rank_cycle_receipts':len(paths),'cgroup_current':mem,'last_worker_progress':progress}),flush=True);next_status=time.monotonic()+60
   time.sleep(.5)
  require(tracker.reap_terminal() and phase['worker_rc']==0,'WORKER_NONZERO_OR_UNREAPED')
  deadline=time.monotonic()+30
  while not tracker.release_ready():require(time.monotonic()<deadline,'SUCCESS_RELEASE_TIMEOUT');tracker.reap_terminal();time.sleep(.2)
  donor.sticky_discovery_check(trackers,rec)
  for rank in range(a.node*8,(a.node+1)*8):
   path=root/'execution/ranks'/('rank_%03d.json'%rank)
   require(path.is_file(),'ALL8_FINAL_RANK_RECEIPTS')
   row=json.loads(path.read_text())
   require(row['status']=='PASS_NATIVE_RUNTIME_CONTRACTS' and row['rank']==rank,'ACTUAL_NATIVE_H3_RANK_PASS')
   require(row['final_versions']==cfg['expected_final_versions'],'G_C_EMA_VERSION_CONTRACT')
   require(len(row['cycles'])==cfg['cycles'] and all(x['finite_losses'] for x in row['cycles']),'BOTH_NATIVE_CYCLES_FINITE')
   from selected_plan_contract import validate_rank
   validate_rank(cfg,row)
  rec['host_cgroup_sampled_peak']=host_peak;rec['driver_free_sampled_min_bytes']=min_free;rec['driver_free_sampling']='actual memory.free query, approximately 5-second samples; micropeaks not guaranteed'
  rec['status']='PASS_LOCAL_OWNED_RUN_PENDING_WORLD_GATE'
 except BaseException as exc:
  rec['status']='FAIL_OWNED_NATIVE_RUN';rec['error']=repr(exc);rec['error_traceback']=__import__('traceback').format_exc()
  try:write(coordination/('abort_'+str(a.node)+'.json'),{'error':repr(exc),'manifest_sha256':a.manifest_sha256})
  except BaseException as e:rec['abort_write_error']=repr(e)
 finally:
  if start is not None:
   for tracker in trackers:
    for sig,secs in ((signal.SIGTERM,8),(signal.SIGKILL,8)):
     deadline=time.monotonic()+secs
     while tracker.safety_pending() and time.monotonic()<deadline:tracker.signal_owned(sig);tracker.reap_terminal();time.sleep(.25)
    while not tracker.release_ready():
     rec['status']='FAIL_SAFETY_HOLD';tracker.signal_owned(signal.SIGKILL);tracker.reap_terminal();time.sleep(.5)
   try:
    rec['after']=r9.audit();rec['cgroup_after']=helper.events();rec['oom_delta']={k:rec['cgroup_after'][k]-rec['cgroup_before'][k] for k in ('oom','oom_kill')}
    final_telemetry=[];final_handles=r9.device_handles(helper,rec.get('bound_ancestor_chain',[]),final_telemetry)
    rec['terminal_FD_audit']={'handles':final_handles,'telemetry':final_telemetry}
    if 'assigned_device_inventory' in rec:r9.validate_assignment_match(rec['assigned_device_inventory'],rec['after'])
    require(not final_handles,'FINAL_GPU_HANDLES_UNRESOLVED');require(all(t.release_ready() for t in trackers),'OWNED_REAP_UNRESOLVED')
    donor.sticky_discovery_check(trackers,rec)
    rec['cleanup_pass']=r9.idle(rec['after']) and not any(rec['oom_delta'].values())
   except BaseException as e:rec['cleanup_pass']=False;rec['cleanup_error']=repr(e)
   rec['held_seconds']=time.monotonic()-start
   if not rec['cleanup_pass'] or rec['held_seconds']>cfg['lease_seconds']:rec['status']='FAIL_CLEANUP_OR_LEASE'
  if log:log.close()
  write(local/'lifecycle.json',rec);lock.close()
 print(json.dumps({k:rec.get(k) for k in ('run_id','node','status','cleanup_pass','held_seconds','oom_delta','error')}),flush=True)
 return 0 if rec['status']=='PASS_LOCAL_OWNED_RUN_PENDING_WORLD_GATE' else 1
if __name__=='__main__':raise SystemExit(main())
