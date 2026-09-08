"""One explicitly selected stage, source-bound NativeTrialRunner, never auto-retry."""
import argparse,json,subprocess,sys,traceback
from pathlib import Path
E=Path(__file__).resolve().parents[1]
R=Path('/home/lyy/Helios/DMD-System/worktrees/joint-planner-20260907')
sys.path[:0]=[str(R/'src'),str(R)]
from distillschedule.search.helios_implementation import NO_GAN,GAN,HeliosImplementation
from distillschedule.search.helios_action_costs import PRECISION,anchors
from distillschedule.search.execution import NativeTrialRunner
from distillschedule.search.runtime_identity import observed_binding
from distillschedule.workloads.best_known.registry import fingerprint,write_new_json
p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True)
p.add_argument('--variant',choices=('no-gan','gan'),required=True)
p.add_argument('--stage',choices=('prepare','calibrate','timing'),required=True)
p.add_argument('--candidate');p.add_argument('--unit-id',required=True);a=p.parse_args()
d=json.loads(a.deployment.read_text());head=d['source_head']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==head
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
w=NO_GAN if a.variant=='no-gan' else GAN
base,inc=anchors(w);impl=HeliosImplementation.parse(a.candidate) if a.candidate else inc
assert impl.workload==w
assets=d['assets'][a.variant]
workspace=E/'controller'/head[:12]/a.variant;workspace.mkdir(parents=True,exist_ok=True)
unit=workspace/'units'/a.unit_id;unit.mkdir(parents=True,exist_ok=False)
identity=dict(source_head=head,asset_fingerprint=fingerprint(assets['assets']),precision_fingerprint=PRECISION)
runner=NativeTrialRunner(workload=w,identity=identity,assets=assets,output=workspace,
 run_prefix='jointp_helios_components_20260908_'+head[:7]+'_'+a.variant,
 reference_id=base.id,bootstrap_peer=inc.id)
write_new_json(unit/'intent.json',dict(status='NOT_RUN',stage=a.stage,variant=a.variant,head=head,
 candidate_id=impl.id,cycles=2,measured_cycle=2,nsight=False,prior_outputs='PRESERVED'))
try:
 probe=runner._rpc('resolve',workload=w,candidate_id=impl.id,assets=assets,
   run_id=runner.run_prefix+'-identity',mode='correctness')
 identity['semantic_fingerprint']=probe['workload_semantics_hash']
 nodes=[runner._rpc('runtime_identity',_node_rank=i,node_rank=i,receipt=probe) for i in (0,1)]
 identity.update(observed_binding(nodes,source_head=head))
 write_new_json(unit/'identity.json',dict(binding=identity,nodes=nodes))
 if a.stage=='prepare':
  result=runner._prepare(impl.candidate())
 elif a.stage=='calibrate':
  result=runner.calibrate(impl.candidate(),directory=unit/'calibration')
 else:
  timing=unit/'timing';timing.mkdir()
  result=runner(impl.candidate(),'native_two_cycle',timing)
 write_new_json(unit/'result.json',result)
 print(json.dumps(dict(stage=a.stage,variant=a.variant,status=result.get('status',result.get('semantics_status')),
  candidate_id=impl.id,cycle_wall_seconds=result.get('cycle_wall_seconds'),evidence=str(unit))),flush=True)
except BaseException as err:
 write_new_json(unit/'failure.json',dict(status='FAIL',stage=a.stage,variant=a.variant,head=head,
   error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc(),
   next_action='PRESERVE_AND_DIAGNOSE_NO_AUTOMATIC_RETRY'))
 raise
