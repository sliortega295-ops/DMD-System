"""Fresh same-revision control/retention, existing H100 launcher and gates."""
from pathlib import Path
import datetime
import json
import signal
import subprocess
import traceback
import sys
import hashlib

BASE_FILE = Path(__file__).resolve().parents[3] / "scripts/run_helios_official_swap_matched_r1.py"
assert hashlib.sha256(BASE_FILE.read_bytes()).hexdigest() == "9542fb6c68e5ff391b96a8f69466aa358a696a504bfee6a5c5c3b1ce3d50a9e6", "BASE_RUNNER_SOURCE_CHANGED"
sys.path.insert(0, str(BASE_FILE.parent))

import run_helios_official_swap_matched_r1 as base

base.REPO=base.ROOT/'repos/DMD-System-h100-helios-nograd-retain-20261015-r1'
base.HEAD='8fc70e8ce9f6e0243604812120119c9f4c078e55'
PROOF=base.PREP/'artifacts/helios_hsdp_retention_api_GPU_r1/result.json'


def source_admission():
    assert subprocess.check_output(['git','-C',str(base.REPO),'rev-parse','HEAD'],text=True).strip()==base.HEAD
    assert not subprocess.check_output(['git','-C',str(base.REPO),'status','--porcelain'],text=True).strip()
    gate=json.load(open(base.PREP/'artifacts/helios_h100_rollout_fence_probe_20261015_r1/manifests/committed_source_r1.json'))
    assert gate['status']=='PASS_CLEAN_COMMITTED_CPU_ONLY' and gate['head']==base.HEAD
    assert gate['tests']==dict(tests=68,errors=0,failures=0,skipped=0)
    assert gate['training_math_changed'] is False
    for name,digest in gate['files'].items():assert base.sha(base.REPO/name)==digest
    assert gate['actual_prior_GPU_API_receipt']=='52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966'
    assert base.sha(PROOF)=='52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966'
    prior=base.PREP/'artifacts/helios_h100_retention_cli_fix_20261015_r1/reports/completed_native_pair_r1/result.json'
    assert base.sha(prior)=='d70f468ffee9af32f84e0285bd02d4e373c5b9c8dfb09838ee787e49593b442f'
    assert gate['actual_prior_full14B_retention_result_sha256']==base.sha(prior)
    assert json.load(open(prior))['owned_cleanup']=='PASS_BOTH_ARMS_EMPTY'
    assert 'torch' not in sys.modules, 'CPU_ADMISSION_IMPORTED_TORCH'
    return dict(status='PASS_CPU_STANDALONE_SOURCE_NO_MODEL',head=base.HEAD,base_runner=str(BASE_FILE),base_runner_sha256=base.sha(BASE_FILE),source_files_verified=True,CUDA_initialized=False,GPU='NOT_RUN',deadline_seconds=14400)


class Runner(base.Runner):
    def __init__(self):
        self.workload='no-gan'
        self.out=base.PREP/'artifacts/helios_h100_rollout_fence_probe_20261015_r1/matched_pair_r1'
        self.out.mkdir(exist_ok=False);(self.out/'logs').mkdir()
        self.ordinal=0;self.records=[]
        base.dump(self.out/'manifest.json',{'head':base.HEAD,'repo':str(base.REPO),
                  'scope':'same-revision retained-G control versus only existing redundant rollout fence elision',
                  'mesh':[2,4],'global_batch':16,'physical_ranks':8,'replications':1,
                  'order':['control','candidate'],'strict_numeric_parity':'NOT_RUN',
                  'reused_official_swap_baseline_seconds':138.3719482459128,
                  'frozen_math_worker_changed':False,'GPU':'NOT_RUN'})

    def status(self,phase,state,**fields):
        base.dump(self.out/'status.json',dict(run_id='helios_h100_retained_fence_probe_20261015_r1',
            phase=phase,status=state,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            exact_head=base.HEAD,strict_claimable=False,**fields),immutable=False)

    def admit(self):
        source_admission()
        self.invoke('inherited_lease',[base.PYTHON,base.REPO/'scripts/distillschedule/verify_inherited_gpu_locks.py',
                    '--project-root',base.ROOT,'--asset-fd','7'],deadline=30)

    def run_pair(self):
        env=base.clean_env();env['DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT']=str(PROOF)
        env.update(HELIOS_RUNTIME_ROOT=str(base.ROOT/'repos/frozen-helios-wan-a100-5d3ffe37'),
                   DISTILLSCHEDULE_ASSET_ROOT=str(base.ASSETS),HF_HOME=str(base.ASSETS/'frozen_a100_wan/hf_home_official_torch210'),
                   DISTILLSCHEDULE_MASTER_PORT='28791',
                   PYTHONPATH=f'{base.REPO}/src:{base.REPO}:{base.ROOT}/repos/frozen-helios-wan-a100-5d3ffe37')
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
        gate_dir=self.out/'asset_gate'
        self.invoke('fresh_live_asset_gate',[base.PYTHON,base.RUNS/'helios_gate_inputs_r1/scripts/build_helios_h100_asset_gate_r1.py',
                    '--repo-root',base.REPO,'--asset-root',base.ASSETS,'--expected-head',base.HEAD,
                    '--plan',base.RUNS/'helios_gate_inputs_r1/manifests/helios_asset_transfer_gzip_plan_r1.json',
                    '--plan-sha256','629869a547249af8c7616ae40f816d5bb41df3b8692bbec09e253397cd0bfd04',
                    '--shared-source',base.RUNS/'helios_gate_inputs_r1/manifests/helios_shared_source_full_sha_r1.json',
                    '--shared-source-sha256','73da6db0e90b5f1ddaea55b44fc0c7c2ecc61953a65b7b43cde076270184b0d6',
                    '--variant','no-gan','--output-dir',gate_dir],cpu_env,deadline=1200)
        gate=gate_dir/'asset_gate.json';launcher=base.REPO/'scripts/distillschedule/launch_helios_hardware_p0_node.sh'
        arms={'control':'h100-nograd-retain','candidate':'h100-nograd-retain-nosync'};correctness={}
        for mode in ('validate','correctness','timing'):
            for label,arm in arms.items():
                rid=f'helios_h100_fence_{label}_{mode}_20261015_r1'
                child=dict(env)
                if mode=='timing':child.update(DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SUMMARY=str(correctness[label]),DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256=base.sha(correctness[label]))
                self.invoke(rid,['bash',launcher,base.ROOT,rid,'no-gan',mode,arm,base.HEAD,gate,base.sha(gate)],child,deadline=4200,gpu=True)
                path=base.RUNS/rid/'summary.json';value=json.load(open(path))
                assert value['status']=='PASS' and value['physical_rank_count']==8 and value['logical_slots']==list(range(16))
                if mode!='validate':assert value['logical_final_versions']=={'generator':3,'critic':15,'ema':15}
                if mode!='validate':
                    for rank in range(8):
                        row=json.load(open(base.RUNS/rid/'execution/ranks'/f'rank_{rank:02d}.json'))
                        fence=row['rollout_fence_policy'];elide=label=='candidate'
                        assert fence['elide_explicit_rollout_sync'] is elide and fence['outer_full5_boundary_sync_preserved']
                        assert fence['generator_update_empty_cache_count']==(0 if elide else 2)
                        assert fence['generator_update_sync_count']==(0 if elide else 2)
                        assert fence['critic_rollout_sync_count']==(0 if elide else 10)
                        persistence=row['generator_nograd_persistent_unshard'];assert persistence['enabled']
                        for m in persistence['microbatches']:
                            state=m['persistence'];assert state['restored'] and state['unsharded_after_exit_count']==0 and state['fsdp_module_count']==41
                            assert state['flush_allocator_cache'] is True

                if mode=='correctness':assert value['execution_correctness_admitted'];correctness[label]=path
                if mode=='timing':self.record_trial(0,label,rid,value,value['full5_seconds_max_rank'],path)
        rows={v['arm']:v for v in self.records};assert set(rows)=={'control','candidate'}
        report={'status':'PASS_SAME_REVISION_SINGLE_DIAGNOSTIC_PAIR','head':base.HEAD,'records':self.records,
                'control_seconds':rows['control']['seconds'],'candidate_seconds':rows['candidate']['seconds'],
                'single_fence_elision_ratio':rows['control']['seconds']/rows['candidate']['seconds'],
                'frozen_official_swap_reference_seconds':138.3719482459128,
                'diagnostic_ratio_to_frozen_official_swap':138.3719482459128/rows['candidate']['seconds'],
                'strict_full_state_parity':'NOT_RUN','replications':1,'CV':'NOT_MEASURED',
                'all_validation_correctness_timing_same_head':True}
        base.dump(self.out/'result.json',report)
        print(json.dumps(report),flush=True)


def main():
    def interrupted(sig,frame):raise InterruptedError(f'owned retention campaign interrupted {sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(14400)
    runner=Runner()
    try:runner.admit();runner.run_pair()
    except BaseException as error:
        base.dump(runner.out/'failure.json',{'status':'FAIL_PRESERVED','error':repr(error),'traceback':traceback.format_exc(),'completed_trials':runner.records})
        raise


if __name__=='__main__':
    if sys.argv[1:] == ['--cpu-admit']:
        print(json.dumps(source_admission()),flush=True)
    elif sys.argv[1:]:
        raise SystemExit('unknown runner argument')
    else:
        main()

