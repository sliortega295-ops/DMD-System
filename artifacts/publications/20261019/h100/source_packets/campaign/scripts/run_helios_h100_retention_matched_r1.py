"""Fresh same-revision control/retention, existing H100 launcher and gates."""
from pathlib import Path
import datetime
import json
import signal
import subprocess
import traceback

import run_helios_official_swap_matched_r1 as base

base.REPO=base.ROOT/'repos/DMD-System-h100-helios-nograd-retain-20261015-r1'
base.HEAD='a065830e550151c56934ec83771ea506708d0e14'
PROOF=base.PREP/'artifacts/helios_hsdp_retention_api_GPU_r1/result.json'


class Runner(base.Runner):
    def __init__(self):
        self.workload='no-gan'
        self.out=base.PREP/'artifacts/helios_h100_retention_matched_20261015_r1'
        self.out.mkdir(exist_ok=False);(self.out/'logs').mkdir()
        self.ordinal=0;self.records=[]
        base.dump(self.out/'manifest.json',{'head':base.HEAD,'repo':str(base.REPO),
                  'scope':'same-revision best versus G no-grad persistent unshard only',
                  'mesh':[2,4],'global_batch':16,'physical_ranks':8,'replications':1,
                  'order':['control','candidate'],'strict_numeric_parity':'NOT_RUN',
                  'reused_official_swap_baseline_seconds':138.3719482459128,
                  'frozen_math_worker_changed':False,'GPU':'NOT_RUN'})

    def admit(self):
        assert subprocess.check_output(['git','-C',str(base.REPO),'rev-parse','HEAD'],text=True).strip()==base.HEAD
        assert not subprocess.check_output(['git','-C',str(base.REPO),'status','--porcelain'],text=True).strip()
        gate=json.load(open(base.PREP/'manifests/helios_retention_source_CPU_r1.json'))
        assert gate['status']=='PASS_CPU_TESTS_AND_ACTUAL_GPU_API_RECEIPT_NO_MODEL' and gate['head']==base.HEAD
        assert base.sha(PROOF)=='52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966'
        self.invoke('inherited_lease',[base.PYTHON,base.REPO/'scripts/distillschedule/verify_inherited_gpu_locks.py',
                    '--project-root',base.ROOT,'--asset-fd','7'],deadline=30)

    def run_pair(self):
        env=base.clean_env();env['DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT']=str(PROOF)
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
        gate_dir=self.out/'asset_gate'
        self.invoke('fresh_live_asset_gate',[base.PYTHON,base.REPO/'scripts/distillschedule/verify_helios_h100_assets.py',
                    '--repo-root',base.REPO,'--asset-root',base.ASSETS,'--expected-head',base.HEAD,
                    '--plan',base.RUNS/'helios_gate_inputs_r1/manifests/helios_asset_transfer_gzip_plan_r1.json',
                    '--plan-sha256','629869a547249af8c7616ae40f816d5bb41df3b8692bbec09e253397cd0bfd04',
                    '--shared-source',base.RUNS/'helios_gate_inputs_r1/manifests/helios_shared_source_full_sha_r1.json',
                    '--shared-source-sha256','73da6db0e90b5f1ddaea55b44fc0c7c2ecc61953a65b7b43cde076270184b0d6',
                    '--variant','no-gan','--output-dir',gate_dir],cpu_env,deadline=1200)
        gate=gate_dir/'asset_gate.json';launcher=base.REPO/'scripts/distillschedule/launch_helios_hardware_p0_node.sh'
        arms={'control':'best','candidate':'h100-nograd-retain'};correctness={}
        for mode in ('validate','correctness','timing'):
            for label,arm in arms.items():
                rid=f'helios_h100_retention_{label}_{mode}_20261015_r1'
                child=dict(env)
                if mode=='timing':child.update(DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SUMMARY=str(correctness[label]),DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256=base.sha(correctness[label]))
                self.invoke(rid,['bash',launcher,base.ROOT,rid,'no-gan',mode,arm,base.HEAD,gate,base.sha(gate)],child,deadline=4200,gpu=True)
                path=base.RUNS/rid/'summary.json';value=json.load(open(path))
                assert value['status']=='PASS' and value['physical_rank_count']==8 and value['logical_slots']==list(range(16))
                if mode!='validate':assert value['logical_final_versions']=={'generator':3,'critic':15,'ema':15}
                if mode=='correctness':assert value['execution_correctness_admitted'];correctness[label]=path
                if mode=='timing':self.record_trial(0,label,rid,value,value['full5_seconds_max_rank'],path)
        rows={v['arm']:v for v in self.records};assert set(rows)=={'control','candidate'}
        report={'status':'PASS_SAME_REVISION_SINGLE_DIAGNOSTIC_PAIR','head':base.HEAD,'records':self.records,
                'control_seconds':rows['control']['seconds'],'candidate_seconds':rows['candidate']['seconds'],
                'single_retention_ratio':rows['control']['seconds']/rows['candidate']['seconds'],
                'frozen_official_swap_reference_seconds':138.3719482459128,
                'diagnostic_ratio_to_frozen_official_swap':138.3719482459128/rows['candidate']['seconds'],
                'strict_full_state_parity':'NOT_RUN','replications':1,'CV':'NOT_MEASURED',
                'all_validation_correctness_timing_same_head':True}
        base.dump(self.out/'result.json',report)
        print(json.dumps(report),flush=True)


def main():
    def interrupted(sig,frame):raise InterruptedError(f'owned retention campaign interrupted {sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    runner=Runner()
    try:runner.admit();runner.run_pair()
    except BaseException as error:
        base.dump(runner.out/'failure.json',{'status':'FAIL_PRESERVED','error':repr(error),'traceback':traceback.format_exc(),'completed_trials':runner.records})
        raise


if __name__=='__main__':main()
