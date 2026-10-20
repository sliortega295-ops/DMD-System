"""W4 orchestration only: fresh source/gates, unchanged comparator/timing."""
from pathlib import Path
import datetime,json,os,signal,subprocess,sys,traceback,types
O=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1');P=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w4_h100_teacher_resident_20261015_r1');REPO=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-livewan-teacher-resident-20261015-r1')
HEAD='4d7c589acd01a4a6c90ca141da266b3090df2b68'
os.environ['DISTILLSCHEDULE_HARDWARE_PROFILE']='h100-8-b16'
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(O/'scripts'))
import run_livewan_resident_matched_r2 as base
base.REPO=REPO;base.HEAD=HEAD
ROOT=base.ROOT;RUNS=base.RUNS;PREP=base.PREP;PYTHON=base.PYTHON
clean_env=base.clean_env;sha=base.sha;dump=base.dump;ARMS=base.ARMS
def livewan(self):
    # Reuse the registered, process-local asset environment; never source shell startup.
    loader = RUNS / 'scripts/livewan_h100_environment_r1.sh'
    code = "import json,os; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('LIVEWAN_','DISTILLSCHEDULE_')) or k in ('PATH','PYTHONPATH','HF_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM')}))"
    values = subprocess.check_output(['bash', '-c', f'source "$1"; "$2" -c "$3"', '_', str(loader), str(PYTHON), code], text=True, timeout=30)
    env = clean_env(); env.update(json.loads(values))
    env.update(DMD_SYSTEM_ROOT=str(REPO), DISTILLSCHEDULE_INHERITED_GPU_LOCKS='1',
               DISTILLSCHEDULE_ASSET_RELOCATION_RECEIPT=str(P / 'manifests/livewan_relocation_r2.json'),
               DISTILLSCHEDULE_ASSET_RELOCATION_SHA256=sha(P / 'manifests/livewan_relocation_r2.json'),
               DISTILLSCHEDULE_CAMPAIGN_ROOT=str(ROOT),
               PYTHONPATH=f'{env["LIVEWAN_PYTHON_OVERLAY"]}:{REPO}/src:{REPO}:{REPO}/third_party/livewan:{env["LIVEWAN_WAN21_SOURCE"]}')
    # Current source CPU gates checked in main; actual per-arm validate follows.
    launcher = REPO / 'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh'
    native_arm = {'baseline': 'baseline', 'candidate': 'pipeline'}
    variant = 'c34-continuous-b16-residual-layout'

    def run_arm(arm, kind, rid):
        run = RUNS / rid
        child_env = dict(env, DISTILLSCHEDULE_OUTPUT_DIR=str(run),
                         LIVEWAN_PIPELINE_VARIANT=variant if arm == 'candidate' else 'b1',
                         LIVEWAN_BASELINE_PLACEMENT='full-resident' if arm == 'baseline' else 'legacy-hybrid',
                         LIVEWAN_PIPELINE_TEACHER_PLACEMENT='full-resident' if arm == 'candidate' else 'legacy-sharded',
                         LIVEWAN_TEACHER_PARAMETER_OFFLOAD='off',
                         LIVEWAN_TRAJECTORY_CYCLES='2', LIVEWAN_TRAJECTORY_SEED='20260830',
                         LIVEWAN_TRAJECTORY_ARM_ID=f'h100-{native_arm[arm]}-{variant if arm == "candidate" else "b1"}',
                         LIVEWAN_TWO_CYCLE_SEED='20260830', LIVEWAN_EXPECTED_HEAD=HEAD)
        child_env['LIVEWAN_TWO_CYCLE_ARM_ID'] = child_env['LIVEWAN_TRAJECTORY_ARM_ID']
        if kind == 'trajectory':
            validate = self.out / ('validate_' + arm)
            self.invoke(rid + '_validate', ['bash', launcher, '0', rid + '_validate', 'validate', native_arm[arm], HEAD, '127.0.0.1', '29732'], dict(child_env, DISTILLSCHEDULE_OUTPUT_DIR=str(validate)), deadline=600)
            checked = json.load(open(validate / 'validate_only.node0.json'))
            assert checked['status']=='PASS' and checked['gpu_workload']=='NOT_RUN'
        self.invoke(rid, ['bash', launcher, '0', rid, kind, native_arm[arm], HEAD, '127.0.0.1', '29732'],
                    child_env, deadline=3600, gpu=True)
        summary = run / ('trajectory_summary.json' if kind == 'trajectory' else 'timing_summary.json')
        cmd = [PYTHON, REPO / f'scripts/distillschedule/summarize_livewan_{"trajectory" if kind == "trajectory" else "two_cycle_timing"}_world16.py',
               '--run-root', run, '--expected-head', HEAD, '--hardware-profile', 'h100-8-b16', '--output', summary]
        if kind == 'trajectory':
            cmd += ['--cycles', '2']
        else:
            cmd += ['--execution-mode', native_arm[arm], '--pipeline-variant', child_env['LIVEWAN_PIPELINE_VARIANT'],
                    '--trace', env['LIVEWAN_OPERATION_TRACE']]
        self.invoke(f'{rid}_summarize', cmd, child_env, deadline=300)
        x = json.loads(summary.read_text())
        if x['status'] != 'PASS':
            raise ValueError('LiveWan actual receipt failed')
        return run, summary, x

    sides = {}
    for arm in ('baseline', 'candidate'):
        sides[arm] = run_arm(arm, 'trajectory', f'livewan_h100_resident_{arm}_correctness_20261015_teacher_r1')
    comparison = self.out / 'trajectory_comparison.json'
    self.invoke('livewan_unchanged_numerical_comparator',
                [PYTHON, REPO / 'scripts/distillschedule/compare_livewan_two_cycle_trajectories.py',
                 '--baseline-run-root', sides['baseline'][0], '--pipeline-run-root', sides['candidate'][0],
                 '--baseline-summary', sides['baseline'][1], '--pipeline-summary', sides['candidate'][1],
                 '--expected-head', HEAD, '--pipeline-variant', variant, '--hardware-profile', 'h100-8-b16',
                 '--output', comparison], env, deadline=1200)
    gate = self.out / 'timing_gate.json'
    self.invoke('livewan_fresh_matched_timing_gate',
                [PYTHON, REPO / 'scripts/distillschedule/build_livewan_two_cycle_timing_gate.py',
                 '--baseline-trajectory-summary', sides['baseline'][1], '--pipeline-trajectory-summary', sides['candidate'][1],
                 '--trajectory-comparison', comparison, '--expected-head', HEAD, '--pipeline-variant', variant,
                 '--hardware-profile', 'h100-8-b16', '--output', gate], env, deadline=300)
    baseline_gate = self.out / 'resident_timing_gate.json'
    comparison_value = json.loads(comparison.read_text())
    baseline_summary = json.loads(sides['baseline'][1].read_text())
    from distillschedule.runtime.livewan_baseline_placement import BaselinePlacement, validate_timing_gate
    from distillschedule.runtime.livewan_hardware import livewan_hardware
    spec = BaselinePlacement('full-resident', 'off')
    value = dict(schema_version='distillschedule.livewan_baseline_placement_timing_gate.v1',
        status='PASS', claimable_for_two_cycle_timing=True, cycles=2,
        final_versions={'generator':2, 'critic':10, 'ema':0}, expected_head=HEAD,
        baseline_placement=spec.receipt(), hardware_profile=livewan_hardware('h100-8-b16').to_dict(),
        identity=baseline_summary['identity'], seed=baseline_summary['seed'], arm_id=baseline_summary['arm_id'],
        numerical_policy='REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW',
        correctness_summary={'path':str(sides['baseline'][1]), 'sha256':sha(sides['baseline'][1])},
        comparison={'path':str(comparison), 'sha256':sha(comparison)},
        common_matched_gate={'path':str(gate),'sha256':sha(gate)})
    validate_timing_gate(value,spec=spec,expected_head=HEAD,identity=baseline_summary['identity'],
                        arm_id=baseline_summary['arm_id'],seed=baseline_summary['seed'])
    dump(baseline_gate,value)
    env.update(LIVEWAN_PAIR_CORRECTNESS_RECEIPT=str(gate), LIVEWAN_PAIR_CORRECTNESS_SHA256=sha(gate))
    for pair, order in enumerate(ARMS):
        for arm in order:
            rid = f'livewan_h100_resident_{arm}_timing_pair{pair}_20261015_teacher_r1'
            env.update(LIVEWAN_PAIR_CORRECTNESS_RECEIPT=str(baseline_gate if arm == 'baseline' else gate),
                       LIVEWAN_PAIR_CORRECTNESS_SHA256=sha(baseline_gate if arm == 'baseline' else gate))
            _, summary_path, x = run_arm(arm, 'two-cycle-timing', rid)
            if x['final_versions'] != {'generator': 2, 'critic': 10, 'ema': 0} or x['rank_count'] != 8:
                raise ValueError('LiveWan actual rank/version gate failed')
            self.record_trial(pair, arm, rid, x, x['cycle_2']['seconds'], summary_path)

def main():
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,lambda s,f:(_ for _ in ()).throw(InterruptedError('ownedW4'+str(s))))
 signal.alarm(7200)
 assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS')=='1'
 assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==HEAD
 assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
 proof=json.load(open(P/'manifests/effective_plan_reporting_fix_r1.json'))
 assert proof['new_reporting_HEAD']==HEAD and proof['tests']['green']=='51 passed +5 subtests' and proof['tests']['skipped']==0
 out=P/'artifacts/matched_resident_teacher_r1';out.mkdir(exist_ok=False);(out/'logs').mkdir()
 runner=object.__new__(base.Runner);runner.workload='livewan';runner.out=out;runner.ordinal=0;runner.records=[]
 runner.livewan=types.MethodType(livewan,runner)
 result=dict(status='FAIL_PRESERVED',head=HEAD,replications=1,profiled=False,strict_full_state_equivalence='NOT_RUN',policy='baseline fullResident; C34pipeline with four fullResident BF16Teacher replicas; no parameterOffload',math_changed=False)
 try:
  runner.invoke('inherited_lease',[PYTHON,REPO/'scripts/distillschedule/verify_inherited_gpu_locks.py','--project-root',ROOT,'--asset-fd','7'],deadline=30)
  runner.livewan();runner.finish();result.update(json.load(open(out/'result.json')))
 except BaseException as exc:
  result.update(error=repr(exc),traceback=traceback.format_exc(),completed_trials=runner.records);dump(out/'failure.json',result)
 finally:signal.alarm(0)
 dump(out/'driver_result.json',result);print(json.dumps(result),flush=True)
 return 0 if result['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
