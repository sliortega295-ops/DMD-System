"""Complete the same-0419 W6 matrix after actual named Resident acceptance."""
import datetime,hashlib,importlib.util,json,os,pathlib,signal,subprocess,sys,traceback
O=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1")
P=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w6_resident_adaln16_staging_20261015_r1")
W=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-w6-lora-unit-inplace-20261015-r1")
HEAD='0419b96b9ed9a8dd76737042e6c5020c80f6673a'
NATIVE=O/'artifacts/w6_resident_boundary_composed_20261015_r1/scripts/run_native_pair_after_C1_signature_audit_r1.py'
assert hashlib.sha256(NATIVE.read_bytes()).hexdigest()=='a4ff993a6824235dc5e52050387c3ba826f0120f61cf69e1f6f5323b371d5019'
spec=importlib.util.spec_from_file_location('previous_actual_native_pair_1abd',NATIVE)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
old=previous.old;base=previous.base
def cpu_admit():
    assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()==HEAD
    assert not subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True).strip()
    proof=json.load(open(P/'manifests/default_off_nonresident_source_equivalence_r1.json'))
    assert proof['status']=='PASS_DEFAULT_OFF_WHOLE_WORKER_AST_EQUIVALENCE_CPU_ONLY' and proof['new']==HEAD
    assert base.sha(W/proof['source'])==proof['new_sha256']
    assert proof['old']=='1abd94656420880f71b0f8adfdcc2974952c33c2'
    assert 'torch' not in sys.modules
    return dict(status='PASS_CPU_SOURCE_ONLY_CONDITIONAL_RESIDENT_PREREQUISITE',
        source_head=HEAD,source_AST_equivalence=proof['status'],GPU='NOT_RUN',
        new_full_model_strict_numerical_parity='NOT_RUN',
        prerequisite='capacity_and_single_timing_r1/result.json must actual PASS before any launch',
        max_new_native_arms=2,native_repetitions=1,budget_seconds=7200)
def main():
    def interrupted(sig,frame):raise InterruptedError('owned same-revision W6 matrix '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(7200);out=P/'same_revision_native_matrix_r1';out.mkdir(exist_ok=False)
    report=dict(status='FAIL_PRESERVED',head=HEAD,arms=[],profiled=False,one_native_trial_per_arm=True,
        formal_speedup_eligibility=False,new_full_model_strict_numerical_parity='NOT_RUN',
        placement_caveat='Resident+boundaryCPU+AdaLN16parameterCPU is named capacity fallback, not pure no-offload placement baseline')
    try:
        report['CPU_admission']=cpu_admit()
        prior_path=P/'capacity_and_single_timing_r1/result.json'
        prior=json.load(open(prior_path))
        if prior['status']!='PASS_RESIDENT_ADALN_CPU_NAMED_CAPACITY_FALLBACK_SINGLE_DIAGNOSTIC':
            report.update(status='NOT_RUN_RESIDENT_PREREQUISITE_FAILED',prerequisite_result=str(prior_path))
        else:
            assert prior['head']==HEAD and prior['single_C2_seconds']>0
            # Verify the actual old27008 C1 proof on its original1abd checkout,
            # without changing or relabelling that evidence as a newC1 run.
            report['inherited_actual_C1']=previous.admit()
            old.HEAD=HEAD;old.W=W
            base.REPO=W;base.HEAD=HEAD;base.WORKER=W/'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
            base.ACTIVATE=P/'scripts/environment_r1.sh'
            base.aggregate=old.aggregate_two
            os.environ['SUDO_UID']=str(W.stat().st_uid)
            assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS')=='1'
            subprocess.run([str(base.PYTHON),str(W/'scripts/distillschedule/verify_inherited_gpu_locks.py'),
                '--project-root',str(base.ROOT),'--asset-fd','7'],check=True,pass_fds=(7,8,9))
            base.source_gate()
            inputs=json.load(open(base.INPUT))
            assert base.sha(inputs['cache_manifest'])==inputs['cache_manifest_sha256']
            assert base.sha(inputs['config_path'])==inputs['config_sha256']
            resident_arm=prior['arms'][-1];assert resident_arm['name']=='resident_timing'
            resident_rows=[json.load(open(pathlib.Path(resident_arm['path'])/f'rank_receipts/rank_{rank:02d}.json')) for rank in range(8)]
            all_rows={};times={'resident_named_CPU_fallback':prior['single_C2_seconds']}
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
            for placement in ('fsdp','candidate'):
                parent=out/('native_'+placement);parent.mkdir()
                base.WORKER_EXTRA_ARGS=('--lora-input-cast-elision','--lora-unit-inplace') if placement=='candidate' else ()
                observer=old.LeaseObserver(parent/'GPU_samples.jsonl');observer.start()
                try:result,path=base.arm(parent,placement,'timing',inputs,False)
                finally:observer.stop()
                report['arms'].append(dict(placement=placement,path=str(path),**result))
                assert result['status']=='PASS_REAL_8_RANK_16_LOGICAL_SLOT_NATIVE_GATE'
                lease=observer.receipt(path/'rank_receipts');base.write(path/'manifests/GPU_lease.json',lease)
                assert lease['status']=='PASS_SAMPLED_8_STABLE_WORKER_PID_SET_8_GPUS'
                rows=[json.load(open(path/f'rank_receipts/rank_{rank:02d}.json')) for rank in range(8)]
                for row in rows:
                    old.policy(row,placement)
                    assert not row['parameter_offload'] and not row['activation_offload']
                    assert not row.get('resident_adaln_CPU_setup')
                all_rows[placement]=rows;times[placement]=result['timed_C2_seconds']
            for rank in range(8):
                for key in ('rank','logical_slots','samples','initial_LoRA_sha256','precision','final_versions'):
                    assert all_rows['fsdp'][rank][key]==all_rows['candidate'][rank][key]==resident_rows[rank][key],('MATRIX_IDENTITY',rank,key)
            report.update(status='PASS_SAME_REVISION_W6_THREE_ARM_SINGLE_DIAGNOSTIC_MATRIX',
                times_seconds=times,fsdp_to_candidate_ratio=times['fsdp']/times['candidate'],
                Resident_prerequisite=dict(path=str(prior_path),sha256=base.sha(prior_path),peak=resident_arm['peak_allocated_bytes']),
                boundary='same0419 source/inputs/dtypes/logicalB16. Prior strict C1 at1abd retained with explicit default-off AST proof. Fresh2cycle native census/versions/ownership onallarms. Resident staging full-model strict parity remainsNOT_RUN. Not a pure placement or formal numerical equivalence claim.')
    except BaseException as error:report.update(error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0)
    report['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    base.write(out/'result.json',report);print(json.dumps({k:v for k,v in report.items() if k!='arms'}),flush=True)
    return 0 if report['status'].startswith(('PASS_','NOT_RUN_')) else 1
if __name__=='__main__':
    if sys.argv[1:]==['--cpu-admit']:print(json.dumps(cpu_admit(),indent=2))
    elif sys.argv[1:]:raise SystemExit('UNKNOWN_ARGUMENT')
    else:raise SystemExit(main())
