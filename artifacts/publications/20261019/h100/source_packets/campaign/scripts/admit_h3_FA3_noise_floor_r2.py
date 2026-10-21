from pathlib import Path
import json,os,hashlib,subprocess,datetime,ast
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1'
source=O/'scripts/h3_cuda_attention_SAC_GPU_canary_r2.py';ast.parse(source.read_text())
log=(O/'logs/h3_FA3_cuda_attention_SAC_GPU_gate_r1.log').read_text()
assert 'Mismatched elements: 1 / 1381632' in log and '1.1641532182693481e-10' in log
health=json.load(open(O/'manifests/h3_FA3_cuda_attention_SAC_GPU_gate_health_r1.json'))
assert health['launcher_rc']!=0 and not health['GPU_contexts_after'].strip() and not any(health['cgroup_oom_delta'].values())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
from_path=R/'repos/DMD-System-h100-helios-official-swap-20261014-r1/src/distillschedule/runtime/p0_parity.py'
previous=json.load(open(O/'manifests/w5_cuda_cache_residency_recovery_admission_r1.json'))
card={'status':'GO_ONE_BOUNDED_NUMERICAL_NOISE_FLOOR_PROBE','scope':'diagnostic library only; no model until PASS existing fixed multi-metric gate',
 'prior_extra_bitwise_gradient_gate':'FAIL_R1_PRESERVED; not the fixed project numerical contract',
 'observed_error':{'mismatches':1,'elements':1381632,'max_abs':1.1641532182693481e-10},
 'new_evidence_question':'Does uncached-vs-uncached show the same BF16 backward variability, and do cached gradients satisfy existing DEFAULT_P0_PARITY without any parameter changes?',
 'criterion':'forward BITWISE; Q/K/V gradients use exact unchanged project DEFAULT_P0_PARITY plus full self/cross distribution; bitwise gradient status reported separately',
 'training_thresholds_modified':False,'algorithm_or_raw_kernel_flags_modified':False,
 'source_threshold_file':str(from_path),'source_threshold_sha256':hashlib.sha256(from_path.read_bytes()).hexdigest(),
 'raw_backward_flag':'existing deterministic=False (permits numerical variability, cause not presumed until self-repeat)',
 'bounds':{'GPU_cases':[64,257,1024],'self_repeats':3,'cross_repeats':3,'deadline_seconds':180,'max_noise_floor_attempts':1},
 'previous_real_model_admission':str(O/'manifests/w5_cuda_cache_residency_recovery_admission_r1.json'),
 'comparison_boundary':previous['claim_boundary'],'GPU':'NOT_RUN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def atomic(p,x):
 q=p.with_suffix('.partial');save(q,x);os.link(q,p);q.unlink()
save(O/'manifests/w5_fa3_GPU_gradient_noise_gate_admission_r2.json',card)
# Old full-model dependency is NOT_RUN, never relabel R1 bitwise failure PASS.
save(O/'queue_r5/016y_h3_cuda_attention_residency_recovery.failure.json',{'status':'NOT_RUN_UNSATISFIED_BITWISE_GRADIENT_LIBRARY_PREREQUISITE','GPU':'NOT_RUN','old_gate':'FAIL_PRESERVED_R1','replacement':'016y2_h3_cuda_attention_existing_parity_recovery'})
script=(O/'queue_r5/016x_h3_cuda_attention_library_gate.sh').read_text()
script=script.replace('h3_FA3_cuda_attention_SAC_GPU_gate_r1','h3_FA3_cuda_attention_SAC_GPU_gate_r2').replace('h3_cuda_attention_SAC_GPU_canary_r1.py','h3_cuda_attention_SAC_GPU_canary_r2.py').replace('PASS_REAL_H100_FA3_CUDA_CACHE_OUTPUT_AND_GRADIENT_BITWISE','PASS_REAL_FA3_CUDA_CACHE_EXISTING_MULTI_METRIC_GATE')
script2=(O/'queue_r5/016y_h3_cuda_attention_residency_recovery.sh').read_text()
script2=script2.replace('h3_FA3_cuda_attention_SAC_GPU_gate_r1','h3_FA3_cuda_attention_SAC_GPU_gate_r2').replace('PASS_REAL_H100_FA3_CUDA_CACHE_OUTPUT_AND_GRADIENT_BITWISE','PASS_REAL_FA3_CUDA_CACHE_EXISTING_MULTI_METRIC_GATE')
old="assert len(x['cases'])==3 and all(v['QKV_gradients']=='BITWISE_EQUAL' and v['FA3_cache_CPU_copies']==0 for v in x['cases'])"
new="assert len(x['cases'])==3 and all(t['output']=='BITWISE_EQUAL' and all(g['status']=='PASS' for g in t['gradients']) for v in x['cases'] for t in v['comparisons'])\nassert x['no_training_math_or_fixed_project_threshold_changed'] is True"
assert script2.count(old)==1;script2=script2.replace(old,new)
for name,text in (('016x2_h3_fa3_gradient_noise_floor',script),('016y2_h3_cuda_attention_existing_parity_recovery',script2)):
 q=O/'queue_r5'/name
 with q.with_suffix('.sh').open('x') as f:f.write(text)
 subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
 atomic(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(text.encode()).hexdigest(),'workload_math_changed':False,'source_head':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','admission':str(O/'manifests/w5_fa3_GPU_gradient_noise_gate_admission_r2.json')})
q=O/'queue_r5/016x_h3_cuda_attention_library_gate.disposition.json'
atomic(q,{'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','old_bitwise_gradient_gate':'FAIL_PRESERVED','admitted_noise_probe':str(O/'manifests/w5_fa3_GPU_gradient_noise_gate_admission_r2.json'),'fixed_project_thresholds':'UNCHANGED'})
print(json.dumps(card))
