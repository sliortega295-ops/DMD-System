"""Read-only A100 helper source audit. No torch import, CUDA context or GPU launch."""
from pathlib import Path
import datetime,hashlib,json,subprocess
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
S=R/'assets/five_workload_h100_20260930/h3_ar_frozen_r1/assets/steady_training_source_helpers_r1/zero3_200s_r2'
files={
 'rms_runtime':'rms_norm_fusion_260s_r1/scripts/native_rms_fusion_r3.py',
 'rms_build':'rms_norm_fusion_260s_r1/scripts/build_cpu_r2.py',
 'rms_bridge':'rms_norm_fusion_260s_r1/scripts/rms_fusion_bridge_r2.cu',
 'lora_runtime':'lora_cutlass_exact_round_260s_r1/scripts/native_cutlass_lora_r1.py',
 'lora_build':'lora_cutlass_exact_round_260s_r1/scripts/build_cpu_r1.py',
 'lora_bridge':'lora_cutlass_exact_round_260s_r1/scripts/round_lora_bridge_r1.cu',
 'lora_parent':'lora_epilogue_260s_r1/scripts/native_lora_epilogue_r1.py'}
needles=('compute_80','code=sm_80','module.weight.dtype==value.dtype==base.dtype',
         'SM80','round_then_add','getattr(p,\'ds_numel\'','DIT_SHA=',
         'H3_50_BLOCK','PINNED_NATIVE_RANK','same_B_module_and_ZeRO_hooks')
evidence={}
for name,rel in files.items():
 p=S/rel;item={'path':str(p),'exists':p.is_file()}
 if p.is_file():
  data=p.read_bytes();lines=data.decode().splitlines()
  item.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
    excerpts=[{'line':i,'text':line.strip()} for i,line in enumerate(lines,1) if any(n in line for n in needles)])
 evidence[name]=item
repo=R/'repos/DMD-System-h100-w6-lora-unit-inplace-20261015-r1'
def cmd(args):return subprocess.check_output(args,text=True).strip()
worker=repo/'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
worker_lines=worker.read_text().splitlines()
report={
 'run_id':'w6_A100_kernel_port_readiness_CPU_20261015_r1',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'PASS_STATIC_SOURCE_AUDIT_ONLY_NOT_GPU_ADMITTED',
 'GPU':'NOT_RUN_FOREIGN_RESERVATION','torch_imported':False,
 'source':{'head':cmd(['git','-C',str(repo),'rev-parse','HEAD']),
           'git_diff':cmd(['git','-C',str(repo),'status','--porcelain']),
           'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
           'precision_excerpts':[{'line':i,'text':line.strip()} for i,line in enumerate(worker_lines,1) if any(s in line for s in ("LoRA_storage",'param_dtype=None','autocast(device_type'))]},
 'A100_helper_sources':evidence,
 'PROVEN_static':[
   'Transferred A100 helper files exist with recorded hashes; no optimization is newly enabled.',
   'Both sampled legacy build commands target compute_80/code=sm_80 and contain original A100-only absolute compiler/Python paths.',
   'Legacy LoRA epilogue asserts B module weight, input and base output are BF16; current H100 arm records FP32 LoRA parameter storage with BF16 autocast.',
   'Current H100 f5d66e9 primitive retains the original native A/B modules and their autocast behavior; it is not the A100 fused GEMM port.'],
 'NOT_PROVEN':[
   'H100 kernel load/execution, numerical parity, hook/recompute safety, memory reduction and speedup of either A100 fused helper.',
   'A100 compiler/library outputs are not admitted as current Hopper binaries.',
   'Existing tiny/native CPU evidence is not full-model cross-arm state equivalence.'],
 'decisions':[
  {'mechanism':'A100 RMSNorm + indexed modulation/RoPE fusion','decision':'DEFER',
   'reason':'Needs current FSDP hook/precision/source gate and bounded SM90 primitive before integration; actual H100 critical-path fraction unknown.',
   'minimum_probe':'Only after own lease and current capacity matrix: actual H3 B1 shapes, BF16 forward and input gradient, unchanged parity, module hook/census audit.',
   'benefit_seconds':'UNKNOWN_NOT_PREDICTED','budget':'not newly admitted by this source audit'},
  {'mechanism':'A100 exact-round LoRA B GEMM epilogue','decision':'DEFER',
   'reason':'Cannot blindly install a BF16-weight assertion on current FP32-storage LoRA; retain storage and native BF16 autocast semantics, never cast persistent trainables to fit helper.',
   'minimum_probe':'Qualify separate H100 default-off implementation against current native module including weight/input gradients and checkpoint replay before tiny/full integration.',
   'benefit_seconds':'UNKNOWN_NOT_PREDICTED','budget':'not newly admitted by this source audit'}],
 'E2E_reference':{'W6_FSDP_seconds':320.4321391545236,'latest_attention_reuse_seconds':290.646278033033,
   'target_1_5x_seconds':320.4321391545236/1.5,
   'additional_reduction_needed_seconds':290.646278033033-320.4321391545236/1.5,
   'evidence_level':'existing single unprofiled diagnostic; not a new measurement'},
 'next_actual_GPU_stage':'Existing R6 f5d66e9 LoRA unit-scale primitive -> actual8rank tiny -> real Resident capacity. Do not combine deferred kernels into that admitted mechanism.',
 'source_environment_shared_process_changes':False,
 'campaign_scope':'W5/W6 remain unfinished; do not release our eventual owned allocation merely for another queued reservation.'}
assert report['source']['head']=='f5d66e96665bb92e3f7f112e24b064b2710903a2' and not report['source']['git_diff']
assert 'code=sm_80' in (S/files['rms_build']).read_text()
assert 'module.weight.dtype==value.dtype==base.dtype==torch.bfloat16' in (S/files['lora_parent']).read_text()
p=O/'reports'/f"{report['run_id']}.json"
with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
