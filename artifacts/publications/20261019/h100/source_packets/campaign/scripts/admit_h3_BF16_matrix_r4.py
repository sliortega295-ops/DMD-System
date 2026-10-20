from pathlib import Path
import hashlib,json,subprocess

ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
p=ROOT/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
r=json.loads((p/'campaign_r5_status.json').read_text())
assert r['phase']=='013_h3_DP8_fixed_three_arm_matrix' and r['status'].startswith('FAIL_PRESERVED')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert json.loads((p.parent/'h3_BF16_compatible_runner_CPU_prepare_20261014_r4/status.json').read_text())['status'].startswith('PASS_CPU_')
for n in ('014_w6_native_tiny_gate','015_w6_native_three_arm_matrix'):
    q=p/'queue_r5'/n;assert not q.with_suffix('.launcher.log').exists()
    with q.with_suffix('.failure.json').open('x') as f:json.dump({'status':'NOT_RUN_DEFERRED_W5_BF16_GRADIENT_GATE_FIX','GPU':'NOT_RUN','attempt_count':0},f)
s=(p/'scripts/queue_h3_BF16_compatible_matrix_r4.sh').read_text()
key='exec "$PY" "$O/scripts/run_h3_h100_three_arm_r4.py"';assert s.count(key)==1
code="""export PYTHONPATH="$ROOT/repos/H3-DMD-for-A100-h100-three-arm-20261014-r1:$ROOT/envs/h3-data-overlay-20261004-r1/site-packages:$ROOT/envs/h3-grid-overlay-20261004/site-packages:$ROOT/repos/h3-diffusers-92846072-20261004-r2/src:$ROOT/repos/h3-lightx2v-d034a6b0-20261004-r2:$ROOT/repos/h3-lightx2v-d034a6b0-20261004-r2/lightx2v_train"
export H3_RESIDENT_COMM_GATE_OUTPUT="$O/artifacts/h3_resident_BF16_communication_GPU_gate_r1"
timeout --signal=TERM --kill-after=30s 180s "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29631 "$O/scripts/h3_resident_BF16_communication_GPU_gate_r1.py"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"""
s=s.replace(key,code+key);new=p/'scripts/queue_h3_BF16_verified_matrix_r4.sh'
with new.open('x') as f:f.write(s)
for index,file in (('016_h3_BF16_verified_three_arm_matrix',new),('017_w6_native_tiny_gate',p/'scripts/queue_w6_native_tiny_gate_r1.sh'),('018_w6_native_three_arm_matrix',p/'scripts/queue_w6_native_matrix_r1.sh')):
    subprocess.run(['bash','-n',file],check=True);q=p/'queue_r5'/index
    with q.with_suffix('.sh').open('x') as f:f.write(file.read_text())
    with q.with_suffix('.ready.json').open('x') as f:json.dump({'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'workload_math_changed':False,'W5_source':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','W6_source':'3003a9903a96e0ccaafa10faec956e1846e2448e','gate':'CPU19tests plus8rank BF16/F32 communication canary beforerealH3; same immutable initial LoRA BF16 storage, same sigma Grid','repeats':1},f,indent=2)
with (p/'queue_r5/013_h3_DP8_fixed_three_arm_matrix.disposition.json').open('x') as f:json.dump({'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','failure_type':'OWNED_ARM_STOPPED_PRE_KNOWN_WRONG_DTYPE_GATE_NOT_OOM','repair_commit':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','training_math_changed':False,'precision':'BF16 parameters/gradients/Adam unchanged, FP32 communication as FSDP2 policy; do not relabel W5 as FP32 LoRA'},f,indent=2)
print('RELEASED013_TO016_BF16VERIFIED_W5_THENW6')
