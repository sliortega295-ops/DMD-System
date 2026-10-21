from pathlib import Path
import json,hashlib,os,subprocess,ast,datetime
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
p=O/'scripts/w6_attention_transport_GPU_canary_r3.py';ast.parse(p.read_text())
assert "source=attention(q.detach(),k.detach(),v.detach())" in p.read_text()
repo=R/'repos/DMD-System-h100-w6-native-logical-b16-20261014-r1'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='3003a9903a96e0ccaafa10faec956e1846e2448e'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def atomic(p,x):
 t=p.with_suffix('.partial');save(t,x);os.link(t,p);t.unlink()
for name in ('017_w6_native_tiny_gate','018_w6_native_three_arm_matrix'):
 q=O/'queue_r5'/name
 assert not q.with_suffix('.done.json').exists() and not q.with_suffix('.failure.json').exists()
 save(q.with_suffix('.failure.json'),{'status':'NOT_RUN_DEFERRED_TO_STRONGER_ACTUAL_NOGRAD_TENSOR_METADATA_GATE','GPU':'NOT_RUN','reason':'No-grad QKV now detach() to match actual native source; fixed project gradient gate, bitwise output stillrequired; training worker/code/math unchanged','source_head':'3003a9903a96e0ccaafa10faec956e1846e2448e'})
script=(O/'queue_r5/017_w6_native_tiny_gate.sh').read_text()
script=script.replace('w6_attention_transport_GPU_canary_r1.py','w6_attention_transport_GPU_canary_r3.py').replace('w6_attention_transport_GPU_canary_r1','w6_attention_transport_GPU_canary_r2')
script=script.replace('PASS_BITWISE_NATIVE_FLASH_OUTPUT_AND_QKV_GRADIENT','PASS_DETACHED_NOGRAD_NATIVE_FLASH_FIXED_GRADIENT_PARITY')
script2=(O/'queue_r5/018_w6_native_three_arm_matrix.sh').read_text().replace('w6_attention_transport_GPU_canary_r1','w6_attention_transport_GPU_canary_r2').replace('PASS_BITWISE_NATIVE_FLASH_OUTPUT_AND_QKV_GRADIENT','PASS_DETACHED_NOGRAD_NATIVE_FLASH_FIXED_GRADIENT_PARITY')
# Tiny/real runIDs are unchanged: neither previous queued arm ran any GPU.
for name,text in (('017r2_w6_actual_nograd_tiny_gate',script),('018r2_w6_native_three_arm_matrix',script2)):
 q=O/'queue_r5'/name
 with q.with_suffix('.sh').open('x') as f:f.write(text)
 subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
 atomic(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(text.encode()).hexdigest(),'workload_math_changed':False,'source_head':'3003a9903a96e0ccaafa10faec956e1846e2448e','primitive_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'strengthened_source_metadata':'actual no-grad detached QKV; real TinyHead96 model after primitive; no version/weights/loss/noise changes'})
note={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'W6_GPU_PRECHECK_CORRECT_NOGRAD_TENSOR_METADATA_BEFORE_FIRST_GPU_RUN','training_head':'3003a9903a96e0ccaafa10faec956e1846e2448e','training_math_changed':False,'old_primitive':'NOT_RUN; q leaf required grad even under no_grad scope; not fullyrepresentative real native QKV','new_primitive':'q/k/v detached in source; Flash compatible Head96; real cache/gradient checked via unchanged project P0 gate; output bitwise required','GPU':'NOT_RUN','tiny_and_real_run_ids':'not changed because neither old queuedrun launched'}
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps(note)+'\n')
print(json.dumps(note))
