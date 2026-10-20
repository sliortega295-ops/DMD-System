import importlib.util,types
from pathlib import Path
p=Path(__file__).with_name('h3_h100_boundary_recording_entry_r1.py')
spec=importlib.util.spec_from_file_location('boundary_observer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
keys={'grad_transformer_forward_count':12,'student_grad_forward_count':2,'fake_grad_forward_count':10,'other_grad_forward_count':0,'grad_checkpoint_call_count':600,'cpu_copy_count':600,'offloaded_logical_bytes':1234,'offloaded_storage_bytes':1234}
t=types.SimpleNamespace(activation_policy='checkpoint_boundary_cpu',boundary_offload_registration=types.SimpleNamespace(stats=keys.copy(),pin_memory=True),_boundary_offload_cycle_start={})
assert m.boundary_policy_receipt(t)['delta']['cpu_copy_count']==600
for key,bad in [('cpu_copy_count',599),('grad_checkpoint_call_count',300),('other_grad_forward_count',1),('offloaded_storage_bytes',0)]:
 old=t.boundary_offload_registration.stats[key];t.boundary_offload_registration.stats[key]=bad
 try:m.boundary_policy_receipt(t)
 except RuntimeError:pass
 else:raise AssertionError(key)
 t.boundary_offload_registration.stats[key]=old
t.activation_policy='none'
try:m.boundary_policy_receipt(t)
except RuntimeError:pass
else:raise AssertionError('none policy cannot impersonate fallback')
print('PASS actual boundary observer valid + five invalid-counter/policy cases; CPU only')
