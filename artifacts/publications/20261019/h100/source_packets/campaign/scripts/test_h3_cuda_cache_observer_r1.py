import importlib.util,types
from pathlib import Path
p=Path(__file__).with_name('h3_h100_cuda_attention_recording_entry_r1.py')
spec=importlib.util.spec_from_file_location('cuda_cache_observer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
keys={'grad_transformer_forwards':12,'completed_grad_transformer_forwards':12,'selected_checkpoint_wraps':600,'selected_scoped_executions':1200,'compact_attention_entries':1200,'compact_forward_impl_calls':600,'compact_backward_calls':600,'cpu_d2h_entries':0,'cpu_h2d_entries':0,'cpu_d2h_bytes':0,'cpu_h2d_bytes':0,'cpu_pool_allocated_bytes':0,'allocator_trim_calls':0,'cached_logical_bytes':12345678}
value={'enabled':True,'storage':'cuda','block_indices':list(range(50)),'trim_before_backward':False,'stats':keys.copy()}
t=types.SimpleNamespace(fa3_replay_cache_registration=types.SimpleNamespace(enabled=True,receipt=lambda:value),_fa3_replay_cache_cycle_start={})
assert m.cuda_attention_cache_receipt(t)['cycle_delta']['compact_backward_calls']==600
for key,bad in [('compact_forward_impl_calls',601),('cpu_d2h_bytes',1),('cpu_h2d_entries',1),('allocator_trim_calls',1),('cached_logical_bytes',0)]:
 old=value['stats'][key];value['stats'][key]=bad
 try:m.cuda_attention_cache_receipt(t)
 except RuntimeError:pass
 else:raise AssertionError(key)
 value['stats'][key]=old
for key,bad in [('storage','cpu_staged'),('trim_before_backward',True),('block_indices',list(range(49)))]:
 old=value[key];value[key]=bad
 try:m.cuda_attention_cache_receipt(t)
 except RuntimeError:pass
 else:raise AssertionError(key)
 value[key]=old
print('PASS actual CUDA-cache observer valid and8 invalid-copy/policy/census cases, no GPU')
