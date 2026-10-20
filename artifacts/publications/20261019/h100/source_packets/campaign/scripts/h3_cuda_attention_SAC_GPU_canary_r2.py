"""Actual FA3 atomic noise-floor probe; use existing frozen P0 metrics, not an invented bitwise gradient gate."""
from pathlib import Path
import dataclasses,json,os,sys,hashlib
import torch
from torch.utils.checkpoint import checkpoint,create_selective_checkpoint_contexts
from h3_a100.model import _configure_local_flash_attn3
from h3_a100 import fa3_replay_cache as m
from diffusers.models.attention_dispatch import AttentionBackendName,_HUB_KERNELS_REGISTRY
D=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-helios-official-swap-20261014-r1')
sys.path.insert(0,str(D/'src'))
from distillschedule.runtime.p0_parity import tensor_parity_metrics,DEFAULT_P0_PARITY
out=Path(os.environ['H3_CUDA_ATTENTION_PRIMITIVE_OUTPUT']);out.mkdir(exist_ok=False)
assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
_configure_local_flash_attn3();cfg=_HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
results=[]
for n in (64,257,1024):
 torch.manual_seed(20260817+n)
 values=[torch.randn(1,n,42,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 def body(q,k,v):return m._CompactFA3Autograd.apply(q,k,v,128**-0.5)
 def execute(cached):
  inputs=[x.detach().clone().requires_grad_(True) for x in values]
  stats=m.FA3ReplayCacheStats();m._RUNTIME=m._Runtime(cfg.wrapped_forward_fn,cfg.wrapped_backward_fn,stats)
  kwargs={'context_fn':lambda:create_selective_checkpoint_contexts([m._fa3_forward_compact._opoverload])} if cached else {}
  prediction=checkpoint(body,*inputs,use_reentrant=False,**kwargs);prediction.float().square().mean().backward()
  assert stats.compact_forward_impl_calls==(1 if cached else 2) and stats.compact_backward_calls==1
  assert stats.cpu_d2h_bytes==stats.cpu_h2d_bytes==0 and stats.cached_logical_bytes>0
  return prediction.detach(),[x.grad.detach().clone() for x in inputs]
 reference,expected=execute(False)
 comparisons=[]
 for group,cached in (('uncached_self_repeat',False),('cuda_cached_cross',True)):
  for trial in range(3):
   prediction,gradients=execute(cached)
   assert torch.equal(prediction,reference),'bitwise forward output changed'
   rows=[]
   for name,wanted,actual in zip(('Q','K','V'),expected,gradients):
    metrics=tensor_parity_metrics(wanted,actual)
    metrics.update(name=name,bitwise_equal=torch.equal(wanted,actual),mismatched_elements=int((wanted!=actual).sum().item()))
    assert metrics['status']=='PASS','unchanged DEFAULT_P0_PARITY failed'
    rows.append(metrics)
   comparisons.append({'group':group,'trial':trial,'output':'BITWISE_EQUAL','gradients':rows})
 results.append({'sequence':n,'head_count':42,'head_dim':128,'comparisons':comparisons})
 torch.cuda.synchronize()
m._RUNTIME=None
result={'status':'PASS_REAL_FA3_CUDA_CACHE_EXISTING_MULTI_METRIC_GATE','kind':'DIAGNOSTIC_LIBRARY_NO_H3_MODEL_CAPACITY_OR_TRAINING_TIME','model_weights':'NOT_LOADED','real_H3_capacity':'NOT_RUN','real_H3_timing':'NOT_RUN','torch':torch.__version__,'torch_cuda':torch.version.cuda,'cases':results,'thresholds':dataclasses.asdict(DEFAULT_P0_PARITY),'threshold_source_sha256':hashlib.sha256((D/'src/distillschedule/runtime/p0_parity.py').read_bytes()).hexdigest(),'old_bitwise_gradient_failure':'PRESERVED_R1; one of1,381,632 BF16 gradient elements differed1.1641532182693481e-10; extra bitwise gradient requirement was not the project numerical gate','no_training_math_or_fixed_project_threshold_changed':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
