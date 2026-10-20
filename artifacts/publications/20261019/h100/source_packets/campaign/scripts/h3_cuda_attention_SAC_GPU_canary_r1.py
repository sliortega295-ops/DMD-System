"""Real pinned FA3 + real selective-checkpoint CUDA cache, no model weights."""
from pathlib import Path
import json,os
import torch
from torch.utils.checkpoint import checkpoint,create_selective_checkpoint_contexts
from h3_a100.model import _configure_local_flash_attn3
from h3_a100 import fa3_replay_cache as m
from diffusers.models.attention_dispatch import AttentionBackendName,_HUB_KERNELS_REGISTRY
out=Path(os.environ['H3_CUDA_ATTENTION_PRIMITIVE_OUTPUT']);out.mkdir(exist_ok=False)
assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
_configure_local_flash_attn3();cfg=_HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
results=[]
for n in (64,257,1024):
 torch.manual_seed(20260817+n)
 inputs=[torch.randn(1,n,42,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
 references=[x.detach().clone().requires_grad_(True) for x in inputs]
 def body(q,k,v):return m._CompactFA3Autograd.apply(q,k,v,128**-0.5)
 baseline_stats=m.FA3ReplayCacheStats();m._RUNTIME=m._Runtime(cfg.wrapped_forward_fn,cfg.wrapped_backward_fn,baseline_stats)
 reference=checkpoint(body,*references,use_reentrant=False);reference.float().square().mean().backward()
 expected_grad=[x.grad.detach().clone() for x in references]
 assert baseline_stats.compact_forward_impl_calls==2 and baseline_stats.compact_backward_calls==1
 cached_stats=m.FA3ReplayCacheStats();m._RUNTIME=m._Runtime(cfg.wrapped_forward_fn,cfg.wrapped_backward_fn,cached_stats)
 actual=checkpoint(body,*inputs,use_reentrant=False,context_fn=lambda:create_selective_checkpoint_contexts([m._fa3_forward_compact._opoverload]))
 actual.float().square().mean().backward()
 torch.testing.assert_close(actual,reference,atol=0,rtol=0)
 for x,wanted in zip(inputs,expected_grad):torch.testing.assert_close(x.grad,wanted,atol=0,rtol=0)
 assert cached_stats.compact_forward_impl_calls==1 and cached_stats.compact_backward_calls==1
 assert cached_stats.cached_logical_bytes>0 and cached_stats.cpu_d2h_bytes==cached_stats.cpu_h2d_bytes==0
 results.append({'sequence':n,'head_count':42,'head_dim':128,'forward_calls_reference':2,'forward_calls_cached':1,'backward_calls_cached':1,'output':'BITWISE_EQUAL','QKV_gradients':'BITWISE_EQUAL','FA3_cache_CPU_copies':0,'cached_logical_bytes':cached_stats.cached_logical_bytes})
 torch.cuda.synchronize()
m._RUNTIME=None
result={'status':'PASS_REAL_H100_FA3_CUDA_CACHE_OUTPUT_AND_GRADIENT_BITWISE','kind':'DIAGNOSTIC_LIBRARY_NO_H3_MODEL_CAPACITY_OR_TRAINING_TIME','model_weights':'NOT_LOADED','real_H3_capacity':'NOT_RUN','real_H3_timing':'NOT_RUN','torch':torch.__version__,'torch_cuda':torch.version.cuda,'cases':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
