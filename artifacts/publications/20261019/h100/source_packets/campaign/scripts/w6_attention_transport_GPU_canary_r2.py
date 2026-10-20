"""Single-GPU exact native Flash return/gradient cache probe, no model weights."""
from pathlib import Path
import json
import os
import torch
import torch.utils.checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend
from selected_attention_pool_r1 import SelectedPool
from distillschedule.runtime.p0_parity import tensor_parity_metrics


out=Path(os.environ['H3_W6_ATTENTION_PRIMITIVE_OUTPUT']);out.mkdir(exist_ok=False)
torch.manual_seed(42)
q,k,v=[torch.randn(1,2,128,96,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
def attention(q,k,v):return torch.nn.functional.scaled_dot_product_attention(q,k,v)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    expected=attention(q,k,v)
    expected.float().square().mean().backward()
    gradients=[t.grad.detach().clone() for t in (q,k,v)]
    for t in (q,k,v):t.grad=None
    pool=SelectedPool(enabled=True)
    with torch.no_grad(),pool.capture('probe',uses=2):source=attention(q.detach(),k.detach(),v.detach())
    assert pool.counters['source_calls']==1,'ACTUAL_NATIVE_FLASH_SOURCE_NOT_CAPTURED'
    pair=pool.contexts_from_source('probe')
    actual=torch.utils.checkpoint.checkpoint(attention,q,k,v,use_reentrant=False,context_fn=lambda:pair)
    torch.testing.assert_close(source,expected,atol=0,rtol=0)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.float().square().mean().backward()
    metrics=[tensor_parity_metrics(wanted,t.grad) for t,wanted in zip((q,k,v),gradients)]
    assert all(m['status']=='PASS' for m in metrics),'UNCHANGED_PROJECT_GRADIENT_PARITY'
    for m,t,wanted in zip(metrics,(q,k,v),gradients):m['bitwise_equal']=torch.equal(t.grad,wanted)
    receipt=pool.finish_selected()
torch.cuda.synchronize()
receipt.update(status='PASS_DETACHED_NOGRAD_NATIVE_FLASH_FIXED_GRADIENT_PARITY',
               source_QKV_requires_grad=False,source_and_returned_latent_bitwise=True,
               QKV_gradient_metrics=metrics,project_thresholds_unchanged=True,model_weights='NOT_LOADED',
               H3_DMD_capacity='NOT_RUN',scope='DIAGNOSTIC_LIBRARY_FLASH_PRIMITIVE_NOT_TRAINING_BACKEND_OVERRIDE')
with (out/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
