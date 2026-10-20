"""Owned-GPU library sanity only; not H3 training or an end-to-end parity result."""
import json
import os
from pathlib import Path

import torch
from h3_a100.model import _configure_local_flash_attn3
from diffusers.models.attention_dispatch import AttentionBackendName, _HUB_KERNELS_REGISTRY

out = Path(os.environ['H3_FA3_CANARY_OUTPUT'])
out.mkdir(exist_ok=False)
assert torch.cuda.device_count() == 1
assert 'H100' in torch.cuda.get_device_name(0)
_configure_local_flash_attn3()
kernel = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB].kernel_fn
assert callable(kernel)
torch.manual_seed(20260817)
q, k, v = [torch.randn(1, 64, 8, 128, device='cuda', dtype=torch.bfloat16,
                      requires_grad=True) for _ in range(3)]
prediction = kernel(q, k, v, causal=False)
reference = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
torch.testing.assert_close(prediction, reference, atol=0.02, rtol=0.02)
prediction.float().square().mean().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))
torch.cuda.synchronize()
receipt = {'status': 'PASS_REAL_H100_FA3_FORWARD_BACKWARD_LIBRARY_SMOKE',
           'shape': list(prediction.shape), 'dtype': str(prediction.dtype),
           'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
           'kernel_module': kernel.__module__, 'kernel_name': kernel.__name__,
           'max_abs_vs_SDPA': float((prediction.float() - reference.float()).abs().max()),
           'training_model': 'NOT_LOADED', 'H3_workload_parity': 'NOT_RUN',
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
           'GPU_driver_free_bytes': torch.cuda.mem_get_info()[0]}
with (out / 'receipt.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
print(json.dumps(receipt), flush=True)
