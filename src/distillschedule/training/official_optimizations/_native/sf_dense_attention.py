"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_dense_attention.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import hashlib
import inspect
from pathlib import Path

def require(value, reason):
    if not value:
        raise ValueError(reason)

def make_dispatch(original, fixed, torch, receipt, *, device_type='cuda'):

    def full_length(lens, length):
        if lens is None:
            return True
        return isinstance(lens, torch.Tensor) and lens.device.type == 'cpu' and (lens.dtype in (torch.int32, torch.int64)) and (tuple(lens.shape) == (1,)) and (int(lens.item()) == length)

    def dispatch(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
        allowed = q.ndim == k.ndim == v.ndim == 4 and q.shape[0] == k.shape[0] == v.shape[0] == 1 and (q.device.type == device_type) and (q.device == k.device == v.device) and (q.shape[-1] <= 256) and (q_scale is None) and (dropout_p == 0.0) and (causal is False) and (tuple(window_size) == (-1, -1)) and (deterministic is False) and (version in (None, 2)) and (dtype in (torch.float16, torch.bfloat16)) and full_length(q_lens, q.shape[1]) and full_length(k_lens, k.shape[1])
        if not allowed:
            receipt['native_varlen_fallbacks'] += 1
            return original(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic, dtype=dtype, version=version)
        output_dtype = q.dtype
        half = lambda x: x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
        q, k, v = (half(q), half(k), half(v))
        q, k = (q.to(v.dtype), k.to(v.dtype))
        result = fixed(q=q, k=k, v=v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, window_size=window_size, deterministic=deterministic)
        receipt['fixed_length_calls'] += 1
        receipt['grad_calls' if torch.is_grad_enabled() else 'nograd_calls'] += 1
        key = str((q.shape[1], k.shape[1], q.shape[2], q.shape[3], str(v.dtype), str(output_dtype)))
        receipt['shape_dtype_histogram'][key] = receipt['shape_dtype_histogram'].get(key, 0) + 1
        return result.type(output_dtype)
    return dispatch

class DenseFA2:

    def __init__(self):
        import torch
        from wan.modules import attention, model
        require(attention.FLASH_ATTN_2_AVAILABLE and (not attention.FLASH_ATTN_3_AVAILABLE), 'PINNED_A100_FA2_REQUIRED')
        require(model.flash_attention is attention.flash_attention, 'NATIVE_FLASH_ALIAS_CHANGED')
        self.attention, self.model = (attention, model)
        self.original = attention.flash_attention
        path = Path(attention.__file__).resolve()
        self.record = dict(mechanism='EXISTING_FA2_FIXED_LENGTH_FULL_B1', fixed_length_calls=0, grad_calls=0, nograd_calls=0, native_varlen_fallbacks=0, shape_dtype_histogram={}, native_attention_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fixed_signature=str(inspect.signature(attention.flash_attn.flash_attn_func)), native_cast_rules_preserved=True, dropout=0.0, causal=False, gpu_lens_value_reads=0)
        self.dispatch = make_dispatch(self.original, attention.flash_attn.flash_attn_func, torch, self.record)
        attention.flash_attention = model.flash_attention = self.dispatch

    def receipt(self):
        return dict(self.record)

    def close(self):
        require(self.attention.flash_attention is self.model.flash_attention is self.dispatch, 'CONCURRENT_ATTENTION_MUTATION')
        self.attention.flash_attention = self.model.flash_attention = self.original
