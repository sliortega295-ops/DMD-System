"""Explicit dense Torch SDPA reference, not the original CUDA dispatcher.

This is exactly the original dispatcher's dense SDPA fallback expression.
NaViT's packed/variable-length convention has no reference implementation here.
"""
import torch


def attn_varlen_func(q, k, v, attention_mask=None):
    if attention_mask is not None:
        raise NotImplementedError("Helios torch-sdpa reference does not support NaViT/varlen attention")
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


def create_navit_attention_masks(*args, **kwargs):
    raise NotImplementedError("Helios torch-sdpa reference does not support NaViT/varlen attention")
