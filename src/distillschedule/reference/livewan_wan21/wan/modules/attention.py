"""Maintained CPU FP32 bridge, NOT FlashAttention despite the upstream name.

Calls the statically retained original LiveWan SDPA definitions, never a CUDA,
FlashAttention or BF16 fallback. The original full model imports flash_attention;
that name is only this explicitly admitted bridge in this private namespace.
"""
from contextlib import contextmanager
import math
from numbers import Real
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import sdpa_original


MAX_ATTENTION_ELEMENTS = 16_777_216
MAX_INPUT_ELEMENTS = 8_388_608
_SDPA = F.scaled_dot_product_attention
_SDPA_KERNEL = sdpa_kernel
_MATH = SDPBackend.MATH


def _require(condition, message):
    if not condition:
        raise ValueError("livewan_wan21 CPU attention: " + message)


def _validate_backend():
    _require(F.scaled_dot_product_attention is _SDPA and sdpa_kernel is _SDPA_KERNEL
             and SDPBackend.MATH is _MATH, "math SDPA backend binding changed")
    _require(os.environ.get("WAN_SDPA_BACKEND", "").strip().upper() in ("", "MATH"),
             "WAN_SDPA_BACKEND override is not CPU MATH")
    _require(not torch.cuda.is_initialized(), "initialized CUDA is outside CPU reference contract")
    _require(not torch.is_autocast_enabled("cpu"), "outer CPU autocast is outside all-FP32 contract")


@contextmanager
def math_sdpa_context():
    _validate_backend()
    with sdpa_kernel([SDPBackend.MATH]):
        yield


def _lengths(value, batch, length, name):
    if value is None:
        return
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("livewan_wan21 CPU attention: invalid " + name) from exc
    _require(tensor.device.type == "cpu" and tensor.layout == torch.strided,
             name + " must be dense CPU lengths")
    _require(tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
             and tuple(tensor.shape) == (batch,), name + " must be integer [B]")
    _require(bool(((tensor >= 0) & (tensor <= length)).all()), name + " outside padded sequence bounds")


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.float32, version=None):
    """Exact original SDPA mask/GQA/scales math under explicit FP32 CPU policy."""
    _validate_backend()
    _require(dtype is torch.float32, "dtype must be torch.float32 (no BF16 default)")
    _require(isinstance(dropout_p, Real) and not isinstance(dropout_p, bool) and dropout_p == 0,
             "nonzero/invalid dropout is not admitted")
    _require(deterministic is False and version is None, "FlashAttention version/deterministic overrides are not admitted")
    _require(isinstance(causal, bool), "causal must be bool")
    _require(isinstance(window_size, (tuple, list)) and tuple(window_size) == (-1, -1),
             "local window is not admitted")
    for value, name in ((softmax_scale, "softmax_scale"), (q_scale, "q_scale")):
        _require(value is None or (isinstance(value, Real) and not isinstance(value, bool)
                                  and math.isfinite(value)), name + " must be a finite scalar")
    for value, name in ((q, "q"), (k, "k"), (v, "v")):
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.layout == torch.strided and value.dtype is torch.float32
                 and value.ndim == 4, name + " must be a dense CPU float32 [B,L,N,C] tensor")
        _require(all(size > 0 for size in value.shape) and value.numel() <= MAX_INPUT_ELEMENTS,
                 name + " empty or above CPU input-element budget")
        _require(bool(torch.isfinite(value).all()), name + " must be finite")
    b, lq, nq, c = q.shape
    bk, lk, nk, ck = k.shape
    bv, lv, nv, cv = v.shape
    _require(b == bk == bv and lk == lv and nk == nv and c == ck and c <= 256 and cv <= 256
             and nq % nk == 0, "incompatible batch/sequence/head/value geometry")
    _require(b * nq * lq * lk <= MAX_ATTENTION_ELEMENTS, "CPU attention-element budget exceeded")
    _lengths(q_lens, b, lq, "q_lens")
    _lengths(k_lens, b, lk, "k_lens")
    return sdpa_original.sdpa_attention(q, k, v, q_lens=q_lens, k_lens=k_lens,
        dropout_p=0., softmax_scale=softmax_scale, q_scale=q_scale,
        causal=causal, window_size=window_size, dtype=torch.float32)


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False,
              dtype=torch.float32, fa_version=None):
    return flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
        softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
        window_size=window_size, deterministic=deterministic, dtype=dtype, version=fa_version)
